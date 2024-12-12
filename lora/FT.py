# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import math
import time
from pathlib import Path
import torch
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils as lora_utils
from mlx.utils import tree_flatten

# Removed: from models import LoRALinear (no longer needed for full fine-tuning)


def build_parser():
    parser = argparse.ArgumentParser(description="Full model fine-tuning.")
    parser.add_argument("--model", default="mlx_model", help="Model path or repo.")
    parser.add_argument("--max-tokens", "-m", type=int, default=100, help="Max tokens.")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Generation prompt.")
    parser.add_argument("--train", action="store_true", help="Enable training.")
    parser.add_argument("--add-eos-token", type=int, default=1, help="Add EOS token.")
    parser.add_argument("--data", type=str, default="data/", help="Dataset directory.")
    parser.add_argument("--adapter-file", type=str, default="adapters.npz", help="Adapter file.")
    # Removed: --lora-layers, as we are no longer doing LoRA
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations.")
    parser.add_argument("--val-batches", type=int, default=25, help="Validation batches.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--steps-per-report", type=int, default=10, help="Steps per report.")
    parser.add_argument("--steps-per-eval", type=int, default=200, help="Steps per evaluation.")
    # Removed adapter file arguments (resume_adapter_file, adapter_file)
    parser.add_argument("--save-every", type=int, default=100, help="Save every N iterations.")
    parser.add_argument("--test", action="store_true", help="Enable testing.")
    parser.add_argument("--test-batches", type=int, default=500, help="Test batches.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load(args):
    def load_and_check(name):
        print(f"Loading {args.data} set")
        dataset_path = Path(args.data) / f"{name}.jsonl"
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid", "test")
    train, valid, test = (load_and_check(n) for n in names)

    if args.train and (train is None or len(train) == 0):
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and (valid is None or len(valid) == 0):
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and (test is None or len(test) == 0):
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    # print(f"logits is {logits}")
    # print(f"input is {inputs}")

    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            if max(lengths) > 2048:
                print(
                    "[WARNING] Some sequences are longer than 2048 tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(model, dataset, loss_fn, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for it, batch in zip(
        index_iterator,
        iterate_batches(dataset, tokenizer, batch_size),
    ):
        losses, toks = loss_fn(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train(model, train_set, val_set, optimizer, loss_fn, tokenizer, args):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(args.iters),
        iterate_batches(train_set, tokenizer, args.batch_size, train=True),
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        print(f"grad is {grad}")
        # Model update
        
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # for name, param in tree_flatten(model.parameters()):
        #     grad = optimizer.state.get(param, {}).get("grad", None)
        #     if grad is not None:
        #         print(f"{name}: Gradient max={grad.max()}, min={grad.min()}")
        #     else:
        #         print(f"{name}: No gradient computed (likely not trainable).")


        # Report training loss if needed
        if (it + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)
            stop = time.perf_counter()
            print(
                f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % args.steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss_fn, tokenizer, args.batch_size, args.val_batches
            )
            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )
            start = time.perf_counter()
        if (it + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file, **dict(tree_flatten(model.trainable_parameters()))
            )
            print(f"Iter {it + 1}: Saved adapter weights to {args.adapter_file}.")

def generate(model, prompt, tokenizer):
    print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    generated_text = ""

    for token, n in zip(
        lora_utils.generate(prompt, model, 0.8),  # Replace with args.temp if needed
        range(200),  # Replace with args.max_tokens if needed
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            generated_text += s[skip:-1]
            skip = len(s) - 1

    remaining_text = tokenizer.decode(tokens)[skip:]
    print(remaining_text, flush=True)
    generated_text += remaining_text

    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return ""

    return generated_text


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Building tokenizer_config
    tokenizer_config = {}
    if args.train:
        print("loading tokenizer")
        tokenizer_config["add_eos_token"] = bool(args.add_eos_token)

    print("Loading pretrained model")
    
    model, tokenizer, _ = lora_utils.load(args.model, tokenizer_config)
    # Ensure all parameters are trainable for full fine-tuning:
    model.train()
    model.unfreeze()

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = load(args)

    if args.train:
        print("Training")

        opt = optim.Adam(learning_rate=args.learning_rate)
        train(model, train_set, valid_set, opt, loss, tokenizer, args)

    if args.test:
        print("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            args.batch_size,
            num_batches=args.test_batches,
        )
        test_ppl = math.exp(test_loss)
        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if args.prompt is not None:
        print("Generating")
        generate(model, args.prompt, tokenizer)

# Example usage:
# python finetune.py --model "/path/to/model" --train --iters 1000
