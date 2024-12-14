import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
from typing import Optional, Dict, Tuple
from tqdm import tqdm

def print_model_parameters(model):
    """Print detailed breakdown of model parameters"""
    trainable_params = 0
    frozen_params = 0
    
    # Print each parameter's details
    print("\nParameter details:")
    print("-" * 80)
    print(f"{'Parameter Name':<50} {'Size':<15} {'Status'}")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            status = "TRAINABLE"
            trainable_params += num_params
        else:
            status = "FROZEN"
            frozen_params += num_params
            
        print(f"{name:<50} {num_params:<15,} {status}")
    
    total_params = trainable_params + frozen_params
    
    print("\nSummary:")
    print("-" * 40)
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    print(f"Frozen parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.4f}%)")
    print(f"Total parameters:     {total_params:,}")


def get_token_from_file(file_path: str) -> str:
    """Read access token from file"""
    with open(file_path, 'r') as f:
        return f.read().strip()

def load_local_dataset(directory: str):
    """Load dataset from local jsonl files"""
    data_files = {
        'train': f"{directory}/train.jsonl",
        'validation': f"{directory}/valid.jsonl"
    }
    return load_dataset('json', data_files=data_files)

def create_custom_lora_config(
    r: int = 8,
    alpha: int = 32,
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]
) -> LoraConfig:
    """Create LoRA-FA configuration"""
    if r <= 0 or alpha <= 0:
        raise ValueError("r and alpha must be positive")
    
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM"
    )

def verify_initialization(model: torch.nn.Module):
    """Verify model initialization meets LoRA-FA requirements"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                # Verify A matrix initialization
                mean = param.mean().item()
                std = param.std().item()
                assert abs(mean) < 0.1, f"A matrix mean {mean} too far from 0"
                assert abs(std - 0.02) < 0.01, f"A matrix std {std} too far from 0.02"
                assert not param.requires_grad, "A matrix should be frozen"
            elif 'lora_B' in name:
                # Verify B matrix initialization
                assert torch.all(param == 0), "B matrix should be initialized to 0"
                assert param.requires_grad, "B matrix should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

def prepare_model_for_lora_fa(
    model_name: str,
    r: int = 8,
    alpha: int = 32,
    token: Optional[str] = None
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """Prepare model with LoRA-FA setup"""
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        token=token
    )
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    # Apply LoRA-FA configuration
    config = create_custom_lora_config(r=r, alpha=alpha)
    model = get_peft_model(model, config)
    
    # Initialize parameters
    for name, param in model.named_parameters():
        if 'lora_A' in name:
            torch.nn.init.normal_(param, mean=0, std=0.02)
            param.requires_grad = False
        elif 'lora_B' in name:
            param.data.zero_()
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model, tokenizer

def tokenize_function(examples: Dict, tokenizer: AutoTokenizer) -> Dict:
    """Process and tokenize examples"""
    input_texts = []
    target_texts = []
    
    for text in examples["text"]:
        if "Answer:" in text:
            parts = text.split("Answer:", 1)
            question_part = parts[0].strip()
            question_formatted = question_part + "\nA:"
            answer_part = parts[1].strip()
            
            input_texts.append(question_formatted)
            target_texts.append(answer_part)
        else:
            input_texts.append(text)
            target_texts.append("")
    
    return tokenizer(
        input_texts,
        text_target=target_texts,
        max_length=512,
        truncation="longest_first",
        padding="max_length"
    )

class LoRAFATrainer(Trainer):
    """Custom trainer for LoRA-FA with gradient verification"""
    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        
        # Verify gradients
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                assert param.grad is None, f"{name} should not have gradients"
            elif 'lora_B' in name:
                assert param.grad is not None, f"{name} should have gradients"
        
        return loss

def train(
    model_name: str,
    data_dir: str,
    output_dir: str,
    token_path: Optional[str] = None,
    r: int = 8,
    alpha: int = 32,
) -> str:
    """Main training function"""
    # Get token if provided
    token = get_token_from_file(token_path) if token_path else None
    
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_for_lora_fa(model_name, r, alpha, token)
    
    # Verify initialization
    verify_initialization(model)
    
    # Load dataset
    ds = load_local_dataset(data_dir)
    train_ds = ds['train']
    val_ds = ds['validation']
    
    # Print dataset info
    print(f"Training examples: {len(train_ds)}")
    print(f"Validation examples: {len(val_ds)}")
    
    # Process datasets
    print("Processing datasets...")
    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Processing train dataset"
    )
    val_ds = val_ds.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
        desc="Processing validation dataset"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        evaluation_strategy='steps',
        logging_steps=100,
        num_train_epochs=1,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to='none',
        fp16=True,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = LoRAFATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer, padding='longest'),
    )
    
    # Train and save
    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training complete. Model saved to {output_dir}")
    return output_dir

def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate answer for a given prompt"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    OUTPUT_DIR = "finetuned_llama3_8b_lorafa_gsm-plus"
    DATA_DIR = "GSM-plus"
    TOKEN_PATH = "access_token.txt"
    
    try:
        train(
            MODEL_NAME,
            DATA_DIR,
            OUTPUT_DIR,
            TOKEN_PATH,
            r=8,
            alpha=32
        )
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
