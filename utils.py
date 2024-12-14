import os
import json
import torch
from datasets import load_dataset
from peft.utils import CONFIG_NAME
from peft import PeftModel, PeftConfig, AdaLoraConfig, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_local_dataset(directory):
    data_files = {
        'train': os.path.join(directory, 'train.jsonl'),
        'validation': os.path.join(directory, 'valid.jsonl'),
        'test': os.path.join(directory, 'test.jsonl'),
    }
    dataset = load_dataset('json', data_files=data_files)
    return dataset


def get_token_from_file(file_path):
    with open(file_path, 'r') as f:
        token = f.read().strip()
    return token


def clean_config_dict(config):
    keys_to_remove = ['eva_config', 'alpha_pattern', 'exclude_modules', 'lora_bias', 'bias']
    for key in keys_to_remove:
        if key in config:
            del config[key]
    return config


def get_trainable_params(model_dir):
    try:
        token = get_token_from_file('access_token.txt')

        # Load and clean the PEFT config
        config_file = os.path.join(model_dir, CONFIG_NAME)
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        config_dict = clean_config_dict(config_dict)

        # Reload tokenizer and base_model
        tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=token
        )
        base_model.resize_token_embeddings(len(tokenizer))

        # Initialize various config from the cleaned dictionary
        if 'beta1' in config_dict:
            config = AdaLoraConfig(**config_dict)
        else:
            config = LoraConfig(**config_dict)

        # Load the model with the cleaned config
        model = PeftModel.from_pretrained(base_model, model_dir, config=config)

        # Count trainable parameters
        all_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Report and calculate percentage
        trainable_percentage = trainable_params / all_params
        print(f'Model {model_dir}: num of trainable params: {trainable_params}, '
              f'total num of params: {all_params}, '
              f'percentage of params: {trainable_percentage}')
        return trainable_params, all_params

    except Exception as e:
        print(f'\nError processing {model_dir}: Error type: {type(e).__name__}, Error message: {str(e)}')
        return None


# Get finetuned model trainable parameter
if __name__ == '__main__':
    dir_list = ['finetuned_llama3_8b_adalora_gsm-plus', 'finetuned_llama3_8b_adalora_gsm8k', 'finetuned_llama3_8b_lorafa_gsm-plus', 'finetuned_llama3_8b_lorafa_gsm8k']
    results = []
    for finetuned_dir in dir_list:
        trainable, all = get_trainable_params(finetuned_dir)
        results.append({
            'model': finetuned_dir,
            'trainable': trainable,
            'all': all,
            'percentage': trainable / all,
        })