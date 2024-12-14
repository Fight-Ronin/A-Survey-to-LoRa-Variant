import os
import json
import torch
from datasets import load_dataset
from peft.utils import CONFIG_NAME
from peft import PeftModel, AdaLoraConfig, LoraConfig, get_peft_model
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


def get_trainable_params(model_dir, checkpoint_dir):
    try:
        token = get_token_from_file('access_token.txt')

        # Load and clean the PEFT config
        config_file = os.path.join(model_dir, CONFIG_NAME)
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        # config_dict = clean_config_dict(config_dict)

        # Reload tokenizer and base_model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=token
        )

        # Initialize various config from the cleaned dictionary
        if 'adalora' in model_dir:
            peft_config = AdaLoraConfig(
                init_r=128,
                target_r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                beta1=0.85,
                beta2=0.85,
                tinit=0,
                tfinal=0,
                deltaT=10,
                task_type='CAUSAL_LM',
            )
            model = get_peft_model(base_model, peft_config)
        else:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                inference_mode=False,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = PeftModel.from_pretrained(base_model, os.path.join(model_dir, checkpoint_dir), config=peft_config)

        # Count trainable parameters
        trainable_params, total_params = 0,0
        for _, params in model.named_parameters():
            total_params += params.numel()
            if params.requires_grad:
                trainable_params += params.numel()

        # Report and calculate percentage
        trainable_percentage = trainable_params / total_params
        print(f'Model {model_dir}: num of trainable params: {trainable_params}, '
              f'total num of params: {total_params}, '
              f'percentage of params: {trainable_percentage}')
        return trainable_params, total_params

    except Exception as e:
        print(f'\nError processing {model_dir}: Error type: {type(e).__name__}, Error message: {str(e)}')
        return None


# Get finetuned model trainable parameter
if __name__ == '__main__':
    dir_list = ['finetuned_llama3_8b_adalora_gsm-plus', 'finetuned_llama3_8b_lorafa_gsm-plus']
    checkpoint_list = ['checkpoint-247', 'checkpoint-247']
    results = []
    for finetuned_dir, checkpoint_dir in zip(dir_list, checkpoint_list):
        trainable, all = get_trainable_params(finetuned_dir, checkpoint_dir)
        results.append({
            'model': finetuned_dir,
            'trainable': trainable,
            'all': all,
            'percentage': trainable / all,
        })