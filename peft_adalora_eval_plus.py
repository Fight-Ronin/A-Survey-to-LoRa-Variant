import os
import json
import torch
import ollama
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding


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


def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_lora(model_dir, data_dir, output_file):
    # Reload finetuned model and adapter
    token = get_token_from_file('access_token.txt')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16,
                                                      device_map='auto', token=token)
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, model_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eso_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    ds = load_local_dataset(data_dir)
    eval_ds = ds['test']
    f = open(output_file, 'w', encoding='utf-8')

    i=0
    print(f"Total number of questions: {len(eval_ds)}")
    true_count = 0
    for example in eval_ds:
        text = example['text']
        if "Answer:" in text:
            parts = text.split('Answer:', 1)
            question_part = parts[0].strip()
            answer_part = parts[1].strip()
            question = question_part + "\nA:"
            generated_answer_full = generate_answer(model, tokenizer, question)

            ollama_model = 'llama3:8b'
            prompt = f''' 
                    You are given a predicted answer and a ground truth solution, determinant whether the predicted answer match the ground truth answer. End your response with <True> or <False>
                    predicted answer: {generated_answer_full}
                    ground truth solution:
                    {answer_part}
                    '''
            response = ollama.generate(model=ollama_model, prompt=prompt)
            evaluation = response['response']
        else:
            generated_answer_full = generate_answer(model, tokenizer, text)
            evaluation = '<False>'

        if '<True>' in evaluation:
            true_count += 1
        new_data = {
            'text': text,
            'generated_answer': generated_answer_full,
            'evaluation': evaluation,
        }
        f.write(json.dumps(new_data) + '\n')
        i += 1
        if i % 5 == 0:
            print(f'question {i} finished, {true_count} / {i} correct, Acc: {true_count / i}')


if __name__ == '__main__':
    pretrained_dir = 'finetuned_llama3_8b_adalora_gsm-plus_v2'
    evaluate_lora(pretrained_dir, '/home/ubuntu/llama3/A-Survey-to-LoRa-Variant/GSM-plus', 'evaluation_result/GSM-plus-adalora-v2.jsonl')



































