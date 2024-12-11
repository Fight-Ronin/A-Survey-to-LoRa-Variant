import os
import json
import torch
import ollama
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, AdaLoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding


def load_local_dataset(directory):
    data_files = {
        'train': os.path.join(directory, 'train.jsonl'),
        'validation': os.path.join(directory, 'valid.jsonl')
    }
    dataset = load_dataset('json', data_files=data_files)
    return dataset


def get_token_from_file(file_path):
    with open(file_path, 'r') as f:
        token = f.read().strip()
    return token


def finetune_lora(model_dir, data_dir, output_dir, init_r=12, target_r=4, lora_alpha=32, lora_dropout=0.1,
                  tinit=False, deltaT=10, beta1=0.9, beta2=0.999):
    # Load Dataset
    ds = load_local_dataset(data_dir)
    train_ds = ds['train']
    val_ds = ds['validation']

    # Load Tokenizer and Model
    token = get_token_from_file('access_token.txt')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='auto', token=token)
    # Introduce padding tokens
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    peft_config = AdaLoraConfig(
        init_r = init_r,
        target_r = target_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        beta1 = beta1,
        beta2 = beta2,
        tinit=tinit,
        deltaT=deltaT,
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, peft_config)

    # Preprocess data
    def tokenize_function(examples):
        input_texts = []
        target_texts = []
        for t in examples["text"]:
            if "Answer:" in t:
                parts = t.split("Answer:", 1)
                question_part = parts[0].strip()
                # Include 'A:' or some marker to help model understand it should produce the answer
                question_formatted = question_part + "\nA:"
                answer_part = parts[1].strip()

                input_texts.append(question_formatted)
                target_texts.append(answer_part)
            else:
                # If formatting isn't as expected, you may need a different parsing strategy.
                # As a fallback, treat the entire text as input and target.
                # This might not be optimal depending on your dataset format.
                input_texts.append(t)
                target_texts.append("")

        model_inputs = tokenizer(input_texts, text_target=target_texts, max_length=512, truncation=True, padding='max_length')
        return model_inputs

    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=val_ds.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy='steps',
        logging_steps=100,
        num_train_epochs=1,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to='none',
        fp16=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding='longest')
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_gsm8k(model_dir, output_file):
    token = get_token_from_file('access_token.txt')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='auto', token=token)

    ds = load_dataset('openai/gsm8k', 'main')
    f = open(output_file, 'w', encoding='utf-8')

    i=0
    print(f"Total number of questions: {len(ds['test'])}")
    for q in ds['test']:
        question = 'Q: ' + q['question'] + '\nA:'
        solution = q['answer']
        generated_answer_full = generate_answer(model, tokenizer, question)
        generated_answer = generated_answer_full.replace(question, '').strip()

        # If we want to model to judge correctness
        prompt = f''' 
        You are given a predicted answer and a ground truth solution, determinant whether the predicted answer match the ground truth answer. End your response with <True> or <False>
        predicted answer: {generated_answer}
        ground truth solution:
        {solution}
        '''

        # Direct comparison
        response = ollama.generate(model=model, prompt=prompt)
        evaluation = response['response']
        # Write result to file
        new_data = {'question': q['question'], 'solution': solution, 'generated_answer': generated_answer, 'evaluation': evaluation}
        f.write(json.dumps(new_data) + '\n')
        i += 1
        print(f'question {i} finished')
    f.close()


def evaluate_gsm_plus(model_dir, output_file):
    token = get_token_from_file('access_token.txt')
    tokenizer = AutoTokenizer.from_pretrained(model_dir, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map='auto', token=token)

    ds = load_dataset('qintongli/GSM-Plus')

    f = open(output_file, 'w', encoding='utf-8')
    i=0
    print(f"Total number of questions: {len(ds['testmini'])}")
    for q in ds['testmini']:
        question = 'Q: ' + q['question'] + '\nA:'
        solution = q['answer']
        generated_answer_full = generate_answer(model, tokenizer, question)
        generated_answer = generated_answer_full.replace(question, '').strip()

        evaluation = "<True>" if generated_answer.strip() == solution.strip() else "<False>"
        new_data = {'question': q['question'], 'solution': solution, 'generated_answer': generated_answer, 'evaluation': evaluation}
        f.write(json.dumps(new_data) + '\n')
        i += 1
        print(f'question {i} finished')
    f.close()


if __name__ == '__main__':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    output_dir = 'finetuned_llama3_8b_adalora'

    # Finetune and evaluate llama3 8B on gsm8k
    finetune_lora(model_name, "gsm8k", output_dir)
    evaluate_gsm8k(output_dir, 'gsm8k_evaluation.jsonl')

    # Finetine and evaluate llama3 8B on gsm-plus
    #finetune_lora(model_name, "gsm-plus", output_dir)
    #evaluate_gsm_plus(output_dir, 'gsm_plus_evaluation.jsonl')



































