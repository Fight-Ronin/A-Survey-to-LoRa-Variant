import json
import os
import ollama
from datasets import load_dataset
from utils import load_local_dataset


def zero_shot_evaluation(data_dir, output_file):
    ds = load_local_dataset(data_dir)
    eval_ds = ds['test']
    f = open(output_file, 'w', encoding='utf-8')

    i, true_count = 0, 0
    new_data = []
    model = 'llama3:8b'
    print(f"Total number of questions: {len(eval_ds)}")
    for example in eval_ds:
        text = example['text']
        if 'Answer' in text:
            parts = text.split('Answer:', 1)
            question_part = parts[0].strip()
            answer_part = parts[1].strip()
            question = question_part + "\nA:"
            # Get zero-shot response
            zero_shot_answer = ollama.generate(model = model, prompt=question)
            generated_answer = zero_shot_answer['response']
            print(f"question {i} finished")
            prompt = f'''
            You are given a predicted answer and a ground truth solution, determinant whether the predicted answer match the ground truth answer. End your response with <True> or <False>
            predicted answer: {generated_answer}
            ground truth solution:
            {answer_part}
            '''
            # Compare zero-shot and ground truth
            response = ollama.generate(model=model, prompt=prompt)
            evaluation = response['response']
            if '<True>' in evaluation:
                true_count += 1
            new_data = {
                'text': text,
                'generated_answer': generated_answer,
                'evaluation': evaluation,
            }
            f.write(json.dumps(new_data) + '\n')
            i += 1
            if i % 5 == 0:
                print(f'question {i} finished, {true_count} / {i} correct, Acc: {true_count / i}')



if __name__ == '__main__':
    zero_shot_evaluation('GSM8k', 'evaluation_result/GSM8k-zero_shot.jsonl')





