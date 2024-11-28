import json
import os
from datasets import load_dataset
import ollama

def zero_shot_gsm8k(file_name):
    ds = load_dataset("openai/gsm8k", "main")
    model = 'llama3:8b'
    f = open(file_name, 'w')

    new_data = []
    i = 0
    print(f"Total number of questions: {len(ds['test'])}")
    for q in ds['test']:
        solution = q['answer']
        response = ollama.generate(model=model, prompt=q['question'])
        generated_answer = response['response']
        print(f"question {i} finished")
        i += 1
        prompt = f''' 
        You are given a predicted answer and a ground truth solution, determinant whether the predicted answer match the ground truth answer. End your response with <True> or <False>
        predicted answer: {generated_answer}
        ground truth solution:
        {solution}

        '''
        response = ollama.generate(model=model, prompt=prompt)
        evaluation = response['response']
        new_data = {'question': q['question'], 'solution': q['answer'], 'generated_answer': generated_answer, 'evaluation': evaluation}
        f.write(json.dumps(new_data) + '\n')


def zero_shot_gsm_plus(saved_file):
    ds = load_dataset("qintongli/GSM-Plus")
    model = 'llama3:8b'
    f = open(saved_file, 'w')

    new_data = []
    i = 0
    print(f"Total number of questions: {len(ds['testmini'])}")
    for q in ds['testmini']:
        solution = q['answer']
        response = ollama.generate(model=model, prompt=q['question'])
        generated_answer = response['response']
        print(f"question {i} finished")
        i += 1
        prompt = f''' 
        You are given a predicted answer and a ground truth solution, determinant whether the predicted answer match the ground truth answer. End your response with <True> or <False>
        predicted answer: {generated_answer}
        ground truth solution:
        {solution}

        '''
        response = ollama.generate(model=model, prompt=prompt)
        evaluation = response['response']
        new_data = {'question': q['question'], 'solution': q['answer'], 'generated_answer': generated_answer, 'evaluation': evaluation}
        f.write(json.dumps(new_data) + '\n')



if __name__ == '__main__':
    zero_shot_gsm8k('zero_shot_gsm8k.jsonl')
        




