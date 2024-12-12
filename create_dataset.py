import os
import json
import random
from pathlib import Path

def sample_jsonl(input_file, output_file, sample_size=500, seed=42):
    with open(input_file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    random.seed(seed)
    sampled_lines = random.sample(lines, sample_size)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        f.writelines(sampled_lines)
    print(f'Sampled {sample_size} entries from {input_file} to {output_file}')

if __name__ == '__main__':
    directories = ['GSM-plus', 'GSM8k']
    for dir_name in directories:
        input_file = Path(dir_name) / 'valid.jsonl'
        output_file = Path(dir_name) / 'test.jsonl'

        try:
            sample_jsonl(str(input_file), str(output_file))
        except Exception as e:
            print(f'Error processing {input_file}: {e}')