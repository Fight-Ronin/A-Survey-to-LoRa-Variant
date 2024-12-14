import os
import math
import json

def compute_accuracy_and_ci(json_file, z=1.96):
    true_count = 0
    total = 0

    with open(json_file, 'r') as f:
        for i, line in enumerate(f, start=1):
            line = line.lstrip('ufeff')
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f'Error decoding JSON at line {i}: {e}')
                raise e
            try:
                evaluation = data.get('evaluation')
                if '<True>' in evaluation:
                    true_count += 1
                total += 1
            except Exception as e:
                continue
    accuracy = true_count / total

    # Compute confidence interval
    epsilon = z * math.sqrt( (accuracy)*(1-accuracy) / total)
    lower_bound = accuracy - epsilon
    upper_bound = accuracy + epsilon

    return accuracy, lower_bound, upper_bound, total

if __name__ == '__main__':
    result_dir = 'evaluation_result'
    GSM8k_files = [pos_json for pos_json in os.listdir(result_dir) if pos_json.endswith('.jsonl') and pos_json.startswith('GSM8k')]
    GSM_plus_files = [pos_json for pos_json in os.listdir(result_dir) if pos_json.endswith('.jsonl') and pos_json.startswith('GSM-plus')]
    print(GSM_plus_files)

    print('GSM8k metrics:')
    for GSM8k_file in GSM8k_files:
        accuracy, lower_bound, upper_bound, n = compute_accuracy_and_ci(result_dir + '/' + GSM8k_file)
        variant_name = GSM8k_file.replace('GSM8k-', '')
        print(f'{variant_name}: Acc: {accuracy * 100:.2f}%, Confidence Interval: ({lower_bound},{upper_bound}), radius: {upper_bound - lower_bound}')

    print('GSM Plus metrics:')
    for GSM_plus_file in GSM_plus_files:
        accuracy, lower_bound, upper_bound, n = compute_accuracy_and_ci(result_dir + '/' + GSM_plus_file)
        variant_name = GSM_plus_file.replace('GSM-plus-', '')
        print(f'{variant_name}: Acc: {accuracy * 100:.2f}%, Confidence Interval: ({lower_bound},{upper_bound}), radius: {upper_bound - lower_bound}')


    # Test area
    # with open('evaluation_result/GSM-8k-zero_shot.jsonl', 'r', encoding = 'utf-8') as f:
    #     print('Open success')
    # with open('evaluation_result/GSM-8k-zero_shot.jsonl', 'r', encoding='utf-8') as f:
    #     for i, line in enumerate(f, start=1):
    #         if i == 501:  # or print only when you catch the exception
    #             print(f"Raw line {i}: {repr(line)}")
    #         line_stripped = line.strip()
    #         if not line_stripped:
    #             continue
    #         try:
    #             data = json.loads(line_stripped)
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding JSON at line {i}: {repr(line_stripped)}")
    #             raise
    #         # ...
