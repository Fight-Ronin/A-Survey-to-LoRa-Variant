



if __name__ == '__main__':
    with open('evaluation_result/GSM-8k-adalora.jsonl', 'r') as f:
        lines = f.readlines()
    print(len(lines))
    f.close()