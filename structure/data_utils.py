# src/data_utils.py
import json
import pandas as pd
from datasets import Dataset, DatasetDict

def load_raw_data(train_path, dev_path, test_path):
    """Loads the raw JSON data from files."""
    print("Step 1: Loading local JSON files...")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(dev_path, 'r') as f:
        dev_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    print("Loading complete.")
    return train_data, dev_data, test_data

def clean_data_for_arrow(datasets):
    """Converts all 'qa.exe_ans' values to strings to prevent Arrow type errors."""
    print("Applying fix: Converting 'qa.exe_ans' to string for dataset compatibility...")
    for dataset in datasets:
        for item in dataset:
            if 'qa' in item and 'exe_ans' in item['qa']:
                item['qa']['exe_ans'] = str(item['qa']['exe_ans'])
    print("Data cleaning complete.")
    return datasets

def create_hf_dataset(train_data, dev_data, test_data):
    """Converts lists of dictionaries to a Hugging Face DatasetDict."""
    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)
    test_dataset = Dataset.from_list(test_data)
    return DatasetDict({
        'train': train_dataset,
        'validation': dev_dataset,
        'test': test_dataset
    })

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """Prepares the FinQA dataset for a T5 model."""
    inputs, targets = [], []
    for i in range(len(examples['id'])):
        question = examples['qa'][i]['question']
        pre_text = " ".join(examples['pre_text'][i])
        post_text = " ".join(examples['post_text'][i])
        
        table_data = examples['table'][i]
        table_str = pd.DataFrame(table_data[1:], columns=table_data[0]).to_string(index=False) if table_data else ""

        input_text = f"finqa: question: {question} pre_text: {pre_text} table: {table_str} post_text: {post_text}"
        inputs.append(input_text)
        
        program = examples['qa'][i]['program']
        targets.append(program)
        
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def correct_program_tokenization(original_program):
    """Official program tokenization logic from evaluate.py."""
    if not isinstance(original_program, str): return ['EOF']
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '': program.append(cur_tok)
                cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '': program.append(cur_tok)
    program.append('EOF')
    return program