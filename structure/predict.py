# predict.py
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import config
from structure.data_utils import load_raw_data, correct_program_tokenization

def run_predictions():
    """Load a trained model and generate predictions on the test set."""
    # 1. Load Tokenizer and the fine-tuned Model
    print(f"Loading tokenizer from '{config.MODEL_NAME}'...")
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)

    print(f"Loading fine-tuned model from '{config.MODEL_SAVE_PATH}'...")
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Error: Model not found at {config.MODEL_SAVE_PATH}. Please run train.py first.")
        return

    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_SAVE_PATH).to(config.DEVICE)
    model.eval() # Set model to evaluation mode

    # 2. Load the original test data (no cleaning needed for prediction input)
    _, _, test_list = load_raw_data(config.TRAIN_FILE, config.DEV_FILE, config.TEST_FILE)

    # Use a smaller subset for demonstration
    test_subset = test_list[:100]

    # 3. Generate predictions
    print(f"\n--- Generating predictions for {len(test_subset)} test samples ---")
    predictions_for_eval = []
    with torch.no_grad():
        for item in tqdm(test_subset, desc="Predicting"):
            question = item['qa']['question']
            pre_text = " ".join(item['pre_text'])
            post_text = " ".join(item['post_text'])
            table_data = item['table']
            table_str = pd.DataFrame(table_data[1:], columns=table_data[0]).to_string(index=False) if table_data else ""
            input_text = f"finqa: question: {question} pre_text: {pre_text} table: {table_str} post_text: {post_text}"

            input_ids = tokenizer(input_text, return_tensors="pt", max_length=config.MAX_INPUT_LENGTH, truncation=True).input_ids.to(config.DEVICE)
            outputs = model.generate(input_ids, max_length=config.MAX_TARGET_LENGTH)
            predicted_program_string = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            tokenized_program_list = correct_program_tokenization(predicted_program_string)
            
            prediction_entry = {
                "id": item["id"],
                "predicted": tokenized_program_list
            }
            predictions_for_eval.append(prediction_entry)

    # 4. Save predictions to file
    with open(config.PREDICTIONS_FILE, 'w') as f:
        json.dump(predictions_for_eval, f, indent=4)
    print(f"'{config.PREDICTIONS_FILE}' created successfully.")

if __name__ == '__main__':
    run_predictions()