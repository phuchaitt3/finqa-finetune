import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

# Assuming 'config.py' is in the same directory or accessible via PYTHONPATH
import config
from structure.data_utils import (
    load_raw_data,
    clean_data_for_arrow,
    create_hf_dataset,
)

# --- Re-use the preprocessing logic from train.py ---
def preprocess_function_for_length_analysis(examples, tokenizer):
    """
    Formats the input data into the full prompt string for length analysis.
    Does NOT tokenize, just returns the prompt strings.
    """
    prompts = []
    for i in range(len(examples["id"])):
        instruction = examples['qa'][i]['question']
        pre_text = " ".join(examples["pre_text"][i])
        post_text = " ".join(examples["post_text"][i])
        
        table_rows = [" | ".join(row) for row in examples["table"][i]]
        table_text = "\n".join(table_rows)

        input_text = f"Pre-text: {pre_text}\n\nTable:\n{table_text}\n\nPost-text: {post_text}"
        answer = examples['qa'][i]['answer']
        
        full_prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: {answer}"
        prompts.append(full_prompt)
    return {"full_prompt_text": prompts} # Return as a dictionary for map to handle

def analyze_dataset_lengths():
    print("--- Starting Dataset Length Analysis ---")

    # 1. Load and clean data (same as train.py)
    train_list, dev_list, _ = load_raw_data(config.TRAIN_FILE, config.DEV_FILE, config.TEST_FILE)
    train_list, dev_list = clean_data_for_arrow([train_list, dev_list])

    # Using the same subsets as your train.py for consistency in analysis
    train_subset = train_list[:800]
    dev_subset = dev_list[:100]

    finqa_subset = create_hf_dataset(train_subset, dev_subset, [])

    # 2. Initialize Tokenizer (same as train.py)
    base_model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Essential for consistent tokenization

    print(f"Tokenizer loaded for '{base_model_name}'.")

    # 3. Apply the modified preprocessing function to get raw prompt strings
    # *** THIS IS THE FIX ***
    # The `remove_columns` argument is removed because the map function already replaces
    # the old columns with the new one returned by our function.
    formatted_datasets = finqa_subset.map(
        partial(preprocess_function_for_length_analysis, tokenizer=tokenizer),
        batched=True,
    )
    print("Prompts formatted for analysis.")

    # 4. Collect and tokenize lengths
    all_lengths = []
    for split_name in formatted_datasets.keys():
        print(f"Analyzing {split_name} split...")
        for i, example in enumerate(formatted_datasets[split_name]):
            # Get the full prompt string
            full_prompt_text = example["full_prompt_text"]
            
            # Tokenize it to get the length without truncation/padding for analysis
            # We don't use max_length or padding/truncation here because we want the *true* length
            tokenized_ids = tokenizer.encode(full_prompt_text, add_special_tokens=True) # Add special tokens to reflect actual input to model
            all_lengths.append(len(tokenized_ids))
        print(f"Finished {split_name} split. Total examples analyzed so far: {len(all_lengths)}")

    if not all_lengths:
        print("No data found for length analysis.")
        return

    # 5. Analyze and visualize lengths
    min_len = np.min(all_lengths)
    max_len = np.max(all_lengths)
    avg_len = np.mean(all_lengths)
    median_len = np.median(all_lengths)
    p90_len = np.percentile(all_lengths, 90)
    p95_len = np.percentile(all_lengths, 95)
    p99_len = np.percentile(all_lengths, 99)

    print("\n--- Length Statistics (in tokens) ---")
    print(f"Min Length: {min_len}")
    print(f"Max Length: {max_len}")
    print(f"Average Length: {avg_len:.2f}")
    print(f"Median Length: {median_len:.2f}")
    print(f"90th Percentile: {p90_len:.2f}")
    print(f"95th Percentile: {p95_len:.2f}")
    print(f"99th Percentile: {p99_len:.2f}")

    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=p90_len, color='r', linestyle='--', label=f'90th Percentile ({int(p90_len)} tokens)')
    plt.axvline(x=p95_len, color='g', linestyle='--', label=f'95th Percentile ({int(p95_len)} tokens)')
    plt.axvline(x=p99_len, color='b', linestyle='--', label=f'99th Percentile ({int(p99_len)} tokens)')
    plt.title('Distribution of Tokenized Prompt Lengths')
    plt.xlabel('Token Length')
    plt.ylabel('Number of Prompts')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    # Save the plot instead of just showing it, which is useful on servers
    plot_filename = "prompt_length_distribution.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

    # Recommendation
    print("\n--- Recommendation for config.MAX_INPUT_LENGTH ---")
    print(f"Consider setting config.MAX_INPUT_LENGTH to around:")
    print(f"- {int(p90_len)} to {int(p95_len)} if you are comfortable truncating the longest 5-10% of examples.")
    print(f"- {int(p99_len)} or higher if you want to capture almost all examples without truncation (memory permitting).")
    print(f"Remember to choose a value that aligns with your model's maximum context window and GPU memory capacity.")

if __name__ == '__main__':
    analyze_dataset_lengths()