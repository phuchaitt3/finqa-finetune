# train.py
import torch
from functools import partial
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
import config
import sys
import os

# Get the directory of the current script (quick_run/)
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to the FinQA/ directory
# (quick_run/ -> FinQA/)
finqa_root_dir = os.path.dirname(current_script_dir)

# Add the FinQA/ directory to sys.path
sys.path.append(finqa_root_dir)

# Now you can use an absolute import relative to FinQA/
# (assuming FinQA/ is now on sys.path)
from structure.data_utils import (  # NO '..' needed here
    load_raw_data,
    clean_data_for_arrow,
    create_hf_dataset,
)

# --- Define the new preprocessing function for FinGPT ---
def preprocess_function_fingpt(examples, tokenizer, max_length=512):
    """
    Correctly formats the input data from the FinQA dataset for training.
    This function handles the nested structure and joins text lists.
    """
    prompts = []
    # The map function works on batches, so we iterate through the batch
    for i in range(len(examples["id"])):
        # First, get the i-th item from the 'qa' list of dictionaries.
        # Then, get the 'question' key from that dictionary.
        instruction = examples['qa'][i]['question']

        # Join the text lists into a single string.
        pre_text = " ".join(examples["pre_text"][i])
        post_text = " ".join(examples["post_text"][i])
        
        # Join the table data into a simple string format.
        table_rows = [" | ".join(row) for row in examples["table"][i]]
        table_text = "\n".join(table_rows)

        # Combine all context into a single input_text field.
        input_text = f"Pre-text: {pre_text}\n\nTable:\n{table_text}\n\nPost-text: {post_text}"

        # Apply the same logic to get the answer.
        answer = examples['qa'][i]['answer']
        
        # Create the full prompt using the correct fields.
        full_prompt = f"Instruction: {instruction}\nInput: {input_text}\nAnswer: {answer}"
        prompts.append(full_prompt)

    # Tokenize the full prompt
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    # For Causal LM, the labels are the same as the input_ids
    model_inputs["labels"] = model_inputs["input_ids"]

    return model_inputs

def run_training():
    """Load data, preprocess, and train a FinGPT model."""
    # 1. Load and clean data
    train_list, dev_list, _ = load_raw_data(config.TRAIN_FILE, config.DEV_FILE, config.TEST_FILE)
    train_list, dev_list = clean_data_for_arrow([train_list, dev_list])

    # >>> CHANGE: Use a tiny subset of the data for a quick bug-checking run.
    # This is the most important change for a fast test.
    # 16 samples for training and 8 for validation is plenty for a dry run.
    train_subset = train_list[:16]
    dev_subset = dev_list[:8]

    finqa_subset = create_hf_dataset(train_subset, dev_subset, [])

    # 2. Initialize Tokenizer and Model
    base_model_name = "mistralai/Mistral-7B-v0.1"
    fingpt_lora_adapter = "diwakartiwari/mistral-7b-financial-sentiment"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, fingpt_lora_adapter)
    
    print("Model and Tokenizer loaded successfully.")

    # 3. Preprocess and tokenize the dataset
    p_preprocess_function = partial(
        preprocess_function_fingpt,
        tokenizer=tokenizer,
        max_length=config.MAX_INPUT_LENGTH,
    )

    tokenized_datasets = finqa_subset.map(
        p_preprocess_function,
        batched=True,
    )
    print("Data preparation complete.")

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./training_checkpoints_fingpt_test", # Use a separate dir for test checkpoints
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir="./logs_fingpt_test", # Separate logs dir
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        report_to="none",
        fp16=False,
    )
    print("Training arguments defined.")

    # 5. Create and run the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    print("\n--- Starting Model Training ---")
    trainer.train()
    print("--- Training Complete ---")

    # 6. Save the final best model
    trainer.save_model(config.MODEL_SAVE_PATH)
    print(f"Best model adapter saved to {config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    run_training()