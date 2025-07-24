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
from data_utils import (
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
    # The test set is not used for training or evaluation in this specific script.
    train_list, dev_list, _ = load_raw_data(config.TRAIN_FILE, config.DEV_FILE, config.TEST_FILE)
    train_list, dev_list = clean_data_for_arrow([train_list, dev_list])

    # For demonstration, use a smaller subset
    train_subset = train_list[:800]
    dev_subset = dev_list[:100]

    finqa_subset = create_hf_dataset(train_subset, dev_subset, [])

    # 2. Initialize Tokenizer and Model
    # Define the base model and the FinGPT LoRA adapter
    base_model_name = "mistralai/Mistral-7B-v0.1"
    fingpt_lora_adapter = "diwakartiwari/mistral-7b-financial-sentiment"
    
    # For memory efficiency, load the base model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16 # Used to be bfloat16
    )

    # Load the base model with the quantization config
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto", # automatically distribute the model's layers across your available devices (GPUs).
        trust_remote_code=True, # Developers can upload models with custom layers, custom forward methods, or custom tokenization logic that isn't part of the standard transformers library. Many cutting-edge research models or specialized models (like FinGPT adaptations) might use this.
    )
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    # Set the padding token to be the end-of-sequence token for Causal LM
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare the model for k-bit training and apply the LoRA adapter
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModel.from_pretrained(base_model, fingpt_lora_adapter)
    
    print("Model and Tokenizer loaded successfully.")

    # 3. Preprocess and tokenize the dataset
    # Note: We are using a new preprocessing function designed for FinGPT's instruction format
    p_preprocess_function = partial(
        preprocess_function_fingpt,
        tokenizer=tokenizer,
        max_length=config.MAX_INPUT_LENGTH, # Re-using config, ensure it's appropriate for the model
    )

    # Important: The column names in your dataset must match what preprocess_function_fingpt expects.
    # Adjust 'pre_text' and 'answer' if your column names are different.
    tokenized_datasets = finqa_subset.map(
        p_preprocess_function,
        batched=True,
        # Keep original columns for potential inspection, trainer will remove them if not needed
        # remove_columns=finqa_subset['train'].column_names
    )
    print("Data preparation complete.")

    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./training_checkpoints_fingpt", # Intermediate checkpoints
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=4, # Often needed for larger models
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir="./logs_fingpt",
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        report_to="none",
        fp16=True, # Use mixed precision for performance
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
    # The trainer automatically handles saving the LoRA adapter, not the whole model
    trainer.save_model(config.MODEL_SAVE_PATH)
    print(f"Best model adapter saved to {config.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    run_training()