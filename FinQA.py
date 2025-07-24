from datasets import load_dataset
import pandas as pd

# Load the FinQA dataset from Hugging Face
try:
    # You can choose from different available versions, e.g., "Aiera/finqa-verified"
    dataset = load_dataset("Aiera/finqa-verified")
    print("Successfully loaded the FinQA dataset from Hugging Face.")
    
    # The dataset is often split into 'train' and 'test'
    train_dataset = dataset['train']
    
    # Print the number of examples in the training split
    print(f"Number of examples in the training set: {len(train_dataset)}")

    # Access and print the first example
    print("\n--- First Example ---")
    first_example = train_dataset[0]
    print(first_example)

    # Extract and display specific parts of the first example
    print("\n--- Parsed First Example ---")
    print(f"Question: {first_example['question']}")
    print(f"Answer: {first_example['answer']}")
    
    # Display the pre-text, post-text, and table
    print(f"\nPre-text: {first_example['pre_text']}")
    
    # Display the table data using pandas
    print("\n--- Table Data ---")
    table_data = first_example.get("table")
    if table_data:
        # The table format might differ slightly in Hugging Face versions
        # This is a general approach; you might need to adapt it
        try:
            # Assuming the table is a list of lists with a header
            header = table_data[0]
            rows = table_data[1:]
            df = pd.DataFrame(rows, columns=header)
            print(df.to_string())
        except Exception as e:
            print(f"Could not format table: {e}")
            print(table_data)

    print(f"\nPost-text: {first_example['post_text']}")


except Exception as e:
    print(f"Error loading dataset from Hugging Face: {e}")