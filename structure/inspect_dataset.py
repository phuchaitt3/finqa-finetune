import config
from structure.data_utils import (
    load_raw_data,
    clean_data_for_arrow,
    create_hf_dataset,
)
import pprint

# Define the name for the output report file
REPORT_FILENAME = "inspection_report.md"

def inspect_and_save_report():
    """
    This script loads the FinQA dataset, inspects its structure, and
    saves the findings into a formatted Markdown file.
    """
    print("--- Starting Dataset Inspection ---")

    # --- Step 1-3: Load and Prepare Data (with console feedback) ---
    try:
        print("\n[Step 1/3] Loading raw data from JSON files...")
        train_list, dev_list, _ = load_raw_data(config.TRAIN_FILE, config.DEV_FILE, config.TEST_FILE)

        print("\n[Step 2/3] Cleaning data for Hugging Face `datasets` compatibility...")
        train_list, dev_list = clean_data_for_arrow([train_list, dev_list])

        print("\n[Step 3/3] Creating Hugging Face `DatasetDict` object...")
        # Use a tiny subset for fast inspection
        finqa_dataset = create_hf_dataset(train_list[:1], dev_list[:1], [])
        print("✅ Data loading and preparation complete.")

    except Exception as e:
        print(f"❌ An error occurred during data loading: {e}")
        print("Aborting report generation.")
        return

    # --- Step 4: Generate Report Content ---
    print(f"\n[Step 4/4] Generating report content...")
    
    report_lines = []
    
    train_split = finqa_dataset['train']
    
    # Add a main title
    report_lines.append("# Dataset Inspection Report")
    report_lines.append(f"This report details the structure of the dataset used in the project.")
    
    # Section 1: Features
    report_lines.append("\n## 1. Dataset Features (Column Names and Data Types)")
    report_lines.append("This section shows the schema of the dataset, including each column's name and its expected data type.")
    report_lines.append("```")
    report_lines.append(str(train_split.features))
    report_lines.append("```")

    # Section 2: Column Names
    report_lines.append("\n## 2. Column Names (as a list)")
    report_lines.append("A simple list of all available column names. **Use these exact names in your code.**")
    report_lines.append("```")
    report_lines.append(str(train_split.column_names))
    report_lines.append("```")

    # Section 3: First Example
    report_lines.append("\n## 3. First Example Record")
    report_lines.append("Below is the full data for the first record in the training set. This helps visualize the content of each column.")
    # Use pprint.pformat to get a nicely formatted string of the dictionary
    first_example_str = pprint.pformat(train_split[0], indent=2)
    report_lines.append("```python")
    report_lines.append(first_example_str)
    report_lines.append("```")

    # --- Step 5: Write to File ---
    try:
        report_content = "\n".join(report_lines)
        with open(REPORT_FILENAME, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\n--- ✅ Inspection Complete ---")
        print(f"Report successfully saved to: {REPORT_FILENAME}")

    except Exception as e:
        print(f"\n--- ❌ Error Saving Report ---")
        print(f"Could not write to file {REPORT_FILENAME}. Error: {e}")


if __name__ == '__main__':
    inspect_and_save_report()