import pandas as pd

def replace_other_in_csv(file_path: str, columns: list):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Replace 'other' with 'Other' only in specified columns
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 'Other' if isinstance(x, str) and x.lower() == 'other' else x)
    
    # Save the modified DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    file_path = "./data/DATA.csv"  # Change this if the file is located elsewhere
    columns_to_modify = ["CV_Category", "ESC_ChromState_Gonzalez2021"]  # Specify the columns to modify
    replace_other_in_csv(file_path, columns_to_modify)
    print(f"Replaced all occurrences of 'other' with 'Other' in specified columns of {file_path}")