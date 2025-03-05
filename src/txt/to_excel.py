import os
import pandas as pd
from pathlib import Path

def convert_csv_to_xlsx(folder_path):
    # Walk through all files and subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):  # Check for CSV files
                csv_path = os.path.join(root, file)
                xlsx_path = os.path.join(root, Path(file).stem + ".xlsx")
                
                try:
                    # Read CSV using pandas
                    df = pd.read_csv(csv_path)
                    
                    # Save as Excel
                    df.to_excel(xlsx_path, index=False)
                    
                    # Remove the original CSV file
                    os.remove(csv_path)
                    print(f"Converted and deleted: {csv_path}")
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")

if __name__ == "__main__":
    #folder = input("Enter the folder path: ")
    folder = "0 - Demo"
    convert_csv_to_xlsx(folder)
    print("Conversion complete!")
