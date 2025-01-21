import os
from PyPDF2 import PdfReader

def extract_text_from_pdfs(directory_input, directory_output):
    """
    Extracts text from all PDF files in the specified directory.
    
    Args:
        directory (str): The path to the directory containing PDF files.
    
    Returns:
        dict: A dictionary with filenames as keys and extracted text as values.
    """
    extracted_texts = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory_input):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_input, filename)
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                extracted_texts[filename] = text

                with open(directory_output+"/"+filename[:-4] +".txt" , "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
    
    return extracted_texts


# Specify the directory containing the PDF files
pdf_directory = "./determine_pdf/MB/Raw_det"  
text_directory= "./src/txt/MB"

# Call the function and process the PDFs
extracted_data = extract_text_from_pdfs(pdf_directory,pdf_directory)

# Save or display the extracted text
for filename, text in extracted_data.items():
    print(f"--- {filename} ---")
    print(text[:500])  # Print the first 500 characters of each PDF text
    print("\n")
