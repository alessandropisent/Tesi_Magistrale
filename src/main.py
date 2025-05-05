import os
import sys
import json
import pandas as pd
import argparse
import re
from PyPDF2 import PdfReader
from tqdm import tqdm

# --- Imports for OPENAI
import dotenv
import openai

# --- Imports for Local LLMs ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig




# Attempt to import ChecklistCompiler, assuming it's in the same directory or Python path
try:
    # Assuming ChecklistCompiler defines LUCCA and LLAMA constants
    from ChecklistCompiler import ChecklistCompiler, LUCCA, LLAMA, MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
except ImportError:
    print("ERROR: Could not import ChecklistCompiler or required constants (LUCCA, LLAMA, MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST).")
    print("Please ensure 'ChecklistCompiler.py' is in the same directory or accessible in your Python path,")
    print("and that it defines LUCCA, LLAMA, and MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST.")
    sys.exit(1)

# --- Configuration ---
# Fixed municipality for this script
TARGET_MUNICIPALITY = LUCCA
# Relative path to the directory containing municipality-specific data
BASE_DATA_PATH = "./src/txt/"
# Default LLM choice changed to LLAMA
DEFAULT_LLM_TYPE = LLAMA
# Default temperature for checklist suggestion (low for consistency)
SUGGEST_TEMPERATURE = 0.01
# Default temperature for checklist execution (moderate for some variability)
EXECUTE_TEMPERATURE = 0.01

# --- Helper Functions (pdf_to_text, load_checklists remain the same) ---

def pdf_to_text(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The full path to the PDF file.

    Returns:
        str: The extracted text, or None if an error occurs.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Check if text extraction returned something
                 text += page_text + "\n" # Add newline between pages
        if not text:
             print(f"Warning: No text could be extracted from {pdf_path}. The PDF might be image-based or empty.")
             # Return empty string instead of None if PDF is valid but has no extractable text
             return ""
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return None

def load_checklists(municipality):
    """
    Loads checklist data for a given municipality from a JSON file.

    Args:
        municipality (str): The name of the municipality (e.g., "Lucca").

    Returns:
        dict: The loaded checklist data, or None if an error occurs.
    """
    checklist_path = os.path.join(BASE_DATA_PATH, municipality, "checklists", "checklists.json")
    try:
        with open(checklist_path, "r", encoding="utf-8") as f:
            checklists_data = json.load(f)
        # Basic validation
        if "checklists" not in checklists_data or not isinstance(checklists_data["checklists"], list):
             print(f"Error: Checklist JSON format is invalid in {checklist_path}. Missing top-level 'checklists' list.")
             return None
        return checklists_data
    except FileNotFoundError:
        print(f"Error: Checklists file not found at {checklist_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {checklist_path}")
        return None
    except Exception as e:
        print(f"Error loading checklists from {checklist_path}: {e}")
        return None

# --- LLM Interaction Functions (Modified for Pipeline) ---

def suggest_checklist(compiler, pdf_text, checklists_data):
    """
    Uses the LLM via ChecklistCompiler to suggest a checklist.
    Now relies on the compiler having its pipeline set.

    Args:
        compiler (ChecklistCompiler): An initialized ChecklistCompiler instance with its pipeline set.
        pdf_text (str): The text extracted from the PDF.
        checklists_data (dict): The loaded checklist data.

    Returns:
        str: The name of the suggested checklist, or None if suggestion fails.
    """
    print("Asking LLM to suggest a checklist based on PDF content...")
    if not compiler.text_gen_pipeline:
         print("Error: ChecklistCompiler's text_gen_pipeline is not set.")
         return None
    try:
        # Generate the prompt for choosing a checklist
        # Ensure the prompt format is suitable for the local LLM (ChecklistCompiler handles this)
        prompt = compiler.generate_prompt_choose(determina=pdf_text, checklists=checklists_data)

        # Generate the response from the LLM using the compiler's method
        response = compiler.generate_response(
            complete_prompt=prompt,
            temperature=SUGGEST_TEMPERATURE # Use configured low temp
            # Compiler's generate_response should handle pipeline args like do_sample, top_p if needed
        )

        # --- Extract the checklist name from the response ---
        valid_checklist_names = [chk['NomeChecklist'] for chk in checklists_data.get('checklists', [])]
        if not valid_checklist_names:
            print("Error: No checklist names found in the loaded checklist data.")
            return None

        # Try to find an exact match (case-insensitive)
        suggested_name = None
        for name in valid_checklist_names:
            # Search for the name as a whole word, potentially surrounded by punctuation/whitespace
            if re.search(rf'\b{re.escape(name)}\b', response, re.IGNORECASE):
                suggested_name = name # Return the correctly capitalized name
                print(f"LLM suggested: '{suggested_name}'")
                return suggested_name

        print(f"Warning: Could not reliably extract a valid checklist name from LLM response.")
        print(f"LLM Raw Response: '{response.strip()}'")
        print(f"Valid names: {valid_checklist_names}")
        return None # Indicate failure to extract

    except Exception as e:
        print(f"Error during LLM checklist suggestion: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return None

def confirm_or_select_checklist(suggested_checklist, checklists_data):
    """
    Prompts the user to confirm the suggested checklist or select another.
    (No changes needed in this function's logic)

    Args:
        suggested_checklist (str or None): The name suggested by the LLM.
        checklists_data (dict): The loaded checklist data containing all options.

    Returns:
        str: The name of the finally chosen checklist, or None if user cancels/error.
    """
    available_checklists = checklists_data.get("checklists", [])
    if not available_checklists:
        print("Error: No checklists available to choose from.")
        return None

    checklist_names = [chk['NomeChecklist'] for chk in available_checklists]

    print("\n--- Checklist Selection ---")
    if suggested_checklist and suggested_checklist in checklist_names:
        print(f"LLM Suggestion: {suggested_checklist}")
        prompt_text = f"Use this suggestion? (y/n) or enter number: "
    else:
        if suggested_checklist:
            print(f"LLM suggestion '{suggested_checklist}' was not found in the available list. Please choose manually.")
        else:
            print("LLM could not suggest a checklist. Please choose manually.")
        prompt_text = "Enter the number of the checklist to use: "

    print("\nAvailable Checklists:")
    for i, name in enumerate(checklist_names):
        print(f"  {i + 1}. {name}")

    while True:
        user_input = input(prompt_text).strip().lower()

        if suggested_checklist and suggested_checklist in checklist_names:
             if user_input == 'y' or user_input == 'yes':
                 return suggested_checklist
             elif user_input == 'n' or user_input == 'no':
                 prompt_text = "Enter the number of the checklist to use: " # Ask for number next time
                 suggested_checklist = None # Clear suggestion, force number input next
                 continue # Re-prompt without suggestion

        # Check if input is a valid number corresponding to a checklist
        if user_input.isdigit():
            try:
                choice_index = int(user_input) - 1
                if 0 <= choice_index < len(checklist_names):
                    chosen_name = checklist_names[choice_index]
                    print(f"You selected: {chosen_name}")
                    return chosen_name
                else:
                    print(f"Invalid number. Please enter a number between 1 and {len(checklist_names)}.")
            except ValueError: # Should not happen due to isdigit, but good practice
                 print("Invalid input. Please enter 'y', 'n', or a number.")
        else:
             print("Invalid input. Please enter 'y', 'n', or a number.")


def execute_checklist(compiler, pdf_text, chosen_checklist_name, checklists_data):
    """
    Executes the chosen checklist against the PDF text using the LLM.
    Relies on the compiler having its pipeline set.

    Args:
        compiler (ChecklistCompiler): Initialized compiler instance with pipeline.
        pdf_text (str): Text from the PDF.
        chosen_checklist_name (str): Name of the checklist to execute.
        checklists_data (dict): Loaded checklist data.

    Returns:
        list: A list of dictionaries, each containing results for a checklist point.
              Returns None if the chosen checklist is not found or execution fails.
    """
    if not compiler.text_gen_pipeline:
         print("Error: ChecklistCompiler's text_gen_pipeline is not set for execution.")
         return None
    try:
        # Retrieve the full details of the chosen checklist
        # Using static method approach if get_checklist is defined that way
        checklist_details = ChecklistCompiler.get_checklist(checklists_data, chosen_checklist_name)
        if not checklist_details:
            print(f"Error: Could not retrieve details for checklist '{chosen_checklist_name}'.")
            return None
    except Exception as e:
        print(f"Error retrieving checklist details: {e}")
        return None

    results = []
    checklist_points = checklist_details.get("Punti", [])
    if not checklist_points:
        print(f"Warning: Checklist '{chosen_checklist_name}' has no points ('Punti').")
        return [] # Return empty list if no points

    print(f"\nExecuting checklist '{chosen_checklist_name}' ({len(checklist_points)} points)...")
    for point in tqdm(checklist_points, desc="Processing Checklist Points"):
        num = point.get("num", "N/A")
        punto_text = point.get("Punto", "")
        istruzioni = point.get("Istruzioni", "")
        # Use compiler's hasSezioni attribute to check if section is needed
        sezione = point.get("Sezione", "") if compiler.hasSezioni else ""

        try:
            # Generate prompt using compiler method
            prompt = compiler.generate_prompt(
                istruzioni=istruzioni,
                punto=punto_text,
                num=num,
                determina=pdf_text,
                sezione=sezione
            )

            # Generate response using compiler method with execution temperature
            llm_response = compiler.generate_response(
                complete_prompt=prompt,
                temperature=EXECUTE_TEMPERATURE # Use configured moderate temp
            )

            # Analyze response using compiler method
            simple_response = compiler.analize_response(llm_response)

            results.append({
                "Numero Punto": num,
                "Testo Punto": punto_text,
                "Risposta Semplice": simple_response,
                "Risposta LLM Completa": llm_response.strip() # Store the full response
            })
        except Exception as e:
            print(f"\nError processing point {num}: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            # Append error information to results
            results.append({
                "Numero Punto": num,
                "Testo Punto": punto_text,
                "Risposta Semplice": "ERROR",
                "Risposta LLM Completa": f"Error during processing: {e}"
            })

    return results

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Process a PDF determination for {TARGET_MUNICIPALITY} using checklists and a local LLM.")
    parser.add_argument("pdf_filename", help="Name of the PDF file (e.g., 'determina_123.pdf')")
    parser.add_argument("pdf_folder", help="Path to the folder containing the PDF file.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model ID for the local LLM (e.g., 'meta-llama/Llama-3.1-8B-Instruct')")
    parser.add_argument("--quantize", action='store_true', help="Load the model using 4-bit quantization (requires bitsandbytes).")
    # Add arguments for max_new_tokens, temperature overrides, etc. if needed
    args = parser.parse_args()

    # --- 1. Setup Paths and Load Data ---
    full_pdf_path = os.path.join(args.pdf_folder, args.pdf_filename)
    # Create output folder based on model name if it doesn't exist
    model_name_safe = args.model_id.split('/')[-1] # Get last part of model ID for folder name
    output_folder = os.path.join(args.pdf_folder, f"results_{model_name_safe}")
    os.makedirs(output_folder, exist_ok=True)

    output_excel_filename = os.path.splitext(args.pdf_filename)[0] + "_checklist_results.xlsx"
    output_excel_path = os.path.join(output_folder, output_excel_filename) # Save Excel in model-specific subfolder

    print(f"Processing PDF: {full_pdf_path}")
    print(f"Municipality: {TARGET_MUNICIPALITY}")
    print(f"Using Model: {args.model_id}")
    print(f"Quantization: {'Enabled' if args.quantize else 'Disabled'}")
    print(f"Output Folder: {output_folder}")


    checklists_data = load_checklists(TARGET_MUNICIPALITY)
    if not checklists_data:
        sys.exit(1) # Error message already printed

    # --- 2. Extract PDF Text ---
    pdf_text = pdf_to_text(full_pdf_path)
    if pdf_text is None: # Check for None specifically for PDF read errors
        sys.exit(1) # Error message already printed
    elif not pdf_text: # Handle case where PDF is valid but no text extracted
         print("PDF processed, but no text content found. Cannot proceed with LLM analysis.")
         sys.exit(0) # Exit cleanly, nothing to analyze
    print(f"Successfully extracted text from PDF (length: {len(pdf_text)} chars).")

    # --- 3. Initialize LLM Model, Tokenizer, Pipeline ---
    print(f"\nLoading local model '{args.model_id}'...")
    try:
        quantization_config = None
        model_kwargs = {
            "torch_dtype": torch.bfloat16, # Or torch.float16, bfloat16 often better if supported
            "device_map": "auto", # Automatically distribute across available GPUs
            "trust_remote_code": True # Often needed for custom architectures
        }

        if args.quantize:
            print("Setting up 4-bit quantization...")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16, # Or bfloat16 if compute supports it
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                print("Quantization config created.")
            except ImportError:
                print("ERROR: 'bitsandbytes' library not found, but --quantize flag was used.")
                print("Please install it: pip install bitsandbytes")
                sys.exit(1)
            except Exception as e_quant:
                 print(f"Error setting up quantization: {e_quant}")
                 # Decide if you want to proceed without quantization or exit
                 print("Proceeding without quantization...")
                 model_kwargs.pop("quantization_config", None) # Remove potentially partial config


        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            **model_kwargs
        )
        print("Model loaded.")

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        print("Tokenizer loaded.")

        # Create Pipeline
        # Adjust max_new_tokens as needed
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500, # Max tokens for the *generated* response
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, # Explicitly set EOS token ID
            truncation=True # Ensure input prompt is truncated if too long
        )
        print("Text generation pipeline created.")

    except ImportError as e_imp:
         print(f"ERROR: Missing library for Hugging Face models: {e_imp}")
         print("Please install required libraries: pip install torch transformers accelerate")
         sys.exit(1)
    except Exception as e_load: # Catch OOM errors, connection errors etc.
        print(f"Error loading local model or creating pipeline: {e_load}")
        import traceback
        traceback.print_exc()
        # Consider adding specific checks for torch.cuda.OutOfMemoryError if relevant
        sys.exit(1)


    # --- 4. Initialize LLM Compiler ---
    print(f"Initializing ChecklistCompiler for {TARGET_MUNICIPALITY} using {DEFAULT_LLM_TYPE}...")
    try:
        # Determine if the municipality needs 'hasSezioni'
        has_sezioni = TARGET_MUNICIPALITY in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
        
        #print(text_gen_pipeline)

        compiler = ChecklistCompiler(
            llm=DEFAULT_LLM_TYPE, # Use LLAMA type
            municipality=TARGET_MUNICIPALITY,
            model=args.model_id, # Store the model ID used
            text_gen_pipeline=text_gen_pipeline, # Pass the pipeline instance
            hasSezioni=has_sezioni
        )
        compiler.set_text_gen_pipeline(text_gen_pipeline)
        print("ChecklistCompiler initialized.")
    except Exception as e_init:
        print(f"Error initializing ChecklistCompiler: {e_init}")
        sys.exit(1)

    # --- 5. Suggest and Confirm Checklist ---
    suggested = suggest_checklist(compiler, pdf_text, checklists_data)
    chosen_checklist = confirm_or_select_checklist(suggested, checklists_data)

    if not chosen_checklist:
        print("No checklist selected. Exiting.")
        sys.exit(0)

    # --- 6. Execute Checklist ---
    final_results = execute_checklist(compiler, pdf_text, chosen_checklist, checklists_data)

    if final_results is None:
        print("Checklist execution failed.")
        sys.exit(1)
    elif not final_results:
         print("Checklist execution finished, but no results were generated (checklist might be empty).")
         sys.exit(0)


    # --- 7. Output to Excel ---
    print(f"\nSaving results to Excel: {output_excel_path}")
    try:
        df_results = pd.DataFrame(final_results)
        # Ensure columns are in a reasonable order
        df_results = df_results[[
            "Numero Punto",
            "Testo Punto",
            "Risposta Semplice",
            "Risposta LLM Completa"
        ]]
        df_results.to_excel(output_excel_path, index=False, engine='openpyxl') # Requires 'openpyxl' install
        print("Excel file saved successfully.")
    except ImportError:
         print("\nError: 'openpyxl' library not found. Cannot write to Excel.")
         print("Please install it using: pip install openpyxl")
         # Optionally, save to CSV as a fallback
         csv_path = os.path.splitext(output_excel_path)[0] + ".csv"
         try:
             df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
             print(f"Saved results to CSV as fallback: {csv_path}")
         except Exception as e_csv:
             print(f"Could not save to CSV either: {e_csv}")
    except Exception as e_save:
        print(f"Error saving results to Excel/CSV: {e_save}")
        sys.exit(1)

    print("\nProcessing complete.")
