import os
import sys
import json
import pandas as pd
import argparse
import re
from PyPDF2 import PdfReader
from tqdm import tqdm

# --- Imports for LLMs (Conditional Loading Later) ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import openai # Needed if OpenAI is chosen
from dotenv import load_dotenv # Needed if OpenAI is chosen
# Ensure accelerate is installed: pip install accelerate
# Ensure bitsandbytes is installed for quantization: pip install bitsandbytes
# Ensure openai is installed: pip install openai
# Ensure python-dotenv is installed: pip install python-dotenv
# -----------------------------------------------------

# Attempt to import ChecklistCompiler, assuming it's in the same directory or Python path
try:
    # Assuming ChecklistCompiler defines LUCCA, LLAMA, OPENAI constants
    from ChecklistCompiler import ChecklistCompiler, LUCCA, LLAMA, OPENAI, MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
except ImportError:
    print("ERROR: Could not import ChecklistCompiler or required constants (LUCCA, LLAMA, OPENAI, MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST).")
    print("Please ensure 'ChecklistCompiler.py' is in the same directory or accessible in your Python path,")
    print("and that it defines LUCCA, LLAMA, OPENAI, and MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST.")
    sys.exit(1)

# --- Configuration ---
# Fixed municipality for this script
TARGET_MUNICIPALITY = LUCCA
# Relative path to the directory containing municipality-specific data
BASE_DATA_PATH = "./src/txt/"
# Default temperature for checklist suggestion (low for consistency)
SUGGEST_TEMPERATURE = 0.1
# Default temperature for checklist execution (moderate for some variability)
EXECUTE_TEMPERATURE = 0.1
# Default OpenAI model
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# --- Predefined Local Models ---
# List of common local models for user convenience
PREDEFINED_LOCAL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

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

# --- LLM Choice and Configuration ---

def prompt_for_llm_config():
    """
    Prompts the user to select LLM type, model, and device map (for local).

    Returns:
        tuple: (llm_type, model_name, quantize, device_map_setting) or (None, None, None, None) on error/cancel.
               llm_type is LLAMA or OPENAI constant.
               quantize is boolean (only relevant for LLAMA).
               device_map_setting is str ('auto', 'cuda:0', 'cuda:1') or None.
    """
    print("\n--- LLM Selection ---")
    while True:
        llm_choice = input("Choose LLM type (1: Local LLM, 2: OpenAI): ").strip()

        # --- Local LLM Choice ---
        if llm_choice == '1':
            llm_type = LLAMA
            model_name = None
            quantize = False
            device_map_setting = None

            print("\nAvailable Local Models:")
            for i, model_id in enumerate(PREDEFINED_LOCAL_MODELS):
                print(f"  {i + 1}. {model_id}")

            # --- Model Selection Loop ---
            while model_name is None:
                model_input = input(f"Enter the number of the model (1-{len(PREDEFINED_LOCAL_MODELS)}) or type a custom Hugging Face model ID: ").strip()

                # Check if input is a number corresponding to the list
                if model_input.isdigit():
                    try:
                        choice_index = int(model_input) - 1
                        if 0 <= choice_index < len(PREDEFINED_LOCAL_MODELS):
                            model_name = PREDEFINED_LOCAL_MODELS[choice_index]
                            print(f"Selected predefined model: {model_name}")
                        else:
                            print(f"Invalid number. Please enter a number between 1 and {len(PREDEFINED_LOCAL_MODELS)} or a custom ID.")
                            continue # Ask for model input again
                    except ValueError:
                        print("Invalid input. Please enter a number or a custom model ID.")
                        continue
                # Check if input is a non-empty string (and not a valid number choice)
                elif model_input:
                    model_name = model_input # Treat as custom model ID
                    print(f"Using custom model ID: {model_name}")
                # Handle empty input
                else:
                    print("Model input cannot be empty.")
                    continue

            # --- Quantization Choice ---
            quantize_choice = input("Use 4-bit quantization? (y/n, default: n): ").strip().lower()
            quantize = quantize_choice == 'y'

            # --- Device Map Choice ---
            while device_map_setting is None:
                print("\nSelect device map:")
                print("  1. auto (Recommended - let Transformers decide)")
                print("  2. cuda:0 (Force to GPU 0)")
                print("  3. cuda:1 (Force to GPU 1)")
                # Add more options like 'cpu' if desired
                device_choice = input("Enter choice (1-3, default: 1): ").strip()

                if not device_choice or device_choice == '1':
                    device_map_setting = 'auto'
                elif device_choice == '2':
                    device_map_setting = torch.device('cuda:0')
                elif device_choice == '3':
                    device_map_setting = torch.device('cuda:1')
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    continue # Ask for device map again
                print(f"Selected device map: '{device_map_setting}'")

            return llm_type, model_name, quantize, device_map_setting # Exit the function with success

        # --- OpenAI Choice ---
        elif llm_choice == '2':
            llm_type = OPENAI
            model_input = input(f"Enter OpenAI model name (default: {DEFAULT_OPENAI_MODEL}): ").strip()
            model_name = model_input if model_input else DEFAULT_OPENAI_MODEL
            # Quantization and device_map don't apply to OpenAI
            return llm_type, model_name, False, None

        # --- Invalid LLM Type Choice ---
        else:
            print("Invalid choice. Please enter 1 or 2.")


# --- LLM Interaction Functions (suggest_checklist, confirm_or_select_checklist, execute_checklist remain the same) ---

def suggest_checklist(compiler, pdf_text, checklists_data):
    """
    Uses the chosen LLM via ChecklistCompiler to suggest a checklist.
    Relies on the compiler having its pipeline set (for Llama) or using API (for OpenAI).

    Args:
        compiler (ChecklistCompiler): An initialized ChecklistCompiler instance.
        pdf_text (str): The text extracted from the PDF.
        checklists_data (dict): The loaded checklist data.

    Returns:
        str: The name of the suggested checklist, or None if suggestion fails.
    """
    print("Asking LLM to suggest a checklist based on PDF content...")
    # Check needed resources based on compiler type
    if compiler.llm == LLAMA and not compiler.text_gen_pipeline:
         print("Error: ChecklistCompiler's text_gen_pipeline is not set for Llama.")
         return None
    # No specific check needed for OpenAI here, compiler handles API calls

    try:
        # Generate the prompt (compiler handles formatting differences)
        prompt = compiler.generate_prompt_choose(determina=pdf_text, checklists=checklists_data)

        # Generate the response (compiler handles pipeline vs API call)
        response = compiler.generate_response(
            complete_prompt=prompt,
            temperature=SUGGEST_TEMPERATURE # Use configured low temp
        )

        # --- Extract the checklist name from the response ---
        valid_checklist_names = [chk['NomeChecklist'] for chk in checklists_data.get('checklists', [])]
        if not valid_checklist_names:
            print("Error: No checklist names found in the loaded checklist data.")
            return None

        # Try to find an exact match (case-insensitive)
        suggested_name = None
        for name in valid_checklist_names:
            if re.search(rf'\b{re.escape(name)}\b', response, re.IGNORECASE):
                suggested_name = name # Return the correctly capitalized name
                print(f"LLM suggested: '{suggested_name}'")
                return suggested_name

        print(f"Warning: Could not reliably extract a valid checklist name from LLM response.")
        print(f"LLM Raw Response: '{response.strip()}'")
        print(f"Valid names: {valid_checklist_names}")
        return None

    except Exception as e:
        print(f"Error during LLM checklist suggestion: {e}")
        import traceback
        traceback.print_exc()
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
    Relies on the compiler having its pipeline set (for Llama) or using API (for OpenAI).

    Args:
        compiler (ChecklistCompiler): Initialized compiler instance.
        pdf_text (str): Text from the PDF.
        chosen_checklist_name (str): Name of the checklist to execute.
        checklists_data (dict): Loaded checklist data.

    Returns:
        list: A list of dictionaries, each containing results for a checklist point.
              Returns None if the chosen checklist is not found or execution fails.
    """
    # Check needed resources based on compiler type
    if compiler.llm == LLAMA and not compiler.text_gen_pipeline:
         print("Error: ChecklistCompiler's text_gen_pipeline is not set for Llama execution.")
         return None

    try:
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
        return []

    print(f"\nExecuting checklist '{chosen_checklist_name}' ({len(checklist_points)} points)...")
    for point in tqdm(checklist_points, desc="Processing Checklist Points"):
        num = point.get("num", "N/A")
        punto_text = point.get("Punto", "")
        istruzioni = point.get("Istruzioni", "")
        sezione = point.get("Sezione", "") if compiler.hasSezioni else ""

        try:
            prompt = compiler.generate_prompt(
                istruzioni=istruzioni,
                punto=punto_text,
                num=num,
                determina=pdf_text,
                sezione=sezione
            )

            llm_response = compiler.generate_response(
                complete_prompt=prompt,
                temperature=EXECUTE_TEMPERATURE
            )

            simple_response = compiler.analize_response(llm_response)

            results.append({
                "Numero Punto": num,
                "Testo Punto": punto_text,
                "Risposta Semplice": simple_response,
                "Risposta LLM Completa": llm_response.strip()
            })
        except Exception as e:
            print(f"\nError processing point {num}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "Numero Punto": num,
                "Testo Punto": punto_text,
                "Risposta Semplice": "ERROR",
                "Risposta LLM Completa": f"Error during processing: {e}"
            })

    return results

# --- Main Execution ---
if __name__ == "__main__":
    # Argument parser now only takes PDF info
    parser = argparse.ArgumentParser(description=f"Process a PDF determination for {TARGET_MUNICIPALITY} using checklists and a user-chosen LLM.")
    parser.add_argument("pdf_filename", help="Name of the PDF file (e.g., 'determina_123.pdf')")
    parser.add_argument("pdf_folder", help="Path to the folder containing the PDF file.")
    args = parser.parse_args()

    # --- 1. Get LLM Configuration from User ---
    # Now unpacks device_map_setting as well
    llm_type, model_name, quantize, device_map_setting = prompt_for_llm_config()
    if not llm_type:
        print("LLM configuration cancelled. Exiting.")
        sys.exit(0)

    # --- 2. Setup Paths and Load Data ---
    full_pdf_path = os.path.join(args.pdf_folder, args.pdf_filename)

    # Create output folder based on LLM type and model name
    model_name_safe = model_name.split('/')[-1].replace('.','_') # Make model name filesystem-safe
    output_folder_name = f"results_{llm_type}_{model_name_safe}"
    output_folder = os.path.join(args.pdf_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)

    output_excel_filename = os.path.splitext(args.pdf_filename)[0] + "_checklist_results.xlsx"
    output_excel_path = os.path.join(output_folder, output_excel_filename)

    print(f"\n--- Processing Setup ---")
    print(f"Processing PDF: {full_pdf_path}")
    print(f"Municipality: {TARGET_MUNICIPALITY}")
    print(f"LLM Type: {llm_type}")
    print(f"Model Name: {model_name}")
    if llm_type == LLAMA:
        print(f"Quantization: {'Enabled' if quantize else 'Disabled'}")
        print(f"Device Map: '{device_map_setting}'") # Display chosen device map
    print(f"Output Folder: {output_folder}")
    print("-" * 24)

    checklists_data = load_checklists(TARGET_MUNICIPALITY)
    if not checklists_data:
        sys.exit(1)

    # --- 3. Extract PDF Text ---
    pdf_text = pdf_to_text(full_pdf_path)
    if pdf_text is None:
        sys.exit(1)
    elif not pdf_text:
         print("PDF processed, but no text content found. Cannot proceed with LLM analysis.")
         sys.exit(0)
    print(f"Successfully extracted text from PDF (length: {len(pdf_text)} chars).")

    # --- 4. Initialize LLM (Conditional) ---
    compiler = None
    text_gen_pipeline = None # Initialize as None

    if llm_type == LLAMA:
        print(f"\nLoading local model '{model_name}'...")
        try:
            quantization_config = None
            # Use the chosen device_map_setting here
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": device_map_setting, # Use the user's choice
                "trust_remote_code": True
            }

            if quantize:
                print("Setting up 4-bit quantization...")
                # Check if device_map is set to a specific device when quantizing
                # Bitsandbytes quantization often works best with device_map='auto' or single device
                if device_map_setting not in ['auto', 'cuda:0', 'cuda:1']: # Add 'cpu' etc. if supported
                     print(f"Warning: Quantization might behave unexpectedly with device_map='{device_map_setting}'. Consider 'auto' or a specific GPU ('cuda:0').")

                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    print("Quantization config created.")
                except ImportError:
                    print("ERROR: 'bitsandbytes' library not found, but quantization was requested.")
                    print("Please install it: pip install bitsandbytes")
                    sys.exit(1)
                except Exception as e_quant:
                    print(f"Error setting up quantization: {e_quant}. Proceeding without it.")
                    model_kwargs.pop("quantization_config", None)

            # Load Model using the constructed arguments
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            print("Model loaded.")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Tokenizer loaded.")

            # Create Pipeline - specify device if not using 'auto' to ensure pipeline runs on the correct device
            pipeline_device_arg = None

            text_gen_pipeline = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                device=pipeline_device_arg, # Pass device index or None
                max_new_tokens=500, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, truncation=True
            )
            print(f"Text generation pipeline created (Target device: {device_map_setting if pipeline_device_arg is None else pipeline_device_arg}).")


        except ImportError as e_imp:
            print(f"ERROR: Missing library for Hugging Face models: {e_imp}")
            print("Please install required libraries: pip install torch transformers accelerate")
            sys.exit(1)
        except Exception as e_load:
            print(f"Error loading local model or creating pipeline: {e_load}")
            import traceback
            traceback.print_exc()
            # Add specific check for CUDA errors if needed
            if "CUDA" in str(e_load):
                 print("\nCUDA-related error detected. Check GPU availability, drivers, and torch compatibility.")
                 if device_map_setting not in ['auto', None]:
                      print(f"Ensure the specified device '{device_map_setting}' is available and has enough memory.")
            sys.exit(1)

    elif llm_type == OPENAI:
        print("\nConfiguring for OpenAI...")
        try:
            load_dotenv() # Load .env file if it exists
            if not os.getenv("OPENAI_API_KEY"):
                print("ERROR: OPENAI_API_KEY not found in environment variables or .env file.")
                print("Please set the environment variable or create a .env file.")
                sys.exit(1)
            # Test connection implicitly during ChecklistCompiler init or first call
            print("OpenAI environment configured.")
        except ImportError:
             print("ERROR: 'python-dotenv' library not found, needed for OpenAI .env loading.")
             print("Please install it: pip install python-dotenv")
             sys.exit(1)
        except Exception as e_env:
            print(f"Error configuring OpenAI environment: {e_env}")
            sys.exit(1)

    # --- 5. Initialize LLM Compiler ---
    print(f"Initializing ChecklistCompiler for {TARGET_MUNICIPALITY} using {llm_type}...")
    try:
        has_sezioni = TARGET_MUNICIPALITY in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
        compiler = ChecklistCompiler(
            llm=llm_type,
            municipality=TARGET_MUNICIPALITY,
            model=model_name, # Store the specific model name
            text_gen_pipeline=text_gen_pipeline, # Will be None for OpenAI, set for Llama
            hasSezioni=has_sezioni
        )
        compiler.set_text_gen_pipeline(text_gen_pipeline)
        print("ChecklistCompiler initialized.")
    except Exception as e_init:
        print(f"Error initializing ChecklistCompiler: {e_init}")
        sys.exit(1)

    # --- 6. Suggest and Confirm Checklist ---
    suggested = suggest_checklist(compiler, pdf_text, checklists_data)
    chosen_checklist = confirm_or_select_checklist(suggested, checklists_data)

    if not chosen_checklist:
        print("No checklist selected. Exiting.")
        sys.exit(0)

    # --- 7. Execute Checklist ---
    final_results = execute_checklist(compiler, pdf_text, chosen_checklist, checklists_data)

    if final_results is None:
        print("Checklist execution failed.")
        sys.exit(1)
    elif not final_results:
         print("Checklist execution finished, but no results were generated (checklist might be empty).")
         sys.exit(0)

    # --- 8. Output to Excel ---
    print(f"\nSaving results to Excel: {output_excel_path}")
    try:
        df_results = pd.DataFrame(final_results)
        df_results = df_results[[
            "Numero Punto", "Testo Punto", "Risposta Semplice", "Risposta LLM Completa"
        ]]
        df_results.to_excel(output_excel_path, index=False, engine='openpyxl')
        print("Excel file saved successfully.")
    except ImportError:
         print("\nError: 'openpyxl' library not found. Cannot write to Excel.")
         print("Please install it: pip install openpyxl")
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
