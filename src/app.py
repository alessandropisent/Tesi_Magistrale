import os
import sys
import json
import pandas as pd
import re
import io # Used for sending file data
import traceback # For detailed error logging
import uuid # For generating unique temporary filenames
import time # For potential delays or progress simulation
# Added session and jsonify
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from tqdm import tqdm # Optional for backend logging, not visible in UI

# --- Imports for LLMs ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import openai
from dotenv import load_dotenv

# --- Import Checklist Compiler Logic ---
try:
    # Added OLBIA to imports
    from ChecklistCompiler import ChecklistCompiler, LUCCA, OLBIA, LLAMA, OPENAI, MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
except ImportError:
    print("ERROR: Could not import ChecklistCompiler or required constants.")
    print("Ensure 'ChecklistCompiler.py' is in the same directory and defines LUCCA, OLBIA, LLAMA, OPENAI, MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST.")
    sys.exit(1)

# --- Flask App Setup ---
app = Flask(__name__)
# Secret key needed for flashing messages and session management
app.config['SECRET_KEY'] = 'your_very_secret_key_here_please_change' # CHANGE THIS!
# Configure upload folder for temporary PDF storage
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # Limit uploads to 32MB (adjust as needed)
ALLOWED_EXTENSIONS = {'pdf'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configuration ---
AVAILABLE_MUNICIPALITIES = [LUCCA, OLBIA] # Define available municipalities
BASE_DATA_PATH = "./src/txt/" # Adjust if your path is different
SUGGEST_TEMPERATURE = 0.1
EXECUTE_TEMPERATURE = 0.5
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
PREDEFINED_LOCAL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

# --- Global variable to hold loaded LLM (simple approach) ---
loaded_llm_objects = {
    "model": None, "tokenizer": None, "pipeline": None,
    "compiler": None, "type": None, "model_name": None
}

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_text_from_file(pdf_filepath):
    """Extracts text from a PDF file path."""
    if not os.path.exists(pdf_filepath):
        print(f"Error: PDF file not found at {pdf_filepath}")
        return None
    try:
        with open(pdf_filepath, 'rb') as f:
            reader = PdfReader(f)
            text = ""
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_e:
                    print(f"Warning: Could not extract text from page {i+1} of {pdf_filepath}. Error: {page_e}")
            if not text:
                 print(f"Warning: No text could be extracted from PDF: {pdf_filepath}")
                 return "" # Return empty string if no text found
            return text
    except Exception as e:
        print(f"Error reading PDF file {pdf_filepath}: {e}")
        traceback.print_exc()
        return None # Return None on error

def load_checklists(municipality):
    """Loads checklist data for the specified municipality."""
    if municipality not in AVAILABLE_MUNICIPALITIES:
        print(f"Error: Municipality '{municipality}' not supported.")
        return None
    checklist_path = os.path.join(BASE_DATA_PATH, municipality, "checklists", "checklists.json")
    try:
        with open(checklist_path, "r", encoding="utf-8") as f:
            checklists_data = json.load(f)
        if "checklists" not in checklists_data or not isinstance(checklists_data["checklists"], list):
             print(f"Error: Checklist JSON format invalid: {checklist_path}")
             return None
        if not all('NomeChecklist' in chk for chk in checklists_data["checklists"]):
            print(f"Error: Some checklists in {checklist_path} are missing 'NomeChecklist'.")
            return None
        return checklists_data
    except FileNotFoundError:
        print(f"Error: Checklists file not found: {checklist_path}")
        return None
    except Exception as e:
        print(f"Error loading checklists for {municipality}: {e}")
        return None

# Modified to accept municipality directly
def initialize_llm_and_compiler(llm_type, model_name, quantize, device_map_setting, municipality):
    """Loads LLM (if local) and initializes the compiler for a given municipality."""
    global loaded_llm_objects
    cleanup_llm() # Clean up any previous model first

    compiler = None
    text_gen_pipeline = None
    model_to_load = None
    tokenizer = None

    try:
        if llm_type == LLAMA:
            print(f"Loading local model '{model_name}'...")
            quantization_config = None
            model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": device_map_setting, "trust_remote_code": True}
            if quantize:
                print("Setting up 4-bit quantization...")
                if device_map_setting not in ['auto', 'cuda:0', 'cuda:1']:
                    print(f"Warning: Quantization might behave unexpectedly with device_map='{device_map_setting}'.")
                try:
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
                    model_kwargs["quantization_config"] = quantization_config
                except ImportError:
                    raise RuntimeError("Bitsandbytes library not found for quantization.")
                except Exception as e_quant:
                    print(f"Error setting up quantization: {e_quant}. Proceeding without it.")
                    model_kwargs.pop("quantization_config", None)

            model_to_load = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Local model and tokenizer loaded.")

            pipeline_device_arg = None
            if device_map_setting != 'auto':
                 try: pipeline_device_arg = int(device_map_setting.split(':')[-1])
                 except: pipeline_device_arg = -1

            text_gen_pipeline = pipeline("text-generation", model=model_to_load, tokenizer=tokenizer, device=pipeline_device_arg, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, truncation=True)
            print("Text generation pipeline created.")

        elif llm_type == OPENAI:
            print("Configuring for OpenAI...")
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not found in environment.")
            print("OpenAI environment configured.")

        # Store loaded objects globally
        loaded_llm_objects["model"] = model_to_load
        loaded_llm_objects["tokenizer"] = tokenizer
        loaded_llm_objects["pipeline"] = text_gen_pipeline
        loaded_llm_objects["type"] = llm_type
        loaded_llm_objects["model_name"] = model_name

        # Initialize Compiler with the specific municipality
        has_sezioni = municipality in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
        compiler = ChecklistCompiler(llm=llm_type, municipality=municipality, model=model_name, text_gen_pipeline=text_gen_pipeline, hasSezioni=has_sezioni)
        loaded_llm_objects["compiler"] = compiler
        print(f"ChecklistCompiler initialized for {municipality}.")
        return compiler

    except Exception as e_llm:
        print(f"Error initializing LLM or Compiler: {e_llm}")
        traceback.print_exc()
        cleanup_llm()
        raise

def cleanup_llm():
    """Releases LLM resources."""
    global loaded_llm_objects
    print("Cleaning up LLM resources...")
    # Safely delete objects if they exist
    try:
        if loaded_llm_objects.get("pipeline"): del loaded_llm_objects["pipeline"]
        if loaded_llm_objects.get("model"): del loaded_llm_objects["model"]
        if loaded_llm_objects.get("tokenizer"): del loaded_llm_objects["tokenizer"]
        if loaded_llm_objects.get("compiler"): del loaded_llm_objects["compiler"]
    except Exception as e_del:
        print(f"Warning: Error during object deletion in cleanup: {e_del}")

    # Reset global dict
    loaded_llm_objects = {
        "model": None, "tokenizer": None, "pipeline": None,
        "compiler": None, "type": None, "model_name": None
    }

    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
        except Exception as e_cuda:
            print(f"Warning: Error clearing CUDA cache: {e_cuda}")
    print("LLM resources released.")


def suggest_checklist(compiler, pdf_text, checklists_data):
    """Uses LLM to suggest a checklist."""
    print("Suggesting checklist...")
    if not compiler:
        print("Error: Compiler not initialized for suggestion.")
        return None
    # Ensure pipeline is set if Llama (defensive check)
    if compiler.llm == LLAMA and not compiler.text_gen_pipeline:
        compiler.text_gen_pipeline = loaded_llm_objects.get("pipeline")
        if not compiler.text_gen_pipeline:
            print("Error: Llama pipeline missing in compiler and global state.")
            return None

    try:
        prompt = compiler.generate_prompt_choose(determina=pdf_text, checklists=checklists_data)
        response = compiler.generate_response(complete_prompt=prompt, temperature=SUGGEST_TEMPERATURE)

        valid_checklist_names = [chk['NomeChecklist'] for chk in checklists_data.get('checklists', [])]
        if not valid_checklist_names: return None

        response_lines = response.strip().split('\n')
        first_line = response_lines[0].strip()

        for name in valid_checklist_names:
             if first_line.lower() == name.lower():
                  print(f"LLM suggested checklist (exact first line): '{name}'")
                  return name
             if re.search(rf'\b{re.escape(name)}\b', response, re.IGNORECASE):
                print(f"LLM suggested checklist (found in text): '{name}'")
                return name # Return the first one found

        print(f"Warning: Could not extract valid checklist name from LLM response: '{response.strip()}'")
        return None
    except Exception as e:
        print(f"Error during LLM suggestion: {e}")
        traceback.print_exc()
        return None

# Modified execute_checklist to accept municipality and compiler directly
def execute_checklist(compiler, pdf_text, chosen_checklist_name, checklists_data, municipality):
    """Executes the chosen checklist, yields progress info."""
    print(f"Executing checklist: {chosen_checklist_name} for {municipality}")
    if not compiler:
        print("Error: Compiler not initialized for execution.")
        yield {"status": "error", "message": "Compiler not initialized."}
        return
    # Ensure compiler's municipality matches the current request
    if compiler.municipality != municipality:
         print(f"Warning: Compiler municipality mismatch ({compiler.municipality} vs {municipality}). Re-setting.")
         compiler.municipality = municipality
         compiler.hasSezioni = municipality in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST

    # Ensure pipeline is set if Llama (defensive check)
    if compiler.llm == LLAMA and not compiler.text_gen_pipeline:
        compiler.text_gen_pipeline = loaded_llm_objects.get("pipeline")
        if not compiler.text_gen_pipeline:
            yield {"status": "error", "message": "Llama pipeline missing."}
            return

    try:
        checklist_details = ChecklistCompiler.get_checklist(checklists_data, chosen_checklist_name)
        if not checklist_details:
            yield {"status": "error", "message": f"Checklist '{chosen_checklist_name}' not found."}
            return
    except Exception as e:
        yield {"status": "error", "message": f"Error retrieving checklist details: {e}"}
        return

    results = []
    checklist_points = checklist_details.get("Punti", [])
    total_points = len(checklist_points)
    if not checklist_points:
        yield {"status": "warning", "message": "Checklist has no points."}
        yield {"status": "completed", "results": []}
        return

    yield {"status": "progress", "current": 0, "total": total_points, "message": "Starting execution..."}

    for i, point in enumerate(checklist_points):
        current_point_num = i + 1
        num = point.get("num", "N/A")
        punto_text = point.get("Punto", "")
        istruzioni = point.get("Istruzioni", "")
        sezione = point.get("Sezione", "") if compiler.hasSezioni else ""

        progress_message = f"Processing point {current_point_num}/{total_points} ('{num}')"
        print(progress_message)
        yield {"status": "progress", "current": current_point_num, "total": total_points, "message": progress_message}

        try:
            prompt = compiler.generate_prompt(istruzioni=istruzioni, punto=punto_text, num=num, determina=pdf_text, sezione=sezione)
            llm_response = compiler.generate_response(complete_prompt=prompt, temperature=EXECUTE_TEMPERATURE)
            simple_response = compiler.analize_response(llm_response)
            results.append({
                "Numero Punto": num, "Testo Punto": punto_text,
                "Risposta Semplice": simple_response, "Risposta LLM Completa": llm_response.strip()
            })
        except Exception as e:
            error_message = f"Error processing point {num}: {e}"
            print(f"\n{error_message}")
            traceback.print_exc()
            results.append({
                "Numero Punto": num, "Testo Punto": punto_text,
                "Risposta Semplice": "ERROR", "Risposta LLM Completa": error_message
            })
            yield {"status": "point_error", "point_num": num, "message": error_message}

    yield {"status": "completed", "results": results}


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Serves the main upload form page."""
    session.clear()
    cleanup_llm()
    return render_template('index.html',
                           predefined_models=PREDEFINED_LOCAL_MODELS,
                           default_openai=DEFAULT_OPENAI_MODEL,
                           municipalities=AVAILABLE_MUNICIPALITIES)

@app.route('/suggest', methods=['POST'])
def suggest():
    """Handles initial PDF upload, LLM setup, and checklist suggestion."""
    # --- File Validation ---
    if 'pdf_file' not in request.files:
        flash('No file part selected.', 'error')
        return redirect(url_for('index'))
    file = request.files['pdf_file']
    if file.filename == '':
        flash('No PDF file selected.', 'error')
        return redirect(url_for('index'))
    if not file or not allowed_file(file.filename):
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))

    # --- Get Config from Form ---
    llm_type_choice = request.form.get('llm_type')
    municipality = request.form.get('municipality')
    model_name = None
    quantize = False
    device_map_setting = None
    llm_type = None

    if municipality not in AVAILABLE_MUNICIPALITIES:
        flash('Invalid municipality selected.', 'error')
        return redirect(url_for('index'))

    if llm_type_choice == 'local':
        llm_type = LLAMA
        model_name = request.form.get('local_model_id')
        custom_model_id = request.form.get('custom_local_model_id', '').strip()
        if custom_model_id: model_name = custom_model_id
        if not model_name:
             flash('Please select or enter a local model ID.', 'error')
             return redirect(url_for('index'))
        quantize = 'quantize' in request.form
        device_map_setting = request.form.get('device_map', 'auto')
    elif llm_type_choice == 'openai':
        llm_type = OPENAI
        model_name = request.form.get('openai_model_id', DEFAULT_OPENAI_MODEL).strip()
        if not model_name: model_name = DEFAULT_OPENAI_MODEL
    else:
        flash('Invalid LLM type selected.', 'error')
        return redirect(url_for('index'))

    print(f"Suggesting for Municipality: {municipality}, LLM Type: {llm_type}, Model: {model_name}")

    # --- Save PDF Temporarily ---
    pdf_filepath = None
    pdf_text = None
    original_filename = secure_filename(file.filename)
    try:
        # Generate unique filename to avoid conflicts
        temp_filename = str(uuid.uuid4()) + ".pdf"
        pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.seek(0) # Reset stream position before saving
        file.save(pdf_filepath)
        print(f"PDF saved temporarily to: {pdf_filepath}")

        # Extract text from the saved file
        pdf_text = pdf_to_text_from_file(pdf_filepath)
        if pdf_text is None:
            raise ValueError("Error extracting text from saved PDF.")
        if not pdf_text:
            flash('Could not extract any text from the PDF.', 'warning')
            # Clean up temp file before redirecting
            if pdf_filepath and os.path.exists(pdf_filepath): os.remove(pdf_filepath)
            return redirect(url_for('index'))
        print(f"PDF text extracted (length: {len(pdf_text)} chars).")

    except Exception as e:
        flash(f'Error saving or processing PDF: {e}', 'error')
        traceback.print_exc()
        # Clean up temp file if it exists
        if pdf_filepath and os.path.exists(pdf_filepath): os.remove(pdf_filepath)
        return redirect(url_for('index'))

    # --- Load Checklists ---
    checklists_data = load_checklists(municipality)
    if not checklists_data:
        flash(f'Error loading checklists for {municipality}.', 'error')
        if pdf_filepath and os.path.exists(pdf_filepath): os.remove(pdf_filepath)
        return redirect(url_for('index'))

    # --- Initialize LLM and Suggest ---
    compiler = None
    suggested_checklist_name = None
    try:
        # Pass municipality to initializer
        compiler = initialize_llm_and_compiler(llm_type, model_name, quantize, device_map_setting, municipality)
        if not compiler:
             raise RuntimeError('Failed to initialize LLM compiler.')

        suggested_checklist_name = suggest_checklist(compiler, pdf_text, checklists_data)
        # Keep LLM loaded for the next step

    except Exception as e_llm:
        flash(f"Error during LLM initialization or suggestion: {e_llm}", 'error')
        traceback.print_exc()
        cleanup_llm() # Cleanup on error
        if pdf_filepath and os.path.exists(pdf_filepath): os.remove(pdf_filepath)
        return redirect(url_for('index'))

    # --- Store necessary info in session (minimal data) ---
    session.clear() # Clear previous session data first
    session['pdf_filepath'] = pdf_filepath # Store path to temp file
    session['original_filename'] = original_filename
    session['municipality'] = municipality
    session['llm_type'] = llm_type
    session['model_name'] = model_name
    session['quantize'] = quantize
    session['device_map'] = device_map_setting
    session['suggested_checklist'] = suggested_checklist_name
    # Store checklist names for the confirmation page dropdown
    session['available_checklist_names'] = [chk['NomeChecklist'] for chk in checklists_data.get('checklists', [])]

    print("Session data set for confirmation.")
    # Redirect to the confirmation page
    return redirect(url_for('confirm_checklist'))


@app.route('/confirm', methods=['GET'])
def confirm_checklist():
    """Displays the suggested checklist and allows user confirmation/override."""
    # Retrieve data needed for the template from session
    suggested_checklist = session.get('suggested_checklist')
    available_checklist_names = session.get('available_checklist_names')
    municipality = session.get('municipality')
    original_filename = session.get('original_filename')

    # Check if essential data is present
    if not available_checklist_names or not municipality or not original_filename:
        flash('Session data missing or expired. Please start over.', 'error')
        return redirect(url_for('index'))

    # Note: We don't need the full checklist_data here anymore, just the names

    return render_template('confirm.html',
                           suggested_checklist=suggested_checklist,
                           # Pass the list of names directly
                           available_checklist_names=available_checklist_names,
                           municipality=municipality,
                           original_filename=original_filename)


@app.route('/execute', methods=['POST'])
def execute():
    """Executes the chosen checklist and returns the CSV."""
    # Retrieve necessary data from session
    pdf_filepath = session.get('pdf_filepath')
    municipality = session.get('municipality')
    original_filename = session.get('original_filename')
    llm_type = session.get('llm_type')
    model_name = session.get('model_name')
    # Get the *chosen* checklist from the confirmation form
    chosen_checklist_name = request.form.get('chosen_checklist')

    # Validate session data
    if not all([pdf_filepath, municipality, original_filename, llm_type, model_name, chosen_checklist_name]):
        flash('Session expired or required data missing. Please start over.', 'error')
        cleanup_llm()
        # Attempt to delete temp file if path exists in session
        if pdf_filepath and os.path.exists(pdf_filepath):
             try: os.remove(pdf_filepath)
             except OSError as e: print(f"Error deleting temp file {pdf_filepath} on session error: {e}")
        session.clear()
        return redirect(url_for('index'))

    # --- Reload necessary data ---
    pdf_text = None
    checklists_data = None
    try:
        print(f"Executing for: {municipality}, Checklist: {chosen_checklist_name}, File: {pdf_filepath}")
        pdf_text = pdf_to_text_from_file(pdf_filepath)
        if pdf_text is None:
            raise ValueError("Failed to read PDF text from temporary file.")

        checklists_data = load_checklists(municipality)
        if not checklists_data:
            raise ValueError(f"Failed to load checklists for {municipality}.")

    except Exception as e_reload:
        flash(f"Error preparing for execution: {e_reload}", 'error')
        cleanup_llm()
        if pdf_filepath and os.path.exists(pdf_filepath):
            try: os.remove(pdf_filepath)
            except OSError as e: print(f"Error deleting temp file {pdf_filepath} on reload error: {e}")
        session.clear()
        return redirect(url_for('index'))


    # --- Get the globally loaded compiler (or re-initialize) ---
    compiler = loaded_llm_objects.get("compiler")
    if not compiler or loaded_llm_objects.get("type") != llm_type or loaded_llm_objects.get("model_name") != model_name:
        print("Compiler missing or mismatched, attempting re-initialization...")
        try:
            quantize = session.get('quantize', False)
            device_map = session.get('device_map', 'auto')
            # Pass municipality explicitly
            compiler = initialize_llm_and_compiler(llm_type, model_name, quantize, device_map, municipality)
            if not compiler: raise RuntimeError("Re-initialization failed.")
        except Exception as e_reinit:
            flash(f"Error re-initializing LLM for execution: {e_reinit}", 'error')
            cleanup_llm()
            if pdf_filepath and os.path.exists(pdf_filepath):
                 try: os.remove(pdf_filepath)
                 except OSError as e: print(f"Error deleting temp file {pdf_filepath} on reinit error: {e}")
            session.clear()
            return redirect(url_for('index'))

    # --- Execute Checklist ---
    final_results = None
    execution_successful = False
    try:
        print("Starting checklist execution generator...")
        all_results_list = []
        # Pass municipality to execute_checklist
        for progress_update in execute_checklist(compiler, pdf_text, chosen_checklist_name, checklists_data, municipality):
            if progress_update["status"] == "completed":
                all_results_list = progress_update["results"]
                execution_successful = True
                print("Checklist execution completed.")
                break
            elif progress_update["status"] == "error":
                flash(f"Execution failed: {progress_update.get('message', 'Unknown error')}", 'error')
                execution_successful = False
                break
            elif progress_update["status"] == "point_error":
                 print(f"Error on point {progress_update.get('point_num')}: {progress_update.get('message')}")
            elif progress_update["status"] == "progress":
                 print(f"Progress: {progress_update.get('message')}")
                 pass

        if not execution_successful and not all_results_list:
             if not any(msg[1] == 'error' for msg in session.get('_flashes', [])):
                  flash('Checklist execution did not complete successfully.', 'error')
             # Don't redirect here, cleanup happens in finally

        final_results = all_results_list

    except Exception as e_exec:
         flash(f"An unexpected error occurred during checklist execution: {e_exec}", 'error')
         traceback.print_exc()
         execution_successful = False
    finally:
        # --- Cleanup LLM and Temporary File ---
        cleanup_llm()
        if pdf_filepath and os.path.exists(pdf_filepath):
            try:
                os.remove(pdf_filepath)
                print(f"Temporary PDF file deleted: {pdf_filepath}")
            except OSError as e:
                print(f"Error deleting temporary PDF file {pdf_filepath}: {e}")
        # Clear session data AFTER potential file download
        session.clear()


    # --- Generate and Send CSV ---
    if execution_successful and final_results is not None:
        try:
            df_results = pd.DataFrame(final_results)
            expected_cols = ["Numero Punto", "Testo Punto", "Risposta Semplice", "Risposta LLM Completa"]
            for col in expected_cols:
                if col not in df_results.columns: df_results[col] = None
            df_results = df_results[expected_cols]

            csv_buffer = io.StringIO()
            df_results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_buffer.seek(0)

            output_filename = os.path.splitext(original_filename)[0] + "_checklist_results.csv"
            print(f"Sending CSV file: {output_filename}")
            return send_file(
                io.BytesIO(csv_buffer.getvalue().encode('utf-8-sig')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=output_filename
            )
        except Exception as e_csv:
            flash(f"Error generating CSV output: {e_csv}", 'error')
            traceback.print_exc()
            return redirect(url_for('index')) # Redirect to index as session is cleared
    else:
        # Redirect to index as session is cleared
        if not any(msg[1] == 'error' for msg in session.get('_flashes', [])):
             flash("Processing did not generate results or failed.", 'warning')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True)