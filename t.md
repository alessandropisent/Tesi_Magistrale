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
# Average estimated seconds per checklist point (highly approximate)
AVG_SECONDS_PER_POINT = 7

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
        # Add number of points to each checklist for estimation
        for chk in checklists_data["checklists"]:
            chk['point_count'] = len(chk.get('Punti', []))
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
        # Check if pipeline exists and call __del__ if available (some pipelines might have specific cleanup)
        pipeline_obj = loaded_llm_objects.get("pipeline")
        if pipeline_obj and hasattr(pipeline_obj, '__del__'):
            pipeline_obj.__del__()
        elif pipeline_obj:
             del pipeline_obj

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
             # More robust search within the response
             if re.search(rf'^\**\s*{re.escape(name)}\s*\**', response, re.IGNORECASE | re.MULTILINE):
                 print(f"LLM suggested checklist (found start of line): '{name}'")
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

# Changed: Now returns the list directly, no longer a generator
def execute_checklist_sync(compiler, pdf_text, chosen_checklist_name, checklists_data, municipality):
    """Executes the chosen checklist synchronously and returns results."""
    print(f"Executing checklist: {chosen_checklist_name} for {municipality}")
    if not compiler:
        print("Error: Compiler not initialized for execution.")
        raise RuntimeError("Compiler not initialized.")

    # Ensure compiler's municipality matches
    if compiler.municipality != municipality:
         print(f"Warning: Compiler municipality mismatch ({compiler.municipality} vs {municipality}). Re-setting.")
         compiler.municipality = municipality
         compiler.hasSezioni = municipality in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST

    # Ensure pipeline is set if Llama
    if compiler.llm == LLAMA and not compiler.text_gen_pipeline:
        compiler.text_gen_pipeline = loaded_llm_objects.get("pipeline")
        if not compiler.text_gen_pipeline:
            raise RuntimeError("Llama pipeline missing.")

    try:
        checklist_details = ChecklistCompiler.get_checklist(checklists_data, chosen_checklist_name)
        if not checklist_details:
            raise ValueError(f"Checklist '{chosen_checklist_name}' not found.")
    except Exception as e:
        raise ValueError(f"Error retrieving checklist details: {e}")

    results = []
    checklist_points = checklist_details.get("Punti", [])
    total_points = len(checklist_points)
    if not checklist_points:
        print("Warning: Checklist has no points.")
        return [] # Return empty list

    print(f"Starting execution of {total_points} points...")
    # Use tqdm for console logging during development
    for i, point in enumerate(tqdm(checklist_points, desc="Checklist Execution", file=sys.stdout)):
        current_point_num = i + 1
        num = point.get("num", "N/A")
        punto_text = point.get("Punto", "")
        istruzioni = point.get("Istruzioni", "")
        sezione = point.get("Sezione", "") if compiler.hasSezioni else ""

        progress_message = f"Processing point {current_point_num}/{total_points} ('{num}')"
        # print(progress_message) # tqdm handles progress display

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
            print(f"\n{error_message}") # Print error distinctly
            traceback.print_exc()
            results.append({
                "Numero Punto": num, "Testo Punto": punto_text,
                "Risposta Semplice": "ERROR", "Risposta LLM Completa": error_message
            })
            # Continue processing other points

    print("Checklist execution finished.")
    return results


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
        temp_filename = str(uuid.uuid4()) + ".pdf"
        pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.seek(0)
        file.save(pdf_filepath)
        print(f"PDF saved temporarily to: {pdf_filepath}")

        pdf_text = pdf_to_text_from_file(pdf_filepath)
        if pdf_text is None: raise ValueError("Error extracting text from saved PDF.")
        if not pdf_text:
            flash('Could not extract any text from the PDF.', 'warning')
            if pdf_filepath and os.path.exists(pdf_filepath): os.remove(pdf_filepath)
            return redirect(url_for('index'))
        print(f"PDF text extracted (length: {len(pdf_text)} chars).")

    except Exception as e:
        flash(f'Error saving or processing PDF: {e}', 'error')
        traceback.print_exc()
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
    estimated_time_str = "N/A" # Default estimate
    try:
        compiler = initialize_llm_and_compiler(llm_type, model_name, quantize, device_map_setting, municipality)
        if not compiler: raise RuntimeError('Failed to initialize LLM compiler.')

        suggested_checklist_name = suggest_checklist(compiler, pdf_text, checklists_data)

        # Calculate estimated time based on suggested (or first if no suggestion)
        checklist_to_estimate = suggested_checklist_name
        if not checklist_to_estimate and checklists_data.get("checklists"):
             checklist_to_estimate = checklists_data["checklists"][0]["NomeChecklist"]

        if checklist_to_estimate:
             for chk in checklists_data.get("checklists", []):
                 if chk['NomeChecklist'] == checklist_to_estimate:
                     point_count = chk.get('point_count', 0)
                     total_seconds = point_count * AVG_SECONDS_PER_POINT
                     minutes = total_seconds // 60
                     seconds = total_seconds % 60
                     estimated_time_str = f"~{minutes} min {seconds} sec (very approximate)"
                     break

    except Exception as e_llm:
        flash(f"Error during LLM initialization or suggestion: {e_llm}", 'error')
        traceback.print_exc()
        cleanup_llm()
        if pdf_filepath and os.path.exists(pdf_filepath): os.remove(pdf_filepath)
        return redirect(url_for('index'))

    # --- Store necessary info in session ---
    session.clear()
    session['pdf_filepath'] = pdf_filepath
    session['original_filename'] = original_filename
    session['municipality'] = municipality
    session['llm_type'] = llm_type
    session['model_name'] = model_name
    session['quantize'] = quantize
    session['device_map'] = device_map_setting
    session['suggested_checklist'] = suggested_checklist_name
    # Store full checklist data now, needed for time estimate on confirm page
    session['checklists_data_json'] = json.dumps(checklists_data)
    session['estimated_time_str'] = estimated_time_str # Store estimate

    print("Session data set for confirmation.")
    return redirect(url_for('confirm_checklist'))


@app.route('/confirm', methods=['GET'])
def confirm_checklist():
    """Displays the suggested checklist and allows user confirmation/override."""
    suggested_checklist = session.get('suggested_checklist')
    checklists_data_json = session.get('checklists_data_json')
    municipality = session.get('municipality')
    original_filename = session.get('original_filename')
    estimated_time_str = session.get('estimated_time_str', "N/A") # Get estimate

    if not checklists_data_json or not municipality or not original_filename:
        flash('Session data missing or expired. Please start over.', 'error')
        return redirect(url_for('index'))

    try:
        checklists_data = json.loads(checklists_data_json)
        # Extract full checklist details for display (including point count for estimate recalc)
        available_checklists = checklists_data.get("checklists", [])
        if not available_checklists:
             flash(f'No checklists found for {municipality}.', 'error')
             return redirect(url_for('index'))
    except json.JSONDecodeError:
         flash('Error reading checklist data from session.', 'error')
         return redirect(url_for('index'))

    # Pass full checklist details to template
    return render_template('confirm.html',
                           suggested_checklist=suggested_checklist,
                           available_checklists=available_checklists, # Pass full list
                           municipality=municipality,
                           original_filename=original_filename,
                           estimated_time_str=estimated_time_str, # Pass initial estimate
                           avg_seconds_per_point=AVG_SECONDS_PER_POINT) # Pass avg time for JS recalc


@app.route('/execute', methods=['POST'])
def execute():
    """Executes the chosen checklist and renders the results page."""
    # Retrieve necessary data from session
    pdf_filepath = session.get('pdf_filepath')
    municipality = session.get('municipality')
    original_filename = session.get('original_filename')
    llm_type = session.get('llm_type')
    model_name = session.get('model_name')
    chosen_checklist_name = request.form.get('chosen_checklist') # Get chosen checklist

    # Validate session data
    if not all([pdf_filepath, municipality, original_filename, llm_type, model_name, chosen_checklist_name]):
        flash('Session expired or required data missing. Please start over.', 'error')
        cleanup_llm()
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
        if pdf_text is None: raise ValueError("Failed to read PDF text from temporary file.")

        checklists_data = load_checklists(municipality) # Reload checklists
        if not checklists_data: raise ValueError(f"Failed to load checklists for {municipality}.")

    except Exception as e_reload:
        flash(f"Error preparing for execution: {e_reload}", 'error')
        cleanup_llm()
        if pdf_filepath and os.path.exists(pdf_filepath):
            try: os.remove(pdf_filepath)
            except OSError as e: print(f"Error deleting temp file {pdf_filepath} on reload error: {e}")
        session.clear()
        return redirect(url_for('index'))

    # --- Get/Re-initialize LLM Compiler ---
    compiler = loaded_llm_objects.get("compiler")
    # Check type and model name match as well
    if not compiler or loaded_llm_objects.get("type") != llm_type or loaded_llm_objects.get("model_name") != model_name:
        print("Compiler missing or mismatched, attempting re-initialization...")
        try:
            quantize = session.get('quantize', False)
            device_map = session.get('device_map', 'auto')
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

    # --- Execute Checklist Synchronously ---
    final_results = None
    execution_successful = False
    try:
        # Call the synchronous version
        final_results = execute_checklist_sync(compiler, pdf_text, chosen_checklist_name, checklists_data, municipality)
        execution_successful = True # Assume success if no exception

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
        # Keep results in session for download, clear other specific items
        session.pop('pdf_filepath', None)
        # Keep municipality, llm_type, model_name for results page display? Or pass via render_template? Let's pass.
        # session.pop('llm_type', None)
        # session.pop('model_name', None)
        session.pop('quantize', None)
        session.pop('device_map', None)
        session.pop('suggested_checklist', None)
        session.pop('checklists_data_json', None) # No longer needed
        session.pop('available_checklist_names', None)
        session.pop('estimated_time_str', None)


    # --- Render Results Page ---
    if execution_successful and final_results is not None:
        # Store results in session for download links
        # Convert DataFrame to list of dicts for JSON serialization
        try:
            session['last_results'] = pd.DataFrame(final_results).to_dict(orient='records')
            session['last_filename_base'] = os.path.splitext(original_filename)[0]
        except Exception as e_session_save:
             flash(f"Warning: Could not store results in session for download: {e_session_save}", 'warning')
             session.pop('last_results', None) # Ensure it's removed if saving failed
             session.pop('last_filename_base', None)

        return render_template('results.html',
                               results=final_results, # Pass results list
                               original_filename=original_filename,
                               municipality=municipality,
                               llm_type=llm_type,
                               model_name=model_name,
                               chosen_checklist=chosen_checklist_name,
                               # Indicate if download is possible
                               download_available=session.get('last_results') is not None
                              )
    else:
        # Redirect back to confirmation page on failure
        flash("Checklist execution failed or produced no results.", 'error')
        # Session is partially cleared in finally, redirect to index might be safer
        session.clear()
        return redirect(url_for('index'))


@app.route('/download/<format>')
def download_results(format):
    """Generates and sends the results file (CSV or Excel)."""
    results_list = session.get('last_results')
    filename_base = session.get('last_filename_base', 'results')

    if results_list is None:
        flash("No results data found in session to download. Please run the process again.", 'error')
        return redirect(url_for('index'))

    try:
        df = pd.DataFrame(results_list)
        # Ensure standard columns
        expected_cols = ["Numero Punto", "Testo Punto", "Risposta Semplice", "Risposta LLM Completa"]
        for col in expected_cols:
             if col not in df.columns: df[col] = None
        df = df[expected_cols]

        output_buffer = io.BytesIO()

        if format == 'csv':
            df.to_csv(output_buffer, index=False, encoding='utf-8-sig')
            mimetype = 'text/csv'
            filename = f"{filename_base}_results.csv"
            output_buffer.seek(0)
            encoded_buffer = io.BytesIO(output_buffer.getvalue().encode('utf-8-sig')) # Re-encode for send_file
            encoded_buffer.seek(0)

        elif format == 'excel':
            try:
                # Use openpyxl engine explicitly
                df.to_excel(output_buffer, index=False, engine='openpyxl')
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                filename = f"{filename_base}_results.xlsx"
                output_buffer.seek(0)
                encoded_buffer = output_buffer # No re-encoding needed for excel bytes
            except ImportError:
                 flash("Cannot generate Excel file: 'openpyxl' library not found. Please install it (`pip install openpyxl`). Downloading CSV instead.", 'warning')
                 # Fallback to CSV
                 output_buffer = io.StringIO() # Use StringIO for text
                 df.to_csv(output_buffer, index=False, encoding='utf-8-sig')
                 mimetype = 'text/csv'
                 filename = f"{filename_base}_results.csv"
                 output_buffer.seek(0)
                 encoded_buffer = io.BytesIO(output_buffer.getvalue().encode('utf-8-sig')) # Re-encode for send_file
                 encoded_buffer.seek(0)

        else:
            flash("Invalid download format requested.", 'error')
            return redirect(url_for('results')) # Or wherever results are shown

        # Clear results from session after preparing download
        session.pop('last_results', None)
        session.pop('last_filename_base', None)

        return send_file(
            encoded_buffer,
            mimetype=mimetype,
            as_attachment=True,
            download_name=filename
        )

    except Exception as e_download:
        flash(f"Error generating download file: {e_download}", 'error')
        traceback.print_exc()
        # Attempt to clear session data even on error
        session.pop('last_results', None)
        session.pop('last_filename_base', None)
        return redirect(url_for('index')) # Redirect home on download error


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True) # Use threaded=True for dev server handling multiple requests
```

**2. Updated `templates/confirm.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Confirm Checklist</title>
    <script src="https://cdn.tailwindcss.com"></script>
     <style>
        /* Simple loading spinner */
        .loader {
            border: 4px solid #f3f3f3; /* Light grey */
            border-top: 4px solid #3498db; /* Blue */
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            display: none; /* Hidden by default */
            margin-left: 10px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        /* Flash message styles (copied from index.html) */
        .flash-message{padding:1rem;margin-bottom:1rem;border-radius:.375rem;font-weight:500}.flash-error{background-color:#fef2f2;color:#dc2626;border:1px solid #fecaca}.flash-warning{background-color:#fffbeb;color:#d97706;border:1px solid #fde68a}.flash-success{background-color:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}
        /* Progress bar styles */
        #progress-container { display: none; margin-top: 1rem; }
        #progress-bar { width: 0%; height: 20px; background-color: #4caf50; text-align: center; line-height: 20px; color: white; border-radius: 5px; transition: width 0.3s ease-in-out; }
        #progress-text { margin-top: 0.5rem; font-size: 0.875rem; color: #4a5568; }

    </style>
</head>
<body class="bg-gray-100 font-sans antialiased">
    <div class="container mx-auto p-6 max-w-2xl">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Confirm Checklist Selection</h1>
        <p class="text-center text-gray-600 mb-2">Municipality: <span class="font-semibold">{{ municipality }}</span></p>
        <p class="text-center text-gray-600 mb-6">File: <span class="font-semibold">{{ original_filename }}</span></p>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="mb-4">
                {% for category, message in messages %}
                    <div class="flash-message flash-{{ category }}">{{ message }}</div>
                {% endfor %}
                </div>
            {% endif %}
        {% endwith %}

        <form id="confirm-form" action="{{ url_for('execute') }}" method="post" class="bg-white p-8 rounded-lg shadow-md space-y-6">

            <div class="bg-blue-50 border border-blue-200 p-4 rounded-md">
                <h2 class="text-lg font-semibold text-blue-800 mb-2">LLM Suggestion:</h2>
                {% if suggested_checklist %}
                    <p class="text-blue-700">The LLM suggests using the checklist: <strong class="font-bold">{{ suggested_checklist }}</strong></p>
                {% else %}
                    <p class="text-orange-700 bg-orange-100 p-2 rounded border border-orange-200">The LLM could not confidently suggest a specific checklist. Please select one below.</p>
                {% endif %}
            </div>

            <div>
                <label class="block text-sm font-medium text-gray-700 mb-2">Select the checklist to execute:</label>
                <div class="space-y-2">
                    {% for checklist in available_checklists %}
                        <label class="flex items-center p-3 border border-gray-200 rounded-md hover:bg-gray-50 cursor-pointer">
                            <input type="radio" name="chosen_checklist" value="{{ checklist.NomeChecklist }}" required
                                   data-point-count="{{ checklist.point_count | default(0) }}" {# Store point count #}
                                   {% if suggested_checklist == checklist.NomeChecklist %}checked{% endif %}
                                   onchange="updateEstimate()" {# Add onchange event #}
                                   class="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300">
                            <span class="ml-3 text-sm font-medium text-gray-900">{{ checklist.NomeChecklist }}</span>
                            {# Display point count #}
                            <span class="ml-auto text-xs text-gray-500 pl-2 text-right">({{ checklist.point_count | default(0) }} points)</span>
                        </label>
                    {% else %}
                        <p class="text-red-600">Error: No checklists available for this municipality.</p>
                    {% endfor %}
                </div>
                 <input type="hidden" name="chosen_checklist_ensure" value="fallback" style="display:none;">
            </div>

             <div class="text-center text-sm text-gray-600 mt-4">
                Estimated Processing Time: <strong id="estimated-time">{{ estimated_time_str }}</strong>
            </div>


            <div class="flex items-center justify-center pt-4">
                <button type="submit" id="submit-button"
                        class="inline-flex items-center justify-center px-6 py-2 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50">
                    Execute Selected Checklist
                </button>
                 <div class="loader" id="loader"></div>
            </div>

             <div id="progress-container" class="border-t pt-4">
                <div class="w-full bg-gray-200 rounded-full h-5 dark:bg-gray-700">
                    <div id="progress-bar" class="bg-blue-600 h-5 rounded-full text-xs font-medium text-blue-100 text-center p-0.5 leading-none" style="width: 0%">0%</div>
                </div>
                <p id="progress-text" class="text-center text-sm text-gray-600 mt-2">Initializing...</p>
            </div>


        </form>
         <div class="text-center mt-4">
             <a href="{{ url_for('index') }}" class="text-sm text-blue-600 hover:text-blue-800">&larr; Start Over</a>
        </div>
    </div>

    <script>
        const form = document.getElementById('confirm-form');
        const submitButton = document.getElementById('submit-button');
        const loader = document.getElementById('loader');
        const progressContainer = document.getElementById('progress-container');
        const progressBar = document.getElementById('progress-bar');
        const progressText = document.getElementById('progress-text');
        const estimatedTimeSpan = document.getElementById('estimated-time');
        const avgSecondsPerPoint = {{ avg_seconds_per_point | default(7) }}; // Get avg time from Flask

        function formatTime(totalSeconds) {
            if (isNaN(totalSeconds) || totalSeconds <= 0) {
                return "N/A";
            }
            const minutes = Math.floor(totalSeconds / 60);
            const seconds = Math.round(totalSeconds % 60);
            return `~${minutes} min ${seconds} sec (very approximate)`;
        }

        function updateEstimate() {
            const selectedRadio = document.querySelector('input[name="chosen_checklist"]:checked');
            if (selectedRadio) {
                const pointCount = parseInt(selectedRadio.getAttribute('data-point-count'), 10);
                const estimatedSeconds = pointCount * avgSecondsPerPoint;
                estimatedTimeSpan.textContent = formatTime(estimatedSeconds);
            } else {
                 estimatedTimeSpan.textContent = "{{ estimated_time_str }}"; // Reset to initial if somehow none selected
            }
        }


        form.addEventListener('submit', function(event) {
            const selectedChecklist = document.querySelector('input[name="chosen_checklist"]:checked');
            if (!selectedChecklist) {
                 alert('Please select a checklist to execute.');
                 event.preventDefault();
                 return;
            }
            // Disable button and show loader immediately
            submitButton.disabled = true;
            submitButton.textContent = 'Executing...';
            loader.style.display = 'inline-block';
            progressContainer.style.display = 'block';
            progressBar.style.width = '5%'; // Show minimal progress
            progressBar.textContent = '';
            progressText.textContent = 'Starting execution, this may take a while... Check server console for progress.';

            // No SSE implementation here, the form submits and waits for the response (which is the results page)
        });

         // Update estimate when the page loads based on the initially checked radio
         document.addEventListener('DOMContentLoaded', updateEstimate);

    </script>

</body>
</html>
```

**3. New `templates/results.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checklist Results</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Optional: Zebra striping for table */
        tbody tr:nth-child(odd) {
            background-color: #f9fafb; /* gray-50 */
        }
        tbody tr:hover {
            background-color: #f3f4f6; /* gray-100 */
        }
    </style>
</head>
<body class="bg-gray-100 font-sans antialiased">
    <div class="container mx-auto p-6 max-w-6xl"> <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Checklist Execution Results</h1>

        <div class="bg-white p-4 rounded-lg shadow-md mb-6 text-sm text-gray-700">
            <p><strong>File:</strong> {{ original_filename }}</p>
            <p><strong>Municipality:</strong> {{ municipality }}</p>
            <p><strong>Checklist Executed:</strong> {{ chosen_checklist }}</p>
            <p><strong>LLM Used:</strong> {{ llm_type }} ({{ model_name }})</p>
        </div>

        {% if download_available %}
        <div class="mb-6 text-center space-x-4">
            <a href="{{ url_for('download_results', format='csv') }}"
               class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500">
               Download CSV
            </a>
            <a href="{{ url_for('download_results', format='excel') }}"
               class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500">
               Download Excel
            </a>
        </div>
        {% else %}
        <div class="mb-6 text-center p-3 bg-yellow-100 border border-yellow-300 text-yellow-800 rounded-md text-sm">
            Download links are unavailable (results might not have been stored correctly).
        </div>
        {% endif %}

        <div class="overflow-x-auto bg-white rounded-lg shadow">
            <table class="min-w-full divide-y divide-gray-200">
                <thead class="bg-gray-50">
                    <tr>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Point #
                        </th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Checklist Point Text
                        </th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Simple Answer
                        </th>
                        <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                            Full LLM Response
                        </th>
                    </tr>
                </thead>
                <tbody class="bg-white divide-y divide-gray-200">
                    {% if results %}
                        {% for row in results %}
                        <tr>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                {{ row['Numero Punto'] | default('N/A') }}
                            </td>
                            <td class="px-6 py-4 text-sm text-gray-700">
                                {{ row['Testo Punto'] | default('') }}
                            </td>
                            <td class="px-6 py-4 whitespace-nowrap text-sm font-semibold
                                       {% if row['Risposta Semplice'] == 'SI' %} text-green-600
                                       {% elif row['Risposta Semplice'] == 'NO' %} text-red-600
                                       {% elif row['Risposta Semplice'] == 'ERROR' %} text-red-700 bg-red-100
                                       {% else %} text-gray-600 {% endif %}">
                                {{ row['Risposta Semplice'] | default('N/A') }}
                            </td>
                            <td class="px-6 py-4 text-sm text-gray-600 break-words"> {# Allow long responses to wrap #}
                                {{ row['Risposta LLM Completa'] | default('') }}
                            </td>
                        </tr>
                        {% endfor %}
                    {% else %}
                        <tr>
                            <td colspan="4" class="px-6 py-4 text-center text-sm text-gray-500">
                                No results generated or checklist was empty.
                            </td>
                        </tr>
                    {% endif %}
                </tbody>
            </table>
        </div>

        <div class="text-center mt-8">
            <a href="{{ url_for('index') }}" class="text-sm text-blue-600 hover:text-blue-800">&larr; Process Another PDF</a>
       </div>

    </div>
</body>
</html>
```

**Summary of Key Changes:**

* **`app.py`:**
    * `execute_checklist` is now synchronous again (`execute_checklist_sync`) as real-time SSE updates are complex to add correctly and reliably without a proper task queue. It returns the full results list upon completion.
    * `/execute` route now calls `execute_checklist_sync`. After completion, it stores the results (as a list of dicts) and a base filename in the session (`session['last_results']`, `session['last_filename_base']`). It then renders the new `results.html` template, passing the results data directly. It also cleans up the LLM and temporary PDF file *after* execution.
    * New `/download/<format>` route added. It retrieves `last_results` and `last_filename_base` from the session. It uses Pandas to create either a CSV or Excel file in memory (`io.BytesIO`) and returns it using `send_file`. It includes basic error handling for missing `openpyxl` and clears the results from the session after preparing the download.
    * Time estimation logic added in `/suggest` and passed to `confirm.html`.
* **`templates/confirm.html`:**
    * Added a display area for the estimated time (`<strong id="estimated-time">...</strong>`).
    * Added JavaScript (`updateEstimate`, `formatTime`) to calculate and update the estimated time displayed based on the selected checklist's point count and the average time passed from Flask. It updates when the page loads and when the radio button selection changes.
    * The progress bar remains but serves mainly as an indicator that the process *is running* during the `/execute` request, as real-time updates are not implemented.
* **`templates/results.html`:**
    * New template created.
    * Displays metadata about the run (filename, municipality, checklist, LLM).
    * Includes "Download CSV" and "Download Excel" buttons linking to `/download/csv` and `/download/excel`.
    * Renders the results list (passed from the `/execute` route) into an HTML table with basic styling.
    * Provides a link to go back and process another PDF.

This version provides the results page and download functionality as requested, along with a basic time estimate, while keeping the backend processing synchronous for simplicity. Remember to install `openpyxl` (`pip install openpyxl`) if you want the Excel download to wo