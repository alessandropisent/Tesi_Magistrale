# app.py
import os
import sys
import json
import pandas as pd
import re
import io # Required for sending file data
import traceback
import uuid
import time
import math # For time estimation
import datetime # For time estimation
import logging # Added for logging
# Use Flask-Session for server-side sessions
from flask import Flask, render_template, request, send_file, flash, redirect, url_for, session, jsonify, Response, after_this_request, g, send_from_directory
from flask_session import Session # Import Flask-Session
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader

# --- LLM Imports ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import openai
from dotenv import load_dotenv

# --- Checklist Compiler Logic ---
try:
    # Ensure ChecklistCompiler.py is accessible
    from ChecklistCompiler import ChecklistCompiler, LUCCA, OLBIA, LLAMA, OPENAI, MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
except ImportError:
    print("ERROR: Could not import ChecklistCompiler or required constants.")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, # Set default level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__) # Get logger for this module

# --- Flask App Setup ---
app = Flask(__name__)
# Configuration from Environment Variables with Defaults
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_please_change_in_prod')

# --- Configure Server-Side Sessions ---
# Option 1: Filesystem Session (Default and simple)
app.config['SESSION_TYPE'] = 'filesystem'
# Option 2: Redis Session (Requires Redis server and `pip install redis`)
# app.config['SESSION_TYPE'] = 'redis'
# app.config['SESSION_REDIS'] = redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379/0'))
# Option 3: SQLAlchemy Session (Requires DB setup and `pip install Flask-SQLAlchemy`)
# app.config['SESSION_TYPE'] = 'sqlalchemy'
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///sessions.db')
# app.config['SESSION_SQLALCHEMY'] = db # Initialize SQLAlchemy 'db' object first
# app.config['SESSION_SQLALCHEMY_TABLE'] = 'sessions'

# Ensure session directory exists if using filesystem
if app.config['SESSION_TYPE'] == 'filesystem':
    session_dir = os.path.join(os.path.dirname(__file__), 'flask_session')
    os.makedirs(session_dir, exist_ok=True)
    app.config['SESSION_FILE_DIR'] = session_dir

app.config['SESSION_PERMANENT'] = False # Session expires when browser closes
app.config['SESSION_USE_SIGNER'] = True # Sign session data for security
# Initialize the session extension
Session(app)
# --- End Session Configuration ---


# Ensure UPLOAD_FOLDER exists
upload_folder_path = os.path.abspath(os.environ.get('UPLOAD_FOLDER', 'uploads'))
os.makedirs(upload_folder_path, exist_ok=True)
app.config['UPLOAD_FOLDER'] = upload_folder_path
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_FILE_MB', 32)) * 1024 * 1024 # e.g., 32MB limit
ALLOWED_EXTENSIONS = {'pdf'}

logger.info(f"Upload folder set to: {app.config['UPLOAD_FOLDER']}")
logger.info(f"Session type set to: {app.config['SESSION_TYPE']}")


# --- Configuration ---
AVAILABLE_MUNICIPALITIES = {
    LUCCA: {"name": "Lucca", "data_path": "./src/txt/Lucca/"},
    OLBIA: {"name": "Olbia", "data_path": "./src/txt/Olbia/"},
    # Add more municipalities here easily
    # "AnotherCity": {"name": "Another City", "data_path": "./src/txt/AnotherCity/"}
}
# Base data path (now fetched from AVAILABLE_MUNICIPALITIES)
SUGGEST_TEMPERATURE = 0.1 # Slightly higher for suggestion flexibility
EXECUTE_TEMPERATURE = 0.01 # As requested for execution
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
# Full list of models
PREDEFINED_LOCAL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
logger.info(f"Available municipalities: {list(AVAILABLE_MUNICIPALITIES.keys())}")
logger.info(f"Predefined local models: {PREDEFINED_LOCAL_MODELS}")


# --- Global LLM State ---
loaded_llm_objects = {
    "model": None, "tokenizer": None, "pipeline": None,
    "compiler": None, "type": None, "model_name": None, "quantization": None
}

# --- Request Timing ---
@app.before_request
def before_request():
    g.start_time = time.time()
    logger.info(f"Request received: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"Request finished: {request.method} {request.path} - Status {response.status_code} - Duration: {duration:.4f}s")
    elif response:
         logger.info(f"Request finished: {request.method} {request.path} - Status {response.status_code}")
    return response

# --- Helper Functions (Unchanged) ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_text_from_file(pdf_filepath):
    """Extracts text from a PDF file path."""
    logger.info(f"Attempting to extract text from PDF: {pdf_filepath}")
    start_pdf_time = time.time()
    if not os.path.exists(pdf_filepath):
        logger.error(f"PDF file not found at {pdf_filepath}")
        return None
    try:
        with open(pdf_filepath, 'rb') as f:
            reader = PdfReader(f)
            text = ""
            if reader.is_encrypted:
                try:
                    reader.decrypt('')
                    logger.info(f"Decrypted PDF {pdf_filepath} with empty password.")
                except Exception as decrypt_e:
                    logger.error(f"PDF {pdf_filepath} is encrypted and could not be decrypted: {decrypt_e}", exc_info=True)
                    return None # Cannot proceed if encrypted and fails

            num_pages = len(reader.pages)
            logger.info(f"Processing {num_pages} pages in {pdf_filepath}")
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as page_e:
                    logger.warning(f"Could not extract text from page {i+1}/{num_pages} of {pdf_filepath}. Error: {page_e}")

            duration_pdf = time.time() - start_pdf_time
            if not text:
                logger.warning(f"No text could be extracted from PDF: {pdf_filepath} (Duration: {duration_pdf:.4f}s)")
                return "" # Return empty string if no text extracted
            logger.info(f"Successfully extracted {len(text)} characters from PDF: {pdf_filepath} (Duration: {duration_pdf:.4f}s)")
            return text
    except Exception as e:
        duration_pdf = time.time() - start_pdf_time
        logger.error(f"Error reading PDF file {pdf_filepath} (Duration: {duration_pdf:.4f}s): {e}", exc_info=True)
        return None # Return None on critical read error


def load_checklists(municipality):
    """Loads checklist data for the specified municipality."""
    logger.info(f"Loading checklists for municipality: {municipality}")
    start_load_time = time.time()
    if municipality not in AVAILABLE_MUNICIPALITIES:
        logger.error(f"Municipality '{municipality}' not configured.")
        return None

    # Construct path using configured base path for the municipality
    base_path = AVAILABLE_MUNICIPALITIES[municipality].get("data_path")
    if not base_path:
         logger.error(f"Configuration error: 'data_path' not set for municipality '{municipality}'.")
         return None
    checklist_path = os.path.join(base_path, "checklists", "checklists.json")

    try:
        with open(checklist_path, "r", encoding="utf-8") as f:
            checklists_data = json.load(f)
        # Basic validation
        if "checklists" not in checklists_data or not isinstance(checklists_data["checklists"], list):
             logger.error(f"Checklist JSON format invalid: 'checklists' key missing or not a list in {checklist_path}")
             return None
        # Ensure essential keys exist
        for i, chk in enumerate(checklists_data["checklists"]):
             if not all(k in chk for k in ['NomeChecklist', 'Descrizione', 'Punti']):
                 logger.error(f"Checklist entry {i} in {checklist_path} is missing required keys ('NomeChecklist', 'Descrizione', 'Punti').")
                 return None
        duration_load = time.time() - start_load_time
        logger.info(f"Successfully loaded {len(checklists_data['checklists'])} checklists for {municipality} from {checklist_path} (Duration: {duration_load:.4f}s)")
        return checklists_data
    except FileNotFoundError:
        logger.error(f"Checklists file not found: {checklist_path}")
        return None
    except json.JSONDecodeError as e_json:
         logger.error(f"Error decoding JSON from {checklist_path}: {e_json}", exc_info=True)
         return None
    except Exception as e:
        duration_load = time.time() - start_load_time
        logger.error(f"Error loading checklists for {municipality} from {checklist_path} (Duration: {duration_load:.4f}s): {e}", exc_info=True)
        return None

# --- initialize_llm_and_compiler and cleanup_llm (Unchanged) ---
def initialize_llm_and_compiler(llm_type, model_name, quantization, device_map_setting, municipality):
    """Loads LLM (if local) and initializes the compiler for a given municipality."""
    global loaded_llm_objects
    init_start_time = time.time()
    needs_init = False
    reason = ""

    # Determine if re-initialization is needed based on type, model, or quantization
    if (loaded_llm_objects.get("type") != llm_type or
        loaded_llm_objects.get("model_name") != model_name or
        loaded_llm_objects.get("quantization") != quantization or # Check quantization change
        not loaded_llm_objects.get("compiler")):
        needs_init = True
        reason = f"LLM config changed (Type: {loaded_llm_objects.get('type')}->{llm_type}, Model: {loaded_llm_objects.get('model_name')}->{model_name}, Quant: {loaded_llm_objects.get('quantization')}->{quantization}) or not loaded."
    elif loaded_llm_objects.get("compiler") and loaded_llm_objects["compiler"].municipality != municipality:
         logger.info(f"Municipality changed ({loaded_llm_objects['compiler'].municipality} -> {municipality}). Updating compiler only.")
         loaded_llm_objects["compiler"].municipality = municipality
         loaded_llm_objects["compiler"].hasSezioni = municipality in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
         duration_init = time.time() - init_start_time
         logger.info(f"Compiler municipality updated. (Duration: {duration_init:.4f}s)")
         return loaded_llm_objects["compiler"]
    else:
        logger.info("Using already loaded LLM and Compiler.")
        return loaded_llm_objects["compiler"]

    if needs_init:
        logger.info(f"Initializing LLM/Compiler. Reason: {reason}")
        cleanup_llm() # Clean up previous model first

    compiler = None
    text_gen_pipeline = None
    model_to_load = None
    tokenizer = None

    try:
        if llm_type == LLAMA:
            logger.info(f"Loading local LLAMA model '{model_name}' (Quantization: {quantization}, Device Map: {device_map_setting})")
            # --- Llama loading logic ---
            quantization_config = None
            load_in_4bit = quantization == '4bit'
            load_in_8bit = quantization == '8bit'

            model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": device_map_setting, "trust_remote_code": True}

            if load_in_4bit or load_in_8bit:
                logger.info(f"Setting up {quantization} quantization...")
                try:
                    # Configure BitsAndBytesConfig based on selection
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=load_in_4bit,
                        load_in_8bit=load_in_8bit,
                        bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None, # Only for 4bit
                        bnb_4bit_quant_type="nf4" if load_in_4bit else None, # Only for 4bit
                        bnb_4bit_use_double_quant=True if load_in_4bit else None # Only for 4bit
                        # 8bit doesn't use these specific 4bit params
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info(f"BitsAndBytesConfig created for {quantization}.")
                except ImportError:
                    logger.error("Bitsandbytes library not found for quantization.", exc_info=True)
                    raise RuntimeError("Bitsandbytes library not found for quantization.")
                except Exception as e_quant:
                    logger.warning(f"Error setting up quantization: {e_quant}. Proceeding without it.", exc_info=True)
                    model_kwargs.pop("quantization_config", None)
                    quantization = 'none' # Reset quantization state if setup failed

            model_load_start = time.time()
            model_to_load = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            logger.info(f"Loaded model '{model_name}' (Duration: {time.time() - model_load_start:.4f}s)")

            tokenizer_load_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer for '{model_name}' (Duration: {time.time() - tokenizer_load_start:.4f}s)")

            # Determine pipeline device argument
            pipeline_device_arg = None
            if device_map_setting != 'auto':
                try:
                    # Extract device index if specified (e.g., 'cuda:0' -> 0)
                    if ':' in device_map_setting:
                        pipeline_device_arg = int(device_map_setting.split(':')[-1])
                    elif device_map_setting == 'cpu':
                         pipeline_device_arg = -1
                    # Add more specific device handling if needed
                except ValueError:
                    logger.warning(f"Could not parse device index from '{device_map_setting}'. Defaulting pipeline device.")
                    pipeline_device_arg = None # Let pipeline decide or use default
            elif device_map_setting == 'auto' and not torch.cuda.is_available():
                 pipeline_device_arg = -1 # Explicitly set to CPU if no CUDA and auto

            pipeline_create_start = time.time()
            text_gen_pipeline = pipeline(
                "text-generation", model=model_to_load, tokenizer=tokenizer,
                device=pipeline_device_arg, # Pass explicit device index or None
                max_new_tokens=500, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, truncation=True
            )
            logger.info(f"Text generation pipeline created on device: {text_gen_pipeline.device} (Duration: {time.time() - pipeline_create_start:.4f}s)")

        elif llm_type == OPENAI:
            logger.info(f"Configuring for OpenAI model '{model_name}'")
            load_dotenv() # Load .env file if present
            if not os.getenv("OPENAI_API_KEY"):
                logger.error("OPENAI_API_KEY not found in environment variables.")
                raise RuntimeError("OPENAI_API_KEY not found in environment.")
            logger.info("OpenAI API Key found.")

        # Store loaded objects globally
        loaded_llm_objects["model"] = model_to_load
        loaded_llm_objects["tokenizer"] = tokenizer
        loaded_llm_objects["pipeline"] = text_gen_pipeline
        loaded_llm_objects["type"] = llm_type
        loaded_llm_objects["model_name"] = model_name
        loaded_llm_objects["quantization"] = quantization # Store quantization level

        # Initialize Compiler
        compiler_init_start = time.time()
        has_sezioni = municipality in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST
        compiler = ChecklistCompiler(llm=llm_type, municipality=municipality, model=model_name, text_gen_pipeline=text_gen_pipeline, hasSezioni=has_sezioni)
        loaded_llm_objects["compiler"] = compiler
        duration_init_total = time.time() - init_start_time
        logger.info(f"ChecklistCompiler initialized for {municipality}. LLM Type: {llm_type}. Model: {model_name}. Quant: {quantization}. (Total Init Duration: {duration_init_total:.4f}s)")
        return compiler

    except Exception as e_llm:
        duration_init_total = time.time() - init_start_time
        logger.error(f"Error initializing LLM or Compiler (Duration: {duration_init_total:.4f}s): {e_llm}", exc_info=True)
        cleanup_llm() # Ensure cleanup on error
        raise # Re-raise the exception

def cleanup_llm():
    """Releases LLM resources."""
    global loaded_llm_objects
    # Check if any non-compiler LLM object exists
    if not any(v is not None for k, v in loaded_llm_objects.items() if k not in ['compiler', 'quantization']):
        logger.info("Cleanup: No LLM model/tokenizer/pipeline resources currently loaded.")
    else:
        logger.info("Cleanup: Releasing LLM model/tokenizer/pipeline resources...")
        cleanup_start_time = time.time()
        # Safely delete objects
        pipeline_obj = loaded_llm_objects.pop("pipeline", None)
        if pipeline_obj: del pipeline_obj
        model_obj = loaded_llm_objects.pop("model", None)
        if model_obj: del model_obj
        tokenizer_obj = loaded_llm_objects.pop("tokenizer", None)
        if tokenizer_obj: del tokenizer_obj

        loaded_llm_objects["type"] = None
        loaded_llm_objects["model_name"] = None
        loaded_llm_objects["quantization"] = None # Reset quantization state

        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("Cleanup: CUDA cache cleared.")
            except Exception as e_cuda:
                logger.warning(f"Cleanup Warning: Error clearing CUDA cache: {e_cuda}", exc_info=True)
        duration_cleanup = time.time() - cleanup_start_time
        logger.info(f"Cleanup: LLM resources released. (Duration: {duration_cleanup:.4f}s)")

    # Clean up compiler separately if it exists
    compiler_obj = loaded_llm_objects.pop("compiler", None)
    if compiler_obj:
        del compiler_obj
        logger.info("Cleanup: Compiler object released.")

    # Ensure dict is fully reset
    loaded_llm_objects = {
        "model": None, "tokenizer": None, "pipeline": None,
        "compiler": None, "type": None, "model_name": None, "quantization": None
    }


# --- suggest_checklist (Unchanged) ---
def suggest_checklist(compiler, pdf_text, checklists_data):
    """Uses LLM to suggest a checklist."""
    logger.info("Attempting to suggest checklist...")
    suggest_start_time = time.time()
    if not compiler:
        logger.error("Suggest Checklist Error: Compiler not initialized.")
        return None

    if compiler.llm == LLAMA and not compiler.text_gen_pipeline:
        logger.warning("Suggest Checklist: Llama pipeline missing in compiler, attempting to get from global state.")
        compiler.text_gen_pipeline = loaded_llm_objects.get("pipeline")
        if not compiler.text_gen_pipeline:
            logger.error("Suggest Checklist Error: Llama pipeline missing in compiler and global state.")
            return None

    try:
        prompt_start_time = time.time()
        prompt = compiler.generate_prompt_choose(determina=pdf_text, checklists=checklists_data)
        logger.debug(f"Generated suggestion prompt (Duration: {time.time() - prompt_start_time:.4f}s)") # Log prompt only in debug

        response_start_time = time.time()
        # *** Use SUGGEST_TEMPERATURE ***
        response = compiler.generate_response(complete_prompt=prompt, temperature=SUGGEST_TEMPERATURE)
        logger.info(f"Received LLM suggestion response (Duration: {time.time() - response_start_time:.4f}s)")
        logger.debug(f"LLM Suggestion Raw Response: '{response.strip()}'") # Log raw response only in debug

        valid_checklist_names = [chk['NomeChecklist'] for chk in checklists_data.get('checklists', [])]
        if not valid_checklist_names:
             logger.warning("Suggest Checklist: No valid checklist names found in loaded data.")
             return None

        # Extraction logic (more robust)
        response_clean = response.strip()
        extracted_name = None

        # 1. Check if the entire response exactly matches a name (case-insensitive)
        for name in valid_checklist_names:
            if response_clean.lower() == name.lower():
                extracted_name = name
                break

        # 2. If not exact match, search for the name within the response (word boundary)
        if not extracted_name:
            for name in valid_checklist_names:
                # Use word boundaries (\b) to avoid partial matches like "Contratti" matching "Contratto"
                if re.search(rf'\b{re.escape(name)}\b', response_clean, re.IGNORECASE):
                    extracted_name = name
                    break # Take the first match found

        duration_suggest = time.time() - suggest_start_time
        if not extracted_name:
            logger.warning(f"Could not extract valid checklist name from LLM response. Response: '{response_clean}' (Duration: {duration_suggest:.4f}s)")
            return None
        else:
            logger.info(f"LLM suggested checklist: '{extracted_name}' (Total Suggest Duration: {duration_suggest:.4f}s)")
            return extracted_name

    except Exception as e:
        duration_suggest = time.time() - suggest_start_time
        logger.error(f"Error during LLM suggestion (Duration: {duration_suggest:.4f}s): {e}", exc_info=True)
        return None


# --- execute_checklist_sse (Unchanged) ---
def execute_checklist_sse(compiler, pdf_text, chosen_checklist_name, checklists_data, municipality, original_filename):
    """
    Executes the chosen checklist and yields SSE formatted progress updates.
    Generates CSV, JSON, and XLSX on completion.
    """
    exec_start_time = datetime.datetime.now()
    logger.info(f"SSE Execute Start: Checklist '{chosen_checklist_name}' for {municipality}. File: {original_filename}")
    results = []
    temp_filenames = {} # To store filenames for csv, json, xlsx

    try:
        # --- Initial Setup & Validation ---
        if not compiler:
            logger.error("SSE Execute Error: Compiler not initialized.")
            yield f"event: error\ndata: {json.dumps({'message': 'Compiler not initialized.'})}\n\n"
            return

        if compiler.municipality != municipality:
             logger.warning(f"SSE Execute: Compiler municipality mismatch ({compiler.municipality} vs {municipality}). Updating.")
             compiler.municipality = municipality
             compiler.hasSezioni = municipality in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST

        if compiler.llm == LLAMA and not compiler.text_gen_pipeline:
            logger.warning("SSE Execute: Llama pipeline missing, attempting reload from global.")
            compiler.text_gen_pipeline = loaded_llm_objects.get("pipeline")
            if not compiler.text_gen_pipeline:
                logger.error("SSE Execute Error: Llama pipeline missing in compiler and global state.")
                yield f"event: error\ndata: {json.dumps({'message': 'Llama pipeline missing.'})}\n\n"
                return

        checklist_details = ChecklistCompiler.get_checklist(checklists_data, chosen_checklist_name)
        if not checklist_details:
            logger.error(f"SSE Execute Error: Checklist '{chosen_checklist_name}' not found in data.")
            yield f"event: error\ndata: {json.dumps({'message': f'Checklist {chosen_checklist_name} not found.'})}\n\n"
            return

        checklist_points = checklist_details.get("Punti", [])
        total_points = len(checklist_points)
        logger.info(f"Executing checklist with {total_points} points.")

        if not checklist_points:
            logger.warning("SSE Execute: Checklist has no points.")
            yield f"event: progress\ndata: {json.dumps({'current': 0, 'total': 0, 'message': 'Checklist has no points.', 'eta_seconds': 0})}\n\n"
            # Generate empty files
            df_results = pd.DataFrame(columns=["Numero Punto", "Testo Punto", "Risposta Semplice", "Risposta LLM Completa"])
            base_filename = str(uuid.uuid4())
            temp_filenames = {
                'csv': base_filename + ".csv",
                'json': base_filename + ".json",
                'xlsx': base_filename + ".xlsx"
            }
            # Save empty files
            df_results.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], temp_filenames['csv']), index=False, encoding='utf-8-sig')
            df_results.to_json(os.path.join(app.config['UPLOAD_FOLDER'], temp_filenames['json']), orient='records', indent=4, force_ascii=False)
            with io.BytesIO() as buffer:
                 with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                     df_results.to_excel(writer, index=False, sheet_name='Results')
                 buffer.seek(0)
                 with open(os.path.join(app.config['UPLOAD_FOLDER'], temp_filenames['xlsx']), 'wb') as f:
                     f.write(buffer.read())
            logger.info(f"Generated empty result files with base name: {base_filename}")
            yield f"event: complete\ndata: {json.dumps({'status': 'success', 'message': 'Completed with no points.', 'download_filenames': temp_filenames, 'results': []})}\n\n"
            return

        # --- Initial Progress Update ---
        yield f"event: progress\ndata: {json.dumps({'current': 0, 'total': total_points, 'message': 'Starting execution...', 'eta_seconds': -1})}\n\n"
        time.sleep(0.1)

        # --- Process Points ---
        point_times = [] # Store duration of each point for better ETA
        for i, point in enumerate(checklist_points):
            point_start_time = time.time() # Use time.time() for duration calculation
            current_point_num = i + 1
            num = point.get("num", "N/A")
            punto_text_full = point.get("Punto", "")
            punto_text_log = punto_text_full[:100] + '...' if len(punto_text_full) > 100 else punto_text_full
            istruzioni = point.get("Istruzioni", "")
            sezione = point.get("Sezione", "") if compiler.hasSezioni else ""

            progress_message = f"Processing point {current_point_num}/{total_points} ('{num}')"
            logger.info(f"SSE Execute: {progress_message} - Text: '{punto_text_log}'")

            # --- Refined Time Estimation ---
            eta_seconds = -1
            if point_times: # If we have completed at least one point
                avg_time_per_point = sum(point_times) / len(point_times)
                remaining_points = total_points - current_point_num
                eta_seconds = math.ceil(avg_time_per_point * remaining_points)

            yield f"event: progress\ndata: {json.dumps({'current': current_point_num, 'total': total_points, 'message': progress_message, 'eta_seconds': eta_seconds})}\n\n"

            try:
                prompt_gen_start = time.time()
                prompt = compiler.generate_prompt(istruzioni=istruzioni, punto=punto_text_full, num=num, determina=pdf_text, sezione=sezione)
                logger.debug(f"Point {num}: Prompt generated (Duration: {time.time() - prompt_gen_start:.4f}s)")

                response_gen_start = time.time()
                # *** Use EXECUTE_TEMPERATURE ***
                llm_response = compiler.generate_response(complete_prompt=prompt, temperature=EXECUTE_TEMPERATURE)
                logger.debug(f"Point {num}: Response generated (Duration: {time.time() - response_gen_start:.4f}s)")

                analyze_start = time.time()
                simple_response = compiler.analize_response(llm_response)
                logger.debug(f"Point {num}: Response analyzed (Duration: {time.time() - analyze_start:.4f}s)")

                results.append({
                    "Numero Punto": num, "Testo Punto": punto_text_full, # Full text in results
                    "Risposta Semplice": simple_response, "Risposta LLM Completa": llm_response.strip()
                })
            except Exception as e_point:
                point_duration = time.time() - point_start_time
                error_message = f"Error processing point {num} (Duration: {point_duration:.4f}s): {e_point}"
                logger.error(error_message, exc_info=True)
                results.append({
                    "Numero Punto": num, "Testo Punto": punto_text_full,
                    "Risposta Semplice": "ERROR", "Risposta LLM Completa": error_message
                })
                yield f"event: point_error\ndata: {json.dumps({'point_num': num, 'message': str(e_point)})}\n\n" # Send simplified error message

            point_duration = time.time() - point_start_time
            point_times.append(point_duration) # Add duration to list for ETA calculation
            logger.info(f"SSE Execute: Point {current_point_num} finished. (Duration: {point_duration:.4f}s)")

        # --- Completion ---
        logger.info("Checklist execution finished. Generating result files...")
        file_gen_start = time.time()
        df_results = pd.DataFrame(results)
        # Ensure columns exist even if results list is empty or malformed
        expected_cols = ["Numero Punto", "Testo Punto", "Risposta Semplice", "Risposta LLM Completa"]
        for col in expected_cols:
             if col not in df_results.columns: df_results[col] = None
        df_results = df_results[expected_cols] # Reorder/select columns

        # Generate unique base filename
        base_filename = str(uuid.uuid4())
        temp_filenames = {
            'csv': base_filename + ".csv",
            'json': base_filename + ".json",
            'xlsx': base_filename + ".xlsx"
        }

        # Save CSV
        csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filenames['csv'])
        df_results.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
        logger.info(f"Generated results CSV: {csv_filepath}")

        # Save JSON
        json_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filenames['json'])
        # Use orient='records' for list of dicts, ensure UTF-8
        df_results.to_json(json_filepath, orient='records', indent=4, force_ascii=False)
        logger.info(f"Generated results JSON: {json_filepath}")

        # Save XLSX
        xlsx_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filenames['xlsx'])
        try:
            # Use BytesIO buffer to avoid saving intermediate file
            with io.BytesIO() as buffer:
                 # Use ExcelWriter context manager
                 with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                     df_results.to_excel(writer, index=False, sheet_name='Results')
                 # Important: Seek back to the start of the buffer before reading
                 buffer.seek(0)
                 # Write buffer content to the actual file
                 with open(xlsx_filepath, 'wb') as f:
                     f.write(buffer.read())
            logger.info(f"Generated results XLSX: {xlsx_filepath}")
        except ImportError:
             logger.error("Cannot generate XLSX file. 'openpyxl' library not found. Please install: pip install openpyxl")
             temp_filenames.pop('xlsx', None) # Remove xlsx from downloadable files
        except Exception as e_xlsx:
            logger.error(f"Error generating XLSX file: {e_xlsx}", exc_info=True)
            temp_filenames.pop('xlsx', None) # Remove xlsx if generation failed

        file_gen_duration = time.time() - file_gen_start
        logger.info(f"Generated result files (Duration: {file_gen_duration:.4f}s)")

        total_exec_duration = (datetime.datetime.now() - exec_start_time).total_seconds()
        completion_message = f"Task completed successfully in {total_exec_duration:.2f} seconds."
        logger.info(f"SSE Execute Complete: {completion_message}")

        # Send completion event with filenames and the actual results data
        yield f"event: complete\ndata: {json.dumps({'status': 'success', 'message': completion_message, 'download_filenames': temp_filenames, 'results': results})}\n\n"

    except GeneratorExit:
        total_exec_duration = (datetime.datetime.now() - exec_start_time).total_seconds()
        logger.warning(f"SSE Client disconnected during execution after {total_exec_duration:.2f}s.")
        # Cleanup happens in finally

    except Exception as e_exec:
        total_exec_duration = (datetime.datetime.now() - exec_start_time).total_seconds()
        error_message = f"An unexpected error occurred during checklist execution after {total_exec_duration:.2f}s: {e_exec}"
        logger.error(error_message, exc_info=True)
        # Ensure error message is JSON serializable
        try:
            error_payload = json.dumps({'message': error_message})
        except TypeError:
            error_payload = json.dumps({'message': f"An non-serializable error occurred: {type(e_exec).__name__}"})
        yield f"event: error\ndata: {error_payload}\n\n"
        # Cleanup happens in finally

    finally:
        # --- Cleanup (called when generator exits/errors/completes) ---
        logger.info("SSE Execute: Stream finished or terminated.")
        # Cleanup LLM? Usually handled elsewhere unless specific to this task.
        # Temp result files (CSV, JSON, XLSX) are cleaned up after download via the /download route.
        # The temporary PDF should be cleaned up here if it's only needed for this execution stream.
        # However, the current logic keeps it in session until download/new upload.


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Serves the main upload form page."""
    logger.info("Route '/'. Clearing session and cleaning up LLM.")
    session.clear()
    cleanup_llm()
    # Pass municipality data structure to template
    return render_template('index.html',
                           predefined_models=PREDEFINED_LOCAL_MODELS,
                           default_openai=DEFAULT_OPENAI_MODEL,
                           municipalities_data=AVAILABLE_MUNICIPALITIES)

# Renamed from /suggest to /upload for clarity
@app.route('/upload', methods=['POST'])
def upload_and_suggest():
    """Handles initial PDF upload, LLM setup, and checklist suggestion."""
    upload_route_start = time.time()
    logger.info("Route '/upload' POST request received.")
    # --- File Validation ---
    if 'pdf_file' not in request.files:
        logger.warning("Upload Error: 'pdf_file' not in request.files.")
        flash('No file part selected.', 'error')
        return redirect(url_for('index'))
    file = request.files['pdf_file']
    if file.filename == '':
        logger.warning("Upload Error: No file selected (filename empty).")
        flash('No PDF file selected.', 'error')
        return redirect(url_for('index'))
    if not file or not allowed_file(file.filename):
        logger.warning(f"Upload Error: Invalid file type or no file for '{file.filename}'.")
        flash('Invalid file type. Please upload a PDF.', 'error')
        return redirect(url_for('index'))
    original_filename = secure_filename(file.filename)
    logger.info(f"Upload: Processing uploaded file '{original_filename}'.")

    # --- Get Config from Form ---
    llm_type_choice = request.form.get('llm_type')
    municipality = request.form.get('municipality')
    quantization = request.form.get('quantization', 'none') # Get quantization level
    model_name = None
    device_map_setting = None
    llm_type = None

    if municipality not in AVAILABLE_MUNICIPALITIES:
        logger.warning(f"Upload Error: Invalid municipality '{municipality}'.")
        flash('Invalid municipality selected.', 'error')
        return redirect(url_for('index'))

    if llm_type_choice == 'local':
        llm_type = LLAMA
        model_name = request.form.get('local_model_id')
        custom_model_id = request.form.get('custom_local_model_id', '').strip()
        if custom_model_id: model_name = custom_model_id
        if not model_name:
             logger.warning("Upload Error: Local model selected but no ID provided.")
             flash('Please select or enter a local model ID.', 'error')
             return redirect(url_for('index'))
        # Quantization already retrieved
        device_map_setting = request.form.get('device_map', 'auto')
    elif llm_type_choice == 'openai':
        llm_type = OPENAI
        model_name = request.form.get('openai_model_id', DEFAULT_OPENAI_MODEL).strip()
        if not model_name: model_name = DEFAULT_OPENAI_MODEL
        quantization = 'none' # Quantization doesn't apply to OpenAI
    else:
        logger.warning(f"Upload Error: Invalid llm_type_choice '{llm_type_choice}'.")
        flash('Invalid LLM type selected.', 'error')
        return redirect(url_for('index'))

    logger.info(f"Upload Config: Municipality={municipality}, LLM Type={llm_type}, Model={model_name}, Quantization={quantization}, DeviceMap={device_map_setting}")

    # --- Save PDF Temporarily ---
    pdf_filepath = None
    pdf_text = None
    temp_pdf_filename = None
    pdf_save_start = time.time()
    try:
        # Use a unique filename for the saved PDF
        temp_pdf_filename = str(uuid.uuid4()) + ".pdf"
        pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_pdf_filename)
        file.seek(0) # Ensure reading from the start of the file stream
        file.save(pdf_filepath)
        pdf_save_duration = time.time() - pdf_save_start
        logger.info(f"PDF saved temporarily to: {pdf_filepath} (Duration: {pdf_save_duration:.4f}s)")

        pdf_text = pdf_to_text_from_file(pdf_filepath) # Logging is inside this function
        if pdf_text is None: # Check for critical extraction error
             flash('Error extracting text from PDF. The file might be corrupted or unreadable. Check logs.', 'error')
             raise ValueError("Critical error extracting text from saved PDF.")
        if not pdf_text: # Check for empty text (valid PDF, but no content)
             flash('Warning: Could not extract any text from the PDF. It might be image-based or empty. Suggestion quality may be low.', 'warning')
             # Proceed even if text is empty, suggestion will likely fail gracefully

    except Exception as e:
        pdf_save_duration = time.time() - pdf_save_start
        logger.error(f"Error saving or processing PDF '{original_filename}' (Duration: {pdf_save_duration:.4f}s): {e}", exc_info=True)
        flash(f'Error saving or processing PDF: {e}', 'error')
        if pdf_filepath and os.path.exists(pdf_filepath):
            try: os.remove(pdf_filepath)
            except OSError as e_del: logger.error(f"Error deleting temp PDF {pdf_filepath} on upload error: {e_del}", exc_info=True)
        return redirect(url_for('index'))

    # --- Load Checklists (Only needed for suggestion here) ---
    checklists_data_for_suggest = load_checklists(municipality) # Logging is inside
    if not checklists_data_for_suggest:
        flash(f'Error loading checklists for {municipality} to perform suggestion. Check server logs.', 'error')
        if pdf_filepath and os.path.exists(pdf_filepath):
             try: os.remove(pdf_filepath)
             except OSError as e_del: logger.error(f"Error deleting temp PDF {pdf_filepath} on checklist load error: {e_del}", exc_info=True)
        return redirect(url_for('index'))

    # --- Initialize LLM and Suggest ---
    compiler = None
    suggested_checklist_name = None
    try:
        compiler = initialize_llm_and_compiler(llm_type, model_name, quantization, device_map_setting, municipality) # Logging inside
        if not compiler:
             raise RuntimeError('Failed to initialize LLM compiler.')

        # Only suggest if PDF text was extracted
        if pdf_text:
             suggested_checklist_name = suggest_checklist(compiler, pdf_text, checklists_data_for_suggest) # Use loaded data
        else:
             logger.warning("Skipping checklist suggestion because no text was extracted from the PDF.")
             flash("Skipped checklist suggestion (no text in PDF). Please choose manually.", "warning")
        # Keep LLM loaded

    except Exception as e_llm:
        logger.error(f"Error during LLM initialization or suggestion: {e_llm}", exc_info=True)
        flash(f"Error during LLM initialization or suggestion. Check server logs.", 'error')
        cleanup_llm() # Cleanup on error
        if pdf_filepath and os.path.exists(pdf_filepath):
             try: os.remove(pdf_filepath)
             except OSError as e_del: logger.error(f"Error deleting temp PDF {pdf_filepath} on LLM init/suggest error: {e_del}", exc_info=True)
        return redirect(url_for('index'))

    # --- Store necessary info in session ---
    # *** DO NOT STORE checklists_data in session ***
    session.clear() # Clear previous session data
    session['pdf_filepath'] = pdf_filepath # Store the path to the saved PDF
    session['original_filename'] = original_filename
    session['municipality'] = municipality
    session['llm_config'] = { 'type': llm_type, 'model_name': model_name, 'quantization': quantization, 'device_map': device_map_setting }
    session['suggested_checklist'] = suggested_checklist_name
    # session['checklists_data'] = checklists_data <-- REMOVED THIS LINE
    session.modified = True
    logger.info("Session data set (excluding full checklist data).")


    upload_route_duration = time.time() - upload_route_start
    logger.info(f"Upload route finished successfully. Redirecting to confirm. (Total Duration: {upload_route_duration:.4f}s)")
    return redirect(url_for('confirm_checklist'))


@app.route('/confirm', methods=['GET'])
def confirm_checklist():
    """Displays the suggested checklist and allows user confirmation/override."""
    logger.info("Route '/confirm' GET request received.")
    # Retrieve data needed for the template from session
    suggested_checklist = session.get('suggested_checklist')
    municipality = session.get('municipality')
    original_filename = session.get('original_filename')
    pdf_filepath = session.get('pdf_filepath')

    # Check if essential data is present
    if not all([municipality, original_filename, pdf_filepath]): # Removed checklists_data check
        logger.warning("Confirm Error: Essential session data missing or expired (municipality, filename, pdf_path).")
        flash('Session data missing or expired. Please start over.', 'error')
        return redirect(url_for('index'))

    # Check if the temporary PDF file still exists
    if not os.path.exists(pdf_filepath):
         logger.error(f"Confirm Error: Temporary PDF file '{pdf_filepath}' is missing.")
         flash('Temporary PDF file is missing. Please start over.', 'error')
         session.clear()
         cleanup_llm()
         return redirect(url_for('index'))

    # *** Load checklists_data here ***
    checklists_data = load_checklists(municipality)
    if not checklists_data:
        logger.error(f"Confirm Error: Could not load checklists for {municipality}.")
        flash(f'Could not load checklist data for {municipality}. Please try again or check server logs.', 'error')
        return redirect(url_for('index'))

    # Extract names for the dropdown
    available_checklist_names = [chk['NomeChecklist'] for chk in checklists_data.get('checklists', [])]

    # Check LLM status
    compiler = loaded_llm_objects.get("compiler")
    llm_status = "LLM Ready" if compiler else "LLM Not Loaded (Will reload on execute)"
    logger.info(f"Confirm page: Municipality={municipality}, File={original_filename}, LLM Status={llm_status}")

    return render_template('confirm.html',
                           suggested_checklist=suggested_checklist,
                           available_checklist_names=available_checklist_names,
                           municipality=municipality,
                           original_filename=original_filename,
                           llm_status=llm_status)

# --- NEW API Endpoint for Checklist Descriptions ---
@app.route('/api/checklist_description/<string:municipality>/<string:checklist_name>')
def get_checklist_description(municipality, checklist_name):
    """API endpoint to fetch the description of a specific checklist."""
    logger.info(f"API request for description: Municipality={municipality}, Checklist={checklist_name}")

    # *** Always load fresh checklist data for API endpoint ***
    # Avoid relying on potentially stale session data here
    checklists_data = load_checklists(municipality)

    if not checklists_data:
        logger.error(f"API Error: Could not load checklists for {municipality}")
        return jsonify({"error": f"Could not load checklists for {municipality}"}), 404

    description = "Description not found."
    for checklist in checklists_data.get("checklists", []):
        if checklist.get("NomeChecklist") == checklist_name:
            description = checklist.get("Descrizione", "No description provided.")
            break

    logger.info(f"API Response: Description found - {len(description)} chars.")
    return jsonify({"description": description})


# --- MODIFIED /execute route ---
@app.route('/execute', methods=['POST'])
def execute_trigger():
    """
    Trigger for the checklist execution. Sets up session parameters. Returns immediately.
    """
    logger.info("Route '/execute' POST request received (trigger).")
    # Retrieve necessary data from session and form
    pdf_filepath = session.get('pdf_filepath')
    municipality = session.get('municipality')
    original_filename = session.get('original_filename')
    llm_config = session.get('llm_config')
    # checklists_data = session.get('checklists_data') # REMOVED - Not needed here
    chosen_checklist_name = request.form.get('chosen_checklist')

    # Validate session data (excluding checklists_data) and file existence
    if not all([pdf_filepath, municipality, original_filename, llm_config, chosen_checklist_name]):
        logger.error("Execute trigger error: Session data missing or incomplete.")
        flash('Session expired or required data missing. Please start over.', 'error')
        # Redirect back to confirm as the session might still hold some info
        return redirect(url_for('confirm_checklist'))

    if not os.path.exists(pdf_filepath):
        logger.error(f"Execute trigger error: Temp PDF not found at {pdf_filepath}")
        flash('Temporary PDF file missing. Please start over.', 'error')
        session.clear()
        cleanup_llm()
        return redirect(url_for('index'))

    # *** Verify chosen checklist exists by loading data again ***
    checklists_data_for_verify = load_checklists(municipality)
    if not checklists_data_for_verify:
         logger.error(f"Execute trigger error: Could not load checklists for {municipality} to verify selection.")
         flash(f"Could not load checklist data for {municipality}. Please try again.", 'error')
         return redirect(url_for('confirm_checklist'))

    if not any(chk['NomeChecklist'] == chosen_checklist_name for chk in checklists_data_for_verify.get('checklists', [])):
         logger.error(f"Execute trigger error: Chosen checklist '{chosen_checklist_name}' not found in available lists for {municipality}.")
         flash(f"Invalid checklist '{chosen_checklist_name}' selected. Please choose again.", 'error')
         return redirect(url_for('confirm_checklist'))


    # Store parameters needed for the SSE stream
    # No need to store checklists_data
    session['task_params'] = {
        'pdf_filepath': pdf_filepath,
        'municipality': municipality,
        'original_filename': original_filename,
        'llm_config': llm_config,
        'chosen_checklist_name': chosen_checklist_name
    }
    session.modified = True

    logger.info(f"Execute trigger successful. Task parameters stored for checklist: '{chosen_checklist_name}', File: '{original_filename}'. Redirecting to progress page.")
    # Redirect to the progress page, which will then connect to the stream
    return redirect(url_for('progress_page'))

# --- /progress route (Unchanged) ---
@app.route('/progress')
def progress_page():
    """Renders the progress page."""
    logger.info("Route '/progress' GET request received.")
    # Retrieve basic info to display initially
    task_params = session.get('task_params')
    if not task_params:
        logger.warning("Progress page accessed without task parameters in session.")
        flash("No active task found. Please start a new analysis.", "warning")
        return redirect(url_for('index'))

    # Pass necessary initial info to the template
    initial_info = {
        'municipality': task_params.get('municipality'),
        'original_filename': task_params.get('original_filename'),
        'chosen_checklist': task_params.get('chosen_checklist_name'),
        'llm_model': task_params.get('llm_config', {}).get('model_name'),
        'llm_type': task_params.get('llm_config', {}).get('type'),
        'quantization': task_params.get('llm_config', {}).get('quantization'),
    }
    return render_template('progress.html', initial_info=initial_info)


# --- MODIFIED /stream route (SSE endpoint) ---
@app.route('/stream')
def progress_stream():
    """Streams progress updates using Server-Sent Events."""
    stream_start_time = time.time()
    logger.info("Route '/stream' GET request received. Client connected for SSE.")

    # Retrieve task parameters from session
    task_params = session.get('task_params')
    # checklists_data = session.get('checklists_data') # REMOVED - Load below

    if not task_params:
        logger.error("Progress stream error: Task parameters not found in session.")
        def error_gen():
             logger.info("Yielding session error message to client.")
             yield f"event: error\ndata: {json.dumps({'message': 'Session data missing. Please start over.'})}\n\n"
        return Response(error_gen(), mimetype='text/event-stream')

    pdf_filepath = task_params.get('pdf_filepath')
    municipality = task_params.get('municipality')
    original_filename = task_params.get('original_filename')
    llm_config = task_params.get('llm_config')
    chosen_checklist_name = task_params.get('chosen_checklist_name')
    logger.info(f"Progress stream started for checklist: '{chosen_checklist_name}', File: '{original_filename}'.")

    # --- Define the event stream generator ---
    def event_stream():
        compiler = None
        pdf_text = None
        checklists_data = None # Will be loaded below
        stream_gen_start_time = time.time()
        temp_pdf_to_delete = pdf_filepath # Keep track of the PDF to delete

        try:
            # --- Load necessary data (PDF Text and Checklists) ---
            logger.info("Stream: Loading data...")
            load_data_start = time.time()

            # Load PDF Text
            if not temp_pdf_to_delete or not os.path.exists(temp_pdf_to_delete):
                 logger.error(f"Stream Error: Temporary PDF file path missing or file not found: {temp_pdf_to_delete}")
                 raise FileNotFoundError("Temporary PDF file path missing or file not found.")
            pdf_text = pdf_to_text_from_file(temp_pdf_to_delete) # Logs inside
            if pdf_text is None: # Critical error reading PDF
                raise ValueError("Failed to read PDF text from temporary file.")
            if not pdf_text: # PDF readable but empty
                 logger.warning("Stream: PDF text is empty.")

            # *** Load Checklists Data ***
            checklists_data = load_checklists(municipality)
            if not checklists_data:
                 logger.error(f"Stream Error: Failed to load checklists for {municipality}.")
                 raise ValueError(f"Failed to load checklists for {municipality}.")

            logger.info(f"Stream: Data loaded (Duration: {time.time() - load_data_start:.4f}s)")

            # --- Get/Initialize LLM ---
            logger.info("Stream: Initializing LLM (if needed)...")
            llm_init_start = time.time()
            compiler = initialize_llm_and_compiler(
                llm_config['type'], llm_config['model_name'],
                llm_config['quantization'], llm_config['device_map'],
                municipality
            ) # Logs inside
            if not compiler:
                 raise RuntimeError("Failed to get or initialize LLM compiler.")
            logger.info(f"Stream: LLM ready (Init Duration: {time.time() - llm_init_start:.4f}s)")

            # --- Execute and Yield Progress ---
            logger.info("Stream: Starting checklist execution generator...")
            # execute_checklist_sse now receives the loaded checklists_data
            yield from execute_checklist_sse(
                compiler, pdf_text, chosen_checklist_name,
                checklists_data, municipality, original_filename
            )
            logger.info("Stream: Checklist execution generator finished normally.")

        except Exception as e_stream:
            stream_duration = time.time() - stream_gen_start_time
            error_message = f"Error during progress streaming (Duration: {stream_duration:.4f}s): {e_stream}"
            logger.error(error_message, exc_info=True)
            try:
                # Ensure error message is JSON serializable
                error_payload = json.dumps({'message': error_message})
            except TypeError:
                error_payload = json.dumps({'message': f"An non-serializable error occurred: {type(e_stream).__name__}"})
            try:
                yield f"event: error\ndata: {error_payload}\n\n"
            except Exception as e_yield:
                 logger.error(f"Stream Error: Could not yield final error message to client: {e_yield}", exc_info=True)

        finally:
            # --- Cleanup ---
            logger.info("Stream: Entering finally block for cleanup.")
            # Clear task params and other related data from session
            session.pop('task_params', None)
            # session.pop('checklists_data', None) # No longer needed
            session.pop('suggested_checklist', None)
            session.pop('pdf_filepath', None) # Clear PDF path now task is done
            session.modified = True
            logger.info("Stream: Cleared task_params, suggested_checklist, pdf_filepath from session.")

            # Cleanup the temporary PDF file associated with THIS task run
            if temp_pdf_to_delete and os.path.exists(temp_pdf_to_delete):
                logger.info(f"Stream: Attempting to delete temporary PDF file: {temp_pdf_to_delete}")
                try:
                    os.remove(temp_pdf_to_delete)
                    logger.info(f"Stream: Temporary PDF file deleted: {temp_pdf_to_delete}")
                except OSError as e_del:
                    logger.error(f"Stream Cleanup Error: Error deleting temporary PDF file {temp_pdf_to_delete}: {e_del}", exc_info=True)
            else:
                 logger.warning(f"Stream Cleanup: Temp PDF file path '{temp_pdf_to_delete}' not found or path missing in finally block.")

            stream_total_duration = time.time() - stream_start_time
            logger.info(f"Progress stream connection closed. (Total Duration: {stream_total_duration:.4f}s)")

    # Return the streaming response
    return Response(event_stream(), mimetype='text/event-stream')


# --- /download route (Unchanged) ---
@app.route('/download/<filename>')
def download_result(filename):
    """Provides the generated result file (CSV, JSON, XLSX) for download and cleans it up."""
    logger.info(f"Route '/download/{filename}' GET request received.")
    download_start_time = time.time()

    # Validate filename format (UUID + extension)
    match = re.match(r'^([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\.(csv|json|xlsx)$', filename.lower())
    if not match:
        logger.error(f"Download Error: Invalid filename format requested: {filename}")
        flash('Invalid download filename format.', 'error')
        return redirect(url_for('index')) # Or maybe progress page?

    base_uuid, extension = match.groups()
    safe_filename = secure_filename(filename) # Further sanitization
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    logger.info(f"Download request for safe path: {filepath}")

    if not os.path.exists(filepath):
        logger.error(f"Download Error: File not found: {filepath}")
        flash('Download file not found. It might have expired or an error occurred.', 'error')
        return redirect(url_for('index')) # Or progress page?

    # Determine mimetype based on extension
    mimetype = None
    if extension == 'csv':
        mimetype = 'text/csv'
    elif extension == 'json':
        mimetype = 'application/json'
    elif extension == 'xlsx':
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    else:
        # Should not happen due to regex, but fallback
        mimetype = 'application/octet-stream'


    # Use after_this_request to delete the file *after* it has been sent
    @after_this_request
    def remove_file(response):
        cleanup_start = time.time()
        logger.info(f"Download Cleanup: Attempting to delete temporary result file: {filepath}")
        try:
            os.remove(filepath)
            logger.info(f"Download Cleanup: Deleted temporary result file: {filepath} (Cleanup Duration: {time.time() - cleanup_start:.4f}s)")
        except Exception as error:
            logger.error(f"Download Cleanup Error: Error removing file {filepath}: {error}", exc_info=True)
        return response

    try:
        logger.info(f"Sending file: {filepath} as attachment '{safe_filename}' with mimetype '{mimetype}'")
        response = send_file(filepath, as_attachment=True, download_name=safe_filename, mimetype=mimetype)
        download_duration = time.time() - download_start_time
        # Note: This log might appear before the remove_file log
        logger.info(f"File '{safe_filename}' sent successfully. (Send Duration: {download_duration:.4f}s)")
        return response
    except Exception as e_send:
         download_duration = time.time() - download_start_time
         logger.error(f"Download Error: Error sending file {filepath} (Duration: {download_duration:.4f}s): {e_send}", exc_info=True)
         flash('Error sending the download file.', 'error')
         return redirect(url_for('index')) # Or progress page?

# --- Main Execution (Unchanged) ---
if __name__ == '__main__':
    logger.info("Starting Flask development server.")
    # Use threaded=True for development ONLY to handle SSE concurrently
    # For production, use a proper WSGI server (Gunicorn, uWSGI) with async workers (gevent/eventlet)
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

