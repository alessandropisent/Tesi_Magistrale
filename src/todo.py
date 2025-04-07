import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
import time
# Assuming ChecklistCompiler, LLAMA, LUCCA, OLBIA are in a file named ChecklistCompiler.py
from ChecklistCompiler import ChecklistCompiler, LLAMA, LUCCA, OLBIA
# import time # No longer needed for sleep
from tqdm import tqdm
import sys
import gc
import traceback # For logging errors before exit
import logging
import argparse

# --- Configure Logging ---
log_file_name = 'main_program.log'
log_format = '%(asctime)s - %(levelname)s - %(message)s'

# Create logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set logger to capture INFO level and above

# Create File Handler (captures INFO and above)
file_handler = logging.FileHandler(log_file_name, mode='a') # Append mode
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(log_format)
file_handler.setFormatter(file_formatter)

# Create Console Handler (captures WARNING and above)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING) # <--- ONLY SHOW WARNINGS/ERRORS ON CONSOLE
console_formatter = logging.Formatter(log_format) # Can use same or different format
console_handler.setFormatter(console_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# --- End Logging Config ---


# OOM Error definition (best effort based on PyTorch)
try:
    # Check if torch and cuda are available before defining the specific error
    if torch.cuda.is_available():
        OOM_ERROR = torch.cuda.OutOfMemoryError
    else:
        # If CUDA not available, OOM error is less likely but define a placeholder
        class DummyOOMError(Exception): pass
        OOM_ERROR = DummyOOMError
except (ImportError, AttributeError):
    # If torch itself is not imported or cuda attribute missing
    class DummyOOMError(Exception): pass
    OOM_ERROR = DummyOOMError


def main_logic(model_id: str, model_folder: str):
    """
    Contains the core task logic for a single run attempt.
    """

    # --- Configuration --- (Consider moving to command-line args or config file later)
    temperatures = [0.0, 0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    #model_ids = ["meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"]
    #model_folders = ["3.3.llama.70B.Instruct", "3.1.llama.70B.Instruct"]
    need_quant = True # Set to False to disable quantization
    max_gpu_memory_per_device = "23GB" # Adjust as needed
    
    
    #need_quant = False
    #device_model = torch.device('cuda:1') # Only used if need_quant=False and device_map='auto' isn't used    
    #model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
    #model_folders = ["3.1.llama.8B.Instruct","3.2.llama.3B.Instruct"]
        

    TOTAL_iterations = len([LUCCA, OLBIA]) * len(temperatures)

    with tqdm(total=TOTAL_iterations, desc="Overall Progress") as main_pbar:
    
        logger.info(f"--- Processing Model: {model_id} ---")

        if need_quant:
            # Assumes 2 GPUs from your nvidia-smi output
            # If you have more/less, adjust the keys (0, 1, ...)
            max_memory_map = {0: max_gpu_memory_per_device, 1: max_gpu_memory_per_device}
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16, # Recommended compute dtype for 4-bit
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            logger.info(f"Loading quantized model '{model_id}'...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                # torch_dtype=torch.bfloat16, # Usually float16 is used with 4-bit quant
                quantization_config=quantization_config,
                device_map='auto', # Automatically distribute across devices respecting max_memory
                max_memory=max_memory_map,
            )
        else:
            logger.info(f"Loading model '{model_id}' (non-quantized)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map='auto' # Let HF handle device placement if not quantizing
                # device_map=device_model, # Or uncomment this to force to a specific device
            )
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info("Model and tokenizer loaded.")
        # --- End Model/Tokenizer Loading ---


        for municipality in [LUCCA, OLBIA]:
            logger.info(f"--- Processing Municipality: {municipality} ---")


            if municipality == LUCCA:
                checklist_path = "./src/txt/Lucca/checklists/checklists.json"
                determine_path = "./src/txt/Lucca/checklists/Lucca_Determine.csv"
            elif municipality == OLBIA:
                checklist_path = "./src/txt/Olbia/checklists/checklists.json"
                determine_path = "./src/txt/Olbia/checklists/Olbia_Determine.csv"
                

            else:
                logger.info(f"WARN: Unknown municipality object {municipality}. Skipping.")
                continue

            try:
                logger.info(f"Loading checklists from {checklist_path}")
                with open(checklist_path, "r", encoding="utf-8") as f:
                    checklists = json.load(f)
                logger.info(f"Loading determine from {determine_path}")
                # Use context manager for file handle, load dataframe
                with open(determine_path, "r", encoding="utf-8") as f:
                    df_determine = pd.read_csv(f)
                    
            except FileNotFoundError as e:
                logger.info(f"ERROR: Data file not found - {e}. Skipping {municipality}.")
                continue

            except Exception as e_load:
                    logger.info(f"ERROR: Failed to load data for {municipality} - {e_load}. Skipping.")
                    continue
            # --- End Data Loading ---


            logger.info("Setting up ChecklistCompiler...")
            compiler = ChecklistCompiler(llm=LLAMA, municipality=municipality, model=model_id)

            for temp in temperatures:
                main_pbar.set_description(f"{model_folder[:10]}.. M:{municipality} T:{temp}")
                logger.info(f"-- Processing Temperature: {temp} --")

                # --- Configure and set pipeline ---
                if temp == 0.0:
                    do_sample = False
                    t_param = None
                    top_p = None
                else:
                    do_sample = True
                    top_p = 0.9
                    t_param = temp

                logger.info("Creating/Updating text generation pipeline...")
                text_gen_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=500,
                    pad_token_id=tokenizer.eos_token_id,
                    truncation=True,
                    eos_token_id=tokenizer.eos_token_id, # Some models need this specific arg
                    do_sample=do_sample,
                    temperature=t_param,
                    top_p=top_p,
                )
                compiler.set_text_gen_pipeline(text_gen_pipeline)
                logger.info("Pipeline ready.")
                # --- End Pipeline Setup ---


                # --- Run checklist generation and choosing ---
                # Define subfolder based on model, municipality, and temp for organization
                sub_cartella_output = f"{model_folder}/{temp}/"

                logger.info(f"Processing {len(df_determine)} determines for checklist generation...")
                for i, row in df_determine.iterrows():
                    num = row["Numero Determina"]
                    che_ass = row["Checklist associata"]
                    
                    compiler.checklist_determina(
                        num,
                        che_ass,
                        checklists,
                        sub_cartella=sub_cartella_output, # Save to specific output folder
                        temperature=temp, # Pass temp if method needs it
                    )
                    logger.info(f"Done determina [{i}] {num}") # Can be too verbose

                logger.info(f"Processing 'choose_checklist' for temp {temp}...")
                compiler.choose_checklist(
                    determine=df_determine, # Pass the dataframe
                    checklists=checklists,
                    sub_cartella=sub_cartella_output, # Save results to same folder
                    temperature=temp)
                logger.info(f"Done choose_checklist temp:{temp}")
                # --- End Checklist Logic ---

                # Update main progress bar after finishing one temp setting for a municipality/model
                main_pbar.update(1)

            # --- Cleanup after finishing temperatures for a municipality ---
            logger.info(f"Finished all temperatures for {municipality}. Cleaning up...")


        # --- Cleanup after finishing municipalities for a model ---
        logger.info(f"Finished all municipalities for model {model_id}. Cleaning up model...")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # --- End Model Cleanup ---

    logger.info(f"****{"\n"} All tasks within main_logic completed. model_id:{model_id}")
    # No need for a finally block here anymore, as the main try/except handles exit


if __name__ == "__main__":
   # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run checklist processing for a specific LLM.")
    parser.add_argument("--model-id", required=True, type=str, help="Hugging Face model ID (e.g., meta-llama/Llama-3.1-70B-Instruct)")
    parser.add_argument("--model-folder", required=True, type=str, help="Subfolder name for output (e.g., 3.1.llama.70B.Instruct)")
    args = parser.parse_args()
    # --- End Argument Parsing ---

    logger.info(f"--- Starting Main Script Execution for Model: {args.model_id} ---")
    exit_code = 0
    try:
        main_logic(model_id=args.model_id, model_folder=args.model_folder)
        logger.info(f"--- Main Script Finished Successfully ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
        exit_code = 0

    except KeyboardInterrupt:
        logger.critical("\n--- Keyboard Interrupt received. Stopping script. ---")
        exit_code = 130 # Standard exit code for Ctrl+C

    except Exception as e_main:
        logger.critical("\n" + "="*60)
        logger.critical(f"!!! AN UNEXPECTED ERROR OCCURRED ({time.strftime('%Y-%m-%d %H:%M:%S')}) !!!")
        logger.critical(f"Error details: {e_main}")
        logger.critical("="*60 + "\n")
        traceback.print_exc() # Print stack trace for debugging
        exit_code = 2 # Use a different code for other errors

    finally:
        # Optional: Final attempt to clear CUDA cache before process truly exits
        # The OS will reclaim the memory anyway when the process terminates.
        if 'torch' in sys.modules and torch.cuda.is_available():
            logger.info("--- Final cleanup: Attempting torch.cuda.empty_cache() before exit ---")
            try:
                torch.cuda.empty_cache()
            except Exception as e_final_clean:
                logger.error(f"Error during final CUDA cache clear: {e_final_clean}")

        logger.info(f"--- Exiting Main Script with Exit Code: {exit_code} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
        sys.exit(exit_code) # Exit with the determined code