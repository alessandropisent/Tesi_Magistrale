# --- File 2: run_wrapper.py (Modified) ---

import subprocess
import time
import sys
import os
import logging

# --- Configure Logging ---
# (Keep the existing wrapper logging setup, writing to wrapper_log_file and console)
wrapper_log_file = 'run_wrapper.log'
log_format = '%(asctime)s - %(levelname)s - WRAPPER - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers: # Avoid adding handlers multiple times
    file_handler = logging.FileHandler(wrapper_log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Wrapper logs everything to console too
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
# --- End Logging Config ---

# --- Configuration ---
python_executable = sys.executable
script_to_run = "src/todo.py" # Script modified to accept args

# --- Define Models to Process ---
model_configs = [
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "folder": "3.3.llama.70B.Instruct"},
    {"id": "meta-llama/Llama-3.1-70B-Instruct", "folder": "3.1.llama.70B.Instruct"},
    # Add more models here as needed
    # {"id": "meta-llama/Llama-3.1-8B-Instruct", "folder": "3.1.llama.8B.Instruct"},
]

retry_delay_seconds = 3600/2 # 1 hour retry delay on failure for a specific model
max_retries_per_model = 48 # Max attempts for a single model before skipping it (0 for infinite)
# --- End Configuration ---

if not os.path.exists(script_to_run):
    logger.error(f"The target script '{script_to_run}' was not found.")
    sys.exit(1)

logger.info(f"Wrapper started. Processing {len(model_configs)} model(s).")
overall_success = True # Track if all models succeeded

# --- Outer loop: Iterate through each model configuration ---
for model_config in model_configs:
    model_id = model_config["id"]
    model_folder = model_config["folder"]
    logger.info(f"\n{'='*30}\n>>> Processing Model: {model_id} <<<\n{'='*30}")

    current_attempt_for_model = 0
    model_succeeded = False

    # --- Inner loop: Retry logic for the CURRENT model ---
    while not model_succeeded:
        current_attempt_for_model += 1
        logger.info(f"--- Starting Attempt #{current_attempt_for_model} for Model: {model_id} ---")

        # Construct the command with arguments
        command = [
            python_executable,
            script_to_run,
            '--model-id', model_id,
            '--model-folder', model_folder
        ]
        logger.info(f"Running command: {' '.join(command)}")

        # Run the main script as a separate process
        process = subprocess.run(command, check=False) # Executes and waits
        exit_code = process.returncode
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"--- Attempt #{current_attempt_for_model} for Model {model_id} finished with Exit Code: {exit_code} ({timestamp}) ---")

        # --- Check Exit Code for the current model attempt ---
        if exit_code == 0:
            logger.info(f"Model {model_id} completed successfully!")
            model_succeeded = True # Exit the inner retry loop
            # No need to break here, loop condition `not model_succeeded` handles it
        else:
            logger.warning(f"Model {model_id} failed on attempt {current_attempt_for_model} (Exit Code: {exit_code}).")
            # Check for specific non-retryable exit codes (like Ctrl+C)
            if exit_code == 130:
                 logger.warning("Subprocess was interrupted (Ctrl+C). Stopping wrapper.")
                 overall_success = False # Mark overall run as failed
                 break # Exit inner loop
            elif exit_code == 2: # Assuming 2 is our general non-OOM error code
                 logger.error(f"Model {model_id} failed with a non-memory error. Check logs.")
                 overall_success = False
                 

            # Check if max retries reached for this model
            if max_retries_per_model > 0 and current_attempt_for_model >= max_retries_per_model:
                 logger.error(f"Maximum retry limit ({max_retries_per_model}) reached for model {model_id}. Skipping this model.")
                 overall_success = False # Mark overall run as incomplete/failed
                 break # Exit the inner retry loop for this model

            # --- Wait before retrying this model ---
            logger.info(f"Waiting for {retry_delay_seconds / 60:.0f} minutes before retrying model {model_id}...")
            try:
                time.sleep(retry_delay_seconds)
            except KeyboardInterrupt:
                 logger.warning("\nKeyboard interrupt received during wait. Stopping wrapper.")
                 overall_success = False
                 # Need to break out of both loops or exit
                 break # Exit inner loop first

    # --- End Inner Retry Loop ---

    # Handle KeyboardInterrupt during sleep (check if we broke from inner loop)
    if not model_succeeded and exit_code != 130 and current_attempt_for_model >= max_retries_per_model and max_retries_per_model > 0 :
         # This condition means we exited the inner loop due to max retries or non-retryable error
         pass # Already logged the reason
    elif not model_succeeded :
         # If we broke out due to Ctrl+C or another reason stopping the outer loop
         logger.warning("Exiting model processing loop due to previous interruption or error.")
         break # Exit the outer model loop

# --- End Outer Model Loop ---

logger.info("\n" + "="*30)
if overall_success:
    logger.info(">>> Wrapper finished: All configured models processed successfully. <<<")
else:
    logger.warning(">>> Wrapper finished: One or more models failed or were skipped. <<<")
logger.info("="*30)

logging.shutdown()
sys.exit(0 if overall_success else 1) # Exit wrapper with 0 if all ok, 1 otherwise