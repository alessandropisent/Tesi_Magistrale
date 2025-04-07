import subprocess
import time
import sys
import os
import logging # Import logging

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
# Path to the python interpreter you want to use
# Uses the same interpreter running this wrapper by default
python_executable = sys.executable

# Name of the main script to run (the modified one above)
script_to_run = "src/todo.py" # Make sure this matches your filename

# Delay between retries in seconds
retry_delay_seconds = 3600 # 1 hour

# --- Define Models to Process ---
model_configs = [
    {"id": "meta-llama/Llama-3.3-70B-Instruct", "folder": "3.3.llama.70B.Instruct", "done":False},
    {"id": "meta-llama/Llama-3.1-70B-Instruct", "folder": "3.1.llama.70B.Instruct", "done":False},
    # Add more models here as needed
    # {"id": "meta-llama/Llama-3.1-8B-Instruct", "folder": "3.1.llama.8B.Instruct"},
]

# Optional: Maximum number of retry attempts (0 for infinite retries)
max_retries = 0
dones = []
# --- End Configuration ---

# Check if the target script exists before starting
if not os.path.exists(script_to_run):
    print(f"ERROR: The target script '{script_to_run}' was not found.")
    print(f"Please make sure it's in the same directory as this wrapper or provide the correct path.")
    sys.exit(1)

current_attempt = 0
while True:
    current_attempt += 1
    print(f"\n{'='*25} Wrapper: Starting Attempt #{current_attempt} {'='*25}")
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Launching '{script_to_run}'...")
    print(f"Using interpreter: {python_executable}")
    
    # --- Outer loop: Iterate through each model configuration ---
    for model_config in model_configs:
        if model_config["done"]:
            continue
        
        model_id = model_config["id"]
        model_folder = model_config["folder"]
        
        logger.info(f"\n{'='*30}\n>>> Processing Model: {model_id} <<<\n{'='*30}")
    
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
        
        # --- Check Exit Code ---
        if exit_code == 0:
            print(f"{timestamp}: Success detected (Exit Code 0).")
            print("Wrapper script finishing.")
            model_config["done"]=True

        # --- Handle Failure ---
        else:
            print(f"{timestamp}: Failure detected (Exit Code {exit_code}).")
            # Optional: Check for specific exit codes if main_program uses them
            if exit_code == 1: # Assuming 1 is our specific OOM code from main_program.py
                print(f"{timestamp}: Failure appears to be due to Out-of-Memory.")
            elif exit_code == 130: # Ctrl+C received in subprocess
                print(f"{timestamp}: Subprocess was interrupted (Ctrl+C). Stopping wrapper.")
                break
            else:
                print(f"{timestamp}: Failure due to an unexpected error in the script (check logs above).")

            # Check if max retries reached
            #if max_retries > 0 and current_attempt >= max_retries:
            #    print(f"{timestamp}: Maximum retry limit ({max_retries}) reached. Stopping wrapper.")
            #    break

            print(f"{timestamp}: Waiting for {retry_delay_seconds / 60:.0f} minutes before next attempt...")
            try:
                time.sleep(retry_delay_seconds)
            except KeyboardInterrupt:
                print(f"\n{timestamp}: Keyboard interrupt received during wait. Stopping wrapper.")
                break # Allow Ctrl+C to stop the wrapper

print("\nWrapper script has finished execution.")
sys.exit(0) # Exit the wrapper cleanly