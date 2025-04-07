import subprocess
import sys
import platform # To check OS

def get_nvidia_gpu_temp():
    """
    Gets the current temperature of the NVIDIA GPU(s).

    Returns:
        int: The maximum temperature found among all NVIDIA GPUs in Celsius,
             or None if nvidia-smi is not found or fails.
    """
    if platform.system() == "Windows":
        # Try the default install path for nvidia-smi on Windows
        nvidia_smi_path = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        command = f'"{nvidia_smi_path}" --query-gpu=temperature.gpu --format=csv,noheader,nounits'
    elif platform.system() == "Linux":
        command = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
    else:
        print(f"Unsupported OS: {platform.system()}")
        return None

    try:
        # Execute the command, capture output, decode to text
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.PIPE)

        # Split lines in case of multiple GPUs, strip whitespace, convert to int
        temps = [int(t.strip()) for t in output.strip().split('\n') if t.strip()]

        if not temps:
            print("Error: nvidia-smi returned empty output.")
            return None

        # Return the maximum temperature if multiple GPUs are present
        return max(temps)

    except FileNotFoundError:
        print("Error: 'nvidia-smi' command not found.")
        print("Ensure NVIDIA drivers are installed and 'nvidia-smi' is in your system's PATH")
        if platform.system() == "Windows":
             print(f" (Expected default path: {nvidia_smi_path})")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        print(f"stderr: {e.stderr}")
        return None
    except ValueError:
        print(f"Error: Could not parse temperature from nvidia-smi output: {output.strip()}")
        return None
    except Exception as e: # Catch unexpected errors
        print(f"An unexpected error occurred while getting GPU temp: {e}")
        return None

# --- Main Execution Logic ---
MAX_GPU_TEMP_THRESHOLD = 90 # Set your desired max temperature in Celsius

print("Checking GPU temperature...")
current_temp = get_nvidia_gpu_temp()

if current_temp is not None:
    print(f"Current GPU Temperature: {current_temp}°C")

    