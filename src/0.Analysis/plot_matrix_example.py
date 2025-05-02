import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast # For parsing string representations of lists/arrays
import os # To handle file paths

def plot_summed_confusion_matrix(df, model_name, temperature, output_dir="Graphs", labels=None, cmap='Blues'):
    """
    Filters a DataFrame, sums confusion matrices, and plots the resulting
    confusion matrix as a heatmap, saving it to a file. Designed for smaller
    output images with larger annotation text.

    Args:
        df (pd.DataFrame): DataFrame with 'Modello', 'Temperature', 'confusion_matrix'.
        model_name (str): The model name to filter by.
        temperature (float): The temperature to filter by.
        output_dir (str): Directory to save the output PNG file.
        labels (list, optional): Class labels for the matrix axes.
        cmap (str, optional): Colormap for the heatmap. Defaults to 'Blues'.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The summed confusion matrix (counts). Returns None on error.
            - str: The path to the saved image file. Returns None on error.
    """
    print(f"Attempting to plot for Model: {model_name}, Temperature: {temperature}")

    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return None, None

    # --- Define output filename ---
    # Sanitize model name for filename (replace slashes, etc.)
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    output_filename = os.path.join(output_dir, f"Confusion_Matrix_{safe_model_name}_Temp_{temperature}.png")

    # 1. Select rows based on model and temperature
    try:
        filtered_df = df[(df['Modello'] == model_name) & (df['Temperature'] == temperature)].copy()
    except KeyError as e:
        print(f"Error: DataFrame is missing required column: {e}")
        return None, None

    if filtered_df.empty:
        print(f"--> No data found for Model='{model_name}' and Temperature={temperature}.")
        return None, None

    # 2. Handle matrices stored as strings (if necessary)
    try:
        # Check the type of the first element safely
        first_matrix = filtered_df['confusion_matrix'].iloc[0]
        if isinstance(first_matrix, str):
            print("Detected string matrices, attempting to parse...")
            # Use a lambda function that handles potential errors during parsing
            def parse_matrix(x):
                try:
                    # Ensure it's a string before trying to evaluate
                    if isinstance(x, str):
                         # Safely evaluate the string representation
                        matrix_data = ast.literal_eval(x)
                        # Convert to NumPy array
                        return np.array(matrix_data)
                    elif isinstance(x, (np.ndarray, list)):
                         # If it's already an array or list, just ensure it's a NumPy array
                        return np.array(x)
                    else:
                        # Handle unexpected types if necessary
                        print(f"Warning: Unexpected type in confusion_matrix column: {type(x)}")
                        return None # Or raise an error
                except (ValueError, SyntaxError, TypeError) as parse_error:
                    print(f"Error parsing matrix string '{x}': {parse_error}")
                    return None # Return None or a default matrix on error

            filtered_df['confusion_matrix'] = filtered_df['confusion_matrix'].apply(parse_matrix)
            # Drop rows where parsing failed
            filtered_df = filtered_df.dropna(subset=['confusion_matrix'])
            if filtered_df.empty:
                 print("Error: No valid confusion matrices found after parsing.")
                 return None, None
        elif isinstance(first_matrix, list):
             # If it's a list of lists, convert to NumPy array
             print("Detected list matrices, converting to NumPy arrays...")
             filtered_df['confusion_matrix'] = filtered_df['confusion_matrix'].apply(np.array)

        # Ensure all elements are NumPy arrays after potential conversion
        if not all(isinstance(m, np.ndarray) for m in filtered_df['confusion_matrix']):
             print("Error: Not all confusion matrices are NumPy arrays after processing.")
             # You might want to inspect the types here:
             # print(filtered_df['confusion_matrix'].apply(type).value_counts())
             return None, None

    except Exception as e:
        print(f"Error processing confusion matrix column: {e}")
        return None, None


    # 3. Sum the confusion matrices
    if 'confusion_matrix' not in filtered_df.columns:
        print(f"Error: 'confusion_matrix' column not found.")
        return None, None

    try:
        # Summing requires all elements to be compatible (e.g., NumPy arrays)
        # Check dimensions if necessary before summing
        first_shape = filtered_df['confusion_matrix'].iloc[0].shape
        if not all(m.shape == first_shape for m in filtered_df['confusion_matrix']):
             print("Error: Confusion matrices have inconsistent shapes.")
             return None, None

        # Sum the matrices - .sum() works directly on a Series of NumPy arrays
        conf_matrix_sum = filtered_df['confusion_matrix'].sum()

    except Exception as e:
        print(f"Error during confusion matrix summation: {e}")
        # Print types for debugging if summation fails
        # print(filtered_df['confusion_matrix'].apply(type))
        return None, None


    # 4. Determine labels if not provided
    num_classes = conf_matrix_sum.shape[0]
    if labels is None:
        labels = [f'Class {i}' for i in range(num_classes)]
        print(f"Using generic labels: {labels}")
    elif len(labels) != num_classes:
        print(f"Warning: Provided labels length ({len(labels)}) doesn't match matrix dimension ({num_classes}). Using generic labels.")
        labels = [f'Class {i}' for i in range(num_classes)]


    # 5. Plot the matrix (smaller figure, larger text)
    plt.style.use('seaborn-v0_8-whitegrid')
    # --- ADJUST figsize and annot_kws size as needed ---
    fig, ax = plt.subplots(figsize=(4.5, 3.8)) # Smaller figure size (width, height in inches)

    sns.heatmap(
        conf_matrix_sum,
        annot=True,
        fmt='d',                 # Format annotations as integers
        cmap=cmap,
        linewidths=0.5,
        linecolor='grey',
        cbar=True,               # Keep color bar for reference
        annot_kws={"size": 14},  # Larger font size for annotations
        xticklabels=labels,
        yticklabels=labels,
        ax=ax                    # Pass the axes object to heatmap
    )

    # Add titles and labels (adjust font size if needed)
    ax.set_title(f'Confusion Matrix\nModel: {model_name}, Temp: {temperature}',
                 fontsize=10, pad=15) # Slightly smaller title font
    ax.set_ylabel('True Label', fontsize=9, labelpad=8) # Smaller axis label font
    ax.set_xlabel('Predicted Label', fontsize=9, labelpad=8) # Smaller axis label font

    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8) # Smaller tick label font
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8) # Smaller tick label font

    # Ensure everything fits
    plt.tight_layout()

    # --- Save the figure ---
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight') # Save with high DPI
        print(f"--> Saved plot to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
        plt.close(fig) # Close the figure even if saving failed
        return conf_matrix_sum, None

    plt.close(fig) # Close the figure window after saving

    return conf_matrix_sum, output_filename # Return the summed matrix and the file path

# === Example Usage ===

# Load your DataFrame
try:
    df = pd.read_csv("src/Evaluation/checklist_compiler/statistics.csv")
    df = df[df["determina_in"]=="system"].copy() # Filter if needed
except FileNotFoundError:
    print("Error: statistics.csv not found. Please check the path.")
    # Create a dummy DataFrame for demonstration if file not found
    data = {'Modello': ['gpt-4o', 'gpt-4o', 'llama-3.1-70B', 'Mistral-v0.3-7B', 'llama-3.1-70B'],
            'Temperature': [0.0, 0.0, 0.0, 0.0, 0.0],
            'confusion_matrix': [
                 '[[4, 5, 2], [1, 48, 4], [2, 0, 3]]', # gpt-4o example 1 (string)
                 np.array([[0, 1, 0], [0, 5, 0], [1, 0, 0]]), # gpt-4o example 2 (array)
                 '[[0, 11, 0], [1, 54, 0], [0, 10, 1]]', # llama example (string)
                 '[[3, 11, 8], [1, 34, 7], [0, 10, 1]]', # mistral example (string)
                 '[[0, 0, 0], [0, 0, 0], [0, 0, 0]]' # llama example 2 (string) - to test summation
                 ]
            }
    df = pd.DataFrame(data)
    print("Using dummy data because statistics.csv was not found.")


# Define class labels
class_labels = ['NO', 'SI', 'NON PERTINETE']

# --- Call the function for each model/temp combination ---

print("\n--- Plotting for gpt-4o @ 0.0 ---")
matrix_gpt, path_gpt = plot_summed_confusion_matrix(
    df,
    model_name='gpt-4o',
    temperature=0.0,
    labels=class_labels,
    cmap='Greens' # Example: Use a different colormap
)

print("\n--- Plotting for llama-3.1-70B @ 0.0 ---")
matrix_llama_70b, path_llama_70b = plot_summed_confusion_matrix(
    df,
    model_name='llama-3.1-70B',
    temperature=0.0,
    labels=class_labels,
    cmap='Blues'
)

print("\n--- Plotting for Mistral-v0.3-7B @ 0.0 ---")
matrix_mistral, path_mistral = plot_summed_confusion_matrix(
    df,
    model_name='Mistral-v0.3-7B',
    temperature=0.0,
    labels=class_labels,
    cmap='Oranges' # Example: Use a different colormap
)

# Now you should have three PNG files (e.g., in a 'Graphs' subdirectory)
# ready to be included in your LaTeX document using the \includegraphics command
# within the subfigure environments as discussed previously.
