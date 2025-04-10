import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast # Uncomment if your matrices are stored as strings like "[[1, 2], [3, 4]]"

def plot_summed_confusion_matrix(df, model_name, temperature, labels=None, cmap='Blues'):
    """
    Filters a DataFrame for a specific model and temperature, sums their
    confusion matrices, calculates row-wise probabilities (P(Predicted|True)),
    and plots the resulting normalized confusion matrix as a heatmap.

    Args:
        df (pd.DataFrame): DataFrame containing experiment results. Must have columns
                           'Modello', 'Temperature', and 'confusion_matrix'.
                           The 'confusion_matrix' column should contain NumPy arrays
                           or list-of-lists representing the confusion matrices.
        model_name (str): The name of the model to select (value in 'Modello').
        temperature (float): The temperature setting to select (value in 'Temperature').
        labels (list, optional): A list of strings representing the class labels
                                 for the rows/columns of the confusion matrix.
                                 If None, generic labels like 'Class 0', 'Class 1'
                                 will be generated.
        cmap (str, optional): The matplotlib colormap to use for the heatmap.
                              Defaults to 'Blues'.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The summed and row-normalized confusion matrix (probabilities).
                          Returns None if no data is found or an error occurs.
            - matplotlib.axes._axes.Axes: The Axes object containing the heatmap plot.
                                           Returns None if no data or error.
    """
    print(f"Attempting to plot for Model: {model_name}, Temperature: {temperature}")

    # 1. Select rows based on model and temperature
    try:
        filtered_df = df[(df['Modello'] == model_name) & (df['Temperature'] == temperature)].copy()
    except KeyError as e:
        print(f"Error: DataFrame is missing required column: {e}")
        return None, None

    if filtered_df.empty:
        print(f"--> No data found for Model='{model_name}' and Temperature={temperature}.")
        return None, None

    # --- Optional: Handle matrices stored as strings ---
    #If your confusion matrices are strings, uncomment and adapt this:
    try:
        if isinstance(filtered_df['confusion_matrix'].iloc[0], str):
            print("Detected string matrices, attempting to parse...")
            # Make sure to import ast: import ast
            filtered_df['confusion_matrix'] = filtered_df['confusion_matrix'].apply(
                lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x)
            )
    except Exception as e:
         print(f"Error parsing string confusion matrix: {e}")
         return None, None
    # --- End Optional ---


    # 2. Sum the confusion matrices
    if 'confusion_matrix' not in filtered_df.columns:
         print(f"Error: 'confusion_matrix' column not found.")
         return None, None

    try:
        filtered_df_g = filtered_df[["Modello","confusion_matrix"]].groupby(by=["Modello"]).sum().reset_index()
        #print(filtered_df_g)

    except Exception as e:
         print(f"Error during confusion matrix summation: {e}")
         return None, None

    # Determine labels if not provided
    
    conf_matrix = filtered_df_g["confusion_matrix"].iloc[0]
    num_classes = conf_matrix.shape[0]
    if labels is None:
        labels = [f'Class {i}' for i in range(num_classes)]
        print(f"Using generic labels: {labels}")
    elif len(labels) != num_classes:
         print(f"Warning: Provided labels length ({len(labels)}) doesn't match matrix dimension ({num_classes}). Using generic labels.")
         labels = [f'Class {i}' for i in range(num_classes)]


    # 4. Plot the matrix beautifully
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice seaborn style
    plt.figure(figsize=(8, 6.5)) # Adjust figure size

    ax = sns.heatmap(
        conf_matrix,
        annot=True,           # Show the probability values in the cells
        #fmt=".2%",            # Format values as percentages with 2 decimal places
        cmap=cmap,            # Use the specified colormap
        linewidths=0.5,       # Add lines between cells
        linecolor='grey',    # Color of the lines
        cbar=True,            # Show the color bar legend
        annot_kws={"size": 10}, # Adjust annotation font size
        xticklabels=labels,
        yticklabels=labels
    )

    # Add titles and labels
    plt.title(f'Confusion Matrix \nModel: {model_name}, Temp: {temperature}',
              fontsize=14, pad=20)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)

    # Rotate labels for better readability if needed
    plt.xticks(rotation=30, ha='right') # Rotate x-axis labels slightly
    plt.yticks(rotation=0)              # Keep y-axis labels horizontal

    # Ensure everything fits
    plt.tight_layout()

    # Display the plot
    plt.show()

    return conf_matrix,ax # Return the calculated matrix and the plot axes

# Assuming your original processing steps:
df = pd.read_csv(f"src/Evaluation/checklist_compiler/statistics.csv")
df = df[df["determina_in"]=="system"] # Filter if needed

# --- Call the function ---
print("\n--- Plotting for gpt-4o @ 0.0 ---")
class_labels = ['NO','SI','NON PERTINETE'] # Example labels for a 2x2 matrix
norm_matrix_gpt, plot_ax_gpt = plot_summed_confusion_matrix(
    df,
    model_name='gpt-4o',
    temperature=0.0,
    labels=class_labels,
    cmap='Blues' # Example: Use a different colormap
)

print("\n--- Plotting for llama-3.1-70B @ 0.2 ---")
norm_matrix_llama, plot_ax_llama = plot_summed_confusion_matrix(
    df,
    model_name='llama-3.1-70B',
    temperature=0.0,
    labels=class_labels,
    cmap='Blues'
)

print("\n--- Plotting for llama-3.1-8B @ 0.2 ---")
norm_matrix_llama, plot_ax_llama = plot_summed_confusion_matrix(
    df,
    model_name='Mistral-v0.3-7B',
    temperature=0.0,
    labels=class_labels,
    cmap='Blues'
)
