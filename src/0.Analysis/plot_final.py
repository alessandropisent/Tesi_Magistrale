import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io # Required for reading the sample CSV data (as per original comment)

# Helper function to categorize model family (used for filtering)
def get_model_family(model_name):
    """Categorizes a model name into a family."""
    model_lower = model_name.lower()
    if 'llama' in model_lower:
        return 'llama'
    elif 'mistral' in model_lower:
        return 'mistral'
    elif 'gpt' in model_lower:
        return 'gpt'
    else:
        return 'other'

# Helper function to create a custom palette based on model families
def create_family_palette(model_list):
    """
    Generates a seaborn color palette dictionary mapping model names to colors,
    grouping by 'llama', 'mistral', and 'gpt' families.
    """
    family_colors = {
        'llama': 'royalblue',    # Base color for Llama family
        'mistral': 'darkorange', # Base color for Mistral family
        'gpt': 'forestgreen',    # Base color for GPT family
    }
    other_color = 'grey'         # Color for models not in these families

    # Categorize models by family
    models_by_family = {'llama': [], 'mistral': [], 'gpt': [], 'other': []}
    # Sort the model list to ensure consistent color assignment across plots
    sorted_model_list = sorted(list(model_list))

    for model in sorted_model_list:
        family = get_model_family(model) # Use the helper function
        models_by_family[family].append(model)

    final_palette = {}

    # Generate specific shades for each model within its family
    for family, models in models_by_family.items():
        if not models: # Skip if no models in this family
            continue

        n_colors = len(models)
        base_color = family_colors.get(family, other_color)

        # Generate a palette of shades for the family
        if n_colors == 1:
            # If only one model, just use the base color
            family_palette_colors = [base_color]
        elif family == 'other':
            # Use sequential grey palette if multiple 'other' models
            family_palette_colors = sns.color_palette("Greys", n_colors=n_colors + 2)[1:-1] # Avoid pure white/black
        else:
            # Use light_palette for main families to get shades
            family_palette_colors = sns.light_palette(base_color, n_colors=n_colors + 1, reverse=False)[1:]

        # Assign the generated colors to the specific models in the sorted list
        for model, color in zip(models, family_palette_colors):
            final_palette[model] = color

    return final_palette

# Modified graph_df function
def graph_df(df, col_to_plot="accuracy", title_ylabel="Accuracy", show_best_per_family=False):
    """
    Generates a grouped bar plot with custom colors highlighting model families.
    The legend is centered and placed below the plot.

    Args:
        df (pd.DataFrame): DataFrame containing the data with 'Temperature',
                           'Modello', and the metric column to plot.
        col_to_plot (str): The name of the column to plot on the y-axis.
        title_ylabel (str): The label for the y-axis.
        show_best_per_family (bool): If True, filters data to show only the best
                                     performing model per family at each temperature.
                                     Defaults to False.
    """
    # Ensure 'Modello' is present
    if 'Modello' not in df.columns:
        raise ValueError("DataFrame must contain a 'Modello' column.")
    if 'Temperature' not in df.columns:
        raise ValueError("DataFrame must contain a 'Temperature' column.")
    if col_to_plot not in df.columns:
        raise ValueError(f"DataFrame must contain the column '{col_to_plot}' to plot.")

    data_to_plot = df.copy() # Work on a copy

    plot_title = f'Model {title_ylabel} per Temperature Setting (Grouped by Family)'

    # --- Filtering Logic ---
    if show_best_per_family:
        print(f"Filtering data to show best model per family based on '{col_to_plot}'...")
        # 1. Assign family to each model
        data_to_plot['Family'] = data_to_plot['Modello'].apply(get_model_family)

        # 2. Find the index of the best model within each Temp/Family group
        best_indices = data_to_plot.loc[data_to_plot.groupby(['Temperature', 'Family'])[col_to_plot].idxmax()].index

        # 3. Filter the DataFrame to keep only those best models
        data_to_plot = data_to_plot.loc[best_indices]
        print(f"Filtered data has {len(data_to_plot)} rows.")
        plot_title = f'Best Model per Family by {title_ylabel} per Temperature Setting' # Update title

        if data_to_plot.empty:
            print("Warning: Filtering resulted in an empty DataFrame. No plot will be generated.")
            return # Exit if no data left

    # --- Plotting ---
    # Get unique models present in the dataframe *to be plotted*
    unique_models = data_to_plot['Modello'].unique()
    # Sort model names for consistent legend ordering
    sorted_unique_models = sorted(list(unique_models))

    # Create the custom palette using the helper function based on models actually being plotted
    custom_palette = create_family_palette(sorted_unique_models)

    # Create the plot
    plt.figure(figsize=(18, 10)) # Increased height slightly to accommodate legend
    sns.set_theme(style="whitegrid")

    barplot = sns.barplot(
        data=data_to_plot, # Use the (potentially filtered) data
        x='Temperature',
        y=col_to_plot,
        hue='Modello',
        palette=custom_palette,   # Use the custom family-based palette
        hue_order=sorted_unique_models # Ensure legend matches palette order
        # errorbar=None # Uncomment if using newer Seaborn and want no error bars
    )

    # --- Add value labels on top of bars ---
    for container in barplot.containers:
        # Check if container is not empty before trying to label
        if container:
            try:
                barplot.bar_label(container, fmt='%.2f', fontsize=8, padding=3)
            except IndexError:
                print("Warning: Could not add labels to a bar container (might be empty or an issue with barplot).")


    # Customize the plot
    plt.title(plot_title, fontsize=16, pad=20) # Use the potentially updated title
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel(title_ylabel, fontsize=12)

    # Adjust y-axis limits dynamically
    if not data_to_plot.empty:
        min_val = data_to_plot[col_to_plot].min()
        max_val = data_to_plot[col_to_plot].max()
        # Give a bit of padding, but don't go below 0 unless necessary
        plt.ylim(0, 1.05)
    else:
        plt.ylim(0, 1) # Default limits if data is empty


    # --- MODIFIED LEGEND SECTION ---
    # Calculate a suitable number of columns for the legend
    num_models = len(sorted_unique_models)
    # Adjust ncol based on the number of unique models. Max 5 columns, or fewer if less models.
    ncol_legend = min(num_models, 5 if num_models > 4 else num_models)


    # Place legend below the plot, centered
    # `loc='upper center'` anchors the legend's upper center to `bbox_to_anchor`.
    # `bbox_to_anchor=(0.5, -0.15)`: 0.5 for horizontal center,
    # negative y-value (e.g., -0.12 to -0.20) to position below the plot.
    # This y-value might need tuning based on legend height and figure aspect ratio.
    legend = plt.legend(
        title='Model (Family Colors:\n Llama=Blue, Mistral=Orange, GPT=Green)',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12), # Adjust -0.12 (y-offset) as needed
        ncol=ncol_legend,
        fontsize='small', # Adjust font size if legend text is too large/small
        title_fontsize='medium' # Adjust legend title font size
    )

    # Center the legend title text if it has multiple lines
    if legend:
        legend.get_title().set_multialignment('center')

    # --- MODIFIED LAYOUT ADJUSTMENT ---
    # Apply tight_layout to fit plot elements neatly
    plt.tight_layout()
    # Adjust subplot parameters to make space for the legend at the bottom.
    # The `bottom` parameter specifies the bottom of the subplots as a fraction of the figure height.
    # This value (e.g., 0.20 or 0.25) might need tuning depending on the actual height of your legend.
    plt.subplots_adjust(bottom=0.22) # Increase if legend is tall and gets cut off

    plt.show()


# --- Main script execution part (assuming you have your CSV) ---
# This part requires your CSV file: "src/Evaluation/checklist_compiler/statistics.csv"
# For demonstration, if you don't have the CSV, you might want to create a dummy DataFrame.
# Example of creating a dummy DataFrame (replace with your actual data loading):
# sample_data_for_testing = {
#     'Temperature': [0.2, 0.2, 0.2, 0.7, 0.7, 0.7, 0.2, 0.7, 0.2, 0.7],
#     'Modello': ['Llama-7B', 'Mistral-7B', 'GPT-3.5', 'Llama-7B', 'Mistral-7B', 'GPT-3.5', 'Llama-13B', 'Llama-13B', 'Other-ModelA', 'Other-ModelA'],
#     'accuracy': [0.8, 0.82, 0.85, 0.78, 0.80, 0.83, 0.81, 0.79, 0.70, 0.68],
#     'precision': [0.78, 0.80, 0.83, 0.76, 0.78, 0.81, 0.79, 0.77, 0.68, 0.66],
#     'recall': [0.82, 0.84, 0.87, 0.80, 0.82, 0.85, 0.83, 0.81, 0.72, 0.70],
#     'f1_score': [0.80, 0.82, 0.85, 0.78, 0.80, 0.83, 0.81, 0.79, 0.70, 0.68],
#     'balanced_accuracy': [0.79, 0.81, 0.84, 0.77, 0.79, 0.82, 0.80, 0.78, 0.69, 0.67]
# }
# group_of_models_reset = pd.DataFrame(sample_data_for_testing)
# print("Using DUMMY data for plotting demonstration.")

try:
    # Assuming your original processing steps:
    df = pd.read_csv(f"src/Evaluation/checklist_compiler/statistics.csv")
    df = df[df["determina_in"]=="system"] # Filter if needed
    keep_rows =[
        "Modello", "Temperature", "accuracy", "precision", "recall",
        "f1_score", "balanced_accuracy",
        ]
    # Use observed=True if Temperature/Modello are pandas Categorical type and you only want to group by observed categories
    group_of_models = df[keep_rows].groupby(by=["Temperature","Modello"], observed=True).mean()
    group_of_models_reset = group_of_models.reset_index()

    print("Sample Data used for plotting (from CSV):")
    print(group_of_models_reset.head())

    # --- Call the plotting function ---
    
    for metric, Label in zip(["accuracy", "precision", "recall","f1_score", "balanced_accuracy"],["Accuracy", "Precision", "Recall","F1 Score", "Balanced Accuracy"]):

        # Example 1: Plot all models (default behavior)
        print(f"\n--- Plotting {Label} (All Models) ---")
        graph_df(df=group_of_models_reset, col_to_plot=metric, title_ylabel=Label, show_best_per_family=False)


    # Example 2: Plot only the best model per family for Accuracy
    print("\n--- Plotting Accuracy (Best per Family) ---")
    graph_df(df=group_of_models_reset, col_to_plot="accuracy", title_ylabel="Accuracy", show_best_per_family=True)

    # Example 3: Plot only the best model per family for F1 Score
    # print("\n--- Plotting F1 Score (Best per Family) ---")
    # graph_df(df=group_of_models_reset, col_to_plot="f1_score", title_ylabel="F1 Score", show_best_per_family=True)

    # Example 4: Plot all models for Balanced Accuracy
    # print("\n--- Plotting Balanced Accuracy (All Models) ---")
    # graph_df(df=group_of_models_reset, col_to_plot="balanced_accuracy", title_ylabel="Balanced Accuracy", show_best_per_family=False)

except FileNotFoundError:
    print("Error: 'src/Evaluation/checklist_compiler/statistics.csv' not found.")
    print("Please ensure the CSV file is in the correct path or use the dummy data example by uncommenting it.")
except Exception as e:
    print(f"An error occurred: {e}")

