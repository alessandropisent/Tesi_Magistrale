import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for sorting

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
    # Use a stable sort like mergesort (pandas default) or Timsort (Python default)
    # or just Python's sorted() which is stable.
    sorted_model_list = sorted(list(model_list))

    for model in sorted_model_list:
        model_lower = model.lower()
        if 'llama' in model_lower:
            models_by_family['llama'].append(model)
        elif 'mistral' in model_lower:
            models_by_family['mistral'].append(model)
        elif 'gpt' in model_lower:
            models_by_family['gpt'].append(model)
        else:
            # Handle potential future models or unexpected names
            models_by_family['other'].append(model)

    final_palette = {}

    # Generate specific shades for each model within its family
    for family, models in models_by_family.items():
        if not models: # Skip if no models in this family
            continue

        n_colors = len(models)
        base_color = family_colors.get(family, other_color)

        # Generate a palette of shades for the family
        # Using light_palette to get variations. Add n_colors+1 and slice [1:]
        # to avoid the lightest color if it's too pale, especially for single models.
        # Reverse=True makes the 'first' model (e.g., smallest size) darker usually.
        if n_colors == 1:
            # If only one model, just use the base color
             family_palette_colors = [base_color]
        elif family == 'other':
             # Use sequential grey palette if multiple 'other' models
             family_palette_colors = sns.color_palette("Greys", n_colors=n_colors + 2)[1:-1] # Avoid pure white/black
        else:
            # Use light_palette for main families to get shades
            # Adjust reverse=True/False based on desired shade order
            family_palette_colors = sns.light_palette(base_color, n_colors=n_colors + 1, reverse=False)[1:]

        # Assign the generated colors to the specific models in the sorted list
        for model, color in zip(models, family_palette_colors):
            final_palette[model] = color

    return final_palette

# Modified graph_df function
def graph_df(df, col_to_plot="accuracy", title_ylabel="Accuracy"):
    """
    Generates a grouped bar plot with custom colors highlighting model families.

    Args:
        df (pd.DataFrame): DataFrame containing the data with 'Temperature',
                           'Modello', and the metric column to plot.
        col_to_plot (str): The name of the column to plot on the y-axis.
        title_ylabel (str): The label for the y-axis.
    """
    # Ensure 'Modello' is present
    if 'Modello' not in df.columns:
        raise ValueError("DataFrame must contain a 'Modello' column.")

    # Get unique models present in this specific dataframe subset
    unique_models = df['Modello'].unique()
    # Sort model names for consistent legend ordering
    sorted_unique_models = sorted(list(unique_models))

    # Create the custom palette using the helper function
    custom_palette = create_family_palette(sorted_unique_models)

    # Create the plot
    plt.figure(figsize=(18, 8)) # Slightly wider figure might be needed
    sns.set_theme(style="whitegrid")

    barplot = sns.barplot(
        data=df,
        x='Temperature',
        y=col_to_plot,
        hue='Modello',
        palette=custom_palette,   # Use the custom family-based palette
        hue_order=sorted_unique_models # Ensure legend matches palette order
        # errorbar=None # Uncomment if using newer Seaborn and want no error bars
    )
    # --- Add horizontal line at 0.5 ---
    plt.axhline(0.5, color='lightcoral', linestyle='--', linewidth=1.5, zorder=2)

    # --- Add value labels on top of bars ---
    for container in barplot.containers:
        barplot.bar_label(container, fmt='%.2f', fontsize=8, padding=3) # Adjust fontsize/padding as needed


    # Customize the plot
    plt.title(f'Model {title_ylabel} per Temperature Setting (Grouped by Family)', fontsize=16, pad=20)
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel(title_ylabel, fontsize=12)

    # Adjust y-axis limits dynamically
    min_val = df[col_to_plot].min()
    max_val = df[col_to_plot].max()
    # Give a bit of padding, but don't go below 0 unless necessary
    plt.ylim(max(0, min_val - (max_val - min_val)*0.1) , min(1.05, max_val + (max_val - min_val)*0.1))

    # Adjust legend position and add a note about colors
    plt.legend(title='Model (Family Colors:\n Llama=Blue, Mistral=Orange, GPT=Green)',
               bbox_to_anchor=(1.02, 1),
               loc='upper left',
               borderaxespad=0.)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust right boundary if needed
    plt.show()



# Assuming your original processing steps:
df = pd.read_csv(f"src/Evaluation/checklist_compiler/statistics.csv")
df = df[df["determina_in"]=="system"] # Filter if needed
keep_rows =[
    "Modello", "Temperature", "accuracy", "precision", "recall",
    "f1_score", "balanced_accuracy"
    ]
group_of_models = df[keep_rows].groupby(by=["Temperature","Modello"], observed=True).mean() # Use observed=True if Temperature/Modello are categoricals
# Use the sample data directly as it's already aggregated
group_of_models_reset = group_of_models.reset_index() # Already in the desired format (like reset_index())

print("Sample Data used for plotting:")
print(group_of_models_reset.head())


# --- Call the plotting function with the modified graph_df ---
print("\nPlotting Accuracy...")
graph_df(df=group_of_models_reset, col_to_plot="accuracy", title_ylabel="Accuracy")

print("\nPlotting Balanced Accuracy...")
#graph_df(df=group_of_models_reset, col_to_plot="balanced_accuracy", title_ylabel="Balanced Accuracy")

print("\nPlotting F1 Score...")
#graph_df(df=group_of_models_reset, col_to_plot="f1_score", title_ylabel="F1 Score")