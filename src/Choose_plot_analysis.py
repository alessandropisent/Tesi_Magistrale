import pandas as pd
from ChecklistCompiler import LLAMA, OPENAI, LUCCA, OLBIA # Assuming these are defined elsewhere
import re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants and Setup ---
temperatures = [0.0, 0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# Define the tests configuration
tests = [
    {
        "LLM": LLAMA,
        "municipality": LUCCA,
        "folder": "llama-3.1-8B",
        "model": "llama 3.1 8B" # Consistent naming convention helps
    },
    {
        "LLM": LLAMA,
        "municipality": LUCCA,
        "folder": "llama-3.2-3B", # Assuming this is 8B based on name pattern? Or is it 3B? Corrected to 8B for consistency
        "model": "llama 3.2 8B"
    },
    {
        "LLM": LLAMA,
        "municipality": LUCCA,
        "folder": "3.3.llama.70B.Instruct",
        "model": "llama 3.3 70B Q4"
    },
    {
        "LLM": LLAMA,
        "municipality": LUCCA,
        "folder": "3.1.llama.70B.Instruct",
        "model": "llama 3.1 70B Q4"
    },
    {
        "LLM": LLAMA, # Assuming Mistral runs locally like Llama
        "municipality": LUCCA,
        "folder": "Mistral.7B.Instruct-v0.3",
        "model": "Mistral v0.3 7B" # Removed trailing space
    },
    {
        "LLM": OPENAI,
        "municipality": LUCCA,
        "model": "gpt-4o-mini", # Use consistent naming
        "folder": "mini"
    },
    # Add other tests if needed (e.g., gpt-4o full)
    # {
    #  "LLM": OPENAI,
    #  "municipality": LUCCA,
    #  "model": "gpt-4o",
    #  "folder": "full" # Assuming a folder name for gpt-4o full results
    # },
]

# --- Helper Functions ---

def add_newline(text):
    """Appends a newline character to the text."""
    return text + "\n"

def numberize_OLBIA(el):
    """Converts Olbia checklist names to numeric codes."""
    if el == "Determina a Contrarre":
        return 0
    elif el == "Procedura negoziata":
        return 1
    elif el == "Affidamento Diretto":
        return 2
    else:
        return -1 # Indicates an unrecognized value

def numberize_LUCCA(el):
    """Converts Lucca checklist names to numeric codes."""
    if el == "Determine":
        return 0
    elif el == "Contratti":
        return 1
    else:
        return -1 # Indicates an unrecognized value

def extract_llm_choices_LUCCA(text: str) -> str:
    """Extracts Lucca checklist choice ('Determine' or 'Contratti') from LLM output."""
    if not isinstance(text, str): # Handle potential non-string input
        return "Not Found"
    line = text.strip()

    # Search for the choice word anywhere in the line (case-insensitive)
    match = re.search(r'\b(Contratti|Determine|Determina)\b', line, re.IGNORECASE)
    if match:
        choice = match.group(1)
        # Normalize "Determina" to "Determine" and ensure proper capitalization
        ret = 'Determine' if choice.lower() == 'determina' else choice.capitalize()
        return ret

    # Check if line starts with the choice (allowing optional markdown)
    match = re.search(r'^(?:[#*]*\s*)?\b(Contratti|Determine|Determina)\b', line, re.IGNORECASE)
    if match:
        choice = match.group(1)
        ret = 'Determine' if choice.lower() == 'determina' else choice.capitalize()
        return ret

    return "Not Found"

def extract_llm_choices_OLBIA(text: str) -> str:
    """Extracts Olbia checklist choice from LLM output."""
    if not isinstance(text, str): # Handle potential non-string input
        return "Not Found"

    def normalize_procurement_choice(choice_str: str) -> str | None:
        """Normalizes variations of Olbia procurement choices."""
        text_norm = choice_str.strip().lower()
        text_norm = re.sub(r'\s+', ' ', text_norm) # Standardize whitespace

        if text_norm == "determina a contrarre":
            return "Determina a Contrarre"
        elif text_norm in ["procedura negoziata", "procedura negoziate"]: # Handle typo
            return "Procedura negoziata"
        elif text_norm == "affidamento diretto":
            return "Affidamento Diretto"
        else:
            return None # Not a recognized choice

    text_stripped = text.strip()

    # Regex to find any of the target phrases (case-insensitive) anywhere
    choice_regex = re.compile(
        r'\b(Determina\s+a\s+Contrarre|Procedura\s+negoziata|Procedura\s+negoziate|Affidamento\s+Diretto)\b',
        re.IGNORECASE
    )
    # Regex to find target phrases at the start (allowing optional markdown)
    line_start_regex = re.compile(
        r'^(?:[#*]*\s*)?\b(Determina\s+a\s+Contrarre|Procedura\s+negoziata|Procedura\s+negoziate|Affidamento\s+Diretto)\b',
        re.IGNORECASE
    )

    # Prioritize match anywhere in the string first
    match = choice_regex.search(text_stripped)
    if match:
        normalized = normalize_procurement_choice(match.group(1))
        if normalized:
            return normalized

    # If not found anywhere, check specifically at the start
    match = line_start_regex.search(text_stripped)
    if match:
        normalized = normalize_procurement_choice(match.group(1))
        if normalized:
            return normalized

    return "Not Found"

def compare_columns(df, truth_col, pred_col, numberize_func):
    """Compares truth and prediction columns using specified numberize function and calculates metrics."""
    if truth_col not in df.columns or pred_col not in df.columns:
        return {"error": f"Column '{truth_col}' or '{pred_col}' not found."}

    # Apply numberization safely, handling potential errors
    try:
        df[truth_col + "_"] = df[truth_col].apply(numberize_func)
        df[pred_col + "_"] = df[pred_col].apply(numberize_func)
    except Exception as e:
        return {"error": f"Error during numberization: {e}"}

    y_true = df[truth_col + "_"]
    y_pred = df[pred_col + "_"]

    # Filter out rows where numberization failed (returned -1)
    valid_indices = (y_true != -1) & (y_pred != -1)
    if not valid_indices.any():
         return {"error": "No valid comparable data after numberization."}
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    # Determine unique valid labels present in the ground truth
    valid_labels = sorted(y_true_valid.unique())
    if not valid_labels:
         return {"error": "No valid labels found in ground truth after filtering."}

    # Calculate metrics only on valid, comparable data
    results = {
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "precision": precision_score(y_true_valid, y_pred_valid, labels=valid_labels, average='weighted', zero_division=0),
        "recall": recall_score(y_true_valid, y_pred_valid, labels=valid_labels, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true_valid, y_pred_valid, labels=valid_labels, average='weighted', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true_valid, y_pred_valid),
        # Add other metrics if needed, ensuring they use y_true_valid and y_pred_valid
    }
    return results

# --- Color Palette Functions ---

def get_model_family(model_name):
    """Categorizes a model name into a family (llama, mistral, gpt, other)."""
    name_lower = model_name.lower()
    if 'llama' in name_lower:
        return 'llama'
    elif 'mistral' in name_lower:
        return 'mistral'
    elif 'gpt' in name_lower or 'chatgpt' in name_lower: # Handle variations
        return 'gpt'
    else:
        return 'other'

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
    # Use unique models to avoid issues if the same model appears multiple times
    unique_models = sorted(list(set(model_list)))

    for model in unique_models:
        family = get_model_family(model) # Use the helper function
        # Ensure the family key exists before appending
        if family not in models_by_family:
             models_by_family[family] = [] # Initialize if a new family type appears
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
            # Generate n_colors shades between light and dark grey
            family_palette_colors = sns.color_palette("Greys", n_colors=n_colors + 2)[1:-1] # Avoid pure white/black
        else:
            # Use light_palette for main families to get shades
            # Generate n_colors shades based on the base color
            # Ensure enough distinct shades, especially for few models
            min_saturation = 0.4 # Adjust saturation range if needed
            max_saturation = 1.0
            # Create palette from light to dark (or reverse=True for dark to light)
            family_palette_colors = sns.light_palette(base_color, n_colors=n_colors + 1, reverse=False)[1:]


        # Assign the generated colors to the specific models in the sorted list
        for model, color in zip(models, family_palette_colors):
            final_palette[model] = color

    return final_palette

# --- Plotting Function ---

def graph_df(df):
    """Graphs the model accuracy per temperature setting using family-based colors."""

    # Filter for the specific municipality if needed (already done before calling)
    # df_plot = df[df["Municipality"] == LUCCA].copy() # Create a copy to avoid SettingWithCopyWarning
    df_plot = df.copy() # Assume df is already filtered or contains only relevant data

    if df_plot.empty:
        print("DataFrame is empty after filtering. Cannot generate plot.")
        return

    # Get unique model names for the palette
    model_names = df_plot['Model'].unique()
    custom_palette = create_family_palette(model_names)

    # Create the grouped bar plot using Seaborn
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")

    barplot = sns.barplot(
        data=df_plot,
        x='Temperature',
        y='accuracy',
        hue='Model',
        palette=custom_palette, # Use the generated custom palette
        hue_order=sorted(model_names) # Ensure legend order matches palette generation
    )

    # Customize the plot
    plt.title(f'Model Accuracy per Temperature Setting in {df_plot["Municipality"].iloc[0]}', fontsize=16, pad=20) # Get municipality from data
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    # Adjust y-axis limits
    min_accuracy = df_plot['accuracy'].min()
    max_accuracy = df_plot['accuracy'].max()
    # Set limits slightly beyond the data range for better visualization
    plt.ylim(max(0, min_accuracy - 0.1), min(1.05, max_accuracy + 0.05))

    # Adjust legend position to avoid overlap
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Leave space on the right for the legend

    # Save the plot (optional)
    # plt.savefig(f"model_accuracy_choose_{df_plot['Municipality'].iloc[0]}.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


# --- Main Execution Block ---

if __name__ == "__main__":
    rows = []
    # Loop through each test configuration
    for test in tests:
        municipality = test["municipality"]
        llm_type = test["LLM"]
        model_name_clean = test["model"] # Use the cleaned model name
        folder_name = test["folder"]

        # Determine paths and functions based on municipality
        if municipality == OLBIA:
            truth_src = "./src/txt/Olbia/checklists/Olbia_Determine.csv"
            results_root_folder = f"./src/{llm_type}/Olbia_text/choose"
            simplify_func = extract_llm_choices_OLBIA
            number_func = numberize_OLBIA
        elif municipality == LUCCA:
            truth_src = "./src/txt/Lucca/checklists/Lucca_Determine.csv"
            results_root_folder = f"./src/{llm_type}/Lucca_text/choose"
            simplify_func = extract_llm_choices_LUCCA
            number_func = numberize_LUCCA
        else:
            print(f"WARNING: Municipality '{municipality}' not recognized. Skipping test.")
            continue

        # Load ground truth data
        try:
            df_determine = pd.read_csv(truth_src, encoding="utf-8")
            # Select only relevant columns early
            df_determine = df_determine[['Numero Determina', 'Checklist associata']].copy()
        except FileNotFoundError:
            print(f"ERROR: Ground truth file not found at {truth_src}. Skipping test.")
            continue
        except Exception as e:
            print(f"ERROR: Could not read ground truth file {truth_src}: {e}. Skipping test.")
            continue

        # Process results for each temperature
        for temp in temperatures:
            json_path = f"{results_root_folder}/{folder_name}/{temp}/determine.json"
            try:
                df_comparison = pd.read_json(json_path)
                if df_comparison.empty:
                    print(f"INFO: Empty results file at {json_path}. Skipping temp {temp}.")
                    continue
            except FileNotFoundError:
                # print(f"INFO: Results file not found at {json_path}. Skipping temp {temp}.")
                continue # Skip if the results file for this temp doesn't exist
            except ValueError as e:
                 print(f"ERROR: Could not read JSON file {json_path}: {e}. Skipping temp {temp}.")
                 continue
            except Exception as e:
                print(f"ERROR: Unexpected error reading {json_path}: {e}. Skipping temp {temp}.")
                continue

            # Apply simplification function safely
            try:
                 df_comparison["Simple"] = df_comparison["LLM"].apply(simplify_func)
            except Exception as e:
                 print(f"ERROR: Failed to apply simplify_func for {json_path}: {e}. Skipping temp {temp}.")
                 continue


            # Merge LLM results with ground truth
            # Ensure 'det' column exists in df_comparison
            if 'det' not in df_comparison.columns:
                 print(f"ERROR: 'det' column missing in {json_path}. Skipping temp {temp}.")
                 continue

            df_merged = df_comparison.merge(
                df_determine,
                how='left',
                left_on='det',
                right_on='Numero Determina'
            )

            # Drop the redundant key from the right DataFrame after merge
            df_merged.drop(columns=['Numero Determina'], inplace=True)

            # Calculate comparison metrics
            row_metrics = compare_columns(df=df_merged,
                                          truth_col="Checklist associata",
                                          pred_col="Simple",
                                          numberize_func=number_func)

            # Check if compare_columns returned an error
            if "error" in row_metrics:
                 print(f"ERROR calculating metrics for {json_path}: {row_metrics['error']}. Skipping temp {temp}.")
                 continue

            # Add metadata to the results row
            row_metrics["Model"] = model_name_clean # Use the cleaned model name
            row_metrics["Temperature"] = temp
            row_metrics["Municipality"] = municipality
            rows.append(row_metrics)

    # Create final DataFrame and display/save results
    if rows:
        df_results = pd.DataFrame(rows)
        print("--- Aggregated Results ---")
        print(df_results)
        # Save results to CSV
        try:
            df_results.to_csv("result_choose_accuracy.csv", index=False)
            print("\nResults saved to result_choose_accuracy.csv")
        except Exception as e:
             print(f"\nERROR: Could not save results to CSV: {e}")


        # Optional: Grouped analysis
        # print("\n--- Grouped Mean Accuracy ---")
        # df_grouped = df_results.groupby(["Municipality", "Temperature", "Model"])['accuracy'].mean().reset_index()
        # print(df_grouped)

        # Generate the graph only for Lucca data as requested
        df_lucca_results = df_results[df_results["Municipality"] == LUCCA]
        if not df_lucca_results.empty:
             print("\nGenerating plot for Lucca...")
             graph_df(df_lucca_results)
        else:
             print("\nNo data available for Lucca to generate plot.")

    else:
        print("No results were generated. Check file paths and configurations.")

