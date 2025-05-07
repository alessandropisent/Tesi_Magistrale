import pandas as pd
# Assuming these are string constants like "Lucca", "Olbia", "Llama", "OpenAI"
# If they are not available, you might need to define them, e.g.:
# LLAMA, OPENAI = "Llama", "OpenAI"
# LUCCA, OLBIA = "Lucca", "Olbia"
# For the purpose of this script, if ChecklistCompiler is not found,
# we'll use string literals where LUCCA and OLBIA are expected.
try:
    from ChecklistCompiler import LLAMA, OPENAI, LUCCA, OLBIA
except ImportError:
    print("Warning: ChecklistCompiler module not found. Using string literals for LLAMA, OPENAI, LUCCA, OLBIA.")
    LLAMA, OPENAI = "Llama", "OpenAI" # Example placeholder values
    LUCCA, OLBIA = "Lucca", "Olbia"   # Example placeholder values

import re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix, # Not used in plotting, but kept from original
    classification_report, # Not used in plotting, but kept from original
    cohen_kappa_score, # Not used in plotting, but kept from original
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import io # Not strictly required here but kept from reference script's imports

# --- Constants and Setup ---
temperatures = [0.0, 0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# Define the tests configuration
tests = [
    {
        "LLM": LLAMA, # Should resolve to a string like "Llama"
        "municipality": LUCCA, # Should resolve to a string like "Lucca"
        "folder": "llama-3.1-8B",
        "model": "llama 3.1 8B"
    },
    {
        "LLM": LLAMA,
        "municipality": LUCCA,
        "folder": "llama-3.2-3B", # Original comment: Assuming this is 8B based on name pattern? Or is it 3B? Corrected to 8B for consistency
        "model": "llama 3.2 8B" # Assuming 8B for consistency as per original comment
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
        "model": "Mistral v0.3 7B"
    },
    {
        "LLM": OPENAI, # Should resolve to a string like "OpenAI"
        "municipality": LUCCA,
        "model": "gpt-4o-mini",
        "folder": "mini"
    },
    # {
    #     "LLM": OPENAI,
    #     "municipality": LUCCA,
    #     "model": "gpt-4o",
    #     "folder": "full"
    # },
]

# --- Helper Functions (from original script) ---

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
    match = re.search(r'\b(Contratti|Determine|Determina)\b', line, re.IGNORECASE)
    if match:
        choice = match.group(1)
        ret = 'Determine' if choice.lower() == 'determina' else choice.capitalize()
        return ret
    match = re.search(r'^(?:[#*]*\s*)?\b(Contratti|Determine|Determina)\b', line, re.IGNORECASE)
    if match:
        choice = match.group(1)
        ret = 'Determine' if choice.lower() == 'determina' else choice.capitalize()
        return ret
    return "Not Found"

def extract_llm_choices_OLBIA(text: str) -> str:
    """Extracts Olbia checklist choice from LLM output."""
    if not isinstance(text, str):
        return "Not Found"
    def normalize_procurement_choice(choice_str: str) -> str | None:
        text_norm = choice_str.strip().lower()
        text_norm = re.sub(r'\s+', ' ', text_norm)
        if text_norm == "determina a contrarre": return "Determina a Contrarre"
        elif text_norm in ["procedura negoziata", "procedura negoziate"]: return "Procedura negoziata"
        elif text_norm == "affidamento diretto": return "Affidamento Diretto"
        else: return None
    text_stripped = text.strip()
    choice_regex = re.compile(r'\b(Determina\s+a\s+Contrarre|Procedura\s+negoziata|Procedura\s+negoziate|Affidamento\s+Diretto)\b', re.IGNORECASE)
    line_start_regex = re.compile(r'^(?:[#*]*\s*)?\b(Determina\s+a\s+Contrarre|Procedura\s+negoziata|Procedura\s+negoziate|Affidamento\s+Diretto)\b', re.IGNORECASE)
    match = choice_regex.search(text_stripped)
    if match:
        normalized = normalize_procurement_choice(match.group(1))
        if normalized: return normalized
    match = line_start_regex.search(text_stripped)
    if match:
        normalized = normalize_procurement_choice(match.group(1))
        if normalized: return normalized
    return "Not Found"

def compare_columns(df, truth_col, pred_col, numberize_func):
    """Compares truth and prediction columns using specified numberize function and calculates metrics."""
    if truth_col not in df.columns or pred_col not in df.columns:
        return {"error": f"Column '{truth_col}' or '{pred_col}' not found."}
    try:
        df[truth_col + "_"] = df[truth_col].apply(numberize_func)
        df[pred_col + "_"] = df[pred_col].apply(numberize_func)
    except Exception as e:
        return {"error": f"Error during numberization: {e}"}
    y_true = df[truth_col + "_"]
    y_pred = df[pred_col + "_"]
    valid_indices = (y_true != -1) & (y_pred != -1)
    if not valid_indices.any():
        return {"error": "No valid comparable data after numberization."}
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]
    valid_labels = sorted(y_true_valid.unique())
    if not valid_labels:
        return {"error": "No valid labels found in ground truth after filtering."}
    results = {
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "precision": precision_score(y_true_valid, y_pred_valid, labels=valid_labels, average='weighted', zero_division=0),
        "recall": recall_score(y_true_valid, y_pred_valid, labels=valid_labels, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true_valid, y_pred_valid, labels=valid_labels, average='weighted', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true_valid, y_pred_valid),
    }
    return results

# --- Color Palette Functions (from original script, already similar to reference) ---

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
        'llama': 'royalblue',
        'mistral': 'darkorange',
        'gpt': 'forestgreen',
    }
    other_color = 'grey'
    models_by_family = {'llama': [], 'mistral': [], 'gpt': [], 'other': []}
    unique_models = sorted(list(set(model_list))) # Use unique models

    for model in unique_models:
        family = get_model_family(model)
        if family not in models_by_family: # Ensure family key exists
            models_by_family[family] = []
        models_by_family[family].append(model)

    final_palette = {}
    for family, models in models_by_family.items():
        if not models: continue
        n_colors = len(models)
        base_color = family_colors.get(family, other_color)
        if n_colors == 1:
            family_palette_colors = [base_color]
        elif family == 'other':
            family_palette_colors = sns.color_palette("Greys", n_colors=n_colors + 2)[1:-1]
        else:
            family_palette_colors = sns.light_palette(base_color, n_colors=n_colors + 1, reverse=False)[1:]
        for model_name, color in zip(models, family_palette_colors): # Corrected variable name from model to model_name for clarity
            final_palette[model_name] = color
    return final_palette

# --- MODIFIED Plotting Function (adapted from matplotlib_legend_update) ---

def graph_df(df, col_to_plot="accuracy", title_ylabel="Accuracy",
             show_best_per_family=False, municipality_name=""):
    """
    Generates a grouped bar plot with custom colors highlighting model families.
    The legend is centered and placed below the plot.

    Args:
        df (pd.DataFrame): DataFrame containing the data. Expected columns:
                           'Temperature', 'Model', and the `col_to_plot`.
        col_to_plot (str): The name of the column to plot on the y-axis.
        title_ylabel (str): The label for the y-axis and part of the title.
        show_best_per_family (bool): If True, filters data to show only the best
                                     performing model per family at each temperature.
        municipality_name (str): Name of the municipality for the plot title.
    """
    # Validate presence of necessary columns
    if 'Model' not in df.columns: # Changed from 'Modello' to 'Model'
        raise ValueError("DataFrame must contain a 'Model' column.")
    if 'Temperature' not in df.columns:
        raise ValueError("DataFrame must contain a 'Temperature' column.")
    if col_to_plot not in df.columns:
        raise ValueError(f"DataFrame must contain the column '{col_to_plot}' to plot.")

    data_to_plot = df.copy()

    # Construct plot title
    base_plot_title = f'Model {title_ylabel} per Temperature Setting'
    if municipality_name:
        base_plot_title += f' in {municipality_name}'
    plot_title = base_plot_title

    # --- Filtering Logic (Adapted from reference) ---
    if show_best_per_family:
        print(f"Filtering data for {municipality_name} to show best model per family based on '{col_to_plot}'...")
        # 1. Assign family to each model
        data_to_plot['Family'] = data_to_plot['Model'].apply(get_model_family) # Use 'Model' column

        # 2. Find the index of the best model within each Temp/Family group
        # Ensure groupby columns exist and handle potential empty groups
        try:
            best_indices = data_to_plot.loc[data_to_plot.groupby(['Temperature', 'Family'])[col_to_plot].idxmax()].index
            # 3. Filter the DataFrame
            data_to_plot = data_to_plot.loc[best_indices]
            print(f"Filtered data for {municipality_name} has {len(data_to_plot)} rows.")
            plot_title = f'Best Model per Family by {title_ylabel} per Temperature Setting'
            if municipality_name:
                plot_title += f' in {municipality_name}'
        except KeyError as e:
            print(f"Warning: Could not group for best_per_family (KeyError: {e}). Plotting all models for {municipality_name}.")
        except Exception as e: # Catch other potential errors during filtering
            print(f"Warning: Error during best_per_family filtering ({e}). Plotting all models for {municipality_name}.")


        if data_to_plot.empty:
            print(f"Warning: Filtering resulted in an empty DataFrame for {municipality_name} and {col_to_plot}. No plot will be generated.")
            return

    # --- Plotting ---
    unique_models = data_to_plot['Model'].unique() # Use 'Model' column
    sorted_unique_models = sorted(list(unique_models))

    custom_palette = create_family_palette(sorted_unique_models)

    plt.figure(figsize=(18, 10)) # Adapted from reference
    sns.set_theme(style="whitegrid")

    barplot = sns.barplot(
        data=data_to_plot,
        x='Temperature',
        y=col_to_plot,
        hue='Model', # Use 'Model' column
        palette=custom_palette,
        hue_order=sorted_unique_models
    )

    # --- Add value labels on top of bars (from reference) ---
    for container in barplot.containers:
        if container: # Check if container is not empty
            try:
                barplot.bar_label(container, fmt='%.2f', fontsize=8, padding=3)
            except IndexError:
                print("Warning: Could not add labels to a bar container (might be empty or an issue with barplot).")
            except Exception as e:
                 print(f"Warning: Error adding bar labels: {e}")


    # Customize the plot
    plt.title(plot_title, fontsize=16, pad=20)
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel(title_ylabel, fontsize=12)

    # Adjust y-axis limits dynamically (from reference)
    if not data_to_plot.empty and pd.api.types.is_numeric_dtype(data_to_plot[col_to_plot]):
        min_val = data_to_plot[col_to_plot].min()
        max_val = data_to_plot[col_to_plot].max()
        plt.ylim(max(0, min_val - abs(max_val - min_val)*0.05), min(1.05, max_val + abs(max_val - min_val)*0.1))
    else:
        plt.ylim(0, 1) # Default limits

    # --- MODIFIED LEGEND SECTION (from reference) ---
    num_models = len(sorted_unique_models)
    # Adjust ncol based on the number of unique models. Max 5 columns.
    ncol_legend = min(num_models, 5 if num_models > 4 else max(1, num_models))


    legend_title_text = 'Model (Family Colors:\n Llama=Blue, Mistral=Orange, GPT=Green)'
    legend = plt.legend(
        title=legend_title_text,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15), # Adjusted y-offset slightly, tune as needed
        ncol=ncol_legend,
        fontsize='small',
        title_fontsize='medium'
    )

    if legend: # Ensure legend object exists
        legend.get_title().set_multialignment('center')

    # --- MODIFIED LAYOUT ADJUSTMENT (from reference) ---
    plt.tight_layout()
    # Adjust subplot parameters to make space for the legend at the bottom.
    plt.subplots_adjust(bottom=0.25) # Tune as needed based on legend height

    # Save the plot (optional)
    # safe_col_name = "".join(c if c.isalnum() else "_" for c in col_to_plot)
    # safe_municipality_name = "".join(c if c.isalnum() else "_" for c in municipality_name)
    # plot_filename = f"model_{safe_col_name}_{safe_municipality_name}{'_best_family' if show_best_per_family else ''}.png"
    # plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    # print(f"Plot saved as {plot_filename}")

    plt.show()


# --- Main Execution Block ---

if __name__ == "__main__":
    rows = []
    # Loop through each test configuration
    for test_config in tests: # Renamed 'test' to 'test_config' to avoid conflict
        municipality = test_config["municipality"]
        llm_type = test_config["LLM"]
        model_name_clean = test_config["model"]
        folder_name = test_config["folder"]

        # Determine paths and functions based on municipality
        # Assuming LUCCA and OLBIA are string constants like "Lucca", "Olbia"
        if municipality == OLBIA: # OLBIA is likely "Olbia"
            truth_src = "./src/txt/Olbia/checklists/Olbia_Determine.csv"
            results_root_folder = f"./src/{llm_type}/Olbia_text/choose"
            simplify_func = extract_llm_choices_OLBIA
            number_func = numberize_OLBIA
        elif municipality == LUCCA: # LUCCA is likely "Lucca"
            truth_src = "./src/txt/Lucca/checklists/Lucca_Determine.csv"
            results_root_folder = f"./src/{llm_type}/Lucca_text/choose"
            simplify_func = extract_llm_choices_LUCCA
            number_func = numberize_LUCCA
        else:
            print(f"WARNING: Municipality '{municipality}' not recognized. Skipping test.")
            continue

        try:
            df_determine = pd.read_csv(truth_src, encoding="utf-8")
            df_determine = df_determine[['Numero Determina', 'Checklist associata']].copy()
        except FileNotFoundError:
            print(f"ERROR: Ground truth file not found at {truth_src}. Skipping test for {model_name_clean}.")
            continue
        except Exception as e:
            print(f"ERROR: Could not read ground truth file {truth_src}: {e}. Skipping test for {model_name_clean}.")
            continue

        for temp in temperatures:
            json_path = f"{results_root_folder}/{folder_name}/{temp}/determine.json"
            try:
                df_comparison = pd.read_json(json_path)
                if df_comparison.empty:
                    # print(f"INFO: Empty results file at {json_path}. Skipping temp {temp} for {model_name_clean}.")
                    continue
            except FileNotFoundError:
                # print(f"INFO: Results file not found at {json_path}. Skipping temp {temp} for {model_name_clean}.")
                continue
            except ValueError as e:
                print(f"ERROR: Could not read JSON file {json_path}: {e}. Skipping temp {temp} for {model_name_clean}.")
                continue
            except Exception as e:
                print(f"ERROR: Unexpected error reading {json_path}: {e}. Skipping temp {temp} for {model_name_clean}.")
                continue

            try:
                df_comparison["Simple"] = df_comparison["LLM"].apply(simplify_func)
            except Exception as e:
                print(f"ERROR: Failed to apply simplify_func for {json_path}: {e}. Skipping temp {temp} for {model_name_clean}.")
                continue

            if 'det' not in df_comparison.columns:
                print(f"ERROR: 'det' column missing in {json_path}. Skipping temp {temp} for {model_name_clean}.")
                continue

            df_merged = df_comparison.merge(
                df_determine, how='left', left_on='det', right_on='Numero Determina'
            )
            df_merged.drop(columns=['Numero Determina'], inplace=True, errors='ignore') # Add errors='ignore'

            row_metrics = compare_columns(df=df_merged, truth_col="Checklist associata",
                                          pred_col="Simple", numberize_func=number_func)

            if "error" in row_metrics:
                print(f"ERROR calculating metrics for {json_path} ({model_name_clean}): {row_metrics['error']}. Skipping temp {temp}.")
                continue

            row_metrics["Model"] = model_name_clean
            row_metrics["Temperature"] = temp
            row_metrics["Municipality"] = municipality # This is the string like "Lucca" or "Olbia"
            rows.append(row_metrics)

    if rows:
        df_results = pd.DataFrame(rows)
        print("\n--- Aggregated Results ---")
        print(df_results.head())
        try:
            df_results.to_csv("result_choose_metrics.csv", index=False) # Changed filename slightly
            print("\nResults saved to result_choose_metrics.csv")
        except Exception as e:
            print(f"\nERROR: Could not save results to CSV: {e}")

        # --- Updated Plotting Calls ---
        # Filter for Lucca data for plotting (assuming LUCCA variable is the string "Lucca")
        df_lucca_results = df_results[df_results["Municipality"] == LUCCA].copy() # Use .copy()

        if not df_lucca_results.empty:
            print(f"\n--- Generating plots for {LUCCA} ---")

            # Plot Accuracy (All Models) for Lucca
            print(f"\nPlotting Accuracy for {LUCCA} (All Models)...")
            graph_df(df_lucca_results,
                     col_to_plot="accuracy",
                     title_ylabel="Accuracy",
                     show_best_per_family=False,
                     municipality_name=LUCCA) # Pass the string name "Lucca"

            # Plot Precision (All Models) for Lucca
            if "precision" in df_lucca_results.columns:
                print(f"\nPlotting Precision for {LUCCA} (All Models)...")
                graph_df(df_lucca_results,
                         col_to_plot="precision",
                         title_ylabel="Precision",
                         show_best_per_family=False,
                         municipality_name=LUCCA)

            # Plot Accuracy (Best per Family) for Lucca
            print(f"\nPlotting Accuracy for {LUCCA} (Best per Family)...")
            graph_df(df_lucca_results,
                     col_to_plot="accuracy",
                     title_ylabel="Accuracy",
                     show_best_per_family=True,
                     municipality_name=LUCCA)

            # Example: Plot F1 Score (Best per Family) for Lucca
            if "f1_score" in df_lucca_results.columns:
                print(f"\nPlotting F1 Score for {LUCCA} (Best per Family)...")
                graph_df(df_lucca_results,
                         col_to_plot="f1_score",
                         title_ylabel="F1 Score",
                         show_best_per_family=True,
                         municipality_name=LUCCA)
        else:
            print(f"\nNo data available for {LUCCA} to generate plots.")

    else:
        print("No results were generated. Check file paths, configurations, and data processing steps.")

