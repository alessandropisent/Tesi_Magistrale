import pandas as pd
from ChecklistCompiler import LLAMA, OPENAI, LUCCA, OLBIA
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




temperatures = [0.0, 0.01, 0.2, 0.4,
                0.5,0.6,0.8,1.0]

tests= [
    {
        "LLM":LLAMA,
        "municipality":LUCCA,
        "folder":"llama-3.1-8B",
        "model":"llama 3.1 8B"
    },
    {
        "LLM":LLAMA,
        "municipality":LUCCA,
        "folder":"llama-3.2-3B",
        "model":"llama 3.2 3B"
    },
    {
        "LLM":LLAMA,
        "municipality":OLBIA,
        "folder":"llama-3.1-8B",
        "model":"llama 3.1 8B"
    },
    {
        "LLM":LLAMA,
        "municipality":OLBIA,
        "folder":"llama-3.2-3B",
        "model":"llama 3.2 3B"
    },
    {
        "LLM":OPENAI,
        "municipality":LUCCA,
        "model":"chatgpt 4o mini",
        "folder":"mini"
    },
    {
        "LLM":OPENAI,
        "municipality":OLBIA,
        "model":"chatgpt 4o mini",
        "folder":"mini"
    }
]

def add_newline(text):
    return text+"\n"

def numberize_OLBIA(el):
    """
    Convert a categorical string value into a numeric code.
,
            , "Affidamento Diretto"

    """
    
    if el == "Determina a Contrarre":
        return 0
    elif el == "Procedura negoziata":
        return 1
    elif el == "Affidamento Diretto":
        return 2
    else:
        return -1

def numberize_LUCCA(el):
    """
    Convert a categorical string value into a numeric code.

    Parameters:
        el (str): Input string expected to be one of "SI", "NO", or "NON PERTINENTE".
                  Any other value returns -1.

    Returns:
        int: Returns 1 for "SI", 0 for "NO", 2 for "NON PERTINENTE", and -1 for any other value.
    """
    if el == "Determine":
        return 0
    elif el == "Contratti":
        return 1
    else:
        return -1


def extract_llm_choices_LUCCA(text: str) -> str:
    """
    """

    line = text.strip()


    # This line *is* the choice following a special marker line
    # Search for the choice word anywhere in this line
    match = re.search(r'\b(Contratti|Determine|Determina)\b', line, re.IGNORECASE)
    if match:
        choice = match.group(1)
        # Normalize "Determina" to "Determine" and ensure proper capitalization
        ret = 'Determine' if choice.lower() == 'determina' else choice.capitalize()
        return ret


    # Regex breakdown:
    # ^                  - Start of the string (content after ||)
    # (?:[#*]*\s*)?    - Optional non-capturing group for markdown (# or *) and spaces
    # \b                 - Word boundary
    # (Contratti|Determine|Determina) - Capture group for the choice words
    # \b                 - Word boundary
    match = re.search(r'^(?:[#*]*\s*)?\b(Contratti|Determine|Determina)\b', line, re.IGNORECASE)
    if match:
        # Extract the captured choice (group 1)
        choice = match.group(1)
        # Normalize "Determina" to "Determine" and ensure proper capitalization
        ret = 'Determine' if choice.lower() == 'determina' else choice.capitalize()
        return ret


    return "Not Found"

def extract_llm_choices_OLBIA(text: str) -> str:
    """

    """
    
    def normalize_procurement_choice(choice_str: str) -> str:
        """
        Normalizes variations of procurement choices and standardizes capitalization.

        Handles multi-word phrases and known typos.

        Args:
            choice_str: The matched string potentially containing a choice.

        Returns:
            The standardized choice string ("Determina a Contrarre",
            "Procedura negoziata", "Affidamento Diretto") or None if not a match.
        """
        # Clean whitespace and convert to lower for matching
        text = choice_str.strip().lower()
        # Replace multiple spaces with single space for robust comparison
        text = re.sub(r'\s+', ' ', text)

        if text == "determina a contrarre":
            return "Determina a Contrarre"
        elif text == "procedura negoziata" or text == "procedura negoziate": # Handle typo
            return "Procedura negoziata"
        elif text == "affidamento diretto":
            return "Affidamento Diretto"
        else:
            # Not one of the expected choices
            return None

    # Regex to find any of the target phrases (case-insensitive)
    # Allows for variable whitespace (\s+) between words
    # Used for searching within the 'Risposta:' block lines
    choice_regex = re.compile(
        r'\b(Determina\s+a\s+Contrarre|Procedura\s+negoziata|Procedura\s+negoziate|Affidamento\s+Diretto)\b',
        re.IGNORECASE
    )

    # Regex to find target phrases at the start of '||' content (case-insensitive)
    # Allows for optional markdown (#, *) and whitespace at the very beginning
    line_start_regex = re.compile(
        r'^(?:[#*]*\s*)?\b(Determina\s+a\s+Contrarre|Procedura\s+negoziata|Procedura\s+negoziate|Affidamento\s+Diretto)\b',
        re.IGNORECASE
    )

    text_stripped = text.strip()


    match = choice_regex.search(text_stripped)
    if match:
        # Normalize and add if it's a valid choice
        normalized = normalize_procurement_choice(match.group(1))
        return normalized
    


    match = line_start_regex.search(text_stripped)
    if match:
        # Normalize and add if it's a valid choice
        normalized = normalize_procurement_choice(match.group(1))
        return normalized

    return "Not Found"


def compare_columns(df, truth_col, pred_col, numberize_func):
    """
    Compare two columns in a pandas DataFrame and compute evaluation metrics.

    This function first maps the values in the truth and prediction columns using the `numberize`
    function. It then calculates:
      - The percentage of exactly matching entries.
      - Accuracy, weighted precision, weighted recall, weighted F1-score.
      - Balanced accuracy and Cohen's kappa.
      - The confusion matrix and a detailed classification report.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        truth_col (str): The name of the column with ground truth labels (expected values: "SI", "NO", "NON PERTINENTE").
        pred_col (str): The name of the column with predicted labels (expected values: "SI", "NO", "NON PERTINENTE").

    Returns:
        dict: A dictionary containing:
              - "percentage_equal": Percentage of entries where the truth and prediction match.
              - "accuracy": Accuracy score.
              - "precision": Weighted precision score.
              - "recall": Weighted recall score.
              - "f1_score": Weighted F1-score.
              - "balanced_accuracy": Balanced accuracy score.
              - "cohen_kappa": Cohen's kappa score.
              - "confusion_matrix": Confusion matrix (as a list of lists).
              - "classification_report": Detailed classification report (as a dictionary).
              If either column is not found in the DataFrame, returns an error dictionary.
    """

    # Check if columns exist
    if truth_col not in df.columns or pred_col not in df.columns:
        return {"error": "One or both columns not found in DataFrame."}
    
    # Convert values to numbers using the provided numberize function
    df[truth_col+"_"] = df[truth_col].apply(numberize_func)
    df[pred_col+"_"] = df[pred_col].apply(numberize_func)

    
    # New column names
    truth_col_ = truth_col + "_"
    pred_col_ = pred_col + "_"
    
    #print(df[["num",truth_col_,truth_col,pred_col_, pred_col]])


    y_true = df[truth_col_]
    y_pred = df[pred_col_]

    # Define valid labels for the ground truth and extra possible prediction (-1)
    valid_labels = [0, 1, 2]
    

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    

    return results

def graph_df(df):
    
    df = df[df["Municipality"]==LUCCA]
    
    # 2. Create the grouped bar plot using Seaborn
    plt.figure(figsize=(15, 8)) # Use a wider figure for grouped bars
    sns.set_theme(style="whitegrid") # Apply a theme

    # Create the bar plot
    # x='Temperature' defines the positions of the groups
    # y='accuracy' defines the height of the bars
    # hue='Model' defines the individual bars within each group by color
    barplot = sns.barplot(
        data=df,
        x='Temperature',
        y='accuracy',
        hue='Model',
        palette='viridis' # Example using the 'viridis' color palette
        # ci=None # Use errorbar=None in newer seaborn versions if you don't want error bars
    )

    # 3. Customize the plot

    plt.title(f'Model Accuracy per Temperature Setting in Lucca', fontsize=16, pad=20)
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    # Adjust y-axis limits to better show variation, but starting near 0 is often good for bars
    # Let's start slightly below the minimum accuracy observed (around 0.66) to give some space
    min_accuracy = df['accuracy'].min()
    plt.ylim(min_accuracy - 0.1, 1.05) # Start just below minimum, end just above maximum

    # Optional: Add value labels on top of bars
    # This can get crowded on grouped bar charts, adjust fontsize/fmt as needed
    # for container in barplot.containers:
    #    barplot.bar_label(container, fmt='%.2f', fontsize=9, padding=3)


    # Adjust legend position
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # 4. Show the plot
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout rect to make space for legend
    plt.show()


if __name__ == "__main__":
    responses = [] 
    rows = []
    for test in tests:
        
        if test["municipality"] == OLBIA:
            truth_src = "./src/txt/Olbia/checklists/Olbia_Determine.csv"   
            results_root_folder = f"./src/{test["LLM"]}/Olbia_text/choose"
            simplify_func = extract_llm_choices_OLBIA
            number_func = numberize_OLBIA    
        
        elif test["municipality"] == LUCCA:
            truth_src = "./src/txt/Lucca/checklists/Lucca_Determine.csv"
            results_root_folder = f"./src/{test["LLM"]}/Lucca_text/choose"
            simplify_func = extract_llm_choices_LUCCA    
            number_func = numberize_LUCCA  
        
        else:
            raise Exception("NOT FOUND MUNICIPALITY")
            
        # load the csv with all the determine da controllare
        with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
            df_determine = pd.read_csv(f)
        
        for temp in temperatures:
            df_comparison = pd.read_json(f"{results_root_folder}/{test["folder"]}/{temp}/determine.json")
            #responses.extend(df_comparison["LLM"].apply(add_newline).to_list())
            df_comparison["Simple"]=df_comparison["LLM"].apply(simplify_func)
            #responses.extend(df_comparison["Simple"].apply(add_newline).to_list())
            df = df_comparison.merge(
                    df_determine,
                    how='left',                   # Specify the type of join
                    left_on='det',                # Column from the left DataFrame (dataframe1)
                    right_on='Numero Determina'   # Column from the right DataFrame (dataframe2)
                )
            row = compare_columns(df=df,
                                  truth_col="Checklist associata",
                                  pred_col="Simple",
                                  numberize_func=number_func)
            
            row["Model"] = test["model"]
            row["Temperature"] = temp
            row["Municipality"] = test["municipality"]
            rows.append(row)
            
    
    df_results = pd.DataFrame(rows)
    print(df_results)
    df_grup = df_results[["Municipality","Temperature","Model","accuracy"]].groupby(["Municipality","Temperature","Model"]).mean()
    df_results.to_csv("result_choose.csv")
    #print(df_grup)
    graph_df(df_results)       
            




        
    #with open("resposese.txt","w",encoding="utf-8") as f:
    #    f.writelines(responses)