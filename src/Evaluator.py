import pandas as pd
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

import os
import json
from ChecklistCompiler import LLAMA,OPENAI, LUCCA,OLBIA

def numberize(el):
    """
    Convert a categorical string value into a numeric code.

    Parameters:
        el (str): Input string expected to be one of "SI", "NO", or "NON PERTINENTE".
                  Any other value returns -1.

    Returns:
        int: Returns 1 for "SI", 0 for "NO", 2 for "NON PERTINENTE", and -1 for any other value.
    """
    if el == "SI":
        return 1
    elif el == "NO":
        return 0
    elif el == "NON PERTINENTE":
        return 2
    else:
        return -1


def compare_columns(df, truth_col, pred_col):
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
    df[truth_col+"_"] = df[truth_col].apply(numberize)
    df[pred_col+"_"] = df[pred_col].apply(numberize)

    
    # New column names
    truth_col_ = truth_col + "_"
    pred_col_ = pred_col + "_"
    
    #print(df[["num",truth_col_,truth_col,pred_col_, pred_col]])


    # Calculate percentage of exactly matching entries
    equal_count = (df[truth_col_] == df[pred_col_]).sum()
    total_count = len(df)
    percentage_equal = (equal_count / total_count) * 100

    y_true = df[truth_col_]
    y_pred = df[pred_col_]

    # Define valid labels for the ground truth and extra possible prediction (-1)
    valid_labels = [0, 1, 2]
    all_labels = valid_labels + [-1]

    # Convert y_true and y_pred to categorical types that include all_labels
    y_true_cat = pd.Categorical(y_true, categories=all_labels)
    y_pred_cat = pd.Categorical(y_pred, categories=all_labels)

    results = {
        "percentage_equal": percentage_equal,
        "accuracy": accuracy_score(y_true_cat, y_pred_cat),
        "precision": precision_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true_cat, y_pred_cat),
        "cohen_kappa": cohen_kappa_score(y_true_cat, y_pred_cat, labels=all_labels),
        "confusion_matrix": confusion_matrix(y_true_cat, y_pred_cat, labels=all_labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=valid_labels, zero_division=0, output_dict=True)
    }

    return results


def calculate_and_return_per_det(nome_determina, 
                                 nome_checklist,
                                 folder,
                                 model,
                                 municipality, 
                                 temperature,
                                 doIWrite=True,
                                 determina_in=""):
    """
    Process and evaluate checklist data for a specific determination.

    This function performs the following steps:
      1. Reads the ground truth CSV file for the specified municipality and determination.
      2. Loads the corresponding JSON file containing checklist responses.
      3. Merges the ground truth data with the checklist responses.
      4. Computes evaluation metrics by comparing the ground truth and predictions.
      5. Optionally writes the merged DataFrame to a CSV file.

    Parameters:
        nome_determina (str): Identifier for the determination.
        nome_checklist (str): Name of the associated checklist.
        folder (str): Directory path containing the JSON response file.
        model (str): Name of the model used to generate the checklist responses.
        municipality (str): Name of the municipality (e.g., LUCCA or OLBIA).
        temperature (float): Temperature parameter used in model evaluation.
        doIWrite (bool, optional): Flag indicating whether to write the resulting DataFrame to a CSV file. Defaults to True.
        determina_in (str,optional): Where is the determina [user, system]

    Returns:
        tuple: A tuple containing:
            - ret (dict): A dictionary with metadata and evaluation metrics for the determination.
            - df (pd.DataFrame): The merged DataFrame that includes both truth and predicted values.
    """
    
    ret = { "Determina" : nome_determina,
            "Modello":model,
            "Temperature":temperature,
            "Checklist":nome_checklist,
            "Municipality":municipality,
            "determina_in":determina_in,
            }
    
    # Aprire Truth as df
    df_truth = pd.read_csv(f"truth/{municipality}/{nome_determina}_TRUTH.csv")
    df_truth["TRUTH"] = df_truth["SI/NO"]
    df_truth["num"] = df_truth["Punto"]
    
    # Apri il file checklist compilata
    with open(f"{folder}{nome_determina}.json","r",encoding="utf-8") as f:
        data = json.load(f)
        
    df_llm = pd.DataFrame(data["Response"])
    
    #print(df_llm)
    #print(df_truth)
    
    df = df_llm.merge(df_truth[["num","TRUTH"]],how="left",on="num")
    
    #print(df)
    
    # compila i risultati
    results = compare_columns(df,"TRUTH","Simple")
    
    # Write to csv the dataframe
    if doIWrite:
        if not os.path.exists(f"src/Evaluation/checklist_chooser/{municipality}/{model}/{temperature}/"):
            os.makedirs(f"src/Evaluation/checklist_chooser/{municipality}/{model}/{temperature}/")  
        
        df.to_csv(f"src/Evaluation/checklist_chooser/{municipality}/{model}/{temperature}/{nome_determina}.csv")
    
    ret.update(results)
    
    return ret,df


if __name__ == "__main__":
    """
    Main section to execute the checklist evaluation pipeline.

    This block performs the following:
      1. Defines temperature values for different municipalities/models.
      2. Sets up a dictionary (dic_todo) containing tasks for each municipality with parameters including:
            - Model name
            - List of JSON response folder paths (one per temperature)
            - Temperature values
            - Checklist JSON file path
            - CSV file path listing determinations to process
            - Municipality identifier
      3. Initializes empty DataFrames to store aggregated results and individual DataFrames.
      4. Iterates over each task in dic_todo:
            a. Loads the checklist JSON and determination CSV.
            b. Processes each determination by calling calculate_and_return_per_det.
            c. Aggregates results and merged DataFrames.
      5. Concatenates and writes the final aggregated DataFrames to CSV files for comprehensive evaluation.

    Note:
        - Output directories and files are created if they do not exist.
        - The results include both statistics (metrics) and the detailed merged DataFrames.
    """
    
    
    temp_values_LUCCA_OPENAI = [0.0, 0.01, 0.2, 0.5, 1.0]
    temp_values_OLBIA_OPENAI = [0.0, 0.01, 0.2, 0.5, 1.0]
    
    
    dic_todo = {"TODO":[
                    {"model": "gpt-4o-mini",
                     "folders_response_json":[f"./src/openai/Lucca_text/responses/mini_1/{temp}/" for temp in temp_values_LUCCA_OPENAI],
                     "temperatures": temp_values_LUCCA_OPENAI,
                     "checklists_json":"src/txt/Lucca/checklists/checklists.json",
                     "csv_determine":"src/txt/Lucca/checklists/Lucca_Determine.csv",
                     "municipality":LUCCA,
                     "determina_in":"user",
                    },

                    {"model": "gpt-4o-mini",
                     "folders_response_json":[f"./src/openai/Lucca_text/responses/mini/{temp}/" for temp in temp_values_LUCCA_OPENAI],
                     "temperatures": temp_values_LUCCA_OPENAI,
                     "checklists_json":"src/txt/Lucca/checklists/checklists.json",
                     "csv_determine":"src/txt/Lucca/checklists/Lucca_Determine.csv",
                     "municipality":LUCCA,
                     "determina_in":"system",
                    },
                    {"model": "gpt-4o-mini",
                     "folders_response_json":[f"./src/openai/Olbia_text/responses/mini/{temp}/" for temp in temp_values_OLBIA_OPENAI],
                     "temperatures":temp_values_OLBIA_OPENAI,
                     "checklists_json":"src/txt/Olbia/checklists/checklists.json",
                     "csv_determine":"src/txt/Olbia/checklists/Olbia_Determine.csv",
                     "municipality":OLBIA,
                     "determina_in":"system",
                    },
                    {"model": "gpt-4o",
                     "folders_response_json":[f"./src/openai/Lucca_text/responses/full/{temp}/" for temp in temp_values_LUCCA_OPENAI],
                     "temperatures": temp_values_LUCCA_OPENAI,
                     "checklists_json":"src/txt/Lucca/checklists/checklists.json",
                     "csv_determine":"src/txt/Lucca/checklists/Lucca_Determine.csv",
                     "municipality":LUCCA,
                     "determina_in":"system",
                    },
                    {"model": "gpt-4o",
                     "folders_response_json":[f"./src/openai/Olbia_text/responses/full/{temp}/" for temp in temp_values_OLBIA_OPENAI],
                     "temperatures":temp_values_OLBIA_OPENAI,
                     "checklists_json":"src/txt/Olbia/checklists/checklists.json",
                     "csv_determine":"src/txt/Olbia/checklists/Olbia_Determine.csv",
                     "municipality":OLBIA,
                     "determina_in":"system",
                    },
                    ],
                }
    
    rows = []
    dfs = []
    
    for todo in dic_todo["TODO"]:
        for i_temp, folder in enumerate(todo["folders_response_json"]):
            #load the json - Dictionary
            with open(todo["checklists_json"],"r", encoding="utf-8") as f:
                checklists = json.load(f)

            # load the csv with all the determine da controllare
            with open(todo["csv_determine"],"r", encoding="utf-8") as f:
                df_determine = pd.read_csv(f)
            
            for i,row in df_determine.iterrows():
                num = row["Numero Determina"]
                che_ass = row["Checklist associata"]
                row_to_append, df_append = calculate_and_return_per_det(nome_determina=num,
                                                         nome_checklist=che_ass,
                                                         folder=folder,
                                                         model=todo["model"],
                                                         temperature=todo["temperatures"][i_temp],
                                                         municipality=todo["municipality"],
                                                         determina_in=todo["determina_in"])
                rows.append(row_to_append)
                dfs.append(df_append)
    
    df_dfs = pd.concat(dfs)
    df_dfs.to_csv(f"src/Evaluation/checklist_chooser/full.csv")
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(f"src/Evaluation/checklist_chooser/statistics.csv")
    
    keep_rows =["Modello",
                "Temperature",
                "determina_in",
                "percentage_equal",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "balanced_accuracy",
                "cohen_kappa",
                ]
    
    group_of_models = df_rows[keep_rows].groupby(by=["Modello","determina_in","Temperature"]).mean()
    
    print(group_of_models)
    
    group_of_models.to_csv(f"src/Evaluation/checklist_chooser/group.csv")
    
    #print(df_dfs)
    
    #print(df_rows)
    
                
                
            
        
    
