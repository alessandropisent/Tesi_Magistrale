import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import json
from ChecklistCompiler import LLAMA,OPENAI, LUCCA,OLBIA

def numberize(el):
    if el == "SI":
        return 1
    elif el == "NO":
        return 0
    elif el == "NON PERTINENTE":
        return 2,
    else:
        return -1

def compare_columns(df, truth_col, pred_col):
    """
    Compares two columns of a pandas DataFrame and returns various metrics.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        truth_col (str): The name of the first column.
        pred_col (str): The name of the second column.

    Returns:
        dict: A dictionary containing the percentage of equal values, accuracy,
              precision, recall, F1-score, and the confusion matrix.
    """

    if truth_col not in df.columns or pred_col not in df.columns:
        return {"error": "One or both columns not found in DataFrame."}
    
    df[truth_col+"_"] = df[truth_col].apply(numberize)
    df[pred_col+"_"] = df[pred_col].apply(numberize)

    truth_col = truth_col+"_"
    pred_col = pred_col+"_"

    equal_count = (df[truth_col] == df[pred_col]).sum()
    total_count = len(df)
    percentage_equal = (equal_count / total_count) * 100

    y_true = df[truth_col]  # Assuming truth_col is the "true" labels
    y_pred = df[pred_col]  # Assuming pred_col is the "predicted" labels

    results = {
        "percentage_equal": percentage_equal,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    return results


def calculate_and_return_per_det(nome_determina, 
                                 nome_checklist,
                                 folder,
                                 model,
                                 municipality, 
                                 temperature):
    
    ret = { "Determina" : nome_determina,
            "Modello":model,
            "Temperature":temperature,
            "Checklist":nome_checklist,
            "municipality":municipality,
            }
    
    # Aprire Truth as df
    df_truth = pd.read_csv(f"truth/{municipality}/{nome_determina}_TRUTH.csv")
    df_truth["TRUTH"] = df_truth["SI/NO"]
    df_truth["num"] = df_truth["Punto"]
    
    # Apri il file checklist compilata
    with open(f"{folder}{nome_determina}.json","r",encoding="utf-8") as f:
        data = json.load(f)
        
    df_llm = pd.DataFrame(data["Response"])
    
    print(df_llm)
    print(df_truth)
    
    df = df_llm.merge(df_truth[["num","TRUTH"]],how="left",on="num")
    
    print(df)
    
    # compila i risultati
    results = compare_columns(df, "TRUTH","Simple")
    
    ret.update(results)
    
    return ret


if __name__ == "__main__":
    
    temp_values_LUCCA_OPENAI = [0.0,0.01,0.5,1.0]
    temp_values_OLBIA_OPENAI = [0.0, 0.2, 0.5, 1.0]
    
    
    dic_todo = {"TODO":[
                    {"model": "gpt-4o-mini",
                     "folders_response_json":[f"./src/openai/Lucca_text/responses/mini/{temp}/" for temp in temp_values_LUCCA_OPENAI],
                     "temperatures": temp_values_LUCCA_OPENAI,
                     "checklists_json":"src/txt/Lucca/checklists/checklists.json",
                     "csv_determine":"src/txt/Lucca/checklists/Lucca_Determine.csv",
                     "municipality":LUCCA,
                    },
                    {"model": "gpt-4o-mini",
                     "folders_response_json":[f"./src/openai/Olbia_text/responses/mini/{temp}/" for temp in temp_values_OLBIA_OPENAI],
                     "temperatures":temp_values_OLBIA_OPENAI,
                     "checklists_json":"src/txt/Olbia/checklists/checklists.json",
                     "csv_determine":"src/txt/Olbia/checklists/Olbia_Determine.csv",
                     "municipality":OLBIA,
                    },
                    ],
                }
    
    df_results = pd.DataFrame(columns=["Determina","Modello",
                                       "Temperature","percentage_equal",
                                       "accuracy","precision", "recall", "f1_score"])
    
    rows = []
    
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
                rows.append(calculate_and_return_per_det(nome_determina=num,
                                                         nome_checklist=che_ass,
                                                         folder=folder,
                                                         model=todo["model"],
                                                         temperature=todo["temperatures"][i_temp],
                                                         municipality=todo["municipality"])
                            )
                break
    
