import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os



#print(df_determine)

def numberize(el):
    if el == "SI":
        return 1
    elif el == "NO":
        return 0
    else:
        return 2
    

def compare_columns(df, col1, col2):
    """
    Compares two columns of a pandas DataFrame and returns various metrics.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns.
        col1 (str): The name of the first column.
        col2 (str): The name of the second column.

    Returns:
        dict: A dictionary containing the percentage of equal values, accuracy,
              precision, recall, F1-score, and the confusion matrix.
    """

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns not found in DataFrame."}
    
    df[col1+"_"] = df[col1].apply(numberize)
    df[col2+"_"] = df[col2].apply(numberize)

    col1 = col1+"_"
    col2 = col2+"_"

    equal_count = (df[col1] == df[col2]).sum()
    total_count = len(df)
    percentage_equal = (equal_count / total_count) * 100

    y_true = df[col1]  # Assuming col1 is the "true" labels
    y_pred = df[col2]  # Assuming col2 is the "predicted" labels

    results = {
        "percentage_equal": percentage_equal,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    return results


def statistics_gen(df_determine,from_truth_path, from_generated_path,results_path):
    statistics = []
    
    if not os.path.exists(f"{results_path}/"):
        os.makedirs(f"{results_path}/")  

    for i,row in df_determine.iterrows():
        if row["checklist"] == "Contratti":
            name_checklist_end = "cont"
        elif row["checklist"] == "Determine":
            name_checklist_end = "det"
        df_generated  = pd.read_csv(f"{from_generated_path}/{row["nome"]}-G_{name_checklist_end}.csv")
        df_generated["Punto"] = df_generated["num"]
        df_generated = df_generated.drop(columns=["num"])
        df_truth  = pd.read_csv(f"{from_truth_path}/{row["nome"]}-T_{name_checklist_end}.csv")
        #print(df_generated)
        #print("\n\n")
        #print(df_truth)
        df = df_truth.merge(df_generated[["Punto","Simple","LLM.generated_text"]],how="left",on="Punto")
        df.to_csv(f"{results_path}/{row["nome"]}.csv")
        df.to_excel(f"{results_path}/{row["nome"]}.xlsx")
        #print(df)
        results = compare_columns(df,"SI/NO","Simple")
        results["num"] = i
        statistics.append(results)

    df_determine = pd.concat([df_determine,pd.DataFrame(statistics)],axis=1)
    print(df_determine)
    df_determine.to_csv(f"{results_path}/results.csv")
    

if __name__ == "__main__":
    df_determine = pd.read_csv("src/Evaluation/Lucca/checklists.csv")
    from_truth_path = "src/Evaluation/Lucca/"
    
    temperatures = [0.01,0.2,0.4,0.6]
    
    for temp in temperatures:
        sub_cartella = f"General/{temp}"
        model_folder = "./src/llama/Lucca_text/responses/"
        from_generated = model_folder+sub_cartella
        out_path = f"src/Evaluation/Lucca/{temp}"
        
        statistics_gen(df_determine,from_truth_path,from_generated,out_path)
        
    
    

    
    
    