import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

df = pd.read_csv("src/Evaluation/checklist_compiler/statistics.csv")
#print(df)

def plot_metrics_separate(df):
    """
    Plot each classification metric vs. Temperature in separate subplots.
    
    Parameters:
        df (DataFrame): DataFrame containing 'Temperature' and metric columns 
                        (e.g., 'accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy').
    """
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
    n_metrics = len(metrics_cols)
    
    fig, axs = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), sharey=True)
    
    for i, metric in enumerate(metrics_cols):
        axs[i].plot(df['Temperature'], df[metric], marker='o', linestyle='-')
        axs[i].set_title(metric.capitalize())
        axs[i].set_xlabel('Temperature')
        axs[i].set_ylim(0, 1)
    axs[0].set_ylabel('Metric Value')
    
    plt.tight_layout()
    plt.show()

def plot_metrics_combined(df):
    """
    Plot all classification metrics vs. Temperature on a single plot.
    
    Parameters:
        df (DataFrame): DataFrame containing 'Temperature' and metric columns 
                        (e.g., 'accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy').
    """
    metrics_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'balanced_accuracy']
    
    plt.figure(figsize=(8, 6))
    for metric in metrics_cols:
        plt.plot(df['Temperature'], df[metric], marker='o', linestyle='-', label=metric.capitalize())
    
    plt.title('Classification Metrics vs. Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def set_matrix(el):
    if isinstance(el, str):  
        el = ast.literal_eval(el)  # Convert string to actual list
    return np.array(el, dtype=int)  # Ensure numeric dtype
    

def plot_confusion_matrix(df):
    t = df["confusion_matrix"].apply(set_matrix)
    agg_conf_matrix = np.sum(np.stack(t.to_list()),axis=0)
    
    labels = ["NO", "SI","NON PERTINENTE"]
    
    # Create a heatmap plot of the aggregated confusion matrix
    plt.figure(figsize=(8, 6))
    
    ax = sns.heatmap(agg_conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    # Set the custom tick labels
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    
    plt.title('Aggregated Confusion Matrix')
    plt.show()

keep_rows =[#"Modello",
            "Temperature",
            #"determina_in",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "balanced_accuracy",
            ]

df_without_noise = df[df["determina_in"] != "user"].reset_index()

#print(df_without_noise)

group_of_models =df_without_noise[keep_rows].groupby(by=["Temperature"]).mean().reset_index()

print(group_of_models)

#plot_metrics_combined(group_of_models)
plot_confusion_matrix(df_without_noise)

