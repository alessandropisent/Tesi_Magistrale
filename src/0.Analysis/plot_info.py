import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def graph_df(df, col_to_plot="accuracy", title_ylabel="Accuracy"):

    
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
        y=col_to_plot,
        hue='Modello',
        palette='viridis' # Example using the 'viridis' color palette
        # ci=None # Use errorbar=None in newer seaborn versions if you don't want error bars
    )

    # 3. Customize the plot

    plt.title(f'Model Accuracy per Temperature Setting', fontsize=16, pad=20)
    plt.xlabel('Temperature Setting', fontsize=12)
    plt.ylabel(title_ylabel, fontsize=12)

    # Adjust y-axis limits to better show variation, but starting near 0 is often good for bars
    # Let's start slightly below the minimum accuracy observed (around 0.66) to give some space
    min_accuracy = df[col_to_plot].min()
    plt.ylim(0, 1.05) # Start just below minimum, end just above maximum

    # Optional: Add value labels on top of bars
    # This can get crowded on grouped bar charts, adjust fontsize/fmt as needed
    # for container in barplot.containers:
    #    barplot.bar_label(container, fmt='%.2f', fontsize=9, padding=3)


    # Adjust legend position
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # 4. Show the plot
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout rect to make space for legend
    plt.show()

df = pd.read_csv(f"src/Evaluation/checklist_compiler/statistics.csv")
keep_rows =["Modello",
            "Temperature",
            "accuracy",
            "precision",
            "recall",
            #"determina_in",
            "f1_score",
            "balanced_accuracy",
            ]

df = df[df["determina_in"]=="system"]

group_of_models = df[keep_rows].groupby(by=["Temperature","Modello"]).mean()
print(group_of_models)


#graph_df(df=group_of_models.reset_index(), col_to_plot="balanced_accuracy", title_ylabel="Balanced Accuracy")
graph_df(df=group_of_models.reset_index())
