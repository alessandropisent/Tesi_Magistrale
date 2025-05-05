import pandas as pd

df = pd.read_csv(f"src/Evaluation/checklist_compiler/statistics.csv")


df = df[df["determina_in"]=="system"] # Filter if needed

for metric in ["precision","recall"]:

    table = pd.pivot_table(df, values=metric, index=['Modello'],
                        columns=['Temperature'], aggfunc="mean", fill_value=0)

    with open(f"table_{metric}.txt","w") as f:
        f.write(table.to_latex())