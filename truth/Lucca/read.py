import pandas as pd

df = pd.read_csv("truth/Lucca/41_2025-T_cont.csv")
df_2 = pd.read_csv("truth/Lucca/41_2025-T_cont.csv")

print(df)
print("\n\n")
print(pd.concat([df,df_2],axis=1))