import pandas as pd

df = pd.read_csv("affidamentiDiretti.tsv",sep="\t")

with open("sez.json","w") as f:
    for i,row in df.iterrows():
        
        gr_str = ""
        if row["Gravita"] == 0 or pd.isna( row["Gravita"]):
            gr_str = ""
        elif row["Gravita"] == 1:
            gr_str = " \\n\\n**GRAVITA' [1/3](lieve) se non presenti**"
        elif row["Gravita"] == 2:
            gr_str = "\\n\\n**GRAVITA' [2/3](rilevante) se non presenti**"
        elif row["Gravita"] == 3:
            gr_str = "\\n\\n**GRAVITA' [3/3](GRAVE) se non presenti**"
    
        toWrite = f'''
{{ 
    "Istruzioni": "{row["Istruzioni"]}{gr_str}",
    "Punti":"{row["Punti"]}",
    "Sezione": "{row["Sezione"]}",
    "num":{row["num"]}

}},
'''
        f.write(toWrite)
    

