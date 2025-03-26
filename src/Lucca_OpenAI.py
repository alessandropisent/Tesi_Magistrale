from dotenv import load_dotenv
import os
import openai
import json
import pandas as pd
import re
from tqdm import tqdm

from ChecklistCompiler import ChecklistCompiler, OPENAI, LUCCA
                



if __name__ == "__main__":
    
    #done = [0,1,3,4,5,6,7,8,9,10]
    #todo = 2
    #done = [0,1,2]
    done = []
    model = "gpt-4o-mini"
    model_folder = "mini/"
    
    #model = "gpt-4o"
    #model_folder = "full"

    #load the json - Dictionary
    with open("./src/txt/Lucca/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)
    
    temp_values = [0.0,0.01,0.5,1.0]
    
    compiler = ChecklistCompiler(
        llm=OPENAI,
        municipality=LUCCA,
        model=model
    )
    
    
    if True:
        for temperature in temp_values:
            for i, row in df_determine.iterrows():
                if i not in done:
                    print(f"DOING determina {i} - temp:{temperature} ")
                    num = row["Numero Determina"]
                    che_ass = row["Checklist associata"]
                    model_folder_t = model_folder + f"{temperature}/"
                    
                    compiler.checklist_determina(nome_determina=num, 
                                        nome_checklist=che_ass, 
                                        checklists=checklists,
                                        sub_cartella=model_folder_t,
                                        temperature=temperature)
                    
                    print(f"Done determina {num} - {che_ass}")
    
    folders = [f"{model_folder}{temp}/" for temp in temp_values]
    
    for i,row in df_determine.iterrows():
        for folder in folders:
            pass

