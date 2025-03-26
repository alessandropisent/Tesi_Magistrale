import json
import pandas as pd
from ChecklistCompiler import ChecklistCompiler, OLBIA, OPENAI


if __name__ == "__main__":
    
    #done = [0,1,3,4,5,6,7,8,9,10]
    #todo = 2
    done = []
    model = "gpt-4o-mini"
    model_folder = "mini/"
    
    #model = "gpt-4o"
    #model_folder = "full/"

    #load the json - Dictionary
    with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)
    
    #temp_values = [round(x * 0.2, 1) for x in range(6)]
    temp_values = [0.0, 0.2, 0.5, 1.0]
    #done = [1,2,3,4,5,6,7,8,9,10]
    
    compiler = ChecklistCompiler(llm=OPENAI,
                                 municipality=OLBIA,
                                 model=model)
    
    if True:
        for temperature in temp_values:
            for i, _ in df_determine.iterrows():
                if i not in done:
                    print(f"DOING determina {i} - temp :{temperature}")
                    num = df_determine["Numero Determina"].loc[i]
                    che_ass = df_determine["Checklist associata"].loc[i]
                    model_folder_t = model_folder + f"{temperature}/"
                    compiler.checklist_determina(nome_determina=num, 
                                        nome_checklist=che_ass, 
                                        checklists=checklists,
                                        sub_cartella=model_folder_t, 
                                        temperature=temperature)
                    
                    print(f"Done determina {num} - {che_ass}")

