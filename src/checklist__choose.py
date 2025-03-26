from ChecklistCompiler import ChecklistCompiler, LLAMA, LUCCA, OPENAI, OLBIA
import json
import pandas as pd

if __name__ == "__main__":
    
    TODO_MUN = [OLBIA]
    TODO_LLM = [OPENAI]
    
    #### OPEN_AI
    ### LUCCA
    
    if (LUCCA in TODO_MUN) and (OPENAI in TODO_LLM):
    
        #load the json - Dictionary
        with open("./src/txt/Lucca/checklists/checklists.json","r", encoding="utf-8") as f:
            checklists_lucca = json.load(f)
        

        # load the csv with all the determine da controllare
        with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
            df_determine = pd.read_csv(f)
        
        model = "gpt-4o-mini"
        model_folder = "mini/"
        
        #model = "gpt-4o"
        #model_folder = "full/"
        
        
        compiler = ChecklistCompiler(llm=OPENAI,municipality=LUCCA,model=model)
        
        for temp in [0.0]:
            subfolder = f"{model_folder}{temp}/"
            compiler.choose_checklist(determine=df_determine,
                                    checklists=checklists_lucca,
                                    sub_cartella=subfolder,
                                    temperature=temp)
        
    ### ----------
    ## OLBIA
    elif (OLBIA in TODO_MUN) and (OPENAI in TODO_LLM):
        
        #load the json - Dictionary
        with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
            checklists = json.load(f)
        
        # load the csv with all the determine da controllare
        with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
            df_determine = pd.read_csv(f)
        
        model = "gpt-4o-mini"
        model_folder = "mini/"
        
        #model = "gpt-4o"
        #model_folder = "full/"
        
        
        compiler = ChecklistCompiler(llm=OPENAI,
                                     municipality=OLBIA,
                                     model=model)
        
        for temp in [0.0]:
            subfolder = f"{model_folder}{temp}/"
            compiler.choose_checklist(determine=df_determine,
                                    checklists=checklists,
                                    sub_cartella=subfolder,
                                    temperature=temp)