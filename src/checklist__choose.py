from ChecklistCompiler import ChecklistCompiler, LLAMA, LUCCA, OPENAI, OLBIA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
import json
import pandas as pd
import sys

if __name__ == "__main__":
    
    TODO_MUN = [LUCCA]
    TODO_LLM = [LLAMA]
    temperatures = [0.0, 0.01, 0.2, 0.4,
                0.5,0.6,0.8,1.0]
    
    try : 
        #### OPEN_AI
        ### LUCCA
        
        if (LUCCA in TODO_MUN) and (OPENAI in TODO_LLM):
        
            #load the json - Dictionary
            with open("./src/txt/Lucca/checklists/checklists.json","r", encoding="utf-8") as f:
                checklists = json.load(f)
            

            # load the csv with all the determine da controllare
            with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
                df_determine = pd.read_csv(f)
            
            model = "gpt-4o-mini"
            model_folder = "mini/"
            
            #model = "gpt-4o"
            #model_folder = "full/"
            
            
            compiler = ChecklistCompiler(llm=OPENAI,municipality=LUCCA,model=model)
            
            for temp in temperatures:
                subfolder = f"{model_folder}{temp}/"
                compiler.choose_checklist(determine=df_determine,
                                        checklists=checklists,
                                        sub_cartella=subfolder,
                                        temperature=temp)
            
        ### ----------
        ## OLBIA
        if (OLBIA in TODO_MUN) and (OPENAI in TODO_LLM):
            
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
            
            for temp in temperatures:
                subfolder = f"{model_folder}{temp}/"
                compiler.choose_checklist(determine=df_determine,
                                        checklists=checklists,
                                        sub_cartella=subfolder,
                                        temperature=temp)
        
        if (LUCCA in TODO_MUN) and (LLAMA in TODO_LLM):
        
            #load the json - Dictionary
            with open("./src/txt/Lucca/checklists/checklists.json","r", encoding="utf-8") as f:
                checklists = json.load(f)
            

            # load the csv with all the determine da controllare
            with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
                df_determine = pd.read_csv(f)
            
            
            #model_ids = ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.2-3B-Instruct"]
            #model_folders = ["llama-3.1-8B","llama-3.2-3B"]
            model_ids = ["mistralai/Mistral-7B-Instruct-v0.3"]
            model_folders = ["Mistral.7B.Instruct-v0.3/"]
            max_memory = {1: "10GB"}
            
            for model_id, model_folder in zip(model_ids,model_folders):
                
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    #quantization_config=quantization_config,
                    device_map= torch.device('cuda:1'),
                    
                    max_memory=max_memory,
                    #device_map='auto',
                    #use_flash_attention_2=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
                
                
                compiler = ChecklistCompiler(llm=LLAMA,municipality=LUCCA,model=model_id)
                
                for temp in temperatures:
                    
                    if temp == 0.0:
                        do_sample = False
                        t = None
                        top_p = None
                    else:
                        do_sample = True
                        top_p = 0.9
                        t = temp
                    
                    # Create the pipeline
                    text_gen_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=1000,
                        #max_lenght=1000,
                        pad_token_id=tokenizer.eos_token_id,  # open-end generation
                        truncation=True,  # Truncates inputs exceeding model max length
                        eos_token_id = tokenizer.eos_token_id,
                        do_sample=do_sample,
                        temperature = t,
                        top_p = top_p,    
                    )
                        
                    compiler.set_text_gen_pipeline(text_gen_pipeline)
                    
                    subfolder = f"{model_folder}/{temp}/"
                    compiler.choose_checklist(determine=df_determine,
                                            checklists=checklists,
                                            sub_cartella=subfolder,
                                            temperature=temp)
        if (OLBIA in TODO_MUN) and (LLAMA in TODO_LLM):
        
            #load the json - Dictionary
            with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
                checklists = json.load(f)
            

            # load the csv with all the determine da controllare
            with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
                df_determine = pd.read_csv(f)
            
            
            #model_ids = ["meta-llama/Llama-3.1-8B-Instruct","meta-llama/Llama-3.2-3B-Instruct"]
            #model_folders = ["llama-3.1-8B","llama-3.2-3B"]
            
            model_ids = ["mistralai/Mistral-7B-Instruct-v0.3"]
            model_folders = ["Mistral.7B.Instruct-v0.3/"]
            max_memory = {0: "23GB", 1: "23GB"}
            
            for model_id, model_folder in zip(model_ids,model_folders):
                
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    #device_map= torch.device('cuda:1'),
                    
                    max_memory=max_memory,
                    #quantization_config=quantization_config,
                    device_map='auto',
                    #use_flash_attention_2=True
                )
                tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
                
                
                compiler = ChecklistCompiler(llm=LLAMA,municipality=OLBIA,model=model_id)
                
                for temp in temperatures:
                    
                    if temp == 0.0:
                        do_sample = False
                        t = None
                        top_p = None
                    else:
                        do_sample = True
                        top_p = 0.9
                        t = temp
                    
                    # Create the pipeline
                    text_gen_pipeline = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=1000,
                        #max_lenght=1000,
                        pad_token_id=tokenizer.eos_token_id,  # open-end generation
                        truncation=True,  # Truncates inputs exceeding model max length
                        eos_token_id = tokenizer.eos_token_id,
                        do_sample=do_sample,
                        temperature = t,
                        top_p = top_p,    
                    )
                        
                    compiler.set_text_gen_pipeline(text_gen_pipeline)
                    
                    subfolder = f"{model_folder}/{temp}/"
                    compiler.choose_checklist(determine=df_determine,
                                            checklists=checklists,
                                            sub_cartella=subfolder,
                                            temperature=temp)
    except Exception as e:
        sys.exit(1) # General failure exit code
    
    sys.exit(0) # Explicitly exit with 0 for success