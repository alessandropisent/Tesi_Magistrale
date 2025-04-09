import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
import sys
from ChecklistCompiler import ChecklistCompiler, LLAMA, OLBIA



if __name__ == "__main__":
    
    ## Multilingual model + 16K of context
    #model_id = "meta-llama/Llama-3.1-8B-Instruct"
    #model_id = "meta-llama/Llama-3.2-3B-Instruct"
    #model_id = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    #model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
    #model_id = "meta-llama/Llama-3.1-70B-Instruct"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    #temperatures = [0.0,0.01,0.2,0.4,0.5,0.6,0.8,1.0]
    temperatures = [0.4,0.5,0.6,0.8,1.0]
    

    # This is to the possiblity to not load the model and just test for errors
    try:

        # Setup the 4-bit quantization configuration. Using nf4 and double quantization are common choices.
        #quantization_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_compute_dtype=torch.float16,
        #    bnb_4bit_quant_type="nf4",  # nf4 is often recommended for good accuracy/speed trade-off
        #    bnb_4bit_use_double_quant=True
        #)
        max_memory = {0: "13GB", 1: "23GB"}
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            
            #device_map= torch.device('cuda:0'),
            
            device_map='auto',
            max_memory=max_memory,
            #quantization_config=quantization_config,
            #use_flash_attention_2=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
        
        #load the json - Dictionary
        with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
            checklists = json.load(f)

        # load the csv with all the determine da controllare
        with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
            df_determine = pd.read_csv(f)
   
        
        compiler = ChecklistCompiler(llm=LLAMA,
                                    municipality=OLBIA,
                                    model=model_id)

        
        
      
        for temp in temperatures:
            for i, _ in df_determine.iterrows():
                   
                num = df_determine["Numero Determina"].loc[i]
                che_ass = df_determine["Checklist associata"].loc[i]
                #sub_cartella = f"3.1.llama.8B.Instruct/{temp}/"
                #sub_cartella = f"3.2.llama.3B.Instruct/{temp}/"
                sub_cartella = f"Mistral.7B.Instruct-v0.3/{temp}/"
                
                
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
                
                
                compiler.checklist_determina(nome_determina=num,
                                            nome_checklist=che_ass, 
                                            checklists=checklists,
                                            sub_cartella=sub_cartella)
                print(f"Done determina [{i}] {num} - {che_ass} - temp:{temp}")
    
    except Exception as e:
        sys.exit(1) # General failure exit code
    
    sys.exit(0) # Explicitly exit with 0 for success
                

## Numero Determina,Checklist associata