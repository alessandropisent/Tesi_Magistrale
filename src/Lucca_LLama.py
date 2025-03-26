import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
import pandas as pd
from ChecklistCompiler import ChecklistCompiler, LLAMA, LUCCA

if __name__ == "__main__":
    
    ## Multilingual model + 16K of context
    #model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_id = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    #model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
    #model_id = "meta-llama/Llama-3.1-70B-Instruct"

    # This is to the possiblity to not load the model and just test for errors
    if True:
        # Define the quantization configuration for 8-bit precision
        #quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #quantization_config=quantization_config,
            device_map= torch.device('cuda:1'),
            
            #device_map='auto',
            #use_flash_attention_2=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")



    #load the json - Dictionary
    with open("./src/txt/Lucca/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)



    ## Choose the right checklist for each determina
    ## i do a loop to make it clear
    if False:
        for index, row in df_determine.iterrows():
            print(f"I'm genenerting the checklist for {index}, det {row['Numero Determina']} - {model_id.split("/", 1)[1]}")
            df_determine.at[index, 'gen'] = choose_checklist(row['Numero Determina'],
                                                            text_gen_pipeline,
                                                            model_id.split("/", 1)[1])
        
        df_determine.to_csv("./src/llama/Lucca_text/Lucca_Determine_gen.csv")
    
    
    temperatures = [0.01,0.2,0.4,0.6]
    
    compiler = ChecklistCompiler(llm=LLAMA, municipality=LUCCA, model=model)
    
    if True:
        for temp in temperatures:
            for i, _ in df_determine.iterrows():
                
                # Create the pipeline
                text_gen_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1000,
                    max_lenght=1000,
                    pad_token_id=tokenizer.eos_token_id,  # open-end generation
                    truncation=True,  # Truncates inputs exceeding model max length
                    eos_token_id = tokenizer.eos_token_id,
                    do_sample=True,
                    temperature = temp,
                    top_p = 0.9,    
                )
                
                compiler.set_text_gen_pipeline(text_gen_pipeline)
        
                
                num = df_determine["Numero Determina"].loc[i]
                che_ass = df_determine["Checklist associata"].loc[i]
                sub_cartella = f"llamantino/{temp}/"
                
                
                compiler.checklist_determina(num,
                                    che_ass,
                                    checklists,
                                    text_gen_pipeline=text_gen_pipeline,
                                    sub_cartella=sub_cartella,
                                    )
                print(f"Done determina {num} - {che_ass}")
    
    compiler.relate(temperatures=temperatures)           

## Numero Determina,Checklist associata