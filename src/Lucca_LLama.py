import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
import pandas as pd
from ChecklistCompiler import ChecklistCompiler, LLAMA, LUCCA

if __name__ == "__main__":
    
    ## Multilingual model + 16K of context
    #model_id = "meta-llama/Llama-3.1-8B-Instruct"
    #model_id = "meta-llama/Llama-3.2-3B-Instruct"
    #model_id = "meta-llama/Llama-3.3-70B-Instruct"
    #model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    #model_id = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    #model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
    #model_id = "meta-llama/Llama-3.1-70B-Instruct"
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    

    # This is to the possiblity to not load the model and just test for errors
    if True:
        # Define the quantization configuration for 4-bit precision
        # Create a dictionary to limit memory per GPU (adjust based on your GPU capacity)
        # Configure maximum GPU memory per device (here, 23GB per GPU)
        max_memory = {0: "13GB", 1: "23GB"}

        # Setup the 4-bit quantization configuration. Using nf4 and double quantization are common choices.
        #quantization_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_compute_dtype=torch.float16,
        #    bnb_4bit_quant_type="nf4",  # nf4 is often recommended for good accuracy/speed trade-off
        #    bnb_4bit_use_double_quant=True
        #)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #device_map= torch.device('cuda:0'),
            
            trust_remote_code = True,
            #quantization_config=quantization_config,
            device_map='auto',
            max_memory=max_memory,   # Ensure each GPU uses at most 23GB of VRAM
            #max_memory=max_memory,
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
    
    
    #temperatures = [0.0,0.01,0.2,0.4,0.5,0.6,0.8,1.0]
    temperatures = [0.0,0.01,0.2,0.4,0.5,0.6,0.8,1.0]
    #temperatures = [0.2,0.5,1.0]
    #temperatures = [0.0,0.1,0.2,0.5,0.7,1.0]
    #temperatures = [0.01,0.5,1.0]
    
    
    
    compiler = ChecklistCompiler(llm=LLAMA, municipality=LUCCA, model=model_id)
    
    if True:
        for temp in temperatures:
            for i, _ in df_determine.iterrows():
                
                #sub_cartella = f"3.2.llama.3B.Instruct/{temp}/"
                #sub_cartella = f"3.1.llama.8B.Instruct/{temp}/"
                #sub_cartella = f"3.1.llama.70B.Instruct/{temp}/"
                #sub_cartella = f"3.3.llama.70B.Instruct/{temp}/"
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
                    max_new_tokens=500,
                    #max_lenght=1000,
                    pad_token_id=tokenizer.eos_token_id,  # open-end generation
                    truncation=True,  # Truncates inputs exceeding model max length
                    eos_token_id = tokenizer.eos_token_id,
                    do_sample=do_sample,
                    temperature = t,
                    top_p = top_p,    
                )
                
                compiler.set_text_gen_pipeline(text_gen_pipeline)
        
                
                num = df_determine["Numero Determina"].loc[i]
                che_ass = df_determine["Checklist associata"].loc[i]
                
                
                
                compiler.checklist_determina(num,
                                    che_ass,
                                    checklists,
                                    sub_cartella=sub_cartella,
                                    temperature=temp,
                                    )
                print(f"Done determina [{i}] {num} - {che_ass} - temp:{temp}")
    
               

## Numero Determina,Checklist associata