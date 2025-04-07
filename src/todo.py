import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch
import pandas as pd
from ChecklistCompiler import ChecklistCompiler, LLAMA, LUCCA, OLBIA
import time
from tqdm import tqdm 
import sys
import gc


def main():
    compiler = None
    text_gen_pipeline = None
    model = None
    tokenizer = None
    
    try:
    
        temperatures = [0.0,0.01,0.2,0.4,0.5,0.6,0.8,1.0]
        
        model_ids = ["meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"]
        model_folders = ["3.3.llama.70B.Instruct","3.1.llama.70B.Instruct"]
        need_quant = True
        
        #need_quant = False
        device_model = torch.device('cuda:1')
        
        #model_ids = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"]
        #model_folders = ["3.1.llama.8B.Instruct","3.2.llama.3B.Instruct"]
        
        TOTAL_determine_json = len(model_ids)*len([LUCCA,OLBIA])*len(temperatures)
        i_tdqm = 0
        
        with tqdm(total=TOTAL_determine_json, desc="Overall Progress") as main_pbar:
            
            # This is to the possiblity to not load the model and just test for errors
            for model_id, model_folder in zip(model_ids,model_folders):
                
                for municipality in [LUCCA, OLBIA]:
                    
                    if need_quant:
                        # Define the quantization configuration for 4-bit precision
                        # Create a dictionary to limit memory per GPU (adjust based on your GPU capacity)
                        # Configure maximum GPU memory per device (here, 23GB per GPU)
                        max_memory = {0: "23GB", 1: "23GB"}

                        # Setup the 4-bit quantization configuration. Using nf4 and double quantization are common choices.
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",  # nf4 is often recommended for good accuracy/speed trade-off
                            bnb_4bit_use_double_quant=True
                        )

                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=torch.bfloat16,
                            #device_map= torch.device('cuda:0'),
                            
                            quantization_config=quantization_config,
                            device_map='auto',
                            max_memory=max_memory,   # Ensure each GPU uses at most 23GB of VRAM
                            #max_memory=max_memory,
                            #use_flash_attention_2=True
                        )
                        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
                    
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id,
                            torch_dtype=torch.bfloat16,
                            device_map= device_model,
                            
                            #quantization_config=quantization_config,
                            #device_map='auto',
                            #max_memory=max_memory,   # Ensure each GPU uses at most 23GB of VRAM
                            #max_memory=max_memory,
                            #use_flash_attention_2=True
                        )
                        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
                        


                    if municipality== LUCCA:
                        #load the json - Dictionary
                        with open("./src/txt/Lucca/checklists/checklists.json","r", encoding="utf-8") as f:
                            checklists = json.load(f)

                        # load the csv with all the determine da controllare
                        with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
                            df_determine = pd.read_csv(f)

                        
                        
                        compiler = ChecklistCompiler(llm=LLAMA, municipality=LUCCA, model=model_id)
                    
                    elif municipality == OLBIA:
                        
                        #load the json - Dictionary
                        with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
                            checklists = json.load(f)

                        # load the csv with all the determine da controllare
                        with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
                            df_determine = pd.read_csv(f)

                        
                        
                        compiler = ChecklistCompiler(llm=LLAMA, municipality=OLBIA, model=model_id)
                    
                    for temp in temperatures:
                        for i, _ in df_determine.iterrows():
                            
                            sub_cartella = f"{model_folder}/{temp}/"
                            
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
                            # Optional: Update description
                            main_pbar.set_description(f"{temp} - {municipality}") 
                            
                            compiler.checklist_determina(num,
                                                che_ass,
                                                checklists,
                                                sub_cartella=sub_cartella,
                                                temperature=temp,
                                                )
                            tqdm.write(f"Done determina [{i}] {num} - {che_ass} - temp:{temp}")
                            # Update the main progress bar manually by 1 step
                        
                        
                        subfolder = f"{model_folder}/{temp}/"
                        
                        compiler.choose_checklist(determine=df_determine,
                                            checklists=checklists,
                                            sub_cartella=subfolder,
                                            temperature=temp)
                        tqdm.write(f"done choose checklists temp:{temp} - {municipality} - {model_id}")
                        main_pbar.update(1)

    # Ensure cleanup happens even if other errors occur within the task
    # Note: This finally block might not be strictly necessary if the objects
    # go out of scope anyway, but explicit deletion can sometimes help guide the GC.
    finally:
        # Attempt to explicitly delete large objects and trigger garbage collection
        # This helps free CPU memory, which *might* release GPU references
        print("Cleaning up task-specific variables (model, inputs etc.)...")
        # Deleting the model might be important if loading it is part of the task
        del compiler
        del text_gen_pipeline
        del model
        del tokenizer
        gc.collect() # Suggest garbage collection for CPU memory

        ## Explicitly clear PyTorch CUDA cache AFTER deleting variables
        #if 'torch' in sys.modules and torch.cuda.is_available():
        print("Attempting to clear PyTorch CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")


if __name__ == "__main__":
    #time.sleep(19_340)
    
    retry_delay_seconds = 3600 # 1 hour
    
    while True:
        try:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Starting attempt...")
            main()
            # If run_main_task() completes without raising an exception, we're done.
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Task completed successfully. Exiting.")
            sys.exit(0) # Exit successfully
        
        except:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: Task failed due to resource issue")
            print(f"Retrying in {retry_delay_seconds / 60} minutes...")
            # Optional: Log the full traceback for debugging
            # traceback.print_exc()
            time.sleep(retry_delay_seconds)