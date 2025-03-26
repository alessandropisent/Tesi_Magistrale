import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
import pandas as pd
import re
import os
from tqdm import tqdm
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def clean_text(text):
  """Cleans text by removing unrecognized special characters.

  Args:
    text: The input text string.

  Returns:
    The cleaned text string.
  """

  # Regular expression to match non-alphanumeric characters and common punctuation
  pattern = r"[^\w\s]"

  # Replace matched characters with an empty string
  cleaned_text = re.sub(pattern, '', text)

  return cleaned_text


def generate_prompt(istruzioni, punti, num, determina,sezione):
    """
    Crea un prompt strutturato per verificare la corrispondenza tra i punti di una checklist normativa e una determina.
    
    Args:
        checklist (str): Contenuto della checklist normativa da verificare.
        checklist (str): Contenuto della checklist normativa da verificare.
        determina (str): Testo della determina dirigenziale.

    Returns:
        str: Prompt strutturato per la verifica normativa.
    """    
    
    return f"""<s>[INST] <<SYS>>
    Sei un assistente esperto in materia di diritto amministrativo. Il tuo compito è supportare un impiegato comunale nel controllo della regolarità amministrativa di una determina dirigenziale.

    Segui i passaggi seguenti:
    1. Leggi la checklist fornita, che contiene punti numerati e specifiche normative da verificare.
    2. Leggi il testo della determina.
    3. Verifica se la norma è citata nella determina.
    4. Rispondi per ogni punto utilizzando uno dei seguenti criteri:
       - **SI**: Il punto della checklist e relative istruzioni sono rispettate (la determina passa il controllo)
       - **NO**: La determina NON passa il controllo, il punto della checklist NON è rispettato
    6. Alla fine, aggiungi eventuali "Note finali" se ci sono problemi generali o ambiguità rilevate nella determina.

    Utilizza un linguaggio semplice e accessibile. Rispondi in maniera chiara e ordinata.
    {istruzioni}
    <</SYS>>

    ##### CHECKLIST 
    {punti}

    ###### DETERMINA:
    {determina}

    ##### OUTPUT: [{sezione}], PUNTO {num}
    **RISPOSTA GENERALE**: [SI/NO], [spiegazione sintetica se necessaria]
    Note finali: [eventuali osservazioni generali]
    [/INST]</s>
    
    <s> 
    Risposta generata dal modello di assistenza amministrativa.
    </s>
    """


def get_checklist(checklists,nome_checklist):
    """
    Recupera una checklist specifica dal dizionario delle checklist disponibili.
    
    Args:
        checklists (dict): Dizionario contenente tutte le checklist disponibili.
        nome_checklist (str): Nome della checklist da recuperare.

    Returns:
        dict: La checklist specificata, se trovata.
    
    Raises:
        Exception: Se la checklist specificata non viene trovata.
    """
    
    checklist = {}
    
    for temp in checklists["checklists"]:
        #print(f"Found-search: {temp["NomeChecklist"]} - {nome_checklist}")
        if temp["NomeChecklist"] == nome_checklist:
            checklist = temp
    
    if not bool(checklist):
        raise Exception("CHECKLIST not found")
    
    return checklist



def checklist_determina(nome_determina,
                        nome_checklist,  
                        text_gen_pipeline,
                        checklists,
                        sub_cartella=""):

    """
    Genera un'analisi dettagliata della corrispondenza tra una checklist normativa e una determina dirigenziale.
    
    Args:
        nome_determina (str): Nome identificativo della determina da analizzare.
        nome_checklist (str): Nome della checklist normativa da applicare alla determina.
        text_gen_pipeline (Pipeline): Pipeline per la generazione del testo tramite un modello di linguaggio.
        checklists (dict): Dizionario contenente tutte le checklist disponibili e i loro punti normativi.
    
    Output:
        Scrive il risultato dell'analisi in file di output, inclusi dettagli sulla corrispondenza normativa.
    """
    
    with open(f"./src/txt/Olbia/determinazioni/{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    checklist = get_checklist(checklists,nome_checklist)
    
    responses_dic = {"array":[]}

    with open(f"./src/llama/Olbia_text/responses/{sub_cartella}{nome_determina}-Complete.txt","w", encoding="utf-8") as f_complete:
        with open(f"./src/llama/Olbia_text/responses/{sub_cartella}{nome_determina}-response.txt","w", encoding="utf-8") as f_response:
            
            for punto in tqdm(checklist["Punti"]):
                
                
                
                complete_prompt = generate_prompt(punto["Istruzioni"],
                                                  punto["Punto"],
                                                  punto["num"], 
                                                  determina,
                                                  punto["Sezione"])
                
                ret = text_gen_pipeline(
                    complete_prompt,
                    max_new_tokens=500,
                    truncation = True,
                    return_full_text=False,
                    #top_p = 0.9,
                    #temperature = 0.8,
                    do_sample=False,  # Disable sampling for deterministic output
                )

                #print(ret)

                #print(ret[0]["generated_text"])

                ### COMPLETE OUPUT
                f_complete.write("\n\n########## ------------ PROMPT --------------\n\n")
                f_complete.write(complete_prompt)
                f_complete.write("\n\n########## -------------------------- GENERATED ---------------------------\n\n")
                f_complete.write(ret[0]["generated_text"])
                
                ### JUST OUTPUT
                
                just_response = f""" 
                #### CHECKLIST
                {punto["Istruzioni"]+"\n\n"+punto["Punto"]}
                
                ### RESPONSE
                {ret[0]["generated_text"]}
                """
                responses_dic["array"].append({
                    "punto":punto["Punto"],
                    "response":ret[0]["generated_text"]
                })
                
                f_response.write(just_response)
                #print(f"Done:{i}")
    with open(f"./src/llama/Olbia_text/responses/{sub_cartella}{nome_determina}-response.json","w", encoding="utf-8") as json_response:
        json.dump(responses_dic,json_response,indent = 6)
        
                
                



if __name__ == "__main__":
    
    ## Multilingual model + 16K of context
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    #model_id = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    #model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
    #model_id = "meta-llama/Llama-3.1-70B-Instruct"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

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


        # Create the pipeline
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,  # open-end generation
            truncation=True,  # Truncates inputs exceeding model max length
            eos_token_id = tokenizer.eos_token_id,
            
        )
    
    #load the json - Dictionary
    with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)



    ## Choose the right checklist for each determina
    ## i do a loop to make it clear
    if False:
        for index, row in df_determine.iterrows():
            print(f"I'm genenerting the checklist for {index}, det {row['Numero Determina']} - {model_id.split("/", 1)[1]}")
            df_determine.at[index, 'gen'] = choose_checklist(row['Numero Determina'],
                                                            text_gen_pipeline,
                                                            model_id.split("/", 1)[1])
        
        df_determine.to_csv("./src/llama/Olbia_text/Olbia_Determine_gen.csv")
    
    done = [1,2]       
    
    if True:
        for i, _ in df_determine.iterrows():
            if i not in done:
                num = df_determine["Numero Determina"].loc[i]
                che_ass = df_determine["Checklist associata"].loc[i]
                sub_cartella = "General/"
                
                
                checklist_determina(num,che_ass, text_gen_pipeline, checklists,sub_cartella)
                print(f"Done determina {num} - {che_ass}")
            

## Numero Determina,Checklist associata