import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd

LOAD = False

#print(determina)

## det_is_contr
#question = """Citare con precisione gli articoli, commi e lettere delle fonti di regolazione.Citare prima le norme di legittimazione e poi le norme procedurali del caso in specie
#Applicazione delle normative pertinenti alla fattispecie, tenuto conto delle sopravvenute disposizioni vigenti: 
#1. D.lgs 267/2000 ( art. 107 c. 2, lettera... art 109...)  183 e 191)																
#2. D.Lgs 163/2006 s.m.i.  e/o contratto servizio escluso																
#3. R.D.  n. 2440/1924  per affidamenti cui non si applica il Codice dei Contratti																
#4. DPR  207/2010, art.. (indicare istituto specifico)																
#5. Normativa sulla privacy nel testo dei provvedimenti e nelle procedure adottate (art. 20 e 21 D.Lgs. 196/2003)																
#6. Eventuali altre normative generali e/o di specificazione"""	

#print(question)

#print(df_determine)
#print(df_determine["Numero Determina"].loc[0])
#print(df_determine["Checklist associata"].loc[0])
#print(df_determine["Oggetto determina"].loc[0])


def checklist_determina(nome_determina,
                        nome_checklist, 
                        ogg_determina, 
                        text_gen_pipeline,
                        checklists):
    
    prompt = """<s>[INST] <<SYS>>
    Sei un avvocato, leggi la seguente checklist con le eventuali note, poi leggi la determina dirigenziale e infine dimmi se i punti della checklist sono rispettati. 
    Rispondi punto per punto della checklist argomentando SE la norma è stata citata nella determina.
    [/INST]
    """
    
    one_shot = """

    ##### Esempio di output desiderato:
    3. D.lgs 267/2000 :-> Rispettato, la norma è citata al interno della determina
    7. articolo 26 bis :-> Non Rispettato, la norma non è citata e non è rilevante

    """
    
    with open(f"./src/llama/MB_text/determinazioni/DET_{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    checklist = {}
    
    for temp in checklists["checklists"]:
        #print(f"Found-search: {temp["NomeChecklist"]} - {nome_checklist}")
        if temp["NomeChecklist"] == nome_checklist:
            checklist = temp
    
    if not bool(checklists):
        print("CHECKLIST NOT FOUND")
        return

    with open(f"./src/llama/MB_text/responses/{nome_determina}-Complete.txt","w", encoding="utf-8") as f_complete:
        with open(f"./src/llama/MB_text/responses/{nome_determina}-response.txt","w", encoding="utf-8") as f_response:
            
            for punto in checklist["Punti"]:
                
                question = punto["Istruzioni"]+"\n"+punto["Punti"]
                
                complete_prompt = prompt+"##### CHECKLIST\n\n"+question+"\n\n#### DETERMINA:\n"+determina+"<</SYS>></s>""\n##### OUTPUT:"

                ret = text_gen_pipeline(
                    complete_prompt,
                    max_length=10_000,    # Limit the length of generated text
                    max_new_tokens=500,
                    truncation = True,
                    temperature=0.01,   # Set the temperature
                    num_beams=2,
                    return_full_text=False,
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
                {question}
                
                ### RESPONSE
                {ret[0]["generated_text"]}
                """
                
                f_response.write(just_response)
                
                


    ### CHECKLIST SELEZIONATE PER LA DETERMINA

    with open("./src/llama/MB_text/possibili_checklists.txt","r",encoding="utf-8") as f:
        possib_checklist = f.read()

    #print(possib_checklist)

    prompt = """<s>[INST] <<SYS>>
    Sei un avvocato, Seleziona la checklist rilevante per questa determina.
    [/INST]
    """
    
    
    
    complete_prompt = prompt+"##### CHECKLISTs POSSIBILI:\n\n" + possib_checklist+"\n\n#### OGGETTO DETERMINA:\n\n"+ogg_determina+"<</SYS>></s>\n\n### OUTPUT:\n"

    ret = text_gen_pipeline(
    complete_prompt,
    max_length=1_000,    # Limit the length of generated text
    #max_new_tokens=500,
    #truncation = True,
    temperature=0.01,   # Set the temperature
    num_return_sequences=3,  # Number of completions to generate
    return_full_text=False,
    )



    with open(f"./src/llama/MB_text/responses/{nome_determina}-pc.txt","w", encoding="utf-8") as f:
        f.write("########## ------------ PROMPT --------------\n\n")
        f.write(complete_prompt)
        f.write("\n\n########## -------------------------- GENERATED ---------------------------\n\n")
        f.write(ret[0]["generated_text"])



if __name__ == "__main__":
    
    ## Multilingual model + 16K of context
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    #model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
    #model_id = "meta-llama/Llama-3.1-70B-Instruct"

    if True:

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map= torch.device('cuda:0'),
            
            #device_map='balanced',
            #use_flash_attention_2=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.add_special_tokens({"pad_token":"<unk>"})

        # Create the pipeline
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            #max_new_tokens=1000,
    )
    
    with open("./src/llama/MB_text/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    #print(checklists)

    with open("./src/llama/MB_text/MB_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)

    for i in range(1):
        num = df_determine["Numero Determina"].loc[i]
        che_ass = df_determine["Checklist associata"].loc[i]
        ogg_det = df_determine["Oggetto determina"].loc[i]
        
        checklist_determina(num,che_ass,ogg_det, text_gen_pipeline, checklists)

