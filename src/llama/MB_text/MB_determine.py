import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import pandas as pd



def generate_prompt(checklist, determina):
    """
    Crea un prompt strutturato per verificare la corrispondenza tra i punti di una checklist normativa e una determina.
    
    Args:
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
    3. Per ogni punto della checklist, verifica se la norma è citata nella determina.
    4. Rispondi per ogni punto utilizzando uno dei seguenti criteri:
       - **SI**: La norma è citata nella determina (specificare dove, se possibile).
       - **Carente**: La norma non è citata ma la mancanza non è grave.
       - **NO**: La norma non è citata e la mancanza è grave.
       - **Ambiguo**: Il punto della checklist è troppo vago o indefinito per fornire una risposta precisa. Aggiungi una spiegazione sintetica.
    5. Alla fine, aggiungi eventuali "Note finali" se ci sono problemi generali o ambiguità rilevate nella determina.

    Utilizza un linguaggio semplice e accessibile. Rispondi in maniera chiara e ordinata.
    <</SYS>>

    ##### CHECKLIST
    {checklist}

    #### DETERMINA:
    {determina}

    ##### OUTPUT:
    Punto 1: [SI/Carente/NO], [spiegazione sintetica se necessaria]
    Punto 2: [SI/Carente/NO], [spiegazione sintetica se necessaria]
    ...

    Note finali: [eventuali osservazioni generali]
    [/INST]</s>
    """

def generate_propt_choose(determina):
    """
    Genera un prompt per suggerire la checklist più adatta per una determina basata sul contenuto della stessa.
    
    Args:
        determina (str): Testo della determina dirigenziale.

    Returns:
        str: Prompt strutturato per il suggerimento della checklist.
    """
    
    return f"""<s>[INST] <<SYS>>
    Sei un assistente virtuale esperto in diritto amministrativo. Il tuo compito è leggere il testo di una determina dirigenziale e consigliare la checklist più adatta per verificare la regolarità amministrativa dell'atto.

    ### Come procedere:
    1. Analizza brevemente la determina.
    2. Confronta il contenuto della determina con le seguenti checklist e i loro ambiti di utilizzo:
       - **DET_IS_CONTR**: DETERMINAZIONI d'IMPEGNO DI SPESA/CONTRATTI.
       - **AUTORINC**: AUTORIZZAZIONI AD INCARICHI EXTRA ISTITUZIONALI.
       - **DET_CONTR**: DETERMINAZIONE/CONCESSIONE DI SOVVENZIONI, CONTRIBUTI, SUSSIDI, ATTRIBUZIONE VANTAGGI ECONOMICI A PERSONE FISICHE ED ENTI PUBBLICI E PRIVATI.
       - **CONCESS**: PROVVEDIMENTO DI CONCESSIONE.
       - **LIQUID**: ATTO DI LIQUIDAZIONE.
       - **INCAREX**: AUTORIZZAZIONI AD INCARICHI EXTRA ISTITUZIONALI.
       - **INCAR**: DETERMINAZIONI D'IMPEGNO DI SPESA INCARICHI/LAVORO AUTONOMO EX Art. 7, comma 6 del D.Lgs. 165/2001.
       - **ACCERT**: ATTI DI ACCERTAMENTO ENTRATE.

    3. Rispondi con **il nome della checklist più adatta** (ad esempio: `DET_IS_CONTR`) e, se necessario, aggiungi una spiegazione sintetica.

    Assicurati di usare un linguaggio chiaro e semplice.

    ### DETERMINA:
    {determina}

    ### OUTPUT:
    Checklist suggerita: [Nome Checklist]
    Motivazione: [Breve spiegazione, se necessaria]

    [/INST]</s>"""


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
    
    if not bool(checklists):
        raise Exception("CHECKLIST not found")
    
    return checklist



def choose_checklist(nome_determina,
                     text_gen_pipeline):
    
    with open(f"./src/llama/MB_text/determinazioni/DET_{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    ### CHECKLIST SELEZIONATE PER LA DETERMINA
    
    
    complete_prompt = generate_propt_choose(determina)

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


def checklist_determina(nome_determina,
                        nome_checklist,  
                        text_gen_pipeline,
                        checklists):

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
    
    with open(f"./src/llama/MB_text/determinazioni/DET_{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    checklist = get_checklist(checklists,nome_checklist)
    

    with open(f"./src/llama/MB_text/responses/{nome_determina}-Complete.txt","w", encoding="utf-8") as f_complete:
        with open(f"./src/llama/MB_text/responses/{nome_determina}-response.txt","w", encoding="utf-8") as f_response:
            
            for punto in checklist["Punti"]:
                
                question = punto["Istruzioni"]+"\n"+punto["Punti"]
                
                complete_prompt = generate_prompt(question,determina)

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
                



if __name__ == "__main__":
    
    ## Multilingual model + 16K of context
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    #model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
    #model_id = "meta-llama/Llama-3.1-70B-Instruct"

    # This is to the possiblity to not load the model and just test for errors
    if True:

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map= torch.device('cuda:0'),
            
            #device_map='balanced',
            #use_flash_attention_2=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        #tokenizer.add_special_tokens({"pad_token":"<unk>"})

        # Create the pipeline
        text_gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            #max_new_tokens=1000,
    )
    
    #load the json - Dictionary
    with open("./src/llama/MB_text/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/llama/MB_text/MB_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)


    # Choose the right checklist for each determina
    # i do a loop to make it clear
    for index, row in df_determine.iterrows():
        df_determine.at[index, 'gen'] = choose_checklist(row['Numero Determina'],
                                                         text_gen_pipeline)
            
    df_determine.to_csv("./src/llama/MB_text/MB_Determine_gen.csv")
    
    for i in range(1):
        num = df_determine["Numero Determina"].loc[i]
        che_ass = df_determine["Checklist associata"].loc[i]
        #ogg_det = df_determine["Oggetto determina"].loc[i]
        
        checklist_determina(num,che_ass, text_gen_pipeline, checklists)

