from dotenv import load_dotenv
import os
import openai
import json
import torch
import pandas as pd
import re

# Load the environment variables from .env file
load_dotenv()

# Now you can access the environment variable just like before
openai.api_key = os.environ.get('OPENAI_APIKEY')

print(openai.api_key)

# Define a function to interact with the model
def generate_text(prompt, info, model="text-davinci-003"):
    try:
        response = openai.Completion.create(
            engine=model,
            prompts=prompt,
        )
        print(f"generated {info}")
        # Extract and return the generated text
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {e}"



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


def generate_prompt(istruzioni, punti, num, sotto, determina):
    """
    Crea un prompt strutturato per verificare la corrispondenza tra i punti di una checklist normativa e una determina.
    
    Args:
        checklist (str): Contenuto della checklist normativa da verificare.
        checklist (str): Contenuto della checklist normativa da verificare.
        determina (str): Testo della determina dirigenziale.

    Returns:
        str: Prompt strutturato per la verifica normativa.
    """    
    
    return [{
        "role": "developer", 
        "content" : f"""Sei un assistente esperto in materia di diritto amministrativo. Il tuo compito è supportare un impiegato comunale nel controllo della regolarità amministrativa di una determina dirigenziale.

    Segui i passaggi seguenti:
    1. Leggi la checklist fornita, che contiene punti numerati e specifiche normative da verificare.
    2. Leggi il testo della determina.
    3. Per ogni punto della checklist, verifica se la norma è citata nella determina.
    4. Rispondi per ogni punto utilizzando uno dei seguenti criteri:
       - **SI**: La norma è citata nella determina (specificare dove, se possibile).
       - **Carente**: La norma non è citata ma la mancanza non è grave.
       - **NO**: La norma non è citata e la mancanza è grave.
       - **Ambiguo**: Il punto della checklist è troppo vago o indefinito per fornire una risposta precisa. Aggiungi una spiegazione sintetica.
    5. Aggiungi una risposta separata per ciascun sottopunto, per giustificare la tua conclusione sul punto principale.
    6. Alla fine, aggiungi eventuali "Note finali" se ci sono problemi generali o ambiguità rilevate nella determina.

    Utilizza un linguaggio semplice e accessibile. Rispondi in maniera chiara e ordinata.
    {istruzioni}

    ##### CHECKLIST 
    {punti}
    
    ##### OUTPUT:
    Punto {num}: [SI/Carente/NO/Ambiguo], [spiegazione sintetica se necessaria]
      {"\n\t\t".join([f"-{num}{lettera}. [SI/Carente/NO/Ambiguo], [spiegazione sintetica se necessaria]" for lettera in sotto])}

    
    QUINDI Dati i sottopunti elencati, il Punto {num}: [SI/Carente/NO/Ambiguo], [spiegazione sintetica se necessaria]
    Note finali: [eventuali osservazioni generali]
    
    <s> 
    Risposta generata dal modello di assistenza amministrativa.
    </s>
    """},
    {"role":"user",
     "content": f"Analizzami questa determina : #### {"\n\n"+determina}"}
    ]

def generate_prompt_choose(determina):
    """
    Genera un prompt per suggerire la checklist più adatta per una determina basata sul contenuto della stessa.
    
    Args:
        determina (str): Testo della determina dirigenziale.

    Returns:
        str: Prompt strutturato per il suggerimento della checklist.
    """
    determina = clean_text(determina)
    determina =  ' '.join(determina.split(" ")[:100])
    
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

    ##### OUTPUT:
    Checklist suggerita: [Nome Checklist]
    Motivazione: [Breve spiegazione, se necessaria]

    <</SYS>>

    ################ DETERMINA:
    {determina}

 
    [/INST]</s>
    ##### OUTPUT:
    
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
    
    if not bool(checklists):
        raise Exception("CHECKLIST not found")
    
    return checklist



def choose_checklist(nome_determina,
                     text_gen_pipeline, 
                     model_id):
    
    with open(f"./src/openai/openai/determinazioni/DET_{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    ### CHECKLIST SELEZIONATE PER LA DETERMINA
    complete_prompt = generate_prompt_choose(determina)

    ret = text_gen_pipeline(
        complete_prompt,
        max_length=1_000,    # Limit the length of generated text
        #max_new_tokens=500,
        #truncation = True,
        temperature=0.01,   # Set the temperature
        return_full_text=False,
    )



    with open(f"./src/openai/openai/responses/{model_id}/{nome_determina}-pc.txt","w", encoding="utf-8") as f:
        f.write("########## ------------ PROMPT --------------\n\n")
        f.write(complete_prompt)
        f.write("\n\n########## -------------------------- GENERATED ---------------------------\n\n")
        f.write(ret[0]["generated_text"])


def checklist_determina(nome_determina,
                        nome_checklist,
                        checklists):

    """
    Genera un'analisi dettagliata della corrispondenza tra una checklist normativa e una determina dirigenziale.
    
    Args:
        nome_determina (str): Nome identificativo della determina da analizzare.
        nome_checklist (str): Nome della checklist normativa da applicare alla determina.
        checklists (dict): Dizionario contenente tutte le checklist disponibili e i loro punti normativi.
    
    Output:
        Scrive il risultato dell'analisi in file di output, inclusi dettagli sulla corrispondenza normativa.
    """
    
    with open(f"./src/openai/MB_text/determinazioni/DET_{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    checklist = get_checklist(checklists,nome_checklist)
    

    with open(f"./src/openai/MB_text/responses/{nome_determina}-Complete.txt","w", encoding="utf-8") as f_complete:
        with open(f"./src/openai/MB_text/responses/{nome_determina}-response.txt","w", encoding="utf-8") as f_response:
            
            for punto in checklist["Punti"]:
                
                
                
                complete_prompt = generate_prompt(punto["Istruzioni"],
                                                  punto["Punti"],
                                                  punto["num"],
                                                  punto["sott"], 
                                                  determina)
                
                ret = generate_text(complete_prompt,punto["num"])

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
                {punto["Istruzioni"]+"\n\n"+punto["Punti"]}
                
                ### RESPONSE
                {ret[0]["generated_text"]}
                """
                
                f_response.write(just_response)
                



if __name__ == "__main__":
    
    ## Multilingual model + 16K of context
    #model_id = "meta-llama/Llama-3.1-8B-Instruct"
    #model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
    #model_id = "meta-llama/Llama-3.1-70B-Instruct"

   
    
    #load the json - Dictionary
    with open("./src/openai/MB_text/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/openai/MB_text/MB_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)



    ## Choose the right checklist for each determina
    ## i do a loop to make it clear
    if False:
        for index, row in df_determine.iterrows():
            print(f"I'm genenerting the checklist for {index}, det {row['Numero Determina']} - {model_id.split("/", 1)[1]}")
            df_determine.at[index, 'gen'] = choose_checklist(row['Numero Determina'],
                                                            text_gen_pipeline,
                                                            model_id.split("/", 1)[1])
        
        df_determine.to_csv("./src/openai/MB_text/MB_Determine_gen.csv")
            
    
    if True:
        for i, _ in df_determine.iterrows():
            num = df_determine["Numero Determina"].loc[i]
            che_ass = df_determine["Checklist associata"].loc[i]
            ogg_det = df_determine["Oggetto determina"].loc[i]
            
            if che_ass == "DET_IS_CONTR":
                checklist_determina(num,che_ass, checklists)
            print(f"Done determina {num} - {che_ass}")

