from dotenv import load_dotenv
import os
import openai
import json
import pandas as pd
import re

# Load the environment variables from .env file
load_dotenv()


# Define a function to interact with the model
def generate_text(prompts, model="gpt-4o-mini"):
    client = openai.OpenAI()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompts,
        )
        #print(f"generated")
        # Extract and return the generated text
        #print(response.choices[0].message)
        return str(response.choices[0].message.content)
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


def generate_user_prompts(punto, return_just_text=False):

        
    if punto["Istruzioni"] != "":
        ist = "### Istruzioni\n\n"+ punto["Istruzioni"] + "\n\n"
    else:
        ist = ""
    
    text = f"""{ist}
        ### Rispondi al punto {punto["num"]}:
        {punto["Punto"]}
        
        ### Output:
        RISPOSTA GENERALE : [SI/NO], [spiegazione sintetica se necessaria]
        """

    if return_just_text:
        return text
    
    return {
        "role":"user",
        "content":[{"type": "text",
            "text": text
        }]
        }

def generate_prompt(determina):
    """
    Crea un prompt strutturato per verificare la corrispondenza tra i punti di una checklist normativa e una determina.
    
    Args:
        determina (str): Testo della determina dirigenziale.

    Returns:
        str: Prompt strutturato per la verifica normativa.
    """    
    

    return [{
        "role": "developer", 
        "content" : [{ "type": "text", 
    "text":f"""Sei un assistente esperto in materia di diritto amministrativo. Il tuo compito è supportare un impiegato comunale nel controllo della regolarità amministrativa di una determina dirigenziale.

    Segui i passaggi seguenti:
    1. Leggi la checklist fornita, che contiene punti numerati e specifiche normative da verificare.
    2. Leggi il testo della determina.
    3. Per ogni punto della checklist, verifica se la norma è citata nella determina.
    4. Rispondi per ogni punto utilizzando uno dei seguenti criteri:
       - **SI**: Il punto della checklist e relative istruzioni sono rispettate (la determina passa il controllo)
       - **NO**: La determina NON passa il controllo, il punto della checklist NON è rispettato
       - **Ambiguo**: Il punto della checklist è troppo vago o indefinito per fornire una risposta precisa. Aggiungi una spiegazione sintetica.
    6. Alla fine, aggiungi eventuali "Note finali" se ci sono problemi generali o ambiguità rilevate nella determina.

    
    <s> 
    Risposta generata dal modello di assistenza amministrativa.
    </s>
    ############# ------------- DETERMINA -------------------
    {determina}
    """}]}]



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
                        checklists,
                        model = "gpt-4o-mini",
                        model_folder="mini"):

    """
    Genera un'analisi dettagliata della corrispondenza tra una checklist normativa e una determina dirigenziale.
    
    Args:
        nome_determina (str): Nome identificativo della determina da analizzare.
        nome_checklist (str): Nome della checklist normativa da applicare alla determina.
        checklists (dict): Dizionario contenente tutte le checklist disponibili e i loro punti normativi.
    
    Output:
        Scrive il risultato dell'analisi in file di output, inclusi dettagli sulla corrispondenza normativa.
    """
    
    with open(f"./src/txt/Lucca/determinazioni/{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    checklist = get_checklist(checklists,nome_checklist)
    

    with open(f"./src/openai/Lucca_text/responses/{model_folder}/{nome_determina}-Complete.json","w", encoding="utf-8") as f_complete:
        with open(f"./src/openai/Lucca_text/responses/{model_folder}/{nome_determina}-response.txt","w", encoding="utf-8") as f_response:
            j_to_append = []
            for i,punto in enumerate(checklist["Punti"]):
                complete_prompt = generate_prompt(determina) + [generate_user_prompts(punto)]
                
                ret = generate_text(complete_prompt, model)

                #print(ret)

                #print(ret[0]["generated_text"])

                ### COMPLETE OUPUT
                j_to_append.append({"index":i,
                           "prompts":complete_prompt,
                           "response":ret})
                print(f"Generated : {i}")
                
                ### Only responses
                f_response.write("#######--------- User prompt\n\n")
                f_response.write(f"{generate_user_prompts(punto,True)}")
                f_response.write("\n\n#####---- RESPONSE:\n\n\n")
                f_response.write(f"{ret}\n\n")
                
            json.dump({"conversations":j_to_append}, f_complete, indent = 6)
                
                



if __name__ == "__main__":
    
    done = []
    #model = "gpt-4o-mini"
    #model_folder = "mini"
    
    model = "gpt-4o"
    model_folder = "full"

    #load the json - Dictionary
    with open("./src/txt/Lucca/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)
    
    
    if True:
        for i, _ in df_determine.iterrows():
            if i not in done:
                print(f"DOING determina {i}")
                num = df_determine["Numero Determina"].loc[i]
                che_ass = df_determine["Checklist associata"].loc[i]
                
                checklist_determina(num,che_ass, checklists,model,model_folder)
                
                print(f"Done determina {num} - {che_ass}")

