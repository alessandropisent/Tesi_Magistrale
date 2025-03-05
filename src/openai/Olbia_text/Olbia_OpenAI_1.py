from dotenv import load_dotenv
import os
import openai
import json
import pandas as pd
import re
from tqdm import tqdm

# Load the environment variables from .env file
load_dotenv()


# Define a function to interact with the model
def generate_text(prompts, temperature, top_p=None, model="gpt-4o-mini"):
    client = openai.OpenAI()
    
    try:
        if top_p is None:
            response = client.chat.completions.create(
                model=model,
                messages=prompts,
                seed=42,
                temperature=temperature,
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
        ### Rispondi al punto {punto["num"]}, SEZIONE: {punto["Sezione"]}:
        {punto["Punto"]}
        
        ### Output:
        RISPOSTA : [SI, NO] (aggiungi note solo se strettamente necessario)
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
    "text":f"""Sei un assistente esperto in diritto amministrativo. Il tuo compito è supportare un impiegato comunale nel controllo della regolarità amministrativa di una determina dirigenziale.

Segui questi passaggi:
1. Leggi la checklist fornita, che include punti numerati con relative istruzioni e riferimenti normativi.
2. Leggi il testo della determina.
3. Per ciascun punto, verifica esclusivamente se la norma è presente o citata nella determina.
4. Rispondi con un semplice "SI" o "NO". Aggiungi note finali solo se emergono ambiguità o problemi rilevanti.
5. Non approfondire ulteriori valutazioni: concentrati solo sulla verifica della presenza della norma.

<s>
Risposta generata dal modello di assistenza amministrativa.
</s>
################# ------------- DETERMINA -------------------
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
                        model_folder="mini",
                        temperature=0.001,
                        top_p_value=None):

    """
    Genera un'analisi dettagliata della corrispondenza tra una checklist normativa e una determina dirigenziale.
    
    Args:
        nome_determina (str): Nome identificativo della determina da analizzare.
        nome_checklist (str): Nome della checklist normativa da applicare alla determina.
        checklists (dict): Dizionario contenente tutte le checklist disponibili e i loro punti normativi.
    
    Output:
        Scrive il risultato dell'analisi in file di output, inclusi dettagli sulla corrispondenza normativa.
    """
    
    with open(f"./src/txt/Olbia/determinazioni/{nome_determina}.txt","r", encoding="utf-8") as f:
        determina= f.read()
    
    checklist = get_checklist(checklists,nome_checklist)
    
    if not os.path.exists(f"./src/openai/Olbia_text/responses/{model_folder}/"):
        os.makedirs(f"./src/openai/Olbia_text/responses/{model_folder}/")    

    with open(f"./src/openai/Olbia_text/responses/{model_folder}/{nome_determina}-Complete.json","w", encoding="utf-8") as f_complete:
        with open(f"./src/openai/Olbia_text/responses/{model_folder}/{nome_determina}-response.txt","w", encoding="utf-8") as f_response:
            j_to_append = []
            
            for i,punto in tqdm(enumerate(checklist["Punti"]),total=len(checklist["Punti"])):
                complete_prompt = generate_prompt(determina) + [generate_user_prompts(punto)]
                
                ret = generate_text(complete_prompt, temperature, top_p_value, model)

                #print(ret)

                #print(ret[0]["generated_text"])

                ### COMPLETE OUPUT
                j_to_append.append({"index":i,
                           "prompts":complete_prompt,
                           "response":ret})
                #print(f"Generated : {i}")
                
                ### Only responses
                f_response.write("#######--------- User prompt\n\n")
                f_response.write(f"{generate_user_prompts(punto,True)}")
                f_response.write("\n\n#####---- RESPONSE:\n\n\n")
                f_response.write(f"{ret}\n\n")
                
            json.dump({"conversations":j_to_append}, f_complete, indent = 6)
                
                



if __name__ == "__main__":
    
    #done = [0,1,3,4,5,6,7,8,9,10]
    #todo = 2
    done = []
    model = "gpt-4o-mini"
    model_folder = "mini_1"
    
    #model = "gpt-4o"
    #model_folder = "full"

    #load the json - Dictionary
    with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)
    
    #temp_values = [round(x * 0.2, 1) for x in range(6)]
    temp_values = [1.0]
    
    
    if True:
        for temperature in temp_values:
            for i, _ in df_determine.iterrows():
                if i not in done:
                    print(f"DOING determina {i}")
                    num = df_determine["Numero Determina"].loc[i]
                    che_ass = df_determine["Checklist associata"].loc[i]
                    model_folder_t = model_folder + f"/temp_{temperature}"
                    checklist_determina(num, 
                                        che_ass, 
                                        checklists,
                                        model,
                                        model_folder_t, 
                                        temperature=temperature)
                    
                    print(f"Done determina {num} - {che_ass}")

