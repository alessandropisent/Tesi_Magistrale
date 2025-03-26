import json
import re
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# --- Funzioni di utilità ---

def clean_text(text):
    """Pulisce il testo eliminando caratteri speciali non riconosciuti."""
    pattern = r"[^\w\s]"
    return re.sub(pattern, '', text)

def generate_prompt(istruzioni, punti, num, determina, sezione):
    """
    Crea un prompt strutturato per verificare la corrispondenza tra i punti di una checklist normativa e una determina dirigenziale.
    """
    return f"""<s>[INST] <<SYS>>
Sei un assistente esperto in diritto amministrativo. Il tuo compito è supportare un impiegato comunale nel controllo della regolarità amministrativa di una determina dirigenziale.

Segui questi passaggi:
1. Leggi la checklist, che contiene punti numerati e specifiche normative da verificare.
2. Leggi il testo della determina.
3. Verifica se la norma è citata nella determina.
4. Per ogni punto, rispondi con:
   - **SI**: se il punto è rispettato
   - **NO**: se il punto NON è rispettato
6. Alla fine, aggiungi eventuali "Note finali" in caso di problematiche o ambiguità.
{istruzioni}
<</SYS>>

##### CHECKLIST 
{punti}

###### DETERMINA:
{determina}

##### OUTPUT: [{sezione}], PUNTO {num}
**RISPOSTA GENERALE**: [SI/NO], [spiegazione sintetica]
Note finali: [eventuali osservazioni]
[/INST]</s>

<s>
Risposta generata dal modello.
</s>
"""

def get_checklist(checklists, nome_checklist):
    """
    Recupera una checklist specifica dal dizionario delle checklist.
    """
    for temp in checklists["checklists"]:
        if temp["NomeChecklist"] == nome_checklist:
            return temp
    raise Exception("CHECKLIST not found")

# --- Costruzione del dataset per determina ---

def build_determinazione_dataset(df_determine, checklists, determinations_folder):
    """
    Per ogni determina (presentata nel CSV) crea un dataset contenente tutti i prompt da processare.
    Ogni dataset corrisponde a una determina.
    Restituisce un dizionario con chiave: numero della determina, e valore: Dataset associato.
    """
    datasets_dict = {}
    for index, row in df_determine.iterrows():
        num = row["Numero Determina"]
        che_ass = row["Checklist associata"]
        # Legge il testo della determina
        with open(f"{determinations_folder}/{num}.txt", "r", encoding="utf-8") as f:
            determina = f.read()
        # Recupera la checklist associata
        checklist = get_checklist(checklists, che_ass)
        data = []
        # Per ogni punto della checklist crea il prompt
        for punto in checklist["Punti"]:
            prompt = generate_prompt(
                punto["Istruzioni"],
                punto["Punto"],
                punto["num"],
                determina,
                punto["Sezione"]
            )
            data.append({
                "numero_determina": num,
                "checklist_associata": che_ass,
                "punto":punto["Punto"],
                "prompt": prompt
            })
        # Crea un dataset per questa determina
        datasets_dict[num] = Dataset.from_list(data)
    return datasets_dict

def generate_response(example, text_gen_pipeline):
    """
    Genera la risposta per un prompt usando la pipeline di generazione.
    """
    ret = text_gen_pipeline(
        example["prompt"],
        max_new_tokens=500,
        truncation=True,
        return_full_text=False,
        do_sample=False  # Output deterministico
    )
    example["response"] = ret
    return example

# --- Script principale ---

if __name__ == "__main__":
    # Carica modello e tokenizer
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
    
    # Crea la pipeline per la generazione del testo, impostando esplicitamente il pad_token_id
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1000,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Carica le checklist e il CSV delle determine
    with open("./src/txt/Olbia/checklists/checklists.json", "r", encoding="utf-8") as f:
        checklists = json.load(f)
    df_determine = pd.read_csv("./src/txt/Olbia/checklists/Olbia_Determine.csv", encoding="utf-8")
    
    # Cartella dove sono salvate le determine (file .txt)
    determinations_folder = "./src/txt/Olbia/determinazioni"
    
    # Crea un dizionario di dataset: una determina = un dataset
    datasets_dict = build_determinazione_dataset(df_determine, checklists, determinations_folder)
    
    # Processa ogni dataset (determine) separatamente e salva i risultati in file JSON
    for num, ds in datasets_dict.items():
        ret = generate_response(ds,text_gen_pipeline)
        
        ret[["numero_determina","checklist_associata","punto","response"]].to_json(
            f"./src/llama/Olbia_text/responses/General/{num}-response.json", orient="records", lines=True)
        print(f"Determinazione {num} processata e salvata.")
