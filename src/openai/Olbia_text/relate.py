import json
import re
import pandas as pd


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

def analize_response(text):

    # Converti il testo in stringa (per sicurezza)
    text = str(text)
    
    # 1. Prova a cercare il pattern "RISPOSTA GENERALE:" seguito da una risposta classificabile
    #    Il pattern gestisce "si" (anche "sì"), "no" e "non richiesto" (case-insensitive)
    general_pattern = re.search(
        r"(?i)(\*?risposta generale\*?:|risposta:|\*?risposta\*?:)\s*(si|sì|no|non richiesto)\b", text
    )
    if general_pattern:
        ans = general_pattern.group(2).lower()
        if ans in ['si', 'sì']:
            return "SI"
        elif ans == "no":
            return "NO"
        elif ans == "non richiesto":
            return "NON RICHIESTO"
    
    # 2. Se non viene trovato il marker "RISPOSTA GENERALE:", controlla se il testo inizia
    #    direttamente con una risposta (con possibili spazi iniziali)
    start_pattern = re.match(
        r"^\s*(si|sì|no|non richiesto)\b", text, flags=re.IGNORECASE
    )
    if start_pattern:
        ans = start_pattern.group(1).lower()
        if ans in ['si', 'sì']:
            return "SI"
        elif ans == "no":
            return "NO"
        elif ans == "non richiesto":
            return "NON RICHIESTO"
    
    # 3. Se ancora non troviamo un pattern chiaro, controlla la parte iniziale del testo (ad esempio, i primi 50 caratteri)
    #    per verificare se contengono le risposte, ma solo se appaiono come parole isolate.
    prefix = text[:100].lower()
    if re.search(r"\bnon richiesto\b", prefix):
        return "NON RICHIESTO"
    if re.search(r"^\s*no\b", prefix) or re.search(r"\bno\b", prefix):
        return "NO"
    if re.search(r"^\s*(si|sì)\b", prefix) or re.search(r"\b(si|sì)\b", prefix):
        return "SI"
    
    # Se nessuna delle regole precedenti risulta soddisfatta, ritorna "Not found"
    return "Not found"

def relate_checklist_determina(nome_determina,
                        nome_checklist,
                        checklists,
                        model = "gpt-4o-mini",
                        model_folder="mini"):
    
    df = pd.DataFrame(columns=["Punto","Response"])
    
    checklist = get_checklist(checklists,nome_checklist)
    
    with open(f"./src/openai/Olbia_text/responses/{model_folder}/{nome_determina}-Complete.json", "r", encoding="utf-8") as j:
        responses_dic = json.load(j)
    
    for i,punto in enumerate(checklist["Punti"]): 
        
        new_row = {
            "Punto": punto["Punto"],
            "Response":responses_dic["conversations"][i]["response"],
            "Simple" : analize_response(responses_dic["conversations"][i]["response"])
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    df.to_csv(f"./src/openai/Olbia_text/responses/{model_folder}/{nome_determina}-response.tsv",sep="\t")
    df.to_excel(f"./src/openai/Olbia_text/responses/{model_folder}/{nome_determina}-response.xlsx", index=False)
    
    print(df)

if __name__ == "__main__":
    
    done = []
    #model = "gpt-4o-mini"
    #model_folder = "mini"
    
    model = "gpt-4o"
    model_folder = "full"

    #load the json - Dictionary
    with open("./src/txt/Olbia/checklists/checklists.json","r", encoding="utf-8") as f:
        checklists = json.load(f)

    # load the csv with all the determine da controllare
    with open("./src/txt/Olbia/checklists/Olbia_Determine.csv","r", encoding="utf-8") as f:
        df_determine = pd.read_csv(f)
    
    
    if True:
        for i, _ in df_determine.iterrows():
           
            
            num = df_determine["Numero Determina"].loc[i]
            che_ass = df_determine["Checklist associata"].loc[i]
            
            relate_checklist_determina(num,che_ass, checklists,model,model_folder)
                
            print(f"Done determina {num} - {che_ass}")
            

    
            
            
            
