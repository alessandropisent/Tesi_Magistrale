import pandas as pd
import json
import re


def analize_response(text):
    # Converti il testo in stringa (per sicurezza)
    text = str(text)
    
    # 1. Prova a cercare il pattern "RISPOSTA GENERALE:" o "RISPOSTA:" seguito da una risposta
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
    
    # 2. Se non viene trovato il marker, controlla se il testo inizia direttamente con una risposta
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
    
    # 3. Controllo nei primi 50 caratteri per verificare la presenza delle risposte in posizioni rilevanti
    prefix = text[:50].lower()
    if re.search(r"\bnon richiesto\b", prefix):
        return "NON RICHIESTO"
    if re.search(r"^\s*no\b", prefix) or re.search(r"\bno\b", prefix):
        return "NO"
    if re.search(r"^\s*(si|sì)\b", prefix) or re.search(r"\b(si|sì)\b", prefix):
        return "SI"
    
    # 4. Controllo in tutto il testo per casi strutturati (ad esempio, RISPOSTA GENERALE all'interno di una sezione)
    structured_pattern = re.search(
        r"(?i)\*{0,2}RISPOSTA GENERALE\*{0,2}\s*:\s*(si|sì|no|non richiesto)\b", text
    )
    if structured_pattern:
        ans = structured_pattern.group(1).lower()
        if ans in ['si', 'sì']:
            return "SI"
        elif ans == "no":
            return "NO"
        elif ans == "non richiesto":
            return "NON RICHIESTO"
    
    # 5. Controllo aggiuntivo: se RISPOSTA GENERALE è seguito da SI anche su una riga successiva
    if re.search(r"(?i)\bRISPOSTA GENERALE\b.*?(\bSI\b|\bSÌ\b)", text, re.DOTALL):
        return "SI"
    
    # Se nessuna delle regole precedenti risulta soddisfatta, ritorna "Not found"
    return "Not found"


def relate_checklist_determina(nome_determina,nome_checklist,sub_cartella):
    
    if nome_checklist == "Contratti":
        name_checklist_end = "cont"
    elif nome_checklist == "Determine":
        name_checklist_end = "det"
    
    with open(f"./src/llama/Lucca_text/responses/{sub_cartella}{nome_determina}-G_{name_checklist_end}.json","r", encoding="utf-8")as file:
        data = json.load(file)


    #df = pd.DataFrame(data["episodes"])
    df = pd.json_normalize(data, record_path=['Response'])
    print(df)
    df["Simple"] = df["LLM.generated_text"].apply(analize_response)
    name_out = f"./src/llama/Lucca_text/responses/{sub_cartella}{nome_determina}-G_{name_checklist_end}"
    df.to_excel(name_out+".xlsx")
    df.to_csv(name_out+".csv")
    






model_folder_t = "General/"

# load the csv with all the determine da controllare
with open("./src/txt/Lucca/checklists/Lucca_Determine.csv","r", encoding="utf-8") as f:
    df_determine = pd.read_csv(f)
    
for i, _ in df_determine.iterrows():
    
    num = df_determine["Numero Determina"].loc[i]
    che_ass = df_determine["Checklist associata"].loc[i]
                
    relate_checklist_determina(num,che_ass,model_folder_t)
        
    print(f"Done determina {num} - {che_ass}")
    