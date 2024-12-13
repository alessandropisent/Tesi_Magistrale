with open("./src/llama/MB_text/determinazioni/DET_MB_2708-2024.txt","r", encoding="utf-8") as f:
    determina= f.read()

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

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

## Multilingual model + 16K of context
model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "swap-uniba/LLaMAntino-2-70b-hf-UltraChat-ITA"
#model_id = "meta-llama/Llama-3.1-70B-Instruct"


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

print(ret)

#print(ret[0]["generated_text"])

with open("./src/llama/MB_text/responses/DET_MB_2708-2024.txt","w", encoding="utf-8") as f:
    f.write("########## ------------ PROMPT --------------\n\n")
    f.write(complete_prompt)
    f.write("\n\n########## -------------------------- GENERATED ---------------------------\n\n")
    f.write(ret[0]["generated_text"])


with open("./src/llama/MB_text/possibili_checklists.txt","r",encoding="utf-8") as f:
    possib_checklist = f.read()

print(possib_checklist)

prompt = """<s>[INST] <<SYS>>
Sei un avvocato, Seleziona la checklist rilevante per questa determina.
[/INST]
"""

ogg_determina = """Oggetto: PROCEDURA APERTA TELEMATICA, AI SENSI DELL'ART. 71 E 107 
COMMA 3 DEL D.LGS 36/2023, PER L'AFFIDAMENTO DEI LAVORI DI 
MANUTENZIONE STRAORDINARIA SS.PP. - RIQUALIFICAZIONE TRATTI 
STRADALI ANNO 2022/23 MIMS 141/2022, PER LA PROVINCIA DI MONZA 
E DELLA BRIANZA, TRAMITE PIATTAFORMA SINTEL DI ARIA S.P.A., CON 
IL CRITERIO DEL PREZZO PIU' BASSO E CON INVERSIONE 
PROCEDIMENTALE, AI SENSI DEGLI ARTT. 71, 107 COMMA 3 E 108 DEL 
D.LGS. 36/2023. CIG: B4474B16DA; CUP: B37H22005330001; CUI: 
L94616010156202300015. DECISIONE DI CONTRARRE. 
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



with open("./src/llama/MB_text/responses/possibili_checklists_DET_MB_2708-2024.txt","w", encoding="utf-8") as f:
    f.write("########## ------------ PROMPT --------------\n\n")
    f.write(complete_prompt)
    f.write("\n\n########## -------------------------- GENERATED ---------------------------\n\n")
    f.write(ret[0]["generated_text"])