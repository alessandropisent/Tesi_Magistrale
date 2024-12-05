with open("./src/llama/MB_text/determinazioni/DET_MB_2708-2024.txt","r", encoding="utf-8") as f:
    determina= f.read()

#print(determina)

## det_is_contr
question = """Applicazione delle normative pertinenti alla fattispecie, tenuto conto delle sopravvenute disposizioni vigenti: 
[NOTE] : Citare con precisione gli articoli, commi e lettere delle fonti di regolazione.Citare prima le norme di legittimazione e poi le norme procedurali del caso in specie
1. D.lgs 267/2000 ( art. 107 c. 2, lettera... art 109...)  183 e 191)																
2. D.Lgs 163/2006 s.m.i.  e/o contratto servizio escluso																
3. R.D.  n. 2440/1924  per affidamenti cui non si applica il Codice dei Contratti																
4. DPR  207/2010, art….. (indicare istituto specifico)																
5. Normativa sulla privacy nel testo dei provvedimenti e nelle procedure adottate (art. 20 e 21 D.Lgs. 196/2003)																
6. Eventuali altre normative generali e/o di specificazione"""	

#print(question)

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

## Multilingual model + 16K of context
model_id = "meta-llama/Llama-3.1-8B"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map= torch.device('cuda:0'),
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create the pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


prompt = "#### ISTRUZIONI\n Sei un avvocato, leggi la determina dirigenziale e poi rispondi a delle domande, rispondi punto per punto. Riporta solo le risposte\n"
one_shot = "##### OUTPUT Esempio di output desiderato:\n 1. D.lgs 267/2000 Rispettato"

ret = text_gen_pipeline(
    prompt+"#### DETERMINA:\n"+determina+"##### DOMANDE\nRispondi punto per punto brevemente\n\n"+question+one_shot,
    max_length=5,    # Limit the length of generated text
    temperature=0.01,   # Set the temperature
    num_return_sequences=1,  # Number of completions to generate
    return_full_text=False,
    )

print(ret)

print(ret[0]["generated_text"])

with open("./src/llama/MB_text/responses/DET_MB_2708-2024.txt","w", encoding="utf-8") as f:
    f.write("########## ------------ PROMPT --------------\n\n")
    f.write(prompt+"#### DETERMINA:\n"+determina+"##### DOMANDE\nRispondi punto per punto brevemente\n\n"+question+one_shot)
    f.write("\n\n########## -------------------------- GENERATED ---------------------------\n\n")
    f.write(ret[0]["generated_text"])