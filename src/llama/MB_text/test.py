with open("./src/llama/MB_text/determinazioni/DET_MB_2708-2024.txt","r", encoding="utf-8") as f:
    determina= f.read()

print(determina)

## det_is_contr
question = """Applicazione delle normative pertinenti alla fattispecie, tenuto conto delle sopravvenute disposizioni vigenti: 
[NOTE] : Citare con precisione gli articoli, commi e lettere delle fonti di regolazione.Citare prima le norme di legittimazione e poi le norme procedurali del caso in specie
1. D.lgs 267/2000 ( art. 107 c. 2, lettera... art 109...)  183 e 191)																
2. D.Lgs 163/2006 s.m.i.  e/o contratto servizio escluso																
3. R.D.  n. 2440/1924  per affidamenti cui non si applica il Codice dei Contratti																
4. DPR  207/2010, art….. (indicare istituto specifico)																
5. Normativa sulla privacy nel testo dei provvedimenti e nelle procedure adottate (art. 20 e 21 D.Lgs. 196/2003)																
6. Eventuali altre normative generali e/o di specificazione"""	

print(question)