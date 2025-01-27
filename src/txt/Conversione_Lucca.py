punti = ["Competenza organo/dirigente individuato con decreto",
"Articolazione dell’atto in intestazione  preambolo/premessa, parte motivazione, dispositivo, indicazione autorità e termini per ricorrere, riferimenti al deposito nel fascicolo digitale degli atti afferenti il procedimento amm.vo ex art. 41 CAD (gestione documentale)",
"Affidabilità riferimenti normativi di carattere generale es. TUEL ",
"Affidabilità riferimenti normativi specifici es. Codice dei contratti, Regolamenti Ente",
"Nel caso di attività inserite negli atti di programmazione Coerenza con: \n- DUP  e PIAO \n- Bilancio di Previsione se trattasi di DD con spesa; \n- cronoprogramma  in fase di esecuzione (lavori, forniture, servizi – SITAT e altre banche dati)",
"Esplicitazione motivazione con puntuale indicazione presupposti di fatto e ragioni giuridiche (riferimenti normativi e/o regolamentari)",
"1) Responsabile del Procedimento ovvero distinzione tra responsabile del procedimento e del provvedimento; se è la EQ responsabile del provvedimento (delega di firma) indicare l’attribuzione/delega del Dirigente\n2)  indicare  RUP delegato per fase",
"Indicazione misure di prevenzione pertinenti rispetto allo specifico provvedimento, ad es:\n- conflitto di interessi (per il PNRR obbligatorio sempre chiedere la dichiarazione)\n- obblighi di pubblicità ai sensi delle Delibere ANAC 2023 per contratti finiti o in corso al 31.1.2023\n- obblighi di pubblicità ai sensi del Dlvo 36/2023, art. 28\nQualora il provvedimento sia funzionale al raggiungimento di un obiettivo di performance, dare atto del rispetto delle misure di protezione inserite nelle fasi di PIAO",
"Rispetto normativa sulla tutela della riservatezza laddove rilevante", 
"Rispetto dei tempi e dei termini del procedimento",
"Altre annotazioni/criticità in merito: a. possibilità di semplificare\nb. possibilità di digitalizzare\nc. fare un focus specifico\nd. pari opportunità\ne. accesso"]

with open("test.txt","w",encoding="utf-8") as f:
    for i,p in enumerate(punti):
        template = f"""[
    "Istruzioni": "",
    "Punto": "{p}",
    "num": {i+1}
]

"""
        f.write(template)
