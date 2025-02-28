el = {
    "ELEMENTI GENERALI IDENTIFICATIVI DELL'ATTO":[1,2,3,4],
    "ELEMENTI RIFERITI AL SOGGETTO CHE ADOTTA L'ATTO":[5,6,7,8],
    "RIFERIMENTI NORMATIVI":[9,10,11,12],
    "ELEMENTI TIPICI DELLA DETERMINAZIONE A CONTRARRE":[13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29],
    "RIFERIMENTI CONTABILI":[30,31,32,33,34],
    "DOCUMENTI RICHIAMATI E ALLEGATI":[35],
    "PRESCRIZIONI ANTICORRUZIONE E TRASPARENZA":[36,37],
    "ASPETTI DI REGOLARITÀ FORMALE":[38,39]
    
}

with open("sez.json","w") as f:
    for sez in el:
        for punt in el[sez]:
            toWrite = f'''
{{ 
    "Istruzioni": "",
    "Punti":"",
    "Sezione": "{sez}",
    "num":{punt}

}},
'''
            f.write(toWrite)
            



