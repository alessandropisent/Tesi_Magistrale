import requests
from bs4 import BeautifulSoup
import re
import os
import pandas as pd



def download_and_rename_pdfs(url):
    
    try:
        response = requests.get(url.strip())
        response.raise_for_status()  # Raise an exception for non-200 status codes

        soup = BeautifulSoup(response.content, 'html.parser')




        for link in soup.find_all('a', href=True):
            if "download" in link['href'] and  link.find('img', alt='File principale'):
                pdf_url = link['href']
                # Assumiamo che il link sia relativo al dominio
                pdf_url = f"https://atti-albopretorio.comune.modena.it/{pdf_url}"

                # Scarica il PDF
                
                try:
                    pdf_response = requests.get(pdf_url, stream=True)
                    pdf_response.raise_for_status()

                

                
                

                    titolo_div = soup.find('div', class_='panel-heading titolo-albo')
                    ## Estrai il testo e rimuovi gli spazi in eccesso
                    titolo = titolo_div.text.split("Oggetto:")[1].strip()
                    
                    nome_file = link.text.strip()
                    
                    with open("download.tsv", "a") as f:
                        f.write(nome_file+" \t " + titolo +"\n")
                    
                    # Salva il PDF con il nuovo nome
                    with open(nome_file, 'wb') as f:
                        f.write(pdf_response.content)
                
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading PDF from {pdf_url}: {e}")
    
    except requests.exceptions.RequestException as e:
        print(f"Error getting page from {url}: {e}")

# URL del sito
url = "https://atti-albopretorio.comune.modena.it/AlboOnline/dettaglioAlbo/37805958"

# Crea una cartella per salvare i PDF (opzionale)
os.makedirs("Modena", exist_ok=True)
os.chdir("Modena")


with open("download.tsv", "w") as f:
    f.write("File\t Oggetto\n")


with open("links.txt","r") as f:
    for line in f.readlines():
        # Scarica e rinomina i PDF
        download_and_rename_pdfs(line)



df = pd.read_csv("download.tsv", sep="\t")
print(df)