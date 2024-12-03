import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

# Configurazione WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Browser senza interfaccia grafica
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))

# Directory dove salvare i PDF
output_dir = "determine_dirigenziali_pdf"
os.makedirs(output_dir, exist_ok=True)

# URL iniziale
base_url = "https://cloud.urbi.it/urbi/progs/urp/ur1ME001.sto?DB_NAME=n1200267&w3cbt=S"

def scarica_pdf():
    try:
        # Vai al sito
        driver.get(base_url)

        # Clicca su "Determine Dirigenziali"
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.LINK_TEXT, "DETERMINAZIONI DIRIGENZIALI (90)"))
        ).click()

        time.sleep(5)

        # Trova tutti i pulsanti "Pubblicazione"
        pubblicazione_buttons = driver.find_elements(By.XPATH, "//button[text()='Pubblicazione']")
        print(f"Trovati {len(pubblicazione_buttons)} pulsanti 'Pubblicazione'.")

        # Itera sui pulsanti
        for index, button in enumerate(pubblicazione_buttons):
            try:
                print(f"Processo il pulsante {index + 1} di {len(pubblicazione_buttons)}.")

                # Clicca sul pulsante "Pubblicazione"
                button.click()

                # Aspetta che si carichi la barra laterale
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "FILE"))
                )

                # Clicca sul link "FILE"
                file_link = driver.find_element(By.LINK_TEXT, "FILE")
                pdf_url = file_link.get_attribute("href")

                # Scarica il PDF
                response = requests.get(pdf_url, stream=True)
                if response.status_code == 200:
                    pdf_name = f"document_{index + 1}.pdf"
                    pdf_path = os.path.join(output_dir, pdf_name)
                    with open(pdf_path, 'wb') as pdf_file:
                        for chunk in response.iter_content(chunk_size=1024):
                            pdf_file.write(chunk)
                    print(f"Scaricato: {pdf_name}")
                else:
                    print(f"Errore nel download del PDF: {response.status_code}")

                # Chiudi la barra laterale (se necessario)
                close_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Chiudi')]")
                close_button.click()

                # Aspetta un attimo prima di procedere
                time.sleep(1)

            except Exception as e:
                print(f"Errore con il pulsante {index + 1}: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    scarica_pdf()
