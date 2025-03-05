import os
import pdfplumber

def table_to_markdown(table):
    """
    Converte una tabella (lista di liste) in una stringa in formato Markdown.
    Assume che la prima riga sia l'header della tabella.
    """
    if not table or len(table) == 0:
        return ""
    
    # Determina il numero massimo di colonne
    num_cols = max(len(row) for row in table)
    # Normalizza ogni riga: se mancano celle, le riempie con stringhe vuote
    normalized_table = [row + [""] * (num_cols - len(row)) for row in table]
    
    # Costruisci la riga dell'header
    header = normalized_table[0]
    md = "| " + " | ".join(header) + " |\n"
    # Riga separatrice
    md += "| " + " | ".join(["---"] * num_cols) + " |\n"
    # Righe del corpo della tabella
    for row in normalized_table[1:]:
        md += "| " + " | ".join(row) + " |\n"
    
    return md

def extract_markdown_from_pdf(pdf_path):
    """
    Estrae il testo da un PDF e converte eventuali tabelle in formato Markdown.
    
    Args:
        pdf_path (str): Percorso del file PDF.
    
    Returns:
        str: Contenuto in Markdown.
    """
    md_output = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Aggiungi intestazione per la pagina
            md_output += f"## Pagina {page_num}\n\n"
            # Estrai il testo della pagina
            text = page.extract_text() or ""
            md_output += text + "\n\n"
            # Estrai tutte le tabelle presenti nella pagina
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    md_output += table_to_markdown(table) + "\n\n"
    return md_output

def extract_markdown_from_pdfs(directory_input, directory_output):
    """
    Legge tutti i file PDF in una directory, li converte in Markdown (testo + tabelle)
    e li salva in file .md nella directory di output.
    
    Args:
        directory_input (str): Directory contenente i PDF.
        directory_output (str): Directory dove salvare i file Markdown.
    """
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)
        
    for filename in os.listdir(directory_input):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory_input, filename)
            try:
                md_text = extract_markdown_from_pdf(pdf_path)
                output_filename = os.path.join(directory_output, filename[:-4] + ".md")
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(md_text)
                print(f"File salvato: {output_filename}")
            except Exception as e:
                print(f"Errore nel processare {filename}: {e}")

# Esempio di utilizzo:
pdf_directory = "./determine_pdf/Lucca"  
md_directory = "./src/txt/Lucca/Raw_det"

extract_markdown_from_pdfs(pdf_directory, md_directory)
