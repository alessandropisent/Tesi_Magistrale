import re
import os
from tqdm import tqdm
import json
import pandas as pd
from dotenv import load_dotenv
import openai


# Load the environment variables from .env file
load_dotenv()

## DEFAULT VALUES
LUCCA = "Lucca"
OLBIA = "Olbia"
LLAMA = "llama"
OPENAI = "openai"

MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST = [OLBIA]

class ChecklistCompiler:
    
    
    
    def __init__(   self, 
                    llm, 
                    municipality,
                    text_gen_pipeline=None, 
                    model="gpt-4o-mini",
                    hasSezioni=None,
                 ):
        self.llm = llm
        self.municipality = municipality
        self.model=model
        self.text_gen_pipeline=text_gen_pipeline,
        
        # Qualche checklist potrebbe avere la sezione "Sezione" nelle checklist
        if hasSezioni is None:
            if municipality in MUNICIPALIIES_WITH_SEZIONE_IN_CHECKLIST:
                self.hasSezioni=True
            else:
                self.hasSezioni=False
        else:
            self.hasSezioni=hasSezioni
        

    def set_model(self,model):
        self.model = model
    
    def set_text_gen_pipeline(self, text_gen_pipeline):
        self.text_gen_pipeline = text_gen_pipeline
    
    @staticmethod
    def analize_response(text):
        # Convert the text to string (for safety)
        text = str(text)
        
        # Define the possible answers
        possible_answers = ['si', 'sì', 'no', 'non richiesto', 'non pertinente']
        
        # 1. Try to find the pattern "RISPOSTA GENERALE:" or "RISPOSTA:" followed by an answer
        general_pattern = re.search(
            r"(?i)(\*?risposta generale\*?:|risposta:|\*?risposta\*?:)\s*(si|sì|no|non richiesto|non pertinente)\b", text
        )
        if general_pattern:
            ans = general_pattern.group(2).lower()
            if ans in possible_answers:
                return ans.upper()
        
        # 2. If the marker is not found, check if the text starts directly with an answer
        start_pattern = re.match(
            r"^\s*(si|sì|no|non richiesto|non pertinente)\b", text, flags=re.IGNORECASE
        )
        if start_pattern:
            ans = start_pattern.group(1).lower()
            if ans in possible_answers:
                return ans.upper()
        
        # 3. Check the first 50 characters for relevant answers in significant positions
        prefix = text[:50].lower()
        for answer in possible_answers:
            if re.search(rf"\b{re.escape(answer)}\b", prefix):
                return answer.upper()
        
        # 4. Check the entire text for structured cases (e.g., RISPOSTA GENERALE within a section)
        structured_pattern = re.search(
            r"(?i)\*{0,2}RISPOSTA GENERALE\*{0,2}\s*:\s*(si|sì|no|non richiesto|non pertinente)\b", text
        )
        if structured_pattern:
            ans = structured_pattern.group(1).lower()
            if ans in possible_answers:
                return ans.upper()
        
        # 5. Additional check: if RISPOSTA GENERALE is followed by an answer, even on a subsequent line
        if re.search(r"(?i)\bRISPOSTA GENERALE\b.*?\b(si|sì|no|non richiesto|non pertinente)\b", text, re.DOTALL):
            ans = re.search(r"(?i)\bRISPOSTA GENERALE\b.*?\b(si|sì|no|non richiesto|non pertinente)\b", text, re.DOTALL).group(1).lower()
            if ans in possible_answers:
                return ans.upper()
        
        # If none of the previous rules are satisfied, return "Not found"
        return "Not found"


    @staticmethod
    def clean_text(text):
        """Cleans text by removing unrecognized special characters.
        
        Args:
            text: The input text string.

        Returns:
            The cleaned text string.
        """

        # Regular expression to match non-alphanumeric characters and common punctuation
        pattern = r"[^\w\s]"

        # Replace matched characters with an empty string
        cleaned_text = re.sub(pattern, '', text)

        return cleaned_text

    @staticmethod
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
    
    
    def generate_user_prompts(  self,
                                punto="",
                                num="",
                                sezione="", 
                                istruzioni="",
                                return_just_text=False):

        
        if istruzioni != "":
            ist = "### Istruzioni\n\n"+ istruzioni + "\n\n"
        else:
            ist = ""
        
        if self.hasSezioni:
            sez = f", SEZIONE: {sezione}:"
        else:
            sez = ""
        
        text = f"""{ist}
            ### Rispondi al punto {num}{sez}:
            {punto}
            
            ### Output:
            RISPOSTA GENERALE : [SI, NO, NON PERTINENTE], [spiegazione sintetica se necessaria]
            """

        if return_just_text:
            return text
        
        return {
            "role":"user",
            "content":[{"type": "text",
                "text": text
            }]
            }


    def generate_prompt(self,
                        istruzioni, 
                        punto,
                        num, 
                        determina,
                        sezione=""):
        """
        Crea un prompt strutturato per verificare la corrispondenza tra i punti di una checklist normativa e una determina.
        
        Args:
            checklist (str): Contenuto della checklist normativa da verificare.
            checklist (str): Contenuto della checklist normativa da verificare.
            determina (str): Testo della determina dirigenziale.

        Returns:
            str: Prompt strutturato per la verifica normativa.
        """    
        
        text_system = f"""
        Sei un assistente esperto in materia di diritto amministrativo. Il tuo compito è supportare un impiegato comunale nel controllo della regolarità amministrativa di una determina dirigenziale.

        Segui i passaggi seguenti:
        1. Leggi la checklist fornita, che contiene punti numerati e specifiche normative da verificare.
        2. Leggi il testo della determina.
        3. Per ogni punto della checklist, verifica se l'istruzione è rispettata.
        4. Rispondi per ogni punto utilizzando uno dei seguenti criteri:
        - **SI**: Il punto della checklist e relative istruzioni sono rispettate [NESSUNA ANOMALIA]
        - **NO**: La determina NON passa il controllo, il punto della checklist NON è rispettato [ANOMALIA RILEVATA: LIEVE o GRAVE]
        - **NON PERTINENTE**: Il punto della checklist non è pertinente alla determina. Aggiungi una spiegazione sintetica. [NESSUNA ANOMALIA]
        6. Alla fine, aggiungi eventuali "Note finali" se ci sono problemi generali o ambiguità rilevate nella determina.

        Utilizza un linguaggio semplice e accessibile. Rispondi in maniera chiara e ordinata.
        """
        
        llama_prompt = f"""<s>[INST] <<SYS>>
        {text_system}
       
        {istruzioni}
        <</SYS>>

        ##### CHECKLIST 
        {punto}

        ###### DETERMINA:
        {determina}

        ##### OUTPUT:
        Punto {num}: [SI/NO/NON PERTINENTE], [spiegazione sintetica se necessaria]

        Note finali: [eventuali osservazioni generali]
        [/INST]</s>
        """
        
        if self.llm == LLAMA:
            return llama_prompt
        
        developer_openAI = [{
            "role": "developer", 
            "content" : [{ "type": "text", 
        "text":f"""{text_system}

        
        ############# ------------- DETERMINA -------------------
        {determina}
        """}]}] 
        
        user_openAI = [self.generate_user_prompts(punto=punto,
                                                  num=num,
                                                  sezione=sezione,
                                                  istruzioni=istruzioni)]
        
        
        if self.llm == OPENAI:
            return developer_openAI + user_openAI
        
                    

    def generate_response(  self,
                            complete_prompt,
                            temperature=1,
                            top_p=None,
                            do_sample=None):
        
        if self.llm == LLAMA:
            if do_sample is None:
                ret = self.text_gen_pipeline(
                            complete_prompt,
                            max_new_tokens=500,
                            truncation = True,
                            return_full_text=False,
                        )
            else:
                ret = self.text_gen_pipeline(
                            complete_prompt,
                            max_new_tokens=500,
                            truncation = True,
                            return_full_text=False,
                            do_sample=do_sample
                        )
            return ret[0]["generated_text"]

        elif self.llm == OPENAI:
            client = openai.OpenAI()
        
            try:
                if top_p is None:
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=complete_prompt,
                        seed=42,
                        temperature=temperature,
                    )
                return str(response.choices[0].message.content)
            except Exception as e:
                return f"Error: {e}"
            
            

    def checklist_determina(self,
                            nome_determina,
                            nome_checklist,  
                            checklists,
                            sub_cartella="",
                            temperature=1,
                            top_p_value=None,
                            debug_files=False,
                            do_sample=None):

        """
        Genera un'analisi dettagliata della corrispondenza tra una checklist normativa e una determina dirigenziale.
        
        Args:
            nome_determina (str): Nome identificativo della determina da analizzare.
            nome_checklist (str): Nome della checklist normativa da applicare alla determina.
            checklists (dict): Dizionario contenente tutte le checklist disponibili e i loro punti normativi.
        
        Output:
            Scrive il risultato dell'analisi in file di output, inclusi dettagli sulla corrispondenza normativa.
        """
        
        if self.municipality == LUCCA:
            pathDet = f"./src/txt/Lucca/determinazioni/{nome_determina}.txt"
            
            if nome_checklist == "Contratti":
                name_checklist_end = "cont"
            elif nome_checklist == "Determine":
                name_checklist_end = "det"
            
            
            if self.llm == LLAMA:
                pathResponse = f"./src/llama/Lucca_text/responses/{sub_cartella}"
                responseComplete = f"./src/llama/Lucca_text/responses/{sub_cartella}{nome_determina}-Complete.txt"
                responseEasy = f"./src/llama/Lucca_text/responses/{sub_cartella}{nome_determina}-response.txt"
                pathJson = f"./src/llama/Lucca_text/responses/{sub_cartella}{nome_determina}.json"
            
            elif self.llm == OPENAI:
                pathResponse = f"./src/openai/Lucca_text/responses/{sub_cartella}"
                pathJson = f"./src/openai/Lucca_text/responses/{sub_cartella}{nome_determina}.json"
        
        elif self.municipality == OLBIA:
            pathDet = f"./src/txt/Olbia/determinazioni/{nome_determina}.txt"
            
            if self.llm == LLAMA:
                pathResponse=f"./src/llama/Olbia_text/responses/{sub_cartella}"
                pathJson = f"{pathResponse}{nome_determina}.json"
            
            elif self.llm == OPENAI:
                pathResponse=f"./src/openai/Olbia_text/responses/{sub_cartella}"
                pathJson = f"{pathResponse}{nome_determina}.json"    
            
            
        
        
        with open(pathDet,"r", encoding="utf-8") as f:
            determina= f.read()
        
        checklist = self.get_checklist(checklists,nome_checklist)
        
        dictionary_response = {"Response":[]}
        
        if not os.path.exists(pathResponse):
            os.makedirs(pathResponse)  

        if debug_files:
            complete_lines = []
            response_lines = []
    
                
        for i,punto in tqdm(enumerate(checklist["Punti"]),
                            total=len(checklist["Punti"])):
            
            
            if self.hasSezioni:
                complete_prompt = self.generate_prompt( istruzioni=punto["Istruzioni"],
                                                        punto=punto["Punto"],
                                                        num=punto["num"],
                                                        sezione=punto["Sezione"], 
                                                        determina=determina)
            
            else:
                complete_prompt = self.generate_prompt( istruzioni=punto["Istruzioni"],
                                                        punto=punto["Punto"],
                                                        num=punto["num"], 
                                                        determina=determina)
            
            

            ret = self.generate_response( complete_prompt,
                                          temperature=temperature,
                                          top_p=top_p_value,
                                          do_sample=do_sample)
            
            # If i want to write all the debus file necessary
            if debug_files:
                ### COMPLETE OUPUT
                complete_lines.append("\n\n########## ------------ PROMPT --------------\n\n")
                complete_lines.append(complete_prompt)
                complete_lines.append("\n\n########## -------------------------- GENERATED ---------------------------\n\n")
                complete_lines.append(ret)
                
                ### JUST OUTPUT
                
                just_response = f""" 
                #### CHECKLIST
                {punto["Istruzioni"]+"\n\n"+punto["Punto"]}
                
                ### RESPONSE
                {ret}
                """
                
                response_lines.append(just_response)
                
            dictionary_response["Response"].append({"num": punto["num"],
                                                    "punto":punto["Punto"],
                                                    "LLM_Response":ret,
                                                    "Simple":self.analize_response(ret),
                                                    "prompt":complete_prompt,
                                                    "llm":self.llm,
                                                    "municipality":self.municipality,
                                                    "temperature":temperature,
                                                    "checklist":nome_checklist,
                                                    "determina":nome_determina,
                                                    "model":self.model,})
        

        with open(pathJson,"w", encoding="utf-8")as f:
            json.dump(dictionary_response,f,indent=3)
        
        if debug_files and self.llm == LLAMA:
            with open(responseComplete,"w", encoding="utf-8") as f_complete:
                f_complete.writelines(complete_lines)
            
            with open(responseEasy,"w", encoding="utf-8") as f_complete:
                f_complete.writelines(response_lines)
    
    
    
    def generate_prompt_choose(determina):
        """
        Genera un prompt per suggerire la checklist più adatta per una determina basata sul contenuto della stessa.
        
        Args:
            determina (str): Testo della determina dirigenziale.

        Returns:
            str: Prompt strutturato per il suggerimento della checklist.
        """
        pass        
    
    def choose_checklist(nome_determina,
                     text_gen_pipeline, 
                     model_id):
        
        return
    
        with open(f"./src/txt/MB/determinazioni/DET_{nome_determina}.txt","r", encoding="utf-8") as f:
            determina= f.read()
        
        ### CHECKLIST SELEZIONATE PER LA DETERMINA
        complete_prompt = generate_prompt_choose(determina)

        ret = text_gen_pipeline(
            complete_prompt,
            max_length=1_000,    # Limit the length of generated text
            #max_new_tokens=500,
            #truncation = True,
            temperature=0.01,   # Set the temperature
            return_full_text=False,
        )
