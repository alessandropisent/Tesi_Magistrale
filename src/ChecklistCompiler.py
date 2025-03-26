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
        
        """
        Initialize a ChecklistCompiler instance.

        Args:
            llm: Identifier for the language model to use (e.g., @ LLAMA or OPENAI).
            municipality (str): Name of the municipality. @ OLBIA, LUCCA
            text_gen_pipeline: Optional text generation pipeline.
            model (str, optional): The model name for generating responses. Defaults to "gpt-4o-mini".
            hasSezioni (bool, optional): Flag indicating if the checklist includes sections. If None, it is
                                         determined based on the municipality.
        """        
        
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
        """
        Set the model name for the ChecklistCompiler instance.

        Args:
            model (str): The new model name to be used.
        """
        self.model = model
    
    def set_text_gen_pipeline(self, text_gen_pipeline):
        """
        Set the text generation pipeline for the ChecklistCompiler instance.

        Args:
            text_gen_pipeline: The text generation pipeline to assign.
        """
        self.text_gen_pipeline = text_gen_pipeline
    
    @staticmethod
    def analize_response(text):
        """
        Analyze the response text to extract a valid answer.

        This method searches for specific markers (e.g., "RISPOSTA GENERALE:" or "RISPOSTA:") in the provided text
        and returns the corresponding answer in uppercase. If no valid answer is found, returns "Not found".

        Args:
            text (str): The text to analyze.

        Returns:
            str: The extracted answer in uppercase, or "Not found" if no valid answer exists.
        """
        
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
        """
        Clean the input text by removing unrecognized special characters.

        Args:
            text (str): The input text string.

        Returns:
            str: The cleaned text string.
        """

        # Regular expression to match non-alphanumeric characters and common punctuation
        pattern = r"[^\w\s]"

        # Replace matched characters with an empty string
        cleaned_text = re.sub(pattern, '', text)

        return cleaned_text

    @staticmethod
    def get_checklist(checklists,nome_checklist):
        """
        Retrieve a specific checklist from the available checklists.

        Args:
            checklists (dict): Dictionary containing all available checklists.
            nome_checklist (str): The name of the checklist to retrieve.

        Returns:
            dict: The specified checklist if found.

        Raises:
            Exception: If the specified checklist is not found.
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
        """
        Generate a user prompt for a checklist point.

        Args:
            punto (str): The text of the checklist point.
            num (str): The identifier number for the checklist point.
            sezione (str, optional): The section of the checklist point, if applicable.
            istruzioni (str, optional): Instructions for the checklist point.
            return_just_text (bool, optional): If True, returns the prompt as plain text instead of a structured dictionary.

        Returns:
            dict or str: The generated prompt in a dictionary format or as plain text if return_just_text is True.
        """

        
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
        Generate a structured prompt for verifying compliance between a checklist point and a determination.

        Args:
            istruzioni (str): Instructions for the checklist point.
            punto (str): The text of the checklist point.
            num (str): The identifier number for the checklist point.
            determina (str): The text of the determination.
            sezione (str, optional): The section of the checklist point, if applicable.

        Returns:
            str or list: The complete prompt formatted for the language model. Returns a string for LLAMA or a list for OPENAI.
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
        
        if self.llm==LLAMA:
            system = [{
            "role": "system", 
            "content" : f"""{text_system}

        
        ############# ------------- DETERMINA -------------------
        {determina}
        """}] 
            user = [{
                "role":"user",
                "content":self.generate_user_prompts(punto=punto,
                                                  num=num,
                                                  sezione=sezione,
                                                  istruzioni=istruzioni,
                                                  return_just_text=True),
            }]
            
            
        elif self.llm == OPENAI:
            
            system = [{
            "role": "developer", 
            "content" : [{ "type": "text", 
        "text":f"""{text_system}

        
        ############# ------------- DETERMINA -------------------
        {determina}
        """}]}] 
        

        
        
        
            user = [self.generate_user_prompts(punto=punto,
                                                  num=num,
                                                  sezione=sezione,
                                                  istruzioni=istruzioni)]
        
        
        
        return system + user
        
                    

    def generate_response(  self,
                            complete_prompt,
                            temperature=1,
                            top_p=None,
                            do_sample=None):
        
        """
        Generate a response from the language model based on the provided prompt.

        Args:
            complete_prompt: The complete prompt to send to the language model.
            temperature (float, optional): The temperature parameter for response generation. Defaults to 1.
            top_p (optional): The top-p sampling parameter.
            do_sample (optional): Flag to determine whether to sample the response.

        Returns:
            str: The generated response text from the language model.
        """
        
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
            return ret[0]["generated_text"][-1]

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
        Generate a detailed analysis of the compliance between a regulatory checklist and a determination.

        This method reads the determination text, retrieves the appropriate checklist, generates prompts for each checklist point,
        obtains responses from the language model, and writes the analysis to output files.

        Args:
            nome_determina (str): Identifier of the determination to analyze.
            nome_checklist (str): Name of the checklist to apply.
            checklists (dict): Dictionary of available checklists.
            sub_cartella (str, optional): Subfolder for storing output files.
            temperature (float, optional): Temperature parameter for response generation. Defaults to 1.
            top_p_value (optional): The top-p sampling parameter.
            debug_files (bool, optional): If True, generates debug output files. Defaults to False.
            do_sample (optional): Flag to control sampling during response generation.
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
    
    
    
    def generate_prompt_choose(self,
                               determina, 
                               checklists):
        """
        Generate a prompt to suggest the most appropriate checklist for a given determination.

        Args:
            determina (str): The text of the determination.

        Returns:
            str: A structured prompt for suggesting the appropriate checklist.
        """
        list_names_checklists = [check["NomeChecklist"] for check in checklists["checklists"]]
        list_basics_checklists = [check["breve"] for check in checklists["checklists"]]
        list_descri_checklists = [check["Descrizione"] for check in checklists["checklists"]]
        
        
        str_list_basics = "        \n".join([f"-**{list_names_checklists[i]}**:{list_basics_checklists[i]}" for i in range(len(list_names_checklists))])
        str_list_heavy = "\n".join([f"3.{i+1}. **{list_names_checklists[i]}**:{list_descri_checklists[i]}" for i in range(len(list_names_checklists))])
        
        
        text_user = f"""### A QUALE CHECKLIST APPARTIENE LA SEGUENTE DETERMINA?:
        
        ### Output (solo nome checklist):
        {" , ".join(list_names_checklists)} -- (note eventuali)  
        
        ############# ------------- DETERMINA -------------------
        {determina}
        
  
            
        """
        
        text_system =f"""
        Sei un assistente esperto in materia di diritto amministrativo. Il tuo compito è supportare un impiegato comunale nel controllo della regolarità amministrativa di una determina dirigenziale.

        Segui i passaggi seguenti:
        1. Leggi le checklists fornite 
        2. Leggi il testo della determina.
        3. Rispondi dicendo quale checklist va applicata alla determina tra le seguenti:
        {str_list_heavy}
        4. Pensa attentamente quali di queste descrizioni sono più pertinenti alla determina
        4. RISPONDI SOLO CON NOME CHECKLIST
        
        ### Output (solo nome checklist):
        {" ".join(list_names_checklists)} -- (note eventuali)
        

        Utilizza un linguaggio semplice e accessibile. Rispondi in maniera chiara e ordinata.
        """
        
        
        if self.llm==LLAMA:
            system = [{
            "role": "system", 
            "content" : text_system}] 
            user = [{
                "role":"user",
                "content":text_user
            }]
            
            
        elif self.llm == OPENAI:
            
            system = [{
            "role": "developer", 
            "content" : [{ "type": "text", 
            "text":text_system}]}] 
        
            user = [ {
            "role":"user",
            "content":[{"type": "text",
                "text": text_user
            }]
            }]
        
        return system+user
    
    
        
    
            
    
    def choose_checklist(   self,
                            determine:pd.DataFrame,
                            checklists,
                            sub_cartella="",
                            temperature=1,
                            top_p_value=None,
                            debug_files=False,
                            do_sample=None):
        
        """
        Choose the most suitable checklist for a given determination.

        Args:
            nome_determina (str): Identifier of the determination.
            text_gen_pipeline: The text generation pipeline to use.
            model_id: The identifier of the model.

        Returns:
            Not explicitly defined (placeholder function).
        """
        if self.municipality == LUCCA:
            
            pathDets = f"./src/txt/Lucca/determinazioni/"
            
            
            if self.llm == LLAMA:
                pathResponse = f"./src/llama/Lucca_text/choose/{sub_cartella}"

                
            
            elif self.llm == OPENAI:
                pathResponse = f"./src/openai/Lucca_text/choose/{sub_cartella}"

        
        elif self.municipality == OLBIA:
            pathDets = f"./src/txt/Olbia/determinazioni/"
            
            if self.llm == LLAMA:
                pathResponse = f"./src/llama/Olbia_text/choose/{sub_cartella}"
                
            
            elif self.llm == OPENAI:
                pathResponse=f"./src/openai/Olbia_text/choose/{sub_cartella}"
   
            
            
        pathJson=f"{pathResponse}determine.json"
        
        rows = []
        
        for name_dets in tqdm(determine["Numero Determina"]):
            
            path = f"{pathDets}{name_dets}.txt"
        
            with open(path,"r", encoding="utf-8") as f:
                determina= f.read()
                
            complete_prompt = self.generate_prompt_choose(determina=determina,
                                                          checklists=checklists)
            
            text_response = self.generate_response(complete_prompt=complete_prompt,
                                                   temperature=temperature)
            
            row = {
                "det":name_dets,
                "model":self.model,
                "temperature":temperature,
                "municipality":self.municipality,
                "LLM":text_response,
                "prompt":complete_prompt
            }
            rows.append(row)

        
        
        
        
        if not os.path.exists(pathResponse):
            os.makedirs(pathResponse)  
        
        df = pd.DataFrame(rows)
        df.to_json(pathJson)
        #print(df)
        
        
        
