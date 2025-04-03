from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import torch

model_id = "meta-llama/Llama-3.3-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map= torch.device('cuda:0'),
            
            #quantization_config=quantization_config,
            #device_map='auto',
            #max_memory=max_memory,   # Ensure each GPU uses at most 23GB of VRAM
            #max_memory=max_memory,
            #use_flash_attention_2=True
        )