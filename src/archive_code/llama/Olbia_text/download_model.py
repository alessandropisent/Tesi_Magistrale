from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            #device_map= torch.device('cuda:0'),
            
            device_map='auto',
            #use_flash_attention_2=True
        )
tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")