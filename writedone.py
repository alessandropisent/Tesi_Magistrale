import json
Municipalities = ["Olbia", "Lucca"]
temps = [0.0, 0.01, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
models = ["meta-llama/Llama-3.3-70B-Instruct"]

{"model":"","municipality":"","temp":""}

l = []

for c in Municipalities:
    for m in models:
        for t in temps:
        
            l.append({"model":m,"municipality":c,"temp":t})

with open("done_1.json","w",encoding="utf-8") as f:
    json.dump({"done":l},f,indent=3)