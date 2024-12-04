from dotenv import load_dotenv
import os
import openai
# Load the environment variables from .env file
load_dotenv()

# Now you can access the environment variable just like before
openai.api_key = os.environ.get('OPENAI_APIKEY')

print(openai.api_key)

# Define a function to interact with the model
def generate_text(prompt, model="text-davinci-003"):
    try:
        response = openai.Completion.create(
            engine=model,
            prompts=prompt,
        )
        # Extract and return the generated text
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {e}"

prompts = [
    { "role":"Avvocato" , "content":"Determina da analizzare"},
    { "role":"Avvocato" , "content":""},
    { "role":"Avvocato" , "content":""},
]





result = generate_text(prompts)
print("Generated Response:", result)

with open("response.md","w",encoding="utf-8") as f:
    f.write(result)