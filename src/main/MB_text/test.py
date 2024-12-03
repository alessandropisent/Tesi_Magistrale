from dotenv import load_dotenv
import os
import openai
# Load the environment variables from .env file
load_dotenv()

# Now you can access the environment variable just like before
openai.api_key = os.environ.get('OPENAI_APIKEY')

print(openai.api_key)

