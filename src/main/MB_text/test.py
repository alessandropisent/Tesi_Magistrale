from dotenv import load_dotenv
import os
# Load the environment variables from .env file
load_dotenv()

# Now you can access the environment variable just like before
api_key = os.environ.get('OPENAI_APIKEY')
print(api_key)