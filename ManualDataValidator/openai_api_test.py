import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt = "tell me abou aI in 2 sentences."
resonse = client.responses.create(
    model="gpt-5-nano",
    input=prompt
)
        

print(resonse.output_text)