import os
import requests
from dotenv import load_dotenv

# Load API keys from your local .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in your .env file.")

url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

response = requests.get(url, headers=headers, timeout=15)
response.raise_for_status()

models_data = response.json().get("data", [])
print(f"Available Groq Models ({len(models_data)} total):")
for model in models_data:
    print(f" - {model['id']} (Owned by: {model.get('owned_by', 'N/A')})")
