import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load API keys from your local .env file
load_dotenv()

# Initialize the Google Gemini chat model using environment variables
model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=1.0,
    max_tokens=1000,
    timeout=15,
    max_retries=2,
)

# Invoke the model and print the text response
response = model.invoke("Write a four-line poem about coding and data.")
print(response.content)
