import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load API keys from your local .env file
load_dotenv()

# Initialize the Groq chat model
model = init_chat_model(
    model_provider="groq",
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Invoke the model and inspect standardized content blocks
response = model.invoke("Describe a cat in detail.")
print(response.content_blocks)
