import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load API keys from your local .env file
load_dotenv()

# Initialize the Groq chat model
model = init_chat_model(
    model_provider="groq",
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
)

# Define a batch of prompts to process in parallel
prompts = [
    "What is an umbrella?",
    "What are key technologies used in modern days?",
]

# Execute batch inference and print each response
responses = model.batch(prompts)

for prompt, res in zip(prompts, responses):
    print(f"Q: {prompt}")
    print(f"A: {res.content}\n")
