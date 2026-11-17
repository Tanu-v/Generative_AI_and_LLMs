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

# 1. Real-time Token Streaming
print("--- 1. STREAMING RESPONSE ---")
for chunk in model.stream("Why is the sky blue? Explain in two sentences."):
    print(chunk.text, end="", flush=True)
print("\n")

# 2. Asynchronous Batching (Yields responses as soon as each finishes)
print("--- 2. BATCH AS COMPLETED ---")
prompts = [
    "What is a healthy snack for weight management?",
    "Give one tip for staying focused while studying.",
    "Suggest a weekend travel destination in India.",
]

for index, result in model.batch_as_completed(prompts):
    print(f"[Prompt {index + 1}]: {result.text}\n")

# 3. Standardized Content Blocks Inspection
print("--- 3. CONTENT BLOCKS ---")
response = model.invoke("Describe a running horse in one sentence.")
print(response.content_blocks)
