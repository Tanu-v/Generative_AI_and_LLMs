import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables (including HF_TOKEN if set in .env)
load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNINGS"] = "1"

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# Load tokenizer and causal language model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create a Hugging Face text-generation pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    return_full_text=False,
)

# Wrap the pipeline inside LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

# Combine with a ChatPromptTemplate using LCEL (|)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Keep answers concise and clear."),
    ("human", "{question}"),
])

chain = prompt | llm
response = chain.invoke({"question": "Explain what a Large Language Model is in two sentences."})
print(response)
