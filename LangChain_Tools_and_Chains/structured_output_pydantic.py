import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

# Load API keys from your local .env file
load_dotenv()

# Initialize the Groq chat model
model = init_chat_model(
    model_provider="groq",
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY"),
)


# Define nested Pydantic schemas for structured JSON output
class Actor(BaseModel):
    name: str = Field(description="Full name of the actor")
    role: str = Field(description="Character name or role played in the movie")


class MovieDetails(BaseModel):
    title: str = Field(description="Official title of the movie")
    year: int = Field(description="Release year of the movie")
    cast: list[Actor] = Field(description="List of main cast members and their roles")
    genres: list[str] = Field(description="Genres associated with the movie")
    budget: float | None = Field(None, description="Estimated budget in millions USD")


# Bind the Pydantic schema to the LLM
structured_model = model.with_structured_output(MovieDetails)

# Invoke the model and print the structured Pydantic object
response = structured_model.invoke("Provide detailed information about the movie Stree.")
print(response)
