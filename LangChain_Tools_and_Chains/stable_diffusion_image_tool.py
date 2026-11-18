import os
import torch
from diffusers import StableDiffusionPipeline
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

# --- 1. SET UP THE FREE DIFFUSION PIPELINE ---
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading Stable Diffusion model on {device.upper()}...")
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)


# --- 2. WRAP IT AS A LANGCHAIN TOOL ---
@tool
def generate_image_tool(prompt: str) -> str:
    """Useful for generating a free AI image from a text prompt.
    Input should be a detailed description of the image."""
    print(f"\n[LangChain Tool] Generating image for prompt: '{prompt}'...")

    # Run the image generation pipeline
    result = pipe(prompt, num_inference_steps=25)
    image = result.images[0]

    # Save the file locally
    output_filename = "langchain_generated_image.png"
    image.save(output_filename)

    return f"Image successfully generated and saved to {output_filename}"


# --- 3. EXECUTE VIA LANGCHAIN ---
# We can use LangChain PromptTemplate to enhance or format the prompt before passing it to our tool
prompt_template = PromptTemplate.from_template(
    "A beautiful, high-quality, professional photograph of {subject}, highly detailed."
)

# Format the final prompt using LangChain
formatted_prompt = prompt_template.format(subject="a futuristic robot reading a book")

# Invoke our tool within the LangChain paradigm
status_message = generate_image_tool.invoke({"prompt": formatted_prompt})
print("\n" + status_message)
