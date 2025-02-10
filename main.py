from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = FastAPI()

# Define the model identifier; update this if you find a different one
MODEL_NAME = "deepseek-ai/deepseek-vl-1.3b-chat"

# If the model requires authentication, set your Hugging Face token as an environment variable (HF_TOKEN)
HF_TOKEN = os.getenv("HF_TOKEN")

# Load the tokenizer and model.
# If your model is public, you can remove the 'use_auth_token' parameter.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, use_auth_token=HF_TOKEN)

@app.get("/")
def read_root():
    return {"message": "DeepSeek Chat 1.3B is running!"}

@app.post("/generate")
def generate_text(prompt: str):
    # Tokenize the input prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response}
