from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load the DeepSeek Chat 1.3B model
MODEL_NAME = "deepseek-ai/deepseek-chat-1.3b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

@app.get("/")
def read_root():
    return {"message": "DeepSeek Chat 1.3B is running!"}

@app.post("/generate/")
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": generated_text}
