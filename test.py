from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Specify the local path where the model files are saved
# local_model_path = "./gemma-2-2b"  # Adjust this path if necessary
local_model_path = "./gemma-2-2b-it"  # Adjust this path if necessary
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Move the model to the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
model.to(device)

def run_prompt(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs["input_ids"], max_length=100)  # Adjust max_length as needed
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

prompt = "What do you know about B2AI?"
print(run_prompt(prompt))
