# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import torch

torch.cuda.empty_cache()

# Step 1: Load and Prepare the Dataset
# Replace 'path/to/your_data.txt' with the path to your text file or dataset.
dataset = load_dataset('text', data_files={'train': './train_data.txt'})

# Step 2: Load the Tokenizer    
# Replace 'path/to/gemma22bit' with the path to your local model files.
local_model_path = "./gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Step 3: Tokenize the Dataset
def tokenize_function(examples):
    # Tokenizes each text sample and handles padding/truncation
    # return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=64)

# Map tokenization function to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Create a Data Collator for Language Modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # mlm=False because this is for causal language modeling
)

# Step 5: Load the Model
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Step 6: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./results",               # Directory to save results and checkpoints
    evaluation_strategy="epoch",          # Evaluate at the end of each epoch
    learning_rate=2e-5,                   # Learning rate for the optimizer
    per_device_train_batch_size=1,        # Batch size for training per device
    per_device_eval_batch_size=1,         # Batch size for evaluation per device
    fp16=True,  # Enable mixed precision
    no_cuda=True,  # Use CPU
    num_train_epochs=3,                   # Number of training epochs
    weight_decay=0.01,                    # Weight decay for regularization
    save_total_limit=2,                   # Limit the number of saved checkpoints
    logging_dir='./logs',                 # Directory for logs
    logging_steps=10                      # Log every 10 steps
)

# Step 7: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],  # Replace with validation set if available
    data_collator=data_collator
)

# Step 8: Train the Model
trainer.train()

# Step 9: Save the Fine-Tuned Model and Tokenizer
fine_tuned_model_path = "./trained_gemma-2-2b-it"
model.save_pretrained(fine_tuned_model_path)
tokenizer.save_pretrained(fine_tuned_model_path)

# Step 10: Inference with the Fine-Tuned Model
def run_prompt(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(inputs["input_ids"], max_length=100)  # Adjust max_length as needed
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage of the fine-tuned model
prompt = "Your custom prompt here"
print(run_prompt(prompt))
