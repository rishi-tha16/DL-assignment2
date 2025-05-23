# -*- coding: utf-8 -*-
"""DL Assignment-2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1FXKxSVGQoWuY9n_QDKuZjOs-oGr05UPj
"""

#Question2
import os
os.environ["WANDB_DISABLED"] = "true"  # Prevents W&B from asking for API key

# === STEP 1: Import Libraries ===
import pandas as pd
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    pipeline
)

# === STEP 2: Load Excel and Extract Lyrics Column ===
df = pd.read_excel("/content/sample_data/Backstreet_Boys_Lyrics_score.xlsx ")

# Auto-detect column with 'lyric' in the name
lyrics_column = None
for col in df.columns:
    if "lyric" in col.lower():
        lyrics_column = col
        break

if not lyrics_column:
    raise ValueError("Could not find a lyrics column in the Excel file.")

# Join lyrics into one big string
lyrics_text = "\n\n".join(df[lyrics_column].dropna().astype(str).tolist())

# Save to plain text file
with open("lyrics.txt", "w", encoding="utf-8") as f:
    f.write(lyrics_text)

# === STEP 3: Load Dataset and Tokenizer ===
dataset = load_dataset("text", data_files={"train": "lyrics.txt"})

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# === STEP 4: Load GPT-2 Model ===
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Support pad token

# === STEP 5: Training Arguments (No W&B logging) ===
training_args = TrainingArguments(
    output_dir="./gpt2-lyrics",
    overwrite_output_dir=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=10,
    report_to="none"  # Don't report to any logger (W&B, TensorBoard, etc.)
)

# === STEP 6: Trainer ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# === STEP 7: Train the Model ===
trainer.train()

# === STEP 8: Save Model and Tokenizer ===
trainer.save_model("./gpt2-lyrics")
tokenizer.save_pretrained("./gpt2-lyrics")

# === STEP 9: Generate Sample Lyrics ===
generator = pipeline("text-generation", model="./gpt2-lyrics", tokenizer=tokenizer)

prompt = "I want it that way"
outputs = generator(prompt, max_length=100, num_return_sequences=1)

print("\n🎤 Generated Lyrics:\n")
print(outputs[0]["generated_text"])



