# Question-2
# 🎵 GPT-2 Fine-Tuning on Backstreet Boys Lyrics

This project fine-tunes OpenAI's GPT-2 language model on a dataset of Backstreet Boys lyrics using Hugging Face Transformers. The goal is to train the model to generate song lyrics similar in style and tone to the band’s original lyrics.


## 📌 Key Highlights

- **Model Used**: GPT-2 (`gpt2` from Hugging Face).
- **Dataset**: Excel file containing lyrics (extracted automatically).
- **Training Framework**: Hugging Face `Trainer`.
- **Tokenization**: GPT-2 tokenizer with padding and truncation.
- **Collation Strategy**: Causal Language Modeling (no MLM).
- **Sample Generation**: Uses Hugging Face `pipeline` for text generation.


## 🔍 Step-by-Step Overview

### 1. **Dependencies and Setup**
- Uses `transformers`, `datasets`, `pandas`, and `torch`.
- Disables Weights & Biases logging by setting `WANDB_DISABLED`.

### 2. **Load and Preprocess Data**
- Reads an Excel file and automatically detects the lyrics column.
- Merges all lyrics into a single `.txt` file for training.

### 3. **Dataset and Tokenizer**
- Loads the text file as a Hugging Face dataset.
- Applies GPT-2 tokenization (max length 512, padded and truncated).

### 4. **Model Loading**
- Loads the pre-trained GPT-2 model and adapts it for the new dataset.

### 5. **Training Configuration**
- 3 training epochs  
- Batch size: 2 per device  
- Learning rate: `2e-5`  
- Weight decay: `0.01`  
- No logging (W&B or TensorBoard)

### 6. **Training**
- Uses `Trainer` API with a `DataCollatorForLanguageModeling` (causal LM setting).
- Trains on the tokenized dataset.

### 7. **Saving the Model**
- Saves the fine-tuned model and tokenizer to `./gpt2-lyrics`.

### 8. **Generating Lyrics**
- Loads the fine-tuned model via `pipeline`.
- Generates lyrics using a custom prompt like:
  ```
  "I want it that way"
  ```

---


##  Output

```
Prompt: I want it that way

Generated Lyrics:
I want it that way, and I need you every night
I can't explain the things you do
You make me feel alright...
```

---
