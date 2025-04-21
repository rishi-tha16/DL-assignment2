# Question-1
# 📝 Latin to Devanagari Transliteration using LSTM (Seq2Seq)

This project builds a Sequence-to-Sequence (Seq2Seq) model using LSTM layers in TensorFlow/Keras to transliterate Latin script (English alphabet) into Devanagari script (Hindi).

---
## 📌 Key Highlights

- **Model Type**: Sequence-to-Sequence with LSTM encoder-decoder.
- **Input Language**: Latin (Romanized Hindi).
- **Target Language**: Devanagari (Hindi).
- **Embedding**: Trainable embeddings for both input and output sequences.
- **Loss Function**: Sparse categorical cross-entropy.
- **Decoder Training**: Teacher forcing used during training.

---

## 🔍 Step-by-Step Overview

### 1. **Load and Preprocess Data**
- Loads `hi.translit.sampled.train.tsv` (Latin ⇄ Devanagari pairs).
- Formats inputs and appends `\t` and `\n` to target texts to denote start/end.

### 2. **Vocabulary Creation**
- Character-level tokenization for both input and output sequences.
- Mapping created for:
  - `char → index`
  - `index → char`

### 3. **Tokenization and Padding**
- Converts text to sequences of integers.
- Pads sequences to match maximum sequence length.

### 4. **Model Architecture**

#### Encoder:
- Input layer → Embedding → LSTM (returns states).

#### Decoder:
- Input layer → Embedding → LSTM (takes encoder states as initial state) → Dense (softmax over output tokens).

### 5. **Training**
- Compiled with Adam optimizer and sparse categorical cross-entropy.
- Trained for 50 epochs with batch size 64 and 20% validation split.

### 6. **Inference Models**
- Encoder model reused to get latent states from input.
- Decoder model created to:
  - Accept previous state and previous token
  - Predict next token
  - Loop until `\n` or max sequence length

### 7. **Interactive Inference**
- Takes user input word in Latin script.
- Transliterates it to Devanagari using the trained model.

---



interaction:
```
Enter a Latin word (or type 'exit' to quit): namaste
Predicted Devanagari: नमस्ते
```

---

## 🧠 Model Summary

- **Embedding Dimension**: 128  
- **Latent Dimension (LSTM units)**: 256  
- **Training Epochs**: 50  
- **Loss Function**: SparseCategoricalCrossentropy  
- **Batch Size**: 64  

---

## 📚 Dataset Format

The dataset should be a tab-separated file (`.tsv`) with no headers:

```
देवनागरी	wordinlatin
...

Example:
हिन्दी	hindi
भारत	bharat
```

---




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
