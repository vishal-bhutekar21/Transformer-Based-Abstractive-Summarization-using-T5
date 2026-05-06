# 📄 Abstractive Text Summarization — Complete Project Documentation

## Table of Contents
1. [What This Project Does](#what-this-project-does)
2. [How the App Works — Step by Step](#how-the-app-works)
3. [What Happens When You Summarize Text](#what-happens-when-you-summarize)
4. [Project Q&A (Viva Prep)](#project-qa-viva-prep)
5. [Technology Deep Dive](#technology-deep-dive)
   - [Streamlit](#streamlit)
   - [Transformers](#transformers)
   - [PyTorch (torch)](#pytorch-torch)
   - [TorchVision](#torchvision)
   - [Datasets](#datasets)
5. [Project File Structure](#project-file-structure)
6. [How to Run the Project](#how-to-run)

---

## 1. What This Project Does

This project is a **Natural Language Processing (NLP)** application that performs **Abstractive Text Summarization**. Given a long piece of text (like a news article), the app produces a **shorter, human-like summary** that captures the most important information.

### Extractive vs Abstractive Summarization

| Type | How it works | Example |
|------|-------------|---------|
| **Extractive** | Selects and copies existing sentences from the text | Like highlighting sentences in a textbook |
| **Abstractive** | Generates NEW sentences that paraphrase the content | Like asking a human to explain the article in their own words |

This project uses **abstractive** summarization — meaning the model doesn't just copy-paste sentences, it generates brand new text that captures the meaning, just like a human would.

---

## 2. How the App Works — Step by Step

```
┌─────────────────────────────────────────────────────────┐
│                   App Startup                           │
│  1. app.py starts                                       │
│  2. T5 model loads from local HuggingFace cache        │
│  3. 100 CNN/DailyMail articles load from JSON file     │
│  4. UI renders in browser                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   User Inputs                           │
│  Option A: Pick a dataset article from dropdown        │
│  Option B: Paste custom text into the text area        │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Click "Summarize →"                        │
│                                                         │
│  Phase 1: Tokenisation                                  │
│  Phase 2: T5 Inference (the heavy NLP work)            │
│  Phase 3: Decoding to human text                        │
│  Phase 4: Display summary with stats                   │
└─────────────────────────────────────────────────────────┘
```

### The CNN/DailyMail Dataset Integration

The app mirrors the original Jupyter notebook exactly. In the notebook, the code was:

```python
article_index = 11
dataset_article = dataset[article_index]['article']
```

In the app, this becomes a **dropdown selector** showing all 100 articles from the test split, labeled with the first few words so you know what each article is about. Selecting an article is the same as changing that `article_index` number.

---

## 3. What Happens When You Summarize Text

When you type or paste text and click **Summarize →**, here is **exactly** what happens inside the code, step by step:

### Step 1 — Input Preparation
```
Your text: "The Palestinian Authority officially became
            the 123rd member of the International Criminal Court..."

Prefix added: "summarize: The Palestinian Authority officially became..."
```
The T5 model was trained to understand tasks via prefixes. Adding `"summarize: "` before your text tells the model **what job to do**. Without this prefix, it wouldn't know whether to translate, classify, or summarize.

---

### Step 2 — Tokenisation (Phase 2 in the Pipeline)

```
Human text  →  Tokenizer  →  Numbers (Token IDs)

"The Palestinian"  →  [37, 7459]
"Authority"        →  [1178]
"officially"       →  [1059]
```

**What is a token?**
A token is a piece of a word. The T5 tokenizer uses a method called **SentencePiece** (a subword tokenizer). This means:
- Common words = 1 token ("the", "is", "and")
- Rare words = multiple tokens ("Palestinian" → "Palestin" + "ian")
- The entire text becomes a list of numbers between 0 and 32,127 (t5-small's vocabulary size)

**Why the 512 token limit?**
The T5-small model can only process up to 512 tokens at once (this is its "context window"). If your text is longer, it gets **truncated** — meaning the end is cut off. The pipeline shows a "truncated" warning if this happens.

```python
inputs = tokenizer(
    "summarize: " + input_text,
    return_tensors="pt",   # "pt" means PyTorch tensors
    max_length=512,
    truncation=True        # cut off if longer than 512 tokens
)
```

**Output of tokenisation:**
- `input_ids` — the list of token numbers, shape: `[1, 312]` (1 sentence, 312 tokens)
- `attention_mask` — a mask of 1s and 0s telling the model which tokens are real vs padding

---

### Step 3 — T5 Model Inference (Phase 3 — the core NLP)

This is the most complex step. The T5 model is an **Encoder-Decoder Transformer**:

```
input_ids (numbers)
       │
       ▼
┌─────────────┐
│   ENCODER   │  ← Reads and understands your full text
│  (6 layers) │    Creates a rich mathematical representation
└─────────────┘    called "hidden states" (vectors of meaning)
       │
       ▼
┌─────────────┐
│   DECODER   │  ← Generates the summary word by word
│  (6 layers) │    At each step, picks the most likely next word
└─────────────┘    using Beam Search
       │
       ▼
summary_ids (token numbers for the summary)
```

**What is Beam Search?**
Normal word selection ("greedy search") always picks the single most probable next word. Beam Search tracks multiple possible continuations at once (controlled by the `num_beams` slider):

```
Beam width = 4 means:
  At each step, keep the 4 most promising partial summaries,
  not just the single best one.
  
  After generating all words, pick the sequence with the
  highest overall probability.
```

Higher beam width = better quality summaries, but slower generation.

**The generation call:**
```python
summary_ids = model.generate(
    inputs["input_ids"],           # the tokenised input
    attention_mask=inputs["attention_mask"],
    max_length=150,                # stop generating after 150 tokens
    min_length=30,                 # force at least 30 tokens
    do_sample=False,               # False = deterministic (Beam Search)
                                   # True  = probabilistic (Creative mode)
    num_beams=4,                   # beam search width
    early_stopping=True            # stop when all beams hit [EOS] token
)
```

---

### Step 4 — Decoding (Phase 4)

The model outputs a list of token IDs (numbers). The tokenizer converts them back to human-readable text:

```
[3, 1178, 7, 1059, 1]  →  "the authority officially became"
```

```python
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# skip_special_tokens=True removes <pad>, </s>, <unk> etc.
```

---

### Step 5 — Stats Calculation & Display

The app then calculates:
- **Time taken** = tokenisation time + inference time
- **Compression ratio** = `(1 - output_words / input_words) × 100%`
  - e.g., 620 words input → 38 words output = **93.9% compression**
- **Token count** = number of tokens in the output

The summary appears in the blue-tinted output box with these stats shown as pills above it.

---

## 4. Project Q&A (Viva Prep)

A detailed list of Frequently Asked Questions (FAQs) and technical answers for your project presentation can be found here:
👉 **[Project_QA.md](./Project_QA.md)**

This document covers:
- Core model architecture (T5).
- Abstractive vs Extractive summarization differences.
- Technical implementation of the pipeline.
- How common errors (Connection, Speed) were solved.

---

## 5. Technology Deep Dive

---

### Streamlit

**What it is:**
Streamlit is a Python library that lets you build interactive web applications **purely in Python** — no HTML, no JavaScript needed. Every time a user interacts with the app (clicks a button, moves a slider), Streamlit re-runs your Python script from top to bottom.

**Key concepts used in this project:**

| Streamlit Feature | What it does in this app |
|---|---|
| `st.set_page_config()` | Sets the browser tab title, icon, and layout |
| `st.html()` | Injects custom HTML/CSS for the iOS white theme |
| `@st.cache_resource` | Loads the T5 model ONCE and keeps it in memory across all reruns |
| `@st.cache_data` | Loads the article JSON file ONCE and caches it |
| `st.columns()` | Creates the side-by-side Input (left) and Output (right) layout |
| `st.empty()` | Creates placeholder slots that can be updated live during processing |
| `st.radio()` | The Dataset / Custom Text toggle |
| `st.selectbox()` | The article dropdown |
| `st.slider()` | The max/min length and beam width controls in the sidebar |
| `st.spinner()` | The "Loading model..." animation during startup |
| `st.session_state` | Persists the summary and pipeline state across Streamlit reruns |

**How `@st.cache_resource` prevents crashes:**
Without caching, every time you click a button, Streamlit would re-download and re-load the 240MB T5 model from disk. With `@st.cache_resource`, the model is loaded once on first run and stored in memory permanently, making all subsequent interactions fast.

**How live pipeline updates work:**
The pipeline phase cards update in real-time because of `st.empty()` placeholders:
```python
pipe_slot = st.empty()          # create a blank placeholder
pipe_slot.html(render_phase_1)  # fill it with phase 1 content
# ... run tokenisation ...
pipe_slot.html(render_phase_2)  # replace with phase 2 content (live update!)
```

---

### Transformers

**What it is:**
The `transformers` library by Hugging Face is the industry-standard library for working with pre-trained NLP models. It provides access to thousands of models (BERT, GPT, T5, BART, etc.) through a unified API.

**The T5 Model:**
T5 stands for **Text-To-Text Transfer Transformer**. Google Research released it in 2019. Its key innovation:

> Every NLP task is reformulated as a text-to-text problem.

| Task | Input format | Output |
|------|-------------|--------|
| Summarization | `"summarize: [article]"` | `"The key points were…"` |
| Translation | `"translate English to French: Hello"` | `"Bonjour"` |
| Question answering | `"question: What is... context: ..."` | `"The answer is…"` |
| Classification | `"classify: I hate this movie"` | `"negative"` |

**Classes used in this app:**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# AutoTokenizer — automatically selects the right tokenizer for t5-small
# (SentencePiece in this case)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# AutoModelForSeq2SeqLM — loads T5's encoder-decoder architecture
# "Seq2Seq" = sequence to sequence (input sequence → output sequence)
# "LM" = Language Model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

**T5-small model size:**
- Parameters: **60.5 million**
- Encoder layers: 6
- Decoder layers: 6
- Hidden dimension: 512
- Attention heads: 8
- Vocabulary size: 32,128 tokens

**Fine-tuning (`finetune_model.py`):**
Fine-tuning means taking the pre-trained T5-small (already trained by Google on 750GB of text) and continuing training it specifically on CNN/DailyMail data so it gets better at news summarization. The `Seq2SeqTrainer` handles:
- Forward pass (feeding data through the model)
- Loss calculation (how wrong was the model's summary vs the reference?)
- Backpropagation (adjusting model weights to reduce the error)
- Checkpoint saving after each epoch

---

### PyTorch (torch)

**What it is:**
PyTorch is a deep learning framework developed by Meta (Facebook). It provides:
1. **Tensors** — multi-dimensional arrays optimised for GPU computation (like NumPy but for neural networks)
2. **Autograd** — automatic differentiation for backpropagation
3. **Neural network layers** — the building blocks that make up T5

**What a Tensor is:**
```
A scalar:       42
A vector:       [1, 2, 3]
A matrix:       [[1,2], [3,4]]
A 3D tensor:    shape [batch, sequence, hidden_dim]
                e.g., [1, 312, 512]
                = 1 article, 312 tokens, 512-dimensional meaning vector
```

**How PyTorch is used in this project:**
```python
# The tokenizer returns PyTorch tensors (return_tensors="pt")
inputs = tokenizer(text, return_tensors="pt")
# inputs["input_ids"] is a tensor of shape [1, 312]

# The model runs its forward pass (encoder + decoder)
summary_ids = model.generate(inputs["input_ids"], ...)
# summary_ids is a tensor of shape [1, 45] 
# (1 article, 45 output tokens)
```

**GPU vs CPU:**
If you have an NVIDIA GPU, PyTorch automatically uses CUDA to run matrix multiplications on the GPU — making inference ~10-50x faster. On CPU only, T5-small inference takes 3-10 seconds per article. The fine-tuning script detects this automatically:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

### TorchVision

**What it is:**
TorchVision is PyTorch's companion library for computer vision tasks. It provides:
- Pre-built datasets (ImageNet, CIFAR-10, etc.)
- Image transformation utilities
- Pre-trained vision models (ResNet, VGG, etc.)

**Why it's required here:**
TorchVision is NOT directly used in the summarization code itself. However, some sub-modules of the `transformers` library (specifically models that handle images + text, like ViT, ViTMatte, YOLOS) **import TorchVision internally**. When Streamlit scans the `transformers` package directory on startup to watch for file changes, it triggers these imports and throws `ModuleNotFoundError: No module named 'torchvision'` — crashing the app before it even starts.

Installing `torchvision` satisfies that dependency and the app starts cleanly.

---

### Datasets

**What it is:**
The `datasets` library by Hugging Face provides easy access to thousands of public machine learning datasets. It handles:
- Downloading datasets from the HuggingFace Hub
- Caching data locally after the first download
- Streaming mode (fetching only what you need without downloading everything)

**The CNN/DailyMail Dataset:**
The `cnn_dailymail` dataset version `3.0.0` is a widely used benchmark for text summarization:

| Split | Size | Used For |
|-------|------|----------|
| `train` | 287,113 articles | Fine-tuning the model |
| `validation` | 13,368 articles | Evaluating during training |
| `test` | 11,490 articles | The dropdown in the app (100 cached) |

Each record contains:
- `article` — the full news article text
- `highlights` — human-written bullet-point summary (used as training labels)

**Why streaming mode is critical:**
```python
# Without streaming — downloads the ENTIRE 800MB dataset first:
ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
# ← this blocks the thread for minutes → Streamlit timeout → connection error!

# With streaming — fetches rows one by one on demand:
ds = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
# ← fetching 100 rows takes ~5 seconds, no timeout
```

**How articles are cached locally:**
The `download_assets.py` and inline commands fetch 100 articles using streaming mode and save them to `articles_cache.json`. When the app starts, it reads this local file instantly (milliseconds) with zero network calls, completely eliminating the connection error.

---

## 5. Project File Structure

```
nlp project/
│
├── app.py                    ← Main Streamlit application
├── download_assets.py        ← One-time script to pre-cache data
├── finetune_model.py         ← Fine-tunes T5 on CNN/DailyMail
├── articles_cache.json       ← 100 CNN/DailyMail articles (local cache)
├── requirements.txt          ← All Python dependencies
├── verify_cache.py           ← Quick check to confirm cache is valid
│
├── .streamlit/
│   └── config.toml           ← Streamlit theme & server settings
│
├── finetuned_t5/             ← Created after running finetune_model.py
│   ├── config.json
│   ├── tokenizer_config.json
│   └── model weights...
│
└── AbstractiveTextSummarization.ipynb  ← Original Colab notebook
```

---

## 6. How to Run the Project

### First-time setup (run once)
```bash
# Install all dependencies
pip install -r requirements.txt

# Pre-cache articles (avoids connection errors in the app)
# Already done if articles_cache.json exists with 100 entries
python download_assets.py
```

### Run the app
```bash
python -m streamlit run app.py
```
Opens at: **http://localhost:8501**

### Optional: Fine-tune the model (improves summary quality)
```bash
# Takes 20-40 min on CPU, ~5 min on GPU
python finetune_model.py

# Then restart the app — it auto-detects and loads the fine-tuned model
python -m streamlit run app.py
```

### Adjust parameters (sidebar)
| Parameter | Effect |
|-----------|--------|
| Max length (50–400) | How long the summary can be |
| Min length (10–150) | Forces a minimum summary length |
| Beam width (1–10) | Higher = better quality, slower |
| Creative sampling | Enables probabilistic word selection for variety |
