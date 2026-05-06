"""
Fine-tunes T5-small on a small subset of CNN/DailyMail training data.
Run ONCE (takes ~10-30 mins on CPU, ~2-5 mins with GPU):

    python finetune_model.py

The fine-tuned model is saved to ./finetuned_t5/ and automatically
used by app.py if that directory exists.
"""

import os, warnings
os.environ["TOKENIZERS_PARALLELISM"]         = "false"
os.environ["TRANSFORMERS_VERBOSITY"]          = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

SAVE_DIR       = "./finetuned_t5"
BASE_MODEL     = "t5-small"
N_TRAIN        = 500   # training samples from CNN/DailyMail train split
N_EVAL         = 50    # validation samples
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 128
BATCH_SIZE     = 4
EPOCHS         = 3
LR             = 5e-5

print("=" * 55)
print("  T5-small Fine-Tuning on CNN/DailyMail")
print("=" * 55)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

# ── Load tokenizer & model ────────────────────────────────
print(f"\n[1/4] Loading {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
print(f"      OK  {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

# ── Load dataset ──────────────────────────────────────────
print(f"\n[2/4] Loading CNN/DailyMail train split (streaming)...")
train_ds = load_dataset("cnn_dailymail", "3.0.0", split="train",      streaming=True)
val_ds   = load_dataset("cnn_dailymail", "3.0.0", split="validation",  streaming=True)

def collect(ds, n):
    rows = []
    for i, item in enumerate(ds):
        if i >= n: break
        rows.append({"article": item["article"], "highlights": item["highlights"]})
    return rows

print(f"      Collecting {N_TRAIN} training samples...")
train_rows = collect(train_ds, N_TRAIN)
print(f"      Collecting {N_EVAL} validation samples...")
val_rows   = collect(val_ds,   N_EVAL)
print(f"      OK  {len(train_rows)} train / {len(val_rows)} val")

# ── Tokenise ──────────────────────────────────────────────
print(f"\n[3/4] Tokenising...")

def tokenise(rows):
    inputs  = ["summarize: " + r["article"]    for r in rows]
    targets = [r["highlights"]                  for r in rows]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )
    model_inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in lab]
        for lab in labels["input_ids"]
    ]
    return model_inputs

class SumDataset(torch.utils.data.Dataset):
    def __init__(self, encoded):
        self.input_ids      = torch.tensor(encoded["input_ids"])
        self.attention_mask = torch.tensor(encoded["attention_mask"])
        self.labels         = torch.tensor(encoded["labels"])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }

train_dataset = SumDataset(tokenise(train_rows))
val_dataset   = SumDataset(tokenise(val_rows))
print(f"      OK  {len(train_dataset)} train / {len(val_dataset)} val tensors")

# ── Train ─────────────────────────────────────────────────
print(f"\n[4/4] Fine-tuning for {EPOCHS} epoch(s)...")
print(f"      Batch size: {BATCH_SIZE}  |  LR: {LR}  |  Device: {device}")

args = Seq2SeqTrainingArguments(
    output_dir                  = SAVE_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LR,
    warmup_steps                = 50,
    weight_decay                = 0.01,
    predict_with_generate       = True,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    logging_steps               = 20,
    load_best_model_at_end      = True,
    fp16                        = (device == "cuda"),
    report_to                   = "none",
)

trainer = Seq2SeqTrainer(
    model         = model,
    args          = args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
    tokenizer     = tokenizer,
)

trainer.train()

# ── Save ──────────────────────────────────────────────────
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print("\n" + "=" * 55)
print(f"  Fine-tuned model saved -> {SAVE_DIR}/")
print("  Restart app.py — it will auto-load the fine-tuned model.")
print("=" * 55)
