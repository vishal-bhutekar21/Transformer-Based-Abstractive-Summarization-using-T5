import os, json, sys
os.environ["TOKENIZERS_PARALLELISM"]         = "false"
os.environ["TRANSFORMERS_VERBOSITY"]          = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["DATASETS_VERBOSITY"]              = "error"

import warnings
warnings.filterwarnings("ignore")

# Redirect stderr to suppress any remaining warnings from HF libraries
import io
_old_stderr = sys.stderr
sys.stderr = io.StringIO()

print("=" * 50)
print("  NLP Project - Pre-download Script")
print("=" * 50)

# ── Step 1: T5-small ─────────────────────────────────
print("\n[1/2] Loading T5-small (from cache or download)...")
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model     = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
params    = sum(p.numel() for p in model.parameters())
print(f"      Model ready: {params/1e6:.1f}M parameters")

# ── Step 2: CNN/DailyMail articles ───────────────────
CACHE_FILE = "articles_cache.json"
N          = 30

print(f"\n[2/2] Fetching {N} CNN/DailyMail articles (streaming)...")
from datasets import load_dataset

ds = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)

arts = []
for i, item in enumerate(ds):
    if i >= N:
        break
    txt   = item["article"]
    label = "Article #{:02d}  -  {}...".format(i, " ".join(txt.split()[:7]))
    arts.append({"index": i, "label": label, "text": txt})
    # restore stderr briefly to print progress
    sys.stderr = _old_stderr
    print(f"      Fetching... {i+1}/{N}", end="\r")
    sys.stderr = io.StringIO()

sys.stderr = _old_stderr
print(f"\n      Done: {N} articles fetched")

with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(arts, f, ensure_ascii=False, indent=2)

print(f"      Saved -> {CACHE_FILE}")
print("\n" + "=" * 50)
print("  Ready! Now start the app:")
print("  python -m streamlit run app.py")
print("=" * 50)
