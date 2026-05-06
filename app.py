import os, json, warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"
warnings.filterwarnings("ignore")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Abstractive Text Summarizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.block-container { padding: 2rem 3rem 3rem !important; max-width: 1300px; }

/* ── Header ── */
.app-hdr { text-align:center; padding:1.8rem 0 0.6rem; }
.app-ttl { font-size:1.85rem; font-weight:700; color:#1C1C1E; letter-spacing:-0.5px; }
.app-sub { font-size:0.86rem; color:#8E8E93; margin-top:0.3rem; }

/* ── Card ── */
.card {
  background:#fff;
  border-radius:16px;
  padding:1.4rem 1.6rem;
  box-shadow:0 1px 6px rgba(0,0,0,0.06), 0 0 0 1px #EBEBEB;
  margin-bottom:1rem;
}
.card-title {
  font-size:0.67rem; font-weight:600; letter-spacing:0.09em;
  text-transform:uppercase; color:#999; margin-bottom:0.9rem;
}

/* ── Mode Toggle Buttons (two separate st.button calls styled) ── */
div[data-testid="stHorizontalBlock"] > div:first-child div.stButton > button:first-child {
  border-radius:8px 0 0 8px !important;
}
div[data-testid="stHorizontalBlock"] > div:last-child div.stButton > button:first-child {
  border-radius:0 8px 8px 0 !important;
  border-left:1px solid #E0E0E0 !important;
}

/* ── Generic segmented toggle container ── */
.toggle-wrap {
  display:flex; background:#F2F2F7; border-radius:10px;
  padding:3px; margin-bottom:1rem;
}
.toggle-opt {
  flex:1; text-align:center; padding:0.4rem 0.8rem;
  border-radius:8px; font-size:0.83rem; font-weight:500;
  color:#636366; transition:all 0.18s;
}
.toggle-opt.on {
  background:#fff; color:#1C1C1E; font-weight:600;
  box-shadow:0 1px 4px rgba(0,0,0,0.12);
}

/* ── Pipeline ── */
.pipe-row { display:flex; align-items:center; gap:0.85rem;
            padding:0.58rem 0; border-bottom:1px solid #F5F5F5; }
.pipe-row:last-child { border-bottom:none; }
.dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dot.w { background:#D1D1D6; }
.dot.a { background:#007AFF; box-shadow:0 0 0 3px #007AFF22; }
.dot.d { background:#34C759; }
.dot.e { background:#FF3B30; }
.plbl  { font-size:0.85rem; font-weight:500; color:#1C1C1E; flex:1; }
.pnote { font-size:0.75rem; color:#AEAEB2; }
.pnote.blue { color:#007AFF; }
.pnote.grn  { color:#34C759; }
.pnote.red  { color:#FF3B30; }

/* ── Article preview ── */
.art-prev {
  background:#FAFAFA; border-radius:10px; padding:0.85rem 1rem;
  border:1px solid #EBEBEB; font-size:0.84rem; color:#3A3A3C;
  line-height:1.65; max-height:155px; overflow-y:auto; margin-top:0.7rem;
}

/* ── Stat badges ── */
.badge {
  display:inline-block; background:#F4F4F6; border-radius:6px;
  padding:0.2rem 0.6rem; font-size:0.74rem; font-weight:500;
  color:#636366; margin-right:0.35rem; margin-top:0.4rem;
  border:1px solid #EBEBEB; letter-spacing:0.01em;
}

/* ── Summary output card ── */
.sum-card {
  margin-top:0.8rem;
  background:linear-gradient(135deg,#FAFCFF 0%,#F5F8FF 100%);
  border:1px solid #DDE8FF;
  border-radius:14px;
  padding:1.3rem 1.5rem;
  position:relative;
}
.sum-label {
  font-size:0.67rem; font-weight:600; letter-spacing:0.09em;
  text-transform:uppercase; color:#6B8CFF; margin-bottom:0.6rem;
}
.sum-text {
  font-size:1rem; line-height:1.85; color:#1C1C1E;
  font-weight:400;
}
.sum-accent {
  position:absolute; top:0; left:0; width:100%; height:3px;
  background:linear-gradient(90deg,#5B9BFF,#7B6BFF);
  border-radius:14px 14px 0 0;
}

/* ── Streamlit overrides ── */
div[data-baseweb="select"] > div {
  border-radius:10px !important; border-color:#EBEBEB !important;
  font-size:0.87rem !important; }
.stTextArea textarea {
  border-radius:10px !important; border-color:#EBEBEB !important;
  font-family:'Inter',sans-serif !important; font-size:0.9rem !important;
  color:#1C1C1E !important; line-height:1.65 !important; }
.stTextArea textarea:focus {
  border-color:#5B9BFF !important;
  box-shadow:0 0 0 3px rgba(91,155,255,0.12) !important; }

div.stButton > button {
  border-radius:50px !important; font-family:'Inter',sans-serif !important;
  font-weight:600 !important; font-size:0.86rem !important;
  padding:0.48rem 1.3rem !important; border:none !important;
  transition:all 0.18s ease !important; }
div.stButton > button:first-child {
  background:#1C1C1E !important; color:#fff !important;
  box-shadow:0 2px 8px rgba(0,0,0,0.18) !important; }
div.stButton > button:first-child:hover {
  background:#333 !important;
  box-shadow:0 4px 14px rgba(0,0,0,0.28) !important;
  transform:translateY(-1px) !important; }

#MainMenu, footer, header { visibility:hidden; }
</style>
""")

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "summary":       "",
    "stats":         {},
    "phases_saved":  None,
    "input_mode":    "dataset",   # "dataset" or "custom"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

CACHE_FILE    = "articles_cache.json"
FINETUNED_DIR = "./finetuned_t5"

if not os.path.exists(CACHE_FILE):
    st.error("Articles cache not found. Run `python download_assets.py` then restart.")
    st.stop()

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path         = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else "t5-small"
    tok          = AutoTokenizer.from_pretrained(path)
    mdl          = AutoModelForSeq2SeqLM.from_pretrained(path)
    params       = sum(p.numel() for p in mdl.parameters())
    is_finetuned = os.path.isdir(FINETUNED_DIR)
    return tok, mdl, params, is_finetuned

@st.cache_data(show_spinner=False)
def load_articles():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Pipeline helpers ──────────────────────────────────────────────────────────
def pipe_html(steps):
    nc = {"a": "blue", "d": "grn", "e": "red", "w": ""}
    rows = "".join(
        f'<div class="pipe-row">'
        f'<div class="dot {s["st"]}"></div>'
        f'<span class="plbl">{s["label"]}</span>'
        f'<span class="pnote {nc[s["st"]]}">{s.get("note","")}</span>'
        f'</div>'
        for s in steps
    )
    return f'<div class="card"><div class="card-title">Processing Pipeline</div>{rows}</div>'

def fresh_steps(note):
    return [
        {"label": "Model Ready",   "note": note, "st": "d"},
        {"label": "Tokenisation",  "note": "—",  "st": "w"},
        {"label": "Inference",     "note": "—",  "st": "w"},
        {"label": "Summary Ready", "note": "—",  "st": "w"},
    ]

# ── Boot ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    tokenizer, model, num_params, is_finetuned = load_model()

model_label = "Fine-tuned T5" if is_finetuned else "T5-small"
model_note  = f"{model_label} · {num_params/1e6:.0f}M params"
articles    = load_articles()

# ── Header ────────────────────────────────────────────────────────────────────
st.html(
    '<div class="app-hdr">'
    '<div class="app-ttl">Abstractive Text Summarizer</div>'
    '<div class="app-sub">Hugging Face T5 &nbsp;·&nbsp; CNN / DailyMail &nbsp;·&nbsp; NLP Course Project</div>'
    '</div>'
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### Parameters")
    max_len   = st.slider("Max length",   50, 400, 150, 10)
    min_len   = st.slider("Min length",   10, 150,  30, 10)
    num_beams = st.slider("Beam width",    1,  10,   4,  1,
                          help="Higher = better quality, slower inference")
    do_sample = st.checkbox("Creative sampling", False)
    st.markdown("---")
    if is_finetuned:
        st.success("Fine-tuned model active")
    else:
        st.caption(f"Model: t5-small · {num_params/1e6:.0f}M params")
        st.caption("Run `python finetune_model.py` to fine-tune")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_app, tab_docs = st.tabs(["Summarizer", "Documentation"])

# ── DOCS tab ─────────────────────────────────────────────────────────────────
with tab_docs:
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("README.md not found.")

# ── APP tab ───────────────────────────────────────────────────────────────────
with tab_app:
    left, right = st.columns([1, 1], gap="large")

    # ── LEFT ─────────────────────────────────────────────────────────────────
    with left:
        # Input source card header + toggle
        st.html('<div class="card"><div class="card-title">Input Source</div></div>')

        # ── Mode toggle — two real buttons that flip session state ────────────
        is_dataset = st.session_state.input_mode == "dataset"
        t1, t2 = st.columns(2, gap="small")

        with t1:
            dataset_btn = st.button(
                "Dataset Article",
                use_container_width=True,
                type="primary" if is_dataset else "secondary",
            )
        with t2:
            custom_btn = st.button(
                "Custom Text",
                use_container_width=True,
                type="primary" if not is_dataset else "secondary",
            )

        if dataset_btn:
            st.session_state.input_mode = "dataset"
            st.rerun()
        if custom_btn:
            st.session_state.input_mode = "custom"
            st.rerun()

        st.write("")
        input_text = ""

        # ── Dataset mode ────────────────────────────────────────────────────
        if st.session_state.input_mode == "dataset":
            labels = [a["label"] for a in articles]
            idx = st.selectbox(
                "Select article",
                range(len(labels)),
                format_func=lambda i: labels[i],
                label_visibility="collapsed",
            )
            input_text = articles[idx]["text"]
            preview    = input_text[:440] + ("…" if len(input_text) > 440 else "")
            st.html(f'<div class="art-prev">{preview}</div>')

        # ── Custom text mode ─────────────────────────────────────────────────
        else:
            input_text = st.text_area(
                "Source text",
                height=240,
                placeholder="Paste the article or document you want to summarize here…",
                label_visibility="collapsed",
            )

        # Word / char count
        if input_text and input_text.strip():
            w, c = len(input_text.split()), len(input_text)
            st.html(
                f'<span class="badge">{w} words</span>'
                f'<span class="badge">{c} characters</span>'
            )

        st.write("")

        # Action buttons
        c1, c2    = st.columns([3, 1])
        run_btn   = c1.button("Generate Summary", use_container_width=True)
        clear_btn = c2.button("Clear", use_container_width=True)

        if clear_btn:
            st.session_state.summary      = ""
            st.session_state.stats        = {}
            st.session_state.phases_saved = None

    # ── RIGHT ────────────────────────────────────────────────────────────────
    with right:
        pipe_slot = st.empty()
        out_slot  = st.empty()

        # Restore pipeline
        pipe_slot.html(
            pipe_html(st.session_state.phases_saved)
            if st.session_state.phases_saved
            else pipe_html(fresh_steps(model_note))
        )

        # Restore / default summary panel
        def render_summary(text, stats):
            s = stats
            badges = (
                f'<span class="badge">{s.get("time","?")} s elapsed</span>'
                f'<span class="badge">{s.get("ratio","?")}% compressed</span>'
                f'<span class="badge">{s.get("tokens","?")} tokens generated</span>'
            )
            return (
                '<div class="card">'
                '<div class="card-title">Generated Summary</div>'
                f'{badges}'
                '<div class="sum-card">'
                '<div class="sum-accent"></div>'
                '<div class="sum-label">Summary</div>'
                f'<div class="sum-text">{text}</div>'
                '</div>'
                '</div>'
            )

        def render_empty():
            return (
                '<div class="card">'
                '<div class="card-title">Generated Summary</div>'
                '<p style="color:#AEAEB2;font-size:0.9rem;padding:0.7rem 0;line-height:1.7;">'
                'Select an article or paste text, then click <strong style="color:#1C1C1E;">'
                'Generate Summary</strong> to see the abstractive summary here.'
                '</p>'
                '</div>'
            )

        if st.session_state.summary:
            out_slot.html(render_summary(st.session_state.summary, st.session_state.stats))
        else:
            out_slot.html(render_empty())

    # ── Summarisation ─────────────────────────────────────────────────────────
    if run_btn:
        if not input_text or not input_text.strip():
            st.warning("Please provide some text before generating a summary.")
        else:
            steps = fresh_steps(model_note)

            def refresh():
                pipe_slot.html(pipe_html(steps))

            # Tokenise
            steps[1]["st"]   = "a"
            steps[1]["note"] = "Running…"
            refresh()

            t0     = time.time()
            raw    = "summarize: " + input_text
            inputs = tokenizer(raw, return_tensors="pt", max_length=512, truncation=True)
            n_tok  = inputs["input_ids"].shape[1]
            t_tok  = time.time() - t0
            trunc  = " · truncated to 512" if len(tokenizer.encode(raw)) > 512 else ""

            steps[1]["st"]   = "d"
            steps[1]["note"] = f"{n_tok} tokens · {t_tok*1000:.0f} ms{trunc}"
            steps[2]["st"]   = "a"
            steps[2]["note"] = f"beam width {num_beams} · sampling {'on' if do_sample else 'off'}"
            refresh()

            # Inference
            try:
                t1  = time.time()
                ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=do_sample,
                    num_beams=num_beams,
                    early_stopping=True,
                )
                t_inf   = time.time() - t1
                summary = tokenizer.decode(ids[0], skip_special_tokens=True)
                t_total = t_tok + t_inf
                in_w    = len(input_text.split())
                out_w   = len(summary.split())
                ratio   = round((1 - out_w / max(in_w, 1)) * 100, 1)

                steps[2]["st"]   = "d"
                steps[2]["note"] = f"{t_inf:.2f} s · {ids.shape[1]} tokens"
                steps[3]["st"]   = "d"
                steps[3]["note"] = f"{out_w} words · {ratio}% reduction"
                refresh()

                stats = {
                    "time":   f"{t_total:.2f}",
                    "ratio":  ratio,
                    "tokens": ids.shape[1],
                }
                st.session_state.summary      = summary
                st.session_state.phases_saved = steps
                st.session_state.stats        = stats

                out_slot.html(render_summary(summary, stats))

            except Exception as ex:
                steps[2]["st"]   = "e"
                steps[2]["note"] = str(ex)
                refresh()
                st.error(f"Summarization error: {ex}")
