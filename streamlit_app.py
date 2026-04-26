# ============================================================
#  Research Paper Explainer Bot  —  Version 2.0
#  Stack : Streamlit · FAISS · SentenceTransformers · Groq
#          langdetect · Plotly · NumPy
#  New   : Multilingual analysis, Citation Extractor,
#           Keyword Cloud, LaTeX math, Language Dashboard
#  Run   : streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re
import json
import time
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from groq import Groq
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter

# ─────────────────────────────────────────────
#  OPTIONAL: langdetect (graceful fallback)
# ─────────────────────────────────────────────
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ─────────────────────────────────────────────
#  LANGUAGE METADATA
# ─────────────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "ar": "Arabic", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "it": "Italian", "nl": "Dutch",
    "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese", "th": "Thai",
    "id": "Indonesian", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "cs": "Czech", "sk": "Slovak", "ro": "Romanian",
    "hu": "Hungarian", "uk": "Ukrainian", "bg": "Bulgarian",
}
LANG_FLAGS = {
    "en": "🇬🇧", "hi": "🇮🇳", "zh-cn": "🇨🇳", "zh-tw": "🇹🇼",
    "ar": "🇸🇦", "fr": "🇫🇷", "de": "🇩🇪", "es": "🇪🇸",
    "pt": "🇧🇷", "ru": "🇷🇺", "ja": "🇯🇵", "ko": "🇰🇷",
    "it": "🇮🇹", "nl": "🇳🇱", "pl": "🇵🇱", "tr": "🇹🇷",
}

# ─────────────────────────────────────────────
#  DEFAULT CONFIGURATION
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "chunk_size":      500,
    "chunk_overlap":   50,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
    "top_k":           3,
    "llm_model":       "llama-3.1-8b-instant",
    "max_tokens":      600,
    "temperature":     0.7,
    "answer_language": "auto",
}

# ─────────────────────────────────────────────
#  PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Research Paper Explainer Bot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  —  refined dark scholarly theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
    --ink:       #0D0D14;
    --ink2:      #181824;
    --ink3:      #22223A;
    --border:    #2E2E50;
    --accent:    #7B6FFF;
    --accent2:   #38BDF8;
    --accent3:   #34D399;
    --warn:      #FB923C;
    --muted:     #7878A0;
    --text:      #E2E2F0;
    --text-dim:  #9090B8;
    --radius:    14px;
    --font-head: 'DM Serif Display', serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* ── Global resets ── */
html, body, [class*="css"] { font-family: var(--font-body); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--ink2); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── App header ── */
.app-hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.app-title {
    font-family: var(--font-head);
    font-size: 3rem;
    font-style: italic;
    background: linear-gradient(135deg, #7B6FFF 0%, #38BDF8 50%, #34D399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: .4rem;
}
.app-sub {
    color: var(--text-dim);
    font-size: .95rem;
    font-weight: 300;
    letter-spacing: .03em;
}

/* ── Tag pills ── */
.tag-row { text-align: center; margin: .8rem 0 1.6rem; }
.tag {
    display: inline-block;
    background: var(--ink3);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: .2rem .85rem;
    font-size: .75rem;
    color: var(--text-dim);
    margin: .2rem .15rem;
    font-family: var(--font-mono);
    letter-spacing: .02em;
}
.tag-new {
    background: linear-gradient(135deg,#7B6FFF22,#38BDF822);
    border-color: #7B6FFF66;
    color: var(--accent2);
}

/* ── Cards ── */
.card {
    background: var(--ink2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
}
.metric-card {
    background: linear-gradient(135deg, var(--ink2), var(--ink3));
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.2rem;
    text-align: center;
    transition: border-color .2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-value { font-size: 2rem; font-weight: 600; color: var(--accent); font-family: var(--font-mono); }
.metric-label { font-size: .75rem; color: var(--muted); margin-top: .2rem; text-transform: uppercase; letter-spacing: .06em; }

/* ── Language badge ── */
.lang-badge {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: linear-gradient(135deg,#34D39922,#38BDF822);
    border: 1px solid #34D39966;
    border-radius: 30px;
    padding: .35rem 1rem;
    font-size: .88rem;
    color: var(--accent3);
    font-weight: 500;
}
.lang-badge-unknown {
    background: linear-gradient(135deg,#FB923C22,#FCD34D22);
    border-color: #FB923C66;
    color: var(--warn);
}

/* ── Chat bubbles ── */
.user-bubble {
    background: linear-gradient(135deg, #7B6FFF, #9B8FFF);
    color: #fff;
    padding: .9rem 1.3rem;
    border-radius: 18px 18px 4px 18px;
    margin: .7rem 0 .7rem 12%;
    box-shadow: 0 6px 20px rgba(123,111,255,.25);
    line-height: 1.7;
    font-size: .93rem;
}
.bot-bubble {
    background: var(--ink2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: .9rem 1.3rem;
    border-radius: 18px 18px 18px 4px;
    margin: .7rem 12% .7rem 0;
    box-shadow: 0 6px 20px rgba(0,0,0,.3);
    line-height: 1.7;
    font-size: .93rem;
}
.bot-bubble code { font-family: var(--font-mono); background: var(--ink3); padding: .1rem .4rem; border-radius: 4px; font-size: .85em; }

/* ── Chunk card ── */
.chunk-card {
    background: var(--ink);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: .9rem 1.1rem;
    margin: .5rem 0;
    font-size: .86rem;
    color: var(--text-dim);
    line-height: 1.75;
    font-family: var(--font-body);
}

/* ── Math block ── */
.math-block {
    background: var(--ink);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-family: var(--font-mono);
    font-size: .88rem;
    color: var(--accent2);
    margin: .8rem 0;
    white-space: pre-wrap;
    line-height: 1.9;
}

/* ── Step badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff;
    border-radius: 50%;
    width: 30px; height: 30px;
    font-weight: 700;
    font-size: .82rem;
    margin-right: .7rem;
    flex-shrink: 0;
}

/* ── Keyword pill ── */
.kw-pill {
    display: inline-block;
    background: var(--ink3);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: .3rem .9rem;
    font-size: .8rem;
    color: var(--accent2);
    margin: .2rem;
    font-family: var(--font-mono);
    cursor: default;
    transition: background .2s, border-color .2s;
}
.kw-pill:hover { background: #38BDF822; border-color: var(--accent2); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--ink);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: var(--font-head);
    font-size: 1.3rem;
    color: var(--text);
}

/* ── Citation card ── */
.citation-card {
    background: var(--ink2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: .85rem 1.1rem;
    margin: .4rem 0;
    font-size: .84rem;
    color: var(--text-dim);
    border-left: 3px solid var(--accent3);
    line-height: 1.6;
}
.citation-num {
    display: inline-block;
    background: var(--accent3);
    color: #000;
    border-radius: 4px;
    padding: .05rem .5rem;
    font-size: .75rem;
    font-weight: 700;
    margin-right: .5rem;
    font-family: var(--font-mono);
}

/* ── Progress dots ── */
.lang-bar {
    display: flex;
    align-items: center;
    gap: .5rem;
    margin: .3rem 0;
}
.lang-bar-fill {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: var(--muted);
    font-size: .78rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  UTILITY: LANGUAGE DETECTION
# ══════════════════════════════════════════════
def detect_language(text: str) -> Tuple[str, str, float]:
    """Returns (lang_code, lang_name, confidence)."""
    sample = text[:3000]
    if LANGDETECT_AVAILABLE:
        try:
            langs = detect_langs(sample)
            top = langs[0]
            code = top.lang
            name = LANGUAGE_NAMES.get(code, code.upper())
            return code, name, round(top.prob, 3)
        except Exception:
            pass
    # Heuristic fallback: check for CJK / Devanagari / Arabic ranges
    arabic_count   = len(re.findall(r'[\u0600-\u06FF]', sample))
    devanagari_cnt = len(re.findall(r'[\u0900-\u097F]', sample))
    cjk_count      = len(re.findall(r'[\u4E00-\u9FFF\u3040-\u30FF]', sample))
    total = max(len(sample), 1)
    if arabic_count / total > 0.05:
        return "ar", "Arabic", 0.8
    if devanagari_cnt / total > 0.05:
        return "hi", "Hindi", 0.8
    if cjk_count / total > 0.05:
        return "zh-cn", "Chinese", 0.8
    return "en", "English", 0.9


def extract_keywords(text: str, top_n: int = 25) -> List[Tuple[str, int]]:
    """Simple TF-based keyword extraction (no external lib needed)."""
    STOPWORDS = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "by","from","is","are","was","were","be","been","being","have","has",
        "had","do","does","did","will","would","could","should","may","might",
        "shall","can","this","that","these","those","i","we","you","he","she",
        "it","they","our","your","their","its","we","us","as","also","which",
        "who","whom","whose","when","where","why","how","all","any","both",
        "each","few","more","most","other","some","such","no","not","only",
        "same","so","than","too","very","just","over","under","again","further",
        "then","once","paper","study","research","show","result","using","used",
    }
    tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    freq   = Counter(t for t in tokens if t not in STOPWORDS)
    return freq.most_common(top_n)


def extract_citations(text: str) -> List[str]:
    """Pull reference-section lines from PDF text."""
    # Find references/bibliography section
    ref_match = re.search(
        r'(?:references|bibliography|works cited)\s*\n([\s\S]{100,})',
        text,
        re.IGNORECASE,
    )
    if not ref_match:
        return []

    ref_block = ref_match.group(1)[:8000]
    # Split on numbered entries like [1], (1), 1.
    entries = re.split(r'\n(?=\[\d+\]|\(\d+\)|\d+\.\s+[A-Z])', ref_block)
    cleaned = []
    for e in entries[:40]:
        e = e.strip()
        if len(e) > 30:
            cleaned.append(e)
    return cleaned


# ══════════════════════════════════════════════
#  RAG BACKEND CLASS
# ══════════════════════════════════════════════
class ResearchPaperRAG:
    """
    Full RAG pipeline built from scratch.
    Steps: PDF → Chunk → Embed → FAISS index → Retrieve → Generate
    Supports multilingual papers and cross-lingual Q&A.
    """

    def __init__(self, pdf_bytes: bytes, config: Dict, groq_api_key: str):
        self.config = config
        self.client = Groq(api_key=groq_api_key)

        # Pipeline
        self.pdf_text    = self._load_pdf(pdf_bytes)
        self.chunks      = self._chunk_text(self.pdf_text,
                                            config["chunk_size"],
                                            config["chunk_overlap"])
        self.embed_model = SentenceTransformer(config["embedding_model"])
        self.embeddings  = self.embed_model.encode(
            self.chunks, show_progress_bar=False, normalize_embeddings=True
        )
        self.index       = self._build_index(self.embeddings)

        # Language metadata (computed once)
        self.lang_code, self.lang_name, self.lang_conf = detect_language(self.pdf_text)
        self.keywords    = extract_keywords(self.pdf_text)
        self.citations   = extract_citations(self.pdf_text)

    # ── Step 1: PDF extraction ──
    def _load_pdf(self, pdf_bytes: bytes) -> str:
        reader = PyPDF2.PdfReader(pdf_bytes)
        pages  = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return re.sub(r"\s+", " ", " ".join(pages)).strip()

    # ── Step 2: Sliding-window chunking ──
    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = text[start: start + size].strip()
            if chunk:
                chunks.append(chunk)
            start += size - overlap
        return chunks

    # ── Step 3: FAISS with cosine similarity (normalized vectors) ──
    def _build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        IndexFlatIP = Inner Product on normalized vectors = Cosine Similarity.
        cos(q, c) = (q · c) / (||q|| · ||c||)  = q · c  when both are unit vectors.
        """
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype("float32"))
        return index

    # ── Step 4: Retrieve top-k with cosine similarity ──
    def retrieve(self, query: str) -> List[Dict]:
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, self.config["top_k"])
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0:
                continue
            results.append({
                "rank":       rank + 1,
                "chunk_id":   int(idx),
                "text":       self.chunks[idx],
                "cosine_sim": float(score),
                "length":     len(self.chunks[idx]),
            })
        return results

    # ── Step 5: LLM generation with language awareness ──
    def generate(self, query: str, chunks: List[Dict],
                 answer_lang: str = "auto") -> Dict:
        context = "\n\n".join(
            f"[Chunk {c['chunk_id']+1}]:\n{c['text']}" for c in chunks
        )

        lang_instruction = ""
        if answer_lang == "auto":
            lang_instruction = (
                "Detect the language of the question and respond in that SAME language. "
                f"The paper appears to be in {self.lang_name}. "
                "You may quote from the paper in its original language when relevant."
            )
        elif answer_lang != "en":
            lang_name = LANGUAGE_NAMES.get(answer_lang, answer_lang)
            lang_instruction = f"Always respond in {lang_name}, regardless of the question language."

        system_prompt = (
            "You are an expert multilingual research assistant. "
            "Answer questions using ONLY the provided context. "
            "Cite chunk numbers when relevant (e.g., 'As stated in Chunk 3…'). "
            "If the context lacks enough information, say so clearly. "
            + lang_instruction
        )

        user_prompt = (
            f"Research paper language: {self.lang_name}\n\n"
            f"Context from paper:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Provide a clear, structured answer:"
        )

        try:
            response = self.client.chat.completions.create(
                model       = self.config["llm_model"],
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens  = self.config["max_tokens"],
                temperature = self.config["temperature"],
            )
            return {
                "answer":  response.choices[0].message.content,
                "success": True,
                "chunks":  chunks,
            }
        except Exception as e:
            return {
                "answer":  f"⚠️ LLM error: {e}",
                "success": False,
                "chunks":  chunks,
            }

    def ask(self, query: str, answer_lang: str = "auto") -> Dict:
        chunks = self.retrieve(query)
        return self.generate(query, chunks, answer_lang)

    def translate_chunk(self, chunk_text: str, target_lang: str) -> str:
        """Use the LLM to translate a chunk to the target language."""
        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        try:
            response = self.client.chat.completions.create(
                model    = self.config["llm_model"],
                messages = [
                    {"role": "system", "content": f"You are a professional translator. Translate the given text to {lang_name} accurately. Return only the translation, no extra commentary."},
                    {"role": "user",   "content": chunk_text},
                ],
                max_tokens  = 600,
                temperature = 0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Translation failed: {e}"

    def summarise(self, language: str = "auto") -> str:
        sample = self.pdf_text[:3000]
        lang_note = ""
        if language != "auto":
            lang_name = LANGUAGE_NAMES.get(language, language)
            lang_note = f" Respond in {lang_name}."
        try:
            response = self.client.chat.completions.create(
                model    = self.config["llm_model"],
                messages = [
                    {"role": "system", "content": f"You are a research paper summariser.{lang_note}"},
                    {"role": "user",   "content":
                        f"The paper is in {self.lang_name}. Summarise it in 6 bullet points covering: "
                        f"objective, methodology, key findings, dataset/experiments, conclusions, and limitations.\n\n{sample}"},
                ],
                max_tokens  = 500,
                temperature = 0.3,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Could not generate summary: {e}"

    def stats(self) -> Dict:
        return {
            "chunks":    len(self.chunks),
            "chars":     len(self.pdf_text),
            "emb_dim":   int(self.embeddings.shape[1]),
            "emb_model": self.config["embedding_model"],
            "pages_est": max(1, len(self.pdf_text) // 3000),
            "lang":      self.lang_name,
            "lang_code": self.lang_code,
            "citations": len(self.citations),
            "keywords":  len(self.keywords),
        }


# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
def init_state():
    defaults = {
        "rag":          None,
        "history":      [],
        "current_file": None,
        "summary":      None,
        "groq_key":     "",
        "answer_lang":  "auto",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔬 Paper Explainer")
    st.markdown("<hr style='border-color:#2E2E50;margin:.5rem 0'>", unsafe_allow_html=True)

    # ── API Key ──
    st.markdown("### 🔑 Groq API Key")
    api_key = st.text_input("Groq API key", type="password",
                             value=st.session_state.groq_key,
                             placeholder="gsk_...")
    if api_key:
        st.session_state.groq_key = api_key
        st.success("✅ Key saved")
    else:
        st.warning("⚠️ Add Groq key to begin")

    st.markdown("<hr style='border-color:#2E2E50'>", unsafe_allow_html=True)

    # ── Embedding Model ──
    st.markdown("### 🧠 Embedding Model")
    emb_model = st.selectbox("Model", [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ], help="Multilingual models handle 50+ languages.")

    if "multilingual" in emb_model:
        st.success("🌍 50+ languages supported")
    else:
        st.info("🇬🇧 English-only model")

    # ── LLM ──
    st.markdown("### ⚡ LLM (Groq)")
    llm_model = st.selectbox("LLM", [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

    # ── RAG params ──
    st.markdown("### 📐 RAG Settings")
    top_k         = st.slider("Top-K chunks",    1,  10,  3)
    chunk_size    = st.slider("Chunk size",     200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap",   0,  200,  50, 10)

    # ── Generation ──
    st.markdown("### 🎛️ Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
    max_tokens  = st.slider("Max tokens",  200, 2000, 600, 50)

    # ── Answer Language ──
    st.markdown("### 🌍 Answer Language")
    lang_options = {"auto": "Auto-detect (match question)", "en": "English",
                    "hi": "Hindi", "fr": "French", "de": "German",
                    "es": "Spanish", "zh-cn": "Chinese", "ar": "Arabic",
                    "ja": "Japanese", "ko": "Korean", "ru": "Russian",
                    "pt": "Portuguese", "it": "Italian"}
    answer_lang = st.selectbox(
        "Respond in",
        list(lang_options.keys()),
        format_func=lambda k: lang_options[k],
    )
    st.session_state.answer_lang = answer_lang

    config = {
        "embedding_model": emb_model,
        "llm_model":       llm_model,
        "top_k":           top_k,
        "chunk_size":      chunk_size,
        "chunk_overlap":   chunk_overlap,
        "temperature":     temperature,
        "max_tokens":      max_tokens,
        "answer_language": answer_lang,
    }

    st.markdown("<hr style='border-color:#2E2E50'>", unsafe_allow_html=True)

    # ── Paper stats (when loaded) ──
    if st.session_state.rag:
        s = st.session_state.rag.stats()
        st.markdown("### 📊 Paper Stats")
        flag = LANG_FLAGS.get(s["lang_code"], "🌐")
        st.markdown(f"**Language:** {flag} {s['lang']}")
        st.metric("Chunks",     s["chunks"])
        st.metric("Characters", f"{s['chars']:,}")
        st.metric("Emb Dim",    s["emb_dim"])
        st.metric("Citations",  s["citations"])
        st.metric("Q&A Count",  len(st.session_state.history))

        st.markdown("<hr style='border-color:#2E2E50'>", unsafe_allow_html=True)
        if st.button("🗑️ Reset / Load New Paper", use_container_width=True):
            for k in ["rag","history","current_file","summary"]:
                st.session_state[k] = None if k == "rag" else ([] if k == "history" else None)
            st.rerun()


# ══════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class="app-hero">
  <div class="app-title">Research Paper Explainer</div>
  <div class="app-sub">Multilingual RAG · Built from Scratch · No Black Boxes</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tag-row">
  <span class="tag">🧠 FAISS + Cosine</span>
  <span class="tag">🤗 SentenceTransformers</span>
  <span class="tag">⚡ Groq (Llama 3)</span>
  <span class="tag">📄 PyPDF2</span>
  <span class="tag">🌊 Streamlit</span>
  <span class="tag tag-new">🌍 Multilingual v2</span>
  <span class="tag tag-new">📚 Citations</span>
  <span class="tag tag-new">🔑 Keywords</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  FILE UPLOAD
# ══════════════════════════════════════════════
uploaded = st.file_uploader("📄 Upload a Research Paper (PDF)", type="pdf")

if uploaded:
    if st.session_state.current_file != uploaded.name:
        for k in ["rag","history","summary"]:
            st.session_state[k] = None if k in ["rag","summary"] else []
        st.session_state.current_file = uploaded.name

    if st.session_state.rag is None:
        if not st.session_state.groq_key:
            st.error("❌ Enter your Groq API key in the sidebar first.")
            st.stop()

        prog = st.progress(0, "Starting pipeline…")
        try:
            prog.progress(10, "📖 Extracting text from PDF…")
            prog.progress(30, "✂️  Chunking with sliding window…")
            prog.progress(50, "🧮 Computing multilingual embeddings…")
            rag = ResearchPaperRAG(uploaded, config, st.session_state.groq_key)
            prog.progress(80, "🗂️  Building FAISS index…")
            time.sleep(0.2)
            prog.progress(100, "✅ Ready!")
            st.session_state.rag = rag
            time.sleep(0.3)
            prog.empty()

            s = rag.stats()
            flag = LANG_FLAGS.get(s["lang_code"], "🌐")
            st.success(
                f"✅ **{uploaded.name}** processed — "
                f"{s['chunks']} chunks · {flag} {s['lang']} detected · "
                f"{s['citations']} citations found"
            )
        except Exception as e:
            prog.empty()
            st.error(f"❌ Pipeline error: {e}")
            st.stop()

    # ── Metric bar ──
    s = st.session_state.rag.stats()
    flag = LANG_FLAGS.get(s["lang_code"], "🌐")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{s["chunks"]}</div><div class="metric-label">Chunks</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{s["chars"]//1000}K</div><div class="metric-label">Characters</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{s["emb_dim"]}</div><div class="metric-label">Emb Dim</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{s["citations"]}</div><div class="metric-label">Citations</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="metric-card"><div class="metric-value">{flag}</div><div class="metric-label">{s["lang"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  TABS
    # ══════════════════════════════════════════
    tab_chat, tab_multilang, tab_summary, tab_explore, tab_cite, tab_math, tab_history = st.tabs([
        "💬 Chat",
        "🌍 Multilingual",
        "📝 Summary",
        "🔍 RAG Explorer",
        "📚 Citations & Keywords",
        "📐 Math & Theory",
        "📜 History",
    ])

    # ──────────────────────────────────────────
    #  TAB 1 · CHAT
    # ──────────────────────────────────────────
    with tab_chat:
        st.markdown("### 💬 Chat with your Paper")
        lang_name = LANGUAGE_NAMES.get(s["lang_code"], "Unknown")
        flag = LANG_FLAGS.get(s["lang_code"], "🌐")
        st.markdown(
            f'<span class="lang-badge">{flag} Paper language: {lang_name} &nbsp;·&nbsp; Ask in any language — the bot will match</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        with st.expander("💡 Sample Questions"):
            samples = [
                "What is the main contribution of this paper?",
                "What methodology or approach did the authors use?",
                "What are the key results or findings?",
                "What datasets were used in the experiments?",
                "What are the limitations of this work?",
                "How does this compare to prior work?",
                "What future directions do the authors suggest?",
                "Explain the technical approach in simple terms.",
            ]
            cols = st.columns(2)
            for i, q in enumerate(samples):
                if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
                    st.session_state["prefill_q"] = q

        # History bubbles
        for entry in st.session_state.history:
            st.markdown(f'<div class="user-bubble">🧑 {entry["q"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-bubble">🤖 {entry["a"]}</div>', unsafe_allow_html=True)

        question = st.text_area(
            "Your question (any language)",
            value=st.session_state.pop("prefill_q", ""),
            height=90,
            placeholder="e.g. What problem does this paper solve? / इस पेपर की मुख्य खोज क्या है? / 这篇论文的主要发现是什么?",
        )

        col_ask, col_opt = st.columns([1, 3])
        ask_btn     = col_ask.button("🚀 Ask", type="primary", use_container_width=True)
        show_chunks = col_opt.checkbox("Show retrieved chunks", value=True)

        if ask_btn and question.strip():
            with st.spinner("🤔 Retrieving + Generating…"):
                result = st.session_state.rag.ask(
                    question,
                    answer_lang=st.session_state.answer_lang
                )

            st.markdown(f'<div class="user-bubble">🧑 {question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-bubble">🤖 {result["answer"]}</div>', unsafe_allow_html=True)

            top_sim = result["chunks"][0]["cosine_sim"] if result["chunks"] else 0
            m1, m2, m3 = st.columns(3)
            m1.metric("Chunks Used",    len(result["chunks"]))
            m2.metric("Top Similarity", f"{top_sim:.3f}")
            m3.metric("Answer Lang",    lang_options.get(st.session_state.answer_lang, "Auto"))

            if show_chunks:
                st.markdown("#### 📚 Retrieved Context")
                for c in result["chunks"]:
                    pct = int(c["cosine_sim"] * 100)
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>Chunk #{c["chunk_id"]+1} &nbsp;·&nbsp; '
                        f'Cosine Similarity: {c["cosine_sim"]:.3f} ({pct}%)</strong><br><br>'
                        f'{c["text"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.session_state.history.append({
                "q":      question,
                "a":      result["answer"],
                "chunks": result["chunks"],
                "ts":     datetime.now().strftime("%H:%M:%S"),
                "lang":   st.session_state.answer_lang,
            })

    # ──────────────────────────────────────────
    #  TAB 2 · MULTILINGUAL
    # ──────────────────────────────────────────
    with tab_multilang:
        st.markdown("### 🌍 Multilingual Analysis")

        rag = st.session_state.rag
        lang_code = rag.lang_code
        lang_name = rag.lang_name
        lang_conf = rag.lang_conf
        flag = LANG_FLAGS.get(lang_code, "🌐")

        # Language detection card
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("#### Detected Paper Language")
            conf_pct = int(lang_conf * 100)
            badge_class = "lang-badge" if lang_code != "unknown" else "lang-badge lang-badge-unknown"
            st.markdown(
                f'<div class="{badge_class}" style="font-size:1.1rem;padding:.6rem 1.5rem;margin:.5rem 0">'
                f'{flag} &nbsp; <strong>{lang_name}</strong> &nbsp; · &nbsp; {conf_pct}% confidence'
                f'</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Language code: `{lang_code}` · Detection library: {'langdetect' if LANGDETECT_AVAILABLE else 'heuristic fallback'}")

            multilingual_model = "multilingual" in config["embedding_model"]
            if multilingual_model:
                st.success("✅ Multilingual embedding model active — cross-lingual search supported")
            else:
                st.warning("⚠️ English-only model selected. Switch to `paraphrase-multilingual-MiniLM-L12-v2` in sidebar for cross-lingual search.")

        with col_r:
            st.markdown("#### How Cross-Lingual Search Works")
            st.markdown("""
            **Multilingual embeddings** map text from 50+ languages into the *same* vector space.
            This means you can:
            - Ask in **English** about a paper written in **Chinese**
            - Ask in **Hindi** about a paper written in **German**
            - Mix languages within the same conversation

            The embedding model understands that *"cat"* (English), *"gato"* (Spanish),
            and *"猫"* (Chinese) are the same concept.
            """)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Cross-lingual Q&A demo
        st.markdown("#### 🔄 Cross-Lingual Q&A")
        st.info("Ask a question in a different language from the paper. The multilingual model retrieves correct context anyway.")

        col_qa, col_opt2 = st.columns([3, 1])
        with col_qa:
            cross_q = st.text_area(
                "Ask in any language",
                height=80,
                placeholder="Try: 'इस पेपर का मुख्य योगदान क्या है?' or '这篇论文用了什么方法?' or 'Quelle est la principale conclusion?'",
                key="cross_q_input"
            )
        with col_opt2:
            target_lang = st.selectbox(
                "Answer in",
                list(lang_options.keys()),
                format_func=lambda k: lang_options[k],
                key="cross_target_lang"
            )

        if st.button("🌐 Cross-Lingual Ask", type="primary", key="cross_ask"):
            if cross_q.strip():
                with st.spinner("🔄 Cross-lingual retrieval + generation…"):
                    result = rag.ask(cross_q, answer_lang=target_lang)
                st.markdown(f'<div class="user-bubble">🧑 {cross_q}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-bubble">🤖 {result["answer"]}</div>', unsafe_allow_html=True)

                # Show which chunks were retrieved
                with st.expander("📚 Retrieved chunks (from original paper)"):
                    for c in result["chunks"]:
                        st.markdown(
                            f'<div class="chunk-card"><strong>Chunk #{c["chunk_id"]+1} · Sim: {c["cosine_sim"]:.3f}</strong><br><br>{c["text"]}</div>',
                            unsafe_allow_html=True
                        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Chunk translation tool
        st.markdown("#### 🔤 Translate Paper Chunks")
        st.markdown("Select a chunk by index and translate it to any language using the LLM.")

        col_cidx, col_tlang = st.columns(2)
        chunk_idx = col_cidx.number_input("Chunk index", 0, len(rag.chunks)-1, 0, key="trans_chunk_idx")
        trans_lang = col_tlang.selectbox("Translate to", list(lang_options.keys())[1:], format_func=lambda k: lang_options[k], key="trans_lang")

        if st.button("🔤 Translate Chunk", key="translate_chunk_btn"):
            orig = rag.chunks[int(chunk_idx)]
            st.markdown(f'<div class="chunk-card"><strong>Original (Chunk #{int(chunk_idx)+1}):</strong><br><br>{orig}</div>', unsafe_allow_html=True)
            with st.spinner("Translating…"):
                translated = rag.translate_chunk(orig, trans_lang)
            target_name = lang_options.get(trans_lang, trans_lang)
            st.markdown(f'<div class="bot-bubble"><strong>Translation ({target_name}):</strong><br><br>{translated}</div>', unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Supported languages info
        st.markdown("#### 🌐 Supported Languages (Multilingual Model)")
        langs_display = [f"{LANG_FLAGS.get(c,'🌐')} {n}" for c, n in list(LANGUAGE_NAMES.items())[:24]]
        cols = st.columns(4)
        for i, lang in enumerate(langs_display):
            cols[i % 4].markdown(f"<span class='tag'>{lang}</span>", unsafe_allow_html=True)
        st.caption("…and 30+ more. The `paraphrase-multilingual-MiniLM-L12-v2` model officially supports 50+ languages.")

    # ──────────────────────────────────────────
    #  TAB 3 · SUMMARY
    # ──────────────────────────────────────────
    with tab_summary:
        st.markdown("### 📝 Auto-Generated Paper Summary")
        st.info("The bot reads the first ~3,000 characters and produces a structured 6-point summary.")

        summ_col1, summ_col2 = st.columns([1, 2])
        with summ_col1:
            sum_lang = st.selectbox("Summary language", list(lang_options.keys()),
                                     format_func=lambda k: lang_options[k], key="sum_lang_sel")
        with summ_col2:
            if st.button("✨ Generate Summary", type="primary", key="gen_sum"):
                with st.spinner("Summarising…"):
                    st.session_state.summary = st.session_state.rag.summarise(language=sum_lang)

        if st.session_state.summary:
            st.markdown(
                f'<div class="bot-bubble" style="margin:0">{st.session_state.summary}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name=f"paper_summary_{sum_lang}.txt",
                mime="text/plain",
            )

    # ──────────────────────────────────────────
    #  TAB 4 · RAG EXPLORER
    # ──────────────────────────────────────────
    with tab_explore:
        st.markdown("### 🔍 RAG Retrieval Explorer")
        st.markdown("Analyse retrieval in real-time. Watch cosine similarity scores and chunk ranking.")

        explore_q = st.text_input("Query to analyse", placeholder="e.g. attention mechanism", key="explore_q")

        if st.button("🔎 Run Retrieval", key="run_retrieval") and explore_q.strip():
            chunks = st.session_state.rag.retrieve(explore_q)

            # Horizontal bar chart
            labels = [f"Chunk #{c['chunk_id']+1}" for c in chunks]
            scores = [c["cosine_sim"] for c in chunks]
            palette = ["#7B6FFF","#38BDF8","#34D399","#FB923C","#F472B6"]

            fig = go.Figure(go.Bar(
                x=scores, y=labels,
                orientation="h",
                marker_color=[palette[i % len(palette)] for i in range(len(chunks))],
                text=[f"{s:.4f}" for s in scores],
                textposition="outside",
            ))
            fig.update_layout(
                title="Cosine Similarity Scores — Retrieved Chunks",
                xaxis_title="Cosine Similarity (higher = more relevant)",
                xaxis=dict(range=[0, 1.05]),
                plot_bgcolor="#0D0D14",
                paper_bgcolor="#0D0D14",
                font=dict(color="#9090B8"),
                height=280,
                margin=dict(l=10, r=60, t=45, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart if top_k >= 3
            if len(chunks) >= 3:
                categories = ["Similarity", "Length", "Rank"]
                max_len = max(c["length"] for c in chunks)
                fig_r = go.Figure()
                for i, c in enumerate(chunks):
                    vals = [
                        c["cosine_sim"],
                        c["length"] / max_len,
                        1 - (c["rank"] - 1) / len(chunks),
                    ]
                    fig_r.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=f"Chunk #{c['chunk_id']+1}",
                        line_color=palette[i % len(palette)],
                        opacity=0.6,
                    ))
                fig_r.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0,1], color="#7878A0"),
                        bgcolor="#0D0D14",
                    ),
                    paper_bgcolor="#0D0D14",
                    font=dict(color="#9090B8"),
                    height=320,
                    title="Multi-Attribute Chunk Comparison",
                    legend=dict(font=dict(color="#C0C0D0")),
                )
                st.plotly_chart(fig_r, use_container_width=True)

            # Chunk cards with expanders
            for c in chunks:
                with st.expander(
                    f"📄 Chunk #{c['chunk_id']+1}  ·  Rank {c['rank']}  ·  "
                    f"Cosine {c['cosine_sim']:.4f}  ·  {c['length']} chars"
                ):
                    st.markdown(f'<div class="chunk-card">{c["text"]}</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cosine Similarity", f"{c['cosine_sim']:.4f}")
                    col2.metric("Rank",              c["rank"])
                    col3.metric("Characters",        c["length"])

            # Stats
            sims = [c["cosine_sim"] for c in chunks]
            st.markdown("#### 📊 Retrieval Summary")
            s1, s2, s3 = st.columns(3)
            s1.metric("Avg Similarity", f"{sum(sims)/len(sims):.4f}")
            s2.metric("Max Similarity", f"{max(sims):.4f}")
            s3.metric("Min Similarity", f"{min(sims):.4f}")

    # ──────────────────────────────────────────
    #  TAB 5 · CITATIONS & KEYWORDS
    # ──────────────────────────────────────────
    with tab_cite:
        rag = st.session_state.rag
        col_cit, col_kw = st.columns([1, 1])

        with col_cit:
            st.markdown("### 📚 Extracted Citations")
            citations = rag.citations
            if citations:
                st.markdown(f"Found **{len(citations)}** reference entries in the paper.")
                for i, cit in enumerate(citations[:30], 1):
                    cit_clean = cit.replace('\n', ' ').strip()
                    st.markdown(
                        f'<div class="citation-card">'
                        f'<span class="citation-num">[{i}]</span>{cit_clean}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if len(citations) > 30:
                    st.caption(f"…{len(citations)-30} more citations truncated.")

                cite_txt = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(citations))
                st.download_button("⬇️ Export Citations (.txt)", cite_txt,
                                   "citations.txt", "text/plain")
            else:
                st.info("No reference section detected. The paper may not have a standard bibliography, or the PDF extraction may have missed it.")

        with col_kw:
            st.markdown("### 🔑 Top Keywords")
            keywords = rag.keywords
            if keywords:
                st.markdown(f"Extracted **{len(keywords)}** significant terms (by term frequency).")

                # Keyword pills
                pills_html = "".join(
                    f'<span class="kw-pill">{kw} <span style="color:#7878A0;font-size:.75em">({cnt})</span></span>'
                    for kw, cnt in keywords[:20]
                )
                st.markdown(f'<div style="line-height:2.5">{pills_html}</div>', unsafe_allow_html=True)

                # Horizontal bar chart for keywords
                kw_words = [k for k, _ in keywords[:15]]
                kw_counts = [c for _, c in keywords[:15]]
                fig_kw = go.Figure(go.Bar(
                    x=kw_counts, y=kw_words,
                    orientation="h",
                    marker=dict(
                        color=kw_counts,
                        colorscale=[[0,"#2E2E50"],[1,"#7B6FFF"]],
                        showscale=False,
                    ),
                    text=kw_counts, textposition="outside",
                ))
                fig_kw.update_layout(
                    title="Term Frequency — Top 15 Keywords",
                    xaxis_title="Occurrences",
                    plot_bgcolor="#0D0D14",
                    paper_bgcolor="#0D0D14",
                    font=dict(color="#9090B8"),
                    height=380,
                    margin=dict(l=10, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_kw, use_container_width=True)
            else:
                st.info("No keywords extracted.")

    # ──────────────────────────────────────────
    #  TAB 6 · MATH & THEORY
    # ──────────────────────────────────────────
    with tab_math:
        st.markdown("### 📐 How RAG Works — Math & Theory")
        st.markdown("Every equation below maps directly to a line of code in this app. Zero black boxes.")

        # ── Pipeline Overview ──
        st.markdown('<span class="step-badge">0</span> **Full Pipeline at a Glance**', unsafe_allow_html=True)
        st.markdown('<div class="math-block">PDF  ──►  Text  ──►  Chunks  ──►  Embeddings  ──►  FAISS Index\n                                                          │\n                              Query (any language)  ───────┘\n                                    │\n                             embed(Query)\n                                    │\n                         Cosine Similarity Search\n                                    │\n                            Top-K Relevant Chunks\n                                    │\n                         Groq LLM (Llama 3 / Mixtral)\n                                    │\n                             Final Answer  ✅</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Step 1 ──
        st.markdown('<span class="step-badge">1</span> **PDF Parsing & Text Extraction**', unsafe_allow_html=True)
        st.markdown("PyPDF2 reads every page and concatenates the raw text. Whitespace is normalised.")
        st.code('reader = PyPDF2.PdfReader(pdf_bytes)\ntext   = " ".join(page.extract_text() for page in reader.pages)\ntext   = re.sub(r"\\s+", " ", text).strip()', language="python")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Step 2 ──
        st.markdown('<span class="step-badge">2</span> **Sliding-Window Chunking**', unsafe_allow_html=True)
        st.markdown("Text is split into overlapping windows so context is not lost at chunk edges.")
        st.latex(r"""
\text{chunk}_i = \text{text}\bigl[\, i \cdot (S - O) \;:\; i \cdot (S - O) + S \,\bigr]
""")
        st.markdown("where **S** = chunk size, **O** = overlap. With S=500, O=50:")
        st.markdown('<div class="math-block">chunk₀  =  text[   0 :  500 ]\nchunk₁  =  text[ 450 :  950 ]   ← 50-char overlap\nchunk₂  =  text[ 900 : 1400 ]   …</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Step 3 ──
        st.markdown('<span class="step-badge">3</span> **Multilingual Dense Embeddings**', unsafe_allow_html=True)
        st.markdown("""
        A pre-trained Sentence-BERT maps any text (in 50+ languages) to a dense vector.
        Semantically similar text — *across languages* — lands close together in vector space.
        """)
        st.latex(r"E : \Sigma^* \;\rightarrow\; \mathbb{R}^d \quad (d = 384 \text{ for MiniLM})")
        st.markdown('<div class="math-block">\"attention mechanism\"       →  [0.12, -0.45,  0.87, … ]  (384 floats)\n\"メカニズムへの注意\"  →  [0.13, -0.44,  0.85, … ]  (very close!)</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Step 4 ──
        st.markdown('<span class="step-badge">4</span> **FAISS Index with Cosine Similarity**', unsafe_allow_html=True)
        st.markdown("""
        Version 2 uses **cosine similarity** (inner product on unit vectors) instead of L2 distance.
        Cosine measures *directional* alignment — better for semantic search.
        """)
        st.latex(r"""
\cos(\mathbf{q}, \mathbf{c}) = \frac{\mathbf{q} \cdot \mathbf{c}}{\|\mathbf{q}\|\,\|\mathbf{c}\|}
\;=\; \mathbf{\hat{q}} \cdot \mathbf{\hat{c}} \quad \in [-1,\, 1]
""")
        st.markdown("Since both vectors are L2-normalised before insertion into `IndexFlatIP`:")
        st.latex(r"\text{FAISS stores} \;\hat{\mathbf{e}}_i = \frac{\mathbf{e}_i}{\|\mathbf{e}_i\|}, \quad \text{then} \quad \hat{\mathbf{q}} \cdot \hat{\mathbf{e}}_i = \cos(\mathbf{q}, \mathbf{e}_i)")
        st.latex(r"\text{Top-K} = \underset{K}{\operatorname{arg\,max}} \;\cos(\mathbf{q}, \mathbf{e}_i)")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Step 5 ──
        st.markdown('<span class="step-badge">5</span> **RAG Prompt Construction**', unsafe_allow_html=True)
        st.markdown("The top-K chunks are concatenated as context and injected into the LLM prompt.")
        st.latex(r"P(\text{answer} \mid \text{query},\, \text{context}_{1..K})")
        st.markdown('<div class="math-block">SYSTEM : You are a multilingual research assistant.\n         Answer ONLY from context. Respond in the question\'s language.\nUSER   : Paper language: {lang}\n         Context:\n           [Chunk 1]: …\n           [Chunk 2]: …\n           [Chunk K]: …\n         Question: {query}\n         Provide a structured answer:</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Multilingual math ──
        st.markdown('<span class="step-badge">✦</span> **Cross-Lingual Retrieval — Why It Works**', unsafe_allow_html=True)
        st.markdown("""
        Multilingual models are trained on **parallel corpora** (same text in many languages).
        This aligns the embedding spaces across languages.
        """)
        st.latex(r"""
E(\text{``cat''}) \approx E(\text{``gato''}) \approx E(\text{``猫''}) \approx E(\text{``बिल्ली''})
""")
        st.markdown("So a query in Hindi retrieves the correct chunk from an English paper — and vice versa.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Hyperparameter guide ──
        st.markdown("#### 🎛️ Hyperparameter Reference")
        params = {
            "chunk_size (S)":     ("500 chars",     "Larger → richer context per chunk, but noisier retrieval"),
            "chunk_overlap (O)":  ("50 chars",       "Prevents information loss at chunk boundaries"),
            "top_k (K)":          ("3",              "More chunks = richer context but longer prompt"),
            "temperature":        ("0.7",            "Lower → deterministic; higher → creative responses"),
            "embedding_dim (d)":  ("384 (MiniLM)",   "Higher d = richer representation, slower encoding"),
            "normalize_embeddings": ("True",         "Required for cosine sim via IndexFlatIP"),
        }
        for param, (default, desc) in params.items():
            st.markdown(f"- **`{param}`** *(default: {default})* — {desc}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🤔 Why Not Use LangChain?")
        st.info(
            "This project hand-rolls every RAG step intentionally — "
            "so you understand what's inside those frameworks. "
            "Once you know this, LangChain is just a convenient wrapper, not magic."
        )

    # ──────────────────────────────────────────
    #  TAB 7 · HISTORY
    # ──────────────────────────────────────────
    with tab_history:
        st.markdown("### 📜 Conversation History")

        if not st.session_state.history:
            st.info("No questions asked yet — head to the Chat tab!")
        else:
            n = len(st.session_state.history)
            st.markdown(f"**{n} question{'s' if n>1 else ''} asked in this session**")

            col_txt, col_json = st.columns(2)
            txt = "\n".join(
                f"[{e['ts']}]\nQ: {e['q']}\nA: {e['a']}\n{'─'*60}"
                for e in st.session_state.history
            )
            col_txt.download_button("⬇️ Export TXT",  txt,  "qa_history.txt", "text/plain", use_container_width=True)
            json_data = json.dumps([
                {"time":e["ts"],"question":e["q"],"answer":e["a"],
                 "top_sim":e["chunks"][0]["cosine_sim"] if e["chunks"] else 0}
                for e in st.session_state.history
            ], indent=2)
            col_json.download_button("⬇️ Export JSON", json_data, "qa_history.json", "application/json", use_container_width=True)

            if n > 1:
                fig3 = go.Figure(go.Scatter(
                    x=list(range(1, n+1)),
                    y=[e["chunks"][0]["cosine_sim"] if e["chunks"] else 0 for e in st.session_state.history],
                    mode="lines+markers",
                    line=dict(color="#7B6FFF", width=2),
                    marker=dict(size=8, color="#38BDF8"),
                ))
                fig3.update_layout(
                    title="Top-Chunk Cosine Similarity per Question",
                    xaxis_title="Question #", yaxis_title="Cosine Similarity",
                    yaxis=dict(range=[0,1]),
                    plot_bgcolor="#0D0D14", paper_bgcolor="#0D0D14",
                    font=dict(color="#9090B8"), height=240,
                    margin=dict(l=10,r=10,t=40,b=40),
                )
                st.plotly_chart(fig3, use_container_width=True)

            for i, entry in enumerate(reversed(st.session_state.history), 1):
                idx = n - i + 1
                with st.expander(f"Q{idx}  [{entry['ts']}]  —  {entry['q'][:80]}…"):
                    st.markdown(f'<div class="user-bubble">🧑 {entry["q"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="bot-bubble">🤖 {entry["a"]}</div>', unsafe_allow_html=True)
                    st.caption(
                        f"Chunks: {len(entry['chunks'])}  ·  "
                        f"Top cosine sim: {entry['chunks'][0]['cosine_sim']:.3f}"
                        if entry["chunks"] else ""
                    )

# ══════════════════════════════════════════════
#  LANDING SCREEN (no file uploaded)
# ══════════════════════════════════════════════
else:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📄", "Step 1 · Upload", "Drop any research PDF — English or multilingual"),
        (c2, "⚙️", "Step 2 · Process", "Auto-chunks, embeds & detects language"),
        (c3, "💬", "Step 3 · Ask", "Chat, explore, translate, extract citations"),
    ]:
        col.markdown(
            f'<div class="card" style="text-align:center;padding:2rem 1.5rem">'
            f'<div style="font-size:2.2rem;margin-bottom:.8rem">{icon}</div>'
            f'<strong style="font-size:1rem">{title}</strong><br>'
            f'<span style="color:#7878A0;font-size:.86rem">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✨ What's New in Version 2.0")
    features = [
        ("🌍", "**Multilingual Analysis**", "Auto-detects paper language, supports 50+ languages, cross-lingual Q&A"),
        ("🔄", "**Cross-Lingual Search**", "Ask in Hindi/Chinese/Arabic — retrieve from an English paper (and vice versa)"),
        ("🔤", "**Chunk Translator**", "Translate any extracted chunk to your preferred language using the LLM"),
        ("📚", "**Citation Extractor**", "Automatically pulls references/bibliography from the paper"),
        ("🔑", "**Keyword Extraction**", "Top terms by frequency + interactive bar chart"),
        ("📐", "**Better Math**", "All equations use st.latex() for proper LaTeX rendering"),
        ("📊", "**Cosine Similarity**", "Upgraded from L2 distance to cosine similarity for better semantic search"),
        ("🕸️", "**Radar Charts**", "Multi-attribute chunk comparison (similarity, length, rank)"),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        cols[i % 2].markdown(f"- {icon} {title} — {desc}")

# ── Footer ──
st.markdown(
    '<div class="footer">'
    'Research Paper Explainer Bot v2.0 &nbsp;·&nbsp; '
    'Streamlit · FAISS · SentenceTransformers · Groq (Llama 3) · langdetect'
    '</div>',
    unsafe_allow_html=True
)
