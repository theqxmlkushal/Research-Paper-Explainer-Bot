# 🔬 Research Paper Explainer Bot — v2.0
### *Multilingual RAG Pipeline, Built from Scratch*

**Project Title:** Research Paper Explainer Bot — Multilingual RAG from Scratch

**GitHub Repository:** https://github.com/theqxmlkushal/21_Days_of_LLM_coding/blob/master/Research_Paper_Explainer_Bot.ipynb

---

## 📌 Project Overview

A fully usable, production-style web application that lets you **chat with any research paper in any language**.

Upload a PDF — in English, Hindi, Chinese, Arabic, French, or 50+ other languages — and the bot:
- **Detects the paper language** automatically
- **Answers your questions** in whichever language you ask
- **Retrieves relevant passages** using cosine similarity over multilingual embeddings
- **Extracts citations** and **top keywords** from the paper
- **Shows the math** behind every pipeline step with LaTeX equations

The entire RAG pipeline is hand-built — no LangChain, no LlamaIndex — so every step is transparent, educational, and modifiable.

---

## 🛠️ Technologies / Concepts Used

| Technology | Role |
|---|---|
| **Streamlit** | Web frontend + backend (single-file app) |
| **PyPDF2** | Extract raw text from uploaded PDFs |
| **SentenceTransformers** — `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual dense embeddings (50+ languages) |
| **FAISS** — `IndexFlatIP` | Cosine similarity search over normalised vectors |
| **Groq API** (Llama 3.1 / Mixtral) | Fast LLM inference for answer generation |
| **langdetect** | Automatic paper language detection |
| **Plotly** | Interactive bar charts, radar charts, heatmaps |
| **RAG Pipeline** | Retrieve relevant chunks before generating answers |
| **Sliding-window chunking** | Overlapping splits to preserve cross-sentence context |
| **Cosine similarity** | `cos(q̂, ĉ) = q̂·ĉ` on L2-normalised vectors |
| **Multilingual embeddings** | Cross-lingual semantic alignment across 50+ languages |

---

## 🏗️ File Structure

```
research_paper_explainer_bot/
│
├── streamlit_app.py                    ← Main app (7 tabs, multilingual, citations)
├── requirements.txt                    ← All dependencies
├── README.md                           ← This file
└── Research_Paper_Explainer_Bot.ipynb  ← Colab notebook (RAG math + multilingual)
```

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run streamlit_app.py

# 3. Open http://localhost:8501
```


## 🔁 RAG Pipeline (v2.0)

```
PDF Upload
   │
   ▼
PDF Parsing (PyPDF2)                   → raw text string
   │
   ▼
Language Detection (langdetect)        → "en", "hi", "zh-cn", …
   │
   ▼
Sliding-Window Chunking                → overlapping text chunks
   chunk_i = text[i·(S−O) : i·(S−O)+S]
   │
   ▼
Multilingual SentenceBERT Embeddings   → matrix [n_chunks × 384]
   Ê(text) ∈ ℝ³⁸⁴,   ‖Ê‖₂ = 1
   │
   ▼
FAISS IndexFlatIP                      → cosine similarity index
   (inner product on unit vectors = cosine sim)
   │            │
   │            ▼  (at query time, any language)
   │       Ê(query) → query unit vector
   │            │
   │       Top-K = argmax_K  q̂ · ĉᵢ
   │            │
   ▼            ▼
Groq LLM  ← context (top-K chunks) + query prompt
   P(answer | query, chunk₁…chunkₖ)
   │
   ▼
Answer ✅  (in question's language, or forced target language)
```

---

## 📐 Core Equations

**Cosine Similarity (retrieval metric):**
```
cos(q̂, ĉ) = q̂ · ĉ = Σᵢ q̂ᵢ·ĉᵢ    ∈ [-1, 1]
```

**L2 Normalisation (required for cosine via IndexFlatIP):**
```
ê = e / ‖e‖₂    so that  ‖ê‖₂ = 1
```

**Cross-lingual alignment (multilingual model property):**
```
E("cat") ≈ E("gato") ≈ E("猫") ≈ E("बिल्ली")
```

**LLM temperature sampling:**
```
P(wₜ = w | w<t) = exp(zw/τ) / Σw' exp(zw'/τ)
```

---

## 👥 Team Member Roles & Contribution Distribution

| Role | Responsibility | Contribution |
|---|---|---|
| **ML Engineer** | RAG pipeline, embeddings, FAISS, multilingual support | 35% |
| **Backend Developer** | Groq API, prompt engineering, language detection | 25% |
| **Frontend Developer** | Streamlit UI, CSS styling, Plotly charts, 7-tab layout | 25% |
| **Research & Docs** | Math derivations, LaTeX equations, README, notebook | 15% |

> *For solo projects: all roles handled by one person — great for showcasing full-stack ML skills.*

---

## 📚 Research Papers / References Related to the Topic

| # | Paper | Authors | Venue | Year |
|---|-------|---------|-------|------|
| [1] | **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | Lewis et al. | NeurIPS | 2020 |
| [2] | **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** | Reimers & Gurevych | EMNLP | 2019 |
| [3] | **Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation** | Reimers & Gurevych | EMNLP | 2020 |
| [4] | **Billion-scale similarity search with GPUs (FAISS)** | Johnson et al. | IEEE Trans. | 2021 |
| [5] | **Llama 3: Open Foundation and Fine-Tuned Chat Models** | Meta AI | — | 2024 |
| [6] | **Dense Passage Retrieval for Open-Domain Question Answering** | Karpukhin et al. | EMNLP | 2020 |
| [7] | **BERT: Pre-training of Deep Bidirectional Transformers** | Devlin et al. | NAACL | 2019 |
| [8] | **Attention Is All You Need** | Vaswani et al. | NeurIPS | 2017 |

---

## ⚙️ Configurable Parameters

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | 500 chars | Larger = more context per chunk |
| `chunk_overlap` | 50 chars | Overlap to avoid losing context at edges |
| `top_k` | 3 | Number of chunks sent to LLM |
| `temperature` | 0.7 | Creativity of answers (0 = deterministic) |
| `max_tokens` | 600 | Maximum answer length |
| `embedding_model` | multilingual-MiniLM | Swap for `mpnet` for higher accuracy |
| `answer_language` | auto | Force responses in a specific language |

---

## 💡 Key Design Decisions

1. **No framework** — Hand-rolled RAG so every step is visible and educational.
2. **Cosine over L2** — Cosine similarity is scale-invariant and better for semantic search.
3. **Multilingual by default** — `paraphrase-multilingual-MiniLM-L12-v2` supports 50+ languages out of the box.
4. **Groq over OpenAI** — Free tier, 400+ tok/sec inference, OpenAI-compatible API.
5. **Streamlit only** — Single `streamlit run` command to deploy. No Docker needed.

---
