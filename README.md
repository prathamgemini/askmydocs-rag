# AskMyDocs: Multimodal Document Q&A with Verified Citations

![Next.js](https://img.shields.io/badge/Next.js-16.1.6-black?style=for-the-badge&logo=next.js)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![ChromaDB](https://img.shields.io/badge/ChromaDB-local_vector-orange?style=for-the-badge&logo=databricks)
![Docling](https://img.shields.io/badge/Docling-Local_OCR-darkblue?style=for-the-badge)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Local_GGUF-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Qwen_3.8_27B-orange?style=for-the-badge)

> **"Accurate, fast, and multi-modal knowledge retrieval from documents, web pages, and images."**

A high-performance Question-Answering system leveraging **Advanced Hybrid Retrieval-Augmented Generation** with **Multimodal Intelligence**. This system answers questions from uploaded documents (PDFs, TXT, DOCX, MD), visual content (Images, Scanned PDFs, Tables), and live web URLs using 100% local OCR (**Docling** + **RapidOCR ONNX**), local **all-MiniLM-L6-v2** embeddings, persistent local ChromaDB, and dual LLM inference options (Cloud Groq vs 100% Local llama.cpp).

---

## ✨ Features

- ⚡ **Dual LLM Architecture**:
  - **Cloud:** **Qwen 3.8 27B** via Groq (default `qwen/qwen3.8-27b`, override with `GROQ_MODEL`). Streams answers; the model's `<think>` reasoning is stripped so only the answer is shown.
  - **Local:** **Ling 3.0 Tiny** via llama.cpp (`bartowski/Ling-3.0-tiny-GGUF:Q4_K_M` running offline on CPU).
  - **Dynamic Model Switcher:** Segmented selector directly in the UI TopBar.
- 🧠 **100% Local Image & Document OCR**: Uses **Docling** and **RapidOCR ONNX** for fast, offline text, table, and layout extraction without external cloud vision dependencies.
- 🔍 **Local Hybrid Retrieval**: Combines **Dense Vector Search** (`all-MiniLM-L6-v2`, 384d, weight 0.7) with a **keyword-match score** (weight 0.3), searched with both the original and an LLM-rewritten query.
- 🗄️ **Local Persistent Vector Store**: Embedded ChromaDB database that survives restarts with zero cloud vector store dependencies.
- 🔗 **Live URL & PDF Ingestion**: Scrapes and parses any web page or web PDF into clean, structured Markdown chunks. A URL pasted into a question is fetched and indexed on the fly.
- 💡 **HyDE Enhancement**: Generates hypothetical questions per chunk during ingestion to improve question-to-question vector matching.
- ♻️ **Safe Re-ingestion**: Uploading a document again replaces its previous chunks; deleting a document also removes its stored file.
- 🎨 **Modern Next.js UI**: Clean dark/light theme, verified source citations, clickable file links, excerpt previews, and knowledge base document management.

---

## 🚀 Quick Start

### Prerequisites
- **Python** 3.11 or higher
- **Node.js** 20.x or higher
- **Groq API Key** from [console.groq.com](https://console.groq.com/keys) (optional, only if using the cloud model)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/prathamgemini/askmydocs-rag.git
cd askmydocs-rag

# Install Python dependencies
pip install -r requirements.txt

# For llama-cpp-python CPU prebuilt wheel on Windows (if building from source fails):
# pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all:

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Configuration

Copy `.env.example` to `.env` in the project root (or `app/`) and fill in your key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_PROVIDER=groq
# GROQ_MODEL=qwen/qwen3.8-27b
# GROQ_REASONING_EFFORT=default
CHROMA_COLLECTION=knowledge_base
```

`.env` is gitignored — never commit your key.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | _(none)_ | Groq API key. Required for the cloud model. |
| `LLM_PROVIDER` | `groq` | Default model: `groq` (cloud) or `llamacpp` (local). The UI switcher overrides it per question. |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model ID. |
| `GROQ_REASONING_EFFORT` | `default` | `default` lets the model reason before answering; `none` is fastest. |
| `CHROMA_COLLECTION` | `knowledge_base` | ChromaDB collection name. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend (set in `frontend/.env.local`). |

### 3. Run the Application

```bash
# Run Backend (FastAPI on port 8000)
.\run_backend.bat
# or: cd app && uvicorn main:app --reload --port 8000

# Run Frontend (Next.js on port 3000)
cd frontend
npm run dev
```

`run_backend.bat` uses `venv\` or `.venv\` in the project root if one exists, otherwise the `python` on your PATH.

Open `http://localhost:3000` in your browser. On first use the embedding model (~80 MB) and the Docling OCR models are downloaded and cached.

### 4. Docker (backend only)

```bash
docker build -t multimodal-rag .
docker run -p 8000:8000 --env-file .env multimodal-rag
```

---

## 💻 Optional: Local Offline Model Setup

To use **Ling 3.0 Tiny** completely offline:
```bash
cd app
python download_local_model.py
```
This downloads `Ling-3.0-tiny-Q4_K_M.gguf` (~4.68 GB) into `app/local_models/`. Once downloaded, select **💻 Ling 3.0 Tiny [Local]** in the TopBar.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│              FRONTEND (Next.js 16)               │
│  - Chat Interface with Markdown & Citations      │
│  - Model Switcher (Qwen 3.8 27B / Ling 3.0 Tiny) │
│  - Knowledge Base Document Modal (Upload / URL)  │
└──────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│              BACKEND API (FastAPI)               │
│  - /ask: Streaming SSE with citations            │
│  - /ingest/file: SSE progress document ingest    │
│  - /ingest/url: SSE progress web URL scraper     │
│  - /models: Model readiness & switcher endpoint  │
│  - /documents: Knowledge base management         │
└──────────────────────────────────────────────────┘
        │                        │
        ▼                        ▼
┌──────────────────┐    ┌──────────────────────────┐
│ Local Image OCR  │    │ Ingestion / Chunker      │
│ Docling + Rapid  │    │ Semantic Markdown Parser │
└──────────────────┘    └──────────────────────────┘
        │                        │
        └───────────┬────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│ CHROMADB (Local Persistent Storage)              │
│  - 100% Local ONNX all-MiniLM-L6-v2 embeddings   │
│  - Hybrid Retrieval: Cosine Vector + Keyword     │
└──────────────────────────────────────────────────┘
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ask` | Answer a question (SSE stream: `status`, `token`, `sources`, `images`, `done`, `error`). Body: `question`, `history`, `provider`. |
| `POST` | `/ingest/file` | Upload and index a file (`.pdf`, `.docx`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`) with SSE progress. |
| `POST` | `/ingest/url` | Scrape and index a web page or web PDF with SSE progress. |
| `GET` | `/documents` | List indexed documents. |
| `DELETE` | `/documents/{doc_title}` | Delete a document's chunks and its stored file. |
| `GET` | `/files/{filename}` | Download an uploaded file or extracted image (used by citations). |
| `GET` | `/models` | Available models and whether each is ready. |
| `GET` | `/health` | Service status and configuration. |

### Project Structure

```
app/
  main.py               FastAPI app: endpoints, retrieval, prompt building
  config.py             Settings and environment variables
  model_service.py      Groq + llama.cpp clients (streaming and non-streaming)
  ingestion_service.py  File / URL ingestion pipeline with SSE progress
  chunker.py            Markdown-aware chunking (headings, tables, overlap)
  hyde_generator.py     Hypothetical questions per chunk
  embedder.py           Local ONNX all-MiniLM-L6-v2 embeddings
  uploader.py           ChromaDB storage, listing and deletion
  scraper.py            Web page / web PDF scraping
  image_scanner.py      Docling + RapidOCR image OCR
frontend/               Next.js chat UI
```

Runtime data lives in `app/chroma_db/`, `app/uploads/`, `app/output/` and `app/local_models/` (all gitignored).

---

## ⚠️ Notes & Limits

- **Rate limits:** ingestion makes one LLM call per chunk for HyDE, so a large document can hit your Groq rate limit. After three consecutive failures HyDE is skipped for the rest of that document and it is indexed without it.
- **Timeouts:** Groq requests time out after 60 seconds instead of hanging.
- **Local use:** the scraper fetches any URL it is given (including `localhost`) and CORS is open to all origins. Restrict both before exposing the backend publicly.

---

## 📄 License
MIT License.
