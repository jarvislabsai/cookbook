# Multimodal RAG with Nemotron

Self-hosted multimodal RAG pipeline that indexes your PDFs, Markdown, and text files into a local vector database and serves a chat interface for document Q&A. Runs on your own GPU infrastructure with no managed cloud dependency.

**Stack:** [Nemotron Parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) (PDF parsing) + [Nemotron Embed VL](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) (embedding) + [Nemotron Rerank VL](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) (reranking) + a local LLM server exposing `v1/chat/completions` with streaming support (generation) + [LanceDB](https://lancedb.github.io/lancedb/) (vector store) + [Chainlit](https://docs.chainlit.io/) (chat UI).

### Get Started Quickly

Spin up a GPU instance (e.g. on [Jarvislabs.ai](https://jarvislabs.ai/)), then:

```bash
# 1. Clone and install
git clone https://github.com/jarvislabsai/cookbook.git
cd cookbook/multimodal-rag-nemotron
MAX_JOBS=4 uv sync

# 2. Parse your PDFs and index them
uv run python src/parse_pdfs_with_vllm.py \
  --pdf-dir /path/to/your/pdfs \
  --output outputs/parse/annotations.jsonl \
  --image-dir outputs/parse/images

uv run python src/index_documents.py \
  --jsonl outputs/parse/annotations.jsonl \
  --db-dir data/lancedb --table chunks --overwrite

# 3. Start your LLM and launch the chat app
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name model --host 0.0.0.0 --port 8000 --trust-remote-code

# (in another terminal)
uv run chainlit run app.py --host 0.0.0.0 --port 6006 -w -h
```

Open the app using your instance endpoint (not necessarily `localhost`) and start asking questions.  
If you are on Jarvislabs, go to the **Instances** dashboard and click **API** for your running instance to open the endpoint for port `6006`.  
For detailed options, see the full [Quick Start](#quick-start) below.

---

## Overview

This cookbook provides a complete multimodal RAG pipeline with the following capabilities:

- **Layout-aware PDF parsing** — [Nemotron Parse v1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) converts PDF pages into structured Markdown with bounding box coordinates for every element (tables, figures, headings, captions). This preserves document structure that plain text extraction loses.
- **Dual-mode embedding** — Each document chunk gets two vector representations: a text-only embedding (`vector_text`) and an image+text embedding (`vector_image_text`). The image-aware mode captures visual signals from charts, diagrams, and tables.
- **Hybrid retrieval + cross-encoder reranking** — First-stage retrieval combines vector similarity with full-text search in LanceDB. A cross-encoder reranker ([Nemotron Rerank VL](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2)) then re-scores the candidates to improve precision before generation.
- **Grounded generation with source overlays** — The LLM generates answers strictly from the retrieved chunks. Each answer includes clickable source buttons; for parsed PDFs, these show the original page image with bounding boxes drawn over parsed layout elements.
- **Conversational query rewriting** — Follow-up questions like "what about that table?" are automatically rewritten into standalone retrieval queries using recent chat history.

---

## Pipeline

Two phases: **offline** (run once per document set) and **online** (runs while you chat).

### Offline: Parse & Index

```
  Your files on disk
  (PDFs, Markdown, text)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: PARSE (optional, recommended for PDFs)                     │
│                                                                     │
│  src/parse_pdfs_with_vllm.py                                        │
│  Nemotron Parse reads each PDF page, outputs:                       │
│    • Structured Markdown (tables, headings, figures preserved)       │
│    • Layout metadata (bounding boxes, element classes)               │
│    • Rendered page images                                            │
│  Saved as JSONL + image files                                        │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: INDEX                                                      │
│                                                                     │
│  src/index_documents.py                                              │
│  For each document chunk, computes:                                  │
│    • Text-only embedding     → stored as vector_text                 │
│    • Image+text embedding    → stored as vector_image_text           │
│  Also creates a full-text search index on the cleaned text.          │
│  Everything is written to a local LanceDB database.                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Online: Query → Answer

```
  User question
        │
        ▼
  1. Query rewriting — resolves references using chat history
        │
        ▼
  2. Retrieve — hybrid search (vector + full-text) in LanceDB → top_k candidates
        │
        ▼
  3. Rerank — cross-encoder rescores candidates → rerank_top_n results
        │
        ▼
  4. Generate — top gen_top_n chunks injected into prompt → LLM streams answer
        │
        ▼
  5. Sources — clickable buttons to inspect chunks + page image overlays
```

Runtime model placement:
- `embed_model` and `rerank_model` load inside the Chainlit app process (`app.py` / `pipeline.py`)
- your chat model runs in a separate inference server process (for example, vLLM)

---

## Models Used

| Model | Role | Details |
|---|---|---|
| [Nemotron Parse v1.1](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1) | PDF parsing | Converts PDF pages to structured Markdown + layout metadata (bounding boxes, element classes) |
| [Nemotron Embed VL 1B v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) | Embedding | Produces text-only query vectors and text/image+text document vectors (when images are available) |
| [Nemotron Rerank VL 1B v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) | Reranking | Cross-encoder that rescores retrieval candidates (supports text, image, or image+text input) |
| Local LLM server with `v1/chat/completions` + streaming support | Generation | Generates the final answer. Examples in this cookbook use `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` served via [vLLM](https://docs.vllm.ai/) |

---

## Prerequisites

- **NVIDIA GPU** — You need a GPU with enough VRAM to run the embedding, reranking, and chat models. A single A100 (40 GB) or H100 works well.
- **Python 3.10+** (this project uses Python 3.12 by default)
- **[uv](https://docs.astral.sh/uv/)** — a fast Python package manager (used instead of pip)

Performance note:
- For best throughput, keep `app.py` (embedding + reranking models) and the vLLM chat server on different GPUs.
- The `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` example is typically H100-class or multi-GPU; on A100 40 GB, use a smaller chat model.

**Don't have a GPU?** Spin up a GPU instance on [Jarvislabs.ai](https://jarvislabs.ai/), clone this repo, and follow the steps below.

---

## Quick Start

### Step 1 — Clone and Install

```bash
git clone https://github.com/jarvislabsai/cookbook.git
cd cookbook/multimodal-rag-nemotron
MAX_JOBS=4 uv sync
```

> `MAX_JOBS=4` limits parallel compilation jobs to avoid running out of memory during install. Adjust based on your machine.

### Step 2 — Prepare Your Documents

Have your files ready on disk. The scripts accept any directory path via their arguments (`--pdf-dir`, `--data-dir`), so you don't need to move files into a specific folder. The output directories (`data/lancedb`, `outputs/parse/`) are created automatically by the scripts.

### Step 3 — Parse PDFs (Recommended for PDFs)

If your documents are PDFs and layout matters (tables, figures, multi-column text), run the parser first. This uses Nemotron Parse to extract structured content from each page:

```bash
uv run python src/parse_pdfs_with_vllm.py \
  --pdf-dir data/pdfs \
  --glob "*.pdf" \
  --output outputs/parse/annotations.jsonl \
  --image-dir outputs/parse/images
```

**What this produces:**
- `outputs/parse/annotations.jsonl` — one JSON record per page, containing the Markdown text, layout classes (Table, Figure, Text, etc.), and bounding box coordinates
- `outputs/parse/images/` — rendered page images used for visual embedding and source previews

**Parser options:**

| Flag | Default | Description |
|---|---|---|
| `--pdf-dir` | *(required)* | Folder containing your PDF files |
| `--glob` | `*.pdf` | Which PDFs to process |
| `--model` | `nvidia/NVIDIA-Nemotron-Parse-v1.1` | The parser model (local path or HuggingFace ID) |
| `--output` | `outputs/parse/annotations.jsonl` | Where to save the parsed output |
| `--image-dir` | `outputs/parse/images` | Where to save rendered page images |
| `--table-format` | `markdown` | Table output format: `markdown`, `latex`, or `HTML` |
| `--text-format` | `markdown` | Text output format: `markdown` or `plain` |
| `--max-chunks` | *(all)* | Limit the number of pages to parse (useful for testing) |
| `--max-num-seqs` | `64` | vLLM batch size setting |

> **Note:** You can skip this step if you only have Markdown or plain text files, or if you want simple text-only extraction from PDFs. But for the best results with PDFs, parsing is recommended.

### Step 4 — Index Your Documents

Indexing reads your documents (or parsed JSONL), computes embeddings, and stores everything in a local LanceDB database.

**Option A — Index parsed PDFs (recommended for PDFs):**

```bash
uv run python src/index_documents.py \
  --jsonl outputs/parse/annotations.jsonl \
  --db-dir data/lancedb \
  --table chunks \
  --overwrite
```

**Option B — Index files directly from a folder:**

```bash
# Markdown files
uv run python src/index_documents.py \
  --data-dir data/markdown \
  --glob "*.md" \
  --db-dir data/lancedb \
  --table chunks \
  --overwrite

# Plain text files
uv run python src/index_documents.py \
  --data-dir data/text \
  --glob "*.txt" \
  --db-dir data/lancedb \
  --table chunks

# PDFs (text-only extraction, no layout/images)
uv run python src/index_documents.py \
  --data-dir data/pdfs \
  --glob "*.pdf" \
  --db-dir data/lancedb \
  --table chunks
```

> Direct PDF indexing extracts only the text from each page. If you need visual signals (table structure, figure regions, image-based retrieval), use the parse-then-index workflow from Option A.
> Direct PDF indexing is not OCR. For scanned/image-only PDFs, extracted text may be empty or low quality.

**How chunking works (important):**
- Parsed PDF JSONL (Option A): one PDF page = one chunk
- Direct PDF indexing (Option B): one PDF page = one text-only chunk via PDFium extraction
- Direct Markdown/Text indexing (Option B): one file = one chunk

**Indexing options:**

| Flag | Default | Description |
|---|---|---|
| `--jsonl` | *(none)* | Path to parsed JSONL records from Step 3 |
| `--data-dir` | *(none)* | Folder of raw files to index (`.md`, `.txt`, `.pdf`) |
| `--glob` | `*.md` | Which files to pick up from `--data-dir` |
| `--db-dir` | `data/lancedb` | Where to store the LanceDB database |
| `--table` | `chunks` | Name of the table inside LanceDB |
| `--embed-model` | `nvidia/llama-nemotron-embed-vl-1b-v2` | Embedding model |
| `--attn-impl` | `flash_attention_2` | Attention backend (`flash_attention_2`, `eager`, `kernels-community/flash-attn2`) |
| `--batch-size` | `32` | Number of records per embedding batch |
| `--overwrite` | `false` | If set, drops the existing table and re-creates it |

> You must provide at least one of `--jsonl` or `--data-dir`. You can provide both to combine sources.

**When your documents change:**
- If you edit existing files and want the index to reflect the new content, rebuild with `--overwrite`
- Without `--overwrite`, previously indexed IDs are skipped (append/dedupe behavior)

### Step 5 — Start the LLM Server

The chat app sends questions to a local `v1/chat/completions` endpoint. The simplest setup is serving a model with vLLM:

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name model \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

Model sizing note:
- The 30B BF16 example above is generally not a single A100 40 GB target.
- If you are on A100 40 GB, pick a smaller chat model.

This starts a server at `http://localhost:8000/v1` that the app will call for answer generation.
Set `llm_endpoint` to an address reachable from where `app.py` runs (for same-VM setup, `http://127.0.0.1:8000/v1` is correct).

> You can use any self-hosted server that exposes a compatible `v1/chat/completions` route and supports streaming responses (e.g., [vLLM](https://docs.vllm.ai/), [Ollama](https://ollama.com/), [TGI](https://huggingface.co/docs/text-generation-inference/)). Update `llm_endpoint` and `llm_model` in `config.toml`.

Compatibility note:
- Streaming is required for normal answer generation
- Query rewriting also uses a structured JSON response path; if your server does not support it, the app automatically falls back to the original user query

### Step 6 — Launch the Chat App

```bash
uv run chainlit run app.py --host 0.0.0.0 --port 6006 -w -h
```

Then open your browser using your instance URL for port `6006` (for local runs, `http://localhost:6006` works).

On Jarvislabs:
- Open **Instances** in the dashboard
- Click **API** for the running instance
- Use the exposed endpoint for port `6006` to access Chainlit

You'll see a chat interface ready to take questions. Ask about your indexed documents, and the system will retrieve chunks, generate a grounded answer (streamed in real time), and attach clickable **Sources** buttons. Clicking a source shows the chunk text, or — for parsed PDFs — the rendered page image with bounding box overlays for parsed layout elements.

---

## Project Structure

```
multimodal-rag-nemotron/
│
├── app.py                          # Chat app entry point (Chainlit)
├── config.toml                     # All runtime settings in one place
├── pyproject.toml                  # Python project metadata and dependencies
├── build_project.sh                # Helper script to install and build
├── chainlit.md                     # Welcome message shown in the chat UI
│
├── src/
│   ├── parse_pdfs_with_vllm.py     # Step 3: Parse PDFs using Nemotron Parse via vLLM
│   ├── index_documents.py          # Step 4: Embed and store documents in LanceDB
│   ├── pipeline.py                 # Core RAG logic: embed, search, rerank, generate
│   │
│   └── nemotron_parse_utils/       # Helper utilities for Nemotron Parse output
│       ├── __init__.py
│       ├── postprocessing.py       # Extract layout classes, bounding boxes, clean text
│       ├── latex2html.py           # Convert LaTeX tables → HTML → Markdown
│       └── chat_template.jinja     # Chat template for model inference
│
├── data/                           # Created by you (not in git)
│   ├── pdfs/                       # Your input PDF files
│   ├── markdown/                   # Your input Markdown files
│   ├── text/                       # Your input plain text files
│   └── lancedb/                    # LanceDB vector database (created during indexing)
│
└── outputs/                        # Created during parsing (not in git)
    └── parse/
        ├── annotations.jsonl       # Parsed page records
        └── images/                 # Rendered page images
```

---

## Configuration Reference

All runtime settings live in `config.toml`. Here is what each setting does:
This repo reads runtime settings from `config.toml` (no environment variable override layer).

### Database & Retrieval

| Setting | Default | Description |
|---|---|---|
| `db_dir` | `data/lancedb` | Path to the LanceDB database folder. Must match the `--db-dir` you used during indexing. |
| `table_name` | `chunks` | Name of the LanceDB table. Must match `--table` from indexing. |
| `vector_column` | `vector_image_text` | Which vector to search: `vector_text` (text only) or `vector_image_text` (image+text). |
| `search_mode` | `hybrid` | Retrieval strategy: `hybrid` (vector + full-text search) or `vector` (vector only). |
| `top_k` | `16` | How many candidates to fetch from the database before reranking. |
| `rerank_top_n` | `16` | How many candidates to keep after reranking. If higher than retrieved candidates, the effective value is clamped automatically. |
| `gen_top_n` | `5` | How many top chunks to include in the LLM prompt for answer generation. |

### LLM (Answer Generation)

| Setting | Default | Description |
|---|---|---|
| `llm_endpoint` | `http://127.0.0.1:8000/v1` | Base URL of your local chat server exposing `v1/chat/completions` with streaming support. |
| `llm_model` | `model` | Model name sent in API requests. Must match your server's model alias (e.g., `--served-model-name` in vLLM). |
| `max_tokens` | `4096` | Maximum tokens in the generated answer. Lower for shorter/faster responses. |
| `temperature` | `0.2` | Controls randomness. Lower = more deterministic/factual. Higher = more creative. |
| `top_p` | `0.95` | Nucleus sampling threshold. Usually keep between 0.9–1.0. |
| `enable_thinking` | `false` | Forwards `enable_thinking` to the model chat template when supported by your server/model. |

### Embedding & Reranking

| Setting | Default | Description |
|---|---|---|
| `embed_model` | `nvidia/llama-nemotron-embed-vl-1b-v2` | Model used to create vector embeddings for queries and documents. |
| `rerank_model` | `nvidia/llama-nemotron-rerank-vl-1b-v2` | Cross-encoder model used to re-score retrieved candidates. |
| `rerank_modality` | `image_text` | What the reranker looks at: `text`, `image`, or `image_text`. |
| `rerank_batch_size` | `16` | How many candidates to rerank in one batch. Lower this if you're running low on GPU memory. |
| `attn_implementation` | `flash_attention_2` | Attention backend for model loading. Supported: `flash_attention_2`, `eager`, `kernels-community/flash-attn2`. |

### Conversation & Query Rewriting

| Setting | Default | Description |
|---|---|---|
| `history_max_turns` | `5` | Number of recent conversation turns used for query rewriting. |
| `condense_max_tokens` | `100` | Max tokens for the rewritten query. |
| `condense_temperature` | `0.6` | Temperature for query rewriting. Lower = more deterministic. |
| `condense_top_p` | `0.95` | Nucleus sampling for query rewriting. |

### Source Visualization

| Setting | Default | Description |
|---|---|---|
| `show_bbox` | `true` | Draw bounding boxes on source images to highlight layout elements. |
| `show_labels` | `true` | Show class labels (Table, Figure, Text, etc.) on bounding boxes. |

---

## Tuning Tips

**Answers missing relevant information?**
- Increase `top_k` to fetch more candidates from the database
- Increase `rerank_top_n` to keep more candidates after reranking
- Increase `gen_top_n` to give the LLM more context (at the cost of more tokens)

**Answers are slow or GPU memory is tight?**
- Reduce `rerank_batch_size` (try 4 or 8)
- Reduce `gen_top_n` to send fewer chunks to the LLM
- Set `attn_implementation = "eager"` if flash attention causes issues
- Lower `max_tokens` for shorter answers

**Working with image-heavy PDFs (charts, tables, diagrams)?**
- Use the parse-then-index workflow (Step 3 → Step 4 Option A)
- Keep defaults: `vector_column = "vector_image_text"` and `rerank_modality = "image_text"`

**Working with plain text or Markdown?**
- Switch to `vector_column = "vector_text"` and `rerank_modality = "text"`
- Direct indexing (Step 4 Option B) is fine — no need for the parser

---

## Troubleshooting

**Answers are generic, empty, or seem unrelated to my documents**
- Most likely the app is reading a different database or table than what you indexed. Check that `db_dir` and `table_name` in `config.toml` match the `--db-dir` and `--table` you used when running `index_documents.py`.

**"FTS index not created" warning during indexing**
- The full-text search index on `text_clean` could not be created. This means hybrid search won't work. You can either re-run indexing (which retries FTS creation), or set `search_mode = "vector"` in `config.toml` as a workaround.

**`flash-attn` fails to install or crashes at runtime**
- Flash Attention can be tricky to build on some systems. Set `attn_implementation = "eager"` in `config.toml`, and when indexing, add `--attn-impl eager`.

**LLM server connection errors**
- Make sure your vLLM (or other) server is running and accessible at the URL in `llm_endpoint`.
- Check that `llm_model` matches the `--served-model-name` your server is using.

**Out of GPU memory**
- Reduce `rerank_batch_size` first (e.g., from 16 to 4)
- Consider running the embedding/reranking models and the chat LLM on separate GPUs
- Use a smaller chat LLM

---

## References

- [Nemotron Parse Overview (NVIDIA Docs)](https://docs.nvidia.com/nim/vision-language-models/1.5.0/examples/nemotron-parse/overview.html)
- [Nemotron Parse v1.1 — Model Card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1)
- [Nemotron Embed VL 1B v2 — Model Card](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2)
- [Nemotron Rerank VL 1B v2 — Model Card](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2)
- [Llama Nemotron Family Overview (NVIDIA Docs)](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/llama-nemotron.html)
- [Chainlit Documentation](https://docs.chainlit.io/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [uv Package Manager](https://docs.astral.sh/uv/)
