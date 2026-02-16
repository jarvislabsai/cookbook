# Generic Multimodal RAG (PDF + Markdown)

Production-oriented RAG recipe for user documents using:
- Local embedding + reranking models
- LanceDB for retrieval storage
- OpenAI-compatible local inference server (vLLM recommended)
- Chainlit chat UI

## Run Directory

```bash
cd /Users/atharva/Work/cookbook/multimodal-rag-nemotron
```

## 1) Install Dependencies

```bash
MAX_JOBS=4 uv sync
```

## 2) Prepare Data and Build Index

### Option A: Parse PDFs with VLM parser, then index

```bash
uv run python src/parse_pdfs_with_vllm.py \
  --pdf-dir data/pdfs \
  --glob "*.pdf" \
  --output outputs/parse/annotations.jsonl \
  --image-dir outputs/parse/images

uv run python src/index_documents.py \
  --jsonl outputs/parse/annotations.jsonl \
  --db-dir data/lancedb \
  --table chunks \
  --overwrite
```

### Option B: Index Markdown directly

```bash
uv run python src/index_documents.py \
  --markdown-dir data/markdown \
  --markdown-glob "*.md" \
  --db-dir data/lancedb \
  --table chunks \
  --overwrite
```

### Option C: Combine parsed PDFs + Markdown in one index

```bash
uv run python src/index_documents.py \
  --jsonl outputs/parse/annotations.jsonl \
  --markdown-dir data/markdown \
  --markdown-glob "*.md" \
  --db-dir data/lancedb \
  --table chunks \
  --overwrite
```

## 3) Start Your Local LLM Server

Example (vLLM):

```bash
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name model \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

## 4) Launch Chat App

```bash
uv run chainlit run app.py --host 0.0.0.0 --port 6006 -w -h
```

## Configuration

Edit `config.toml` for defaults.

Useful environment overrides (optional):

```bash
export RAG_LLM_ENDPOINT="http://127.0.0.1:8000/v1"
export RAG_LLM_MODEL="model"
export RAG_DB_DIR="data/lancedb"
export RAG_TABLE_NAME="chunks"
export RAG_VECTOR_COLUMN="vector_text"
export RAG_RERANK_MODALITY="text"
```

See `.env.example` for full override list.

## Troubleshooting

- OOM during reranking:
  - Lower `rerank_batch_size`.
  - Use `rerank_modality = "text"`.
- Empty answers:
  - Confirm table exists in LanceDB and `table_name` matches.
- Missing local model response:
  - Check `llm_endpoint` and `llm_model` in `config.toml`.
