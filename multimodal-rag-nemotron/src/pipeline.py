"""Shared multimodal RAG pipeline logic for the Chainlit app."""

import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import lancedb
import pandas as pd
import requests
import tomli
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor

DEFAULT_ATTN_IMPL = "flash_attention_2"

SYSTEM_PROMPT = (
    "You are a document-grounded AI assistant. "
    "For document-grounded questions, use only the provided sources to answer. "
    "If the user asks a general conversational question (for example: greetings, who you are, or what you can do), "
    "answer briefly from your assistant role. "
    "If a document-grounded answer is not in the sources, reply exactly: "
    "\"I am sorry, I can't answer that. Please rephrase your question or ask a different question.\" "
    "Be precise and complete: include all relevant details found in the sources and avoid omissions. "
    "Avoid terse responses; provide a fuller explanation with any qualifiers or context from the sources. "
    "Prefer readable prose with short paragraphs; use bullets only when they add clarity. "
    "If the answer is a list, include brief context for each item in sentence form. "
    "Start with a 1-2 sentence summary, then give details. "
    "Format the answer clearly in Markdown with short headings when appropriate. "
    "Do not include any thoughts or reasoning in the final answer. "
    "Do not include citations in the answer; citations will be added by the system."
)

CONDENSE_PROMPT = (
    "You rewrite a follow-up user message into one standalone search query for retrieval. "
    "Rules: "
    "1) Do not answer the question. "
    "2) Use chat history only to resolve references like it/that/they/there. "
    "3) Keep the output concise and specific for document retrieval. "
    '4) Return strict JSON only: {"query": "..."}. '
    "5) The query must end with a question mark."
)

ENV_OVERRIDES = {
    "db_dir": ("RAG_DB_DIR", str),
    "table_name": ("RAG_TABLE_NAME", str),
    "vector_column": ("RAG_VECTOR_COLUMN", str),
    "search_mode": ("RAG_SEARCH_MODE", str),
    "top_k": ("RAG_TOP_K", int),
    "rerank_top_n": ("RAG_RERANK_TOP_N", int),
    "gen_top_n": ("RAG_GEN_TOP_N", int),
    "llm_endpoint": ("RAG_LLM_ENDPOINT", str),
    "llm_model": ("RAG_LLM_MODEL", str),
    "max_tokens": ("RAG_MAX_TOKENS", int),
    "temperature": ("RAG_TEMPERATURE", float),
    "top_p": ("RAG_TOP_P", float),
    "rerank_batch_size": ("RAG_RERANK_BATCH_SIZE", int),
    "rerank_modality": ("RAG_RERANK_MODALITY", str),
    "enable_thinking": ("RAG_ENABLE_THINKING", bool),
    "attn_implementation": ("RAG_ATTN_IMPLEMENTATION", str),
    "embed_model": ("RAG_EMBED_MODEL", str),
    "rerank_model": ("RAG_RERANK_MODEL", str),
    "history_max_turns": ("RAG_HISTORY_MAX_TURNS", int),
    "condense_max_tokens": ("RAG_CONDENSE_MAX_TOKENS", int),
    "condense_temperature": ("RAG_CONDENSE_TEMPERATURE", float),
    "condense_top_p": ("RAG_CONDENSE_TOP_P", float),
    "show_bbox": ("RAG_SHOW_BBOX", bool),
    "show_labels": ("RAG_SHOW_LABELS", bool),
}


def _parse_bool(raw_value):
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got {raw_value!r}")


def _coerce_env(raw_value, caster):
    if caster is bool:
        return _parse_bool(raw_value)
    return caster(raw_value)


def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as handle:
        data = tomli.load(handle)
    for key, (env_name, caster) in ENV_OVERRIDES.items():
        raw_value = os.getenv(env_name)
        if raw_value is None or raw_value == "":
            continue
        try:
            data[key] = _coerce_env(raw_value, caster)
        except ValueError as exc:
            raise ValueError(f"Invalid value for {env_name}: {raw_value!r}") from exc
    return SimpleNamespace(**data)


def l2_normalize(tensor, eps=1e-12):
    return tensor / (tensor.norm(p=2, dim=-1, keepdim=True) + eps)


def load_embed_model(model_id, attn_impl=DEFAULT_ATTN_IMPL):
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()
    model.processor.max_input_tiles = 6
    model.processor.use_thumbnail = True
    return model


def load_rerank_model(model_id, attn_impl=DEFAULT_ATTN_IMPL, modality="text"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.max_input_tiles = 6
    processor.use_thumbnail = True
    if modality == "image":
        processor.rerank_max_length = 2048
    elif modality == "text":
        processor.rerank_max_length = 8192
    else:
        processor.rerank_max_length = 10240
    return model, processor


def parse_elements(elements_json):
    if not elements_json:
        return []
    try:
        return json.loads(elements_json)
    except json.JSONDecodeError:
        return []


def color_for_class(cls):
    palette = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
    ]
    return palette[hash(cls) % len(palette)]


def load_image_from_bytes(raw_bytes):
    return Image.open(io.BytesIO(raw_bytes)).convert("RGB")


def draw_annotations(image, elements, show_labels=True):
    drawn = image.copy()
    draw = ImageDraw.Draw(drawn)
    font = ImageFont.load_default()

    for element in elements:
        bbox = element.get("bbox_abs") or element.get("bbox")
        cls = element.get("class", "Unknown")
        if not bbox or len(bbox) != 4:
            continue
        left, top, right, bottom = bbox
        left = max(0, min(left, image.width))
        right = max(0, min(right, image.width))
        top = max(0, min(top, image.height))
        bottom = max(0, min(bottom, image.height))
        color = color_for_class(cls)
        draw.rectangle([left, top, right, bottom], outline=color, width=2)
        if show_labels:
            draw.text((left + 2, top + 2), cls, fill=color, font=font)
    return drawn


def render_overlay_bytes(image_bytes, elements_json, show_boxes=True, show_labels=True, fmt="PNG"):
    if not image_bytes:
        return None
    image = load_image_from_bytes(image_bytes)
    if show_boxes:
        elements = parse_elements(elements_json)
        image = draw_annotations(image, elements, show_labels)
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def build_chunk_label(row):
    chunk_num = int(row.get("chunk_index", 0)) + 1
    return f"{row.get('source_name', 'document')}:{chunk_num}"


def build_chunk_display_label(row):
    chunk_num = int(row.get("chunk_index", 0)) + 1
    return f"{row.get('source_name', 'document')} · chunk {chunk_num}"


class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.attn_impl = config.attn_implementation
        self.vector_db = lancedb.connect(self.config.db_dir)
        self.table = self.vector_db.open_table(self.config.table_name)
        self.embed_model = load_embed_model(config.embed_model, attn_impl=self.attn_impl)
        self.rerank_model, self.rerank_processor = load_rerank_model(
            config.rerank_model,
            attn_impl=self.attn_impl,
            modality=config.rerank_modality,
        )

    def embed_query(self, query):
        self.embed_model.processor.p_max_length = 8192
        with torch.inference_mode():
            vec = self.embed_model.encode_queries([query])
        vec = l2_normalize(vec).float().cpu().numpy()[0]
        return vec

    def vector_search(self, query_vec, vector_column, limit):
        return (
            self.table.search(query_vec, vector_column_name=vector_column).distance_type("dot").limit(limit).to_pandas()
        )

    def hybrid_search(self, query_vec, query_text, vector_column, limit):
        return (
            self.table.search(
                query_type="hybrid",
                vector_column_name=vector_column,
                fts_columns="text_clean",
            )
            .vector(query_vec)
            .text(query_text)
            .distance_type("dot")
            .limit(limit)
            .to_pandas()
        )

    def rerank_candidates(
        self,
        query,
        rows,
        modality="text",
        batch_size=16,
    ):
        include_image = modality in ("image", "image_text")
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            examples = []
            for row in batch_rows:
                doc_image = ""
                if include_image and row.get("image_bytes"):
                    doc_image = load_image_from_bytes(row.get("image_bytes"))
                examples.append(
                    {
                        "question": query,
                        "doc_text": row.get("content_markdown", "") or "",
                        "doc_image": doc_image,
                    }
                )
            batch = self.rerank_processor.process_queries_documents_crossencoder(examples)
            batch = {k: (v.to(self.rerank_model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.inference_mode():
                outputs = self.rerank_model(**batch)
            scores = outputs.logits.squeeze(-1).float().cpu()
            probs = torch.sigmoid(scores).tolist()
            for row, score in zip(batch_rows, probs):
                row["rerank_score"] = score
        return sorted(rows, key=lambda r: r.get("rerank_score", 0.0), reverse=True)

    def retrieve_and_rerank(
        self,
        query,
        vector_column=None,
        search_mode=None,
        top_k=None,
        rerank_top_n=None,
        rerank_modality=None,
        rerank_batch_size=None,
    ):
        timings = {}
        start = time.perf_counter()
        query_vec = self.embed_query(query)
        mode = (search_mode or self.config.search_mode).lower()
        if mode == "hybrid":
            raw_results = self.hybrid_search(
                query_vec,
                query,
                vector_column or self.config.vector_column,
                top_k or self.config.top_k,
            )
        else:
            raw_results = self.vector_search(
                query_vec,
                vector_column or self.config.vector_column,
                top_k or self.config.top_k,
            )
        timings["retrieval_s"] = time.perf_counter() - start

        rerank_start = time.perf_counter()
        results = raw_results
        if not raw_results.empty:
            rows = raw_results.to_dict("records")
            reranked = self.rerank_candidates(
                query,
                rows,
                modality=rerank_modality or self.config.rerank_modality,
                batch_size=rerank_batch_size or self.config.rerank_batch_size,
            )
            top_n = min(rerank_top_n or self.config.rerank_top_n, len(reranked))
            results = pd.DataFrame(reranked[:top_n])
        timings["rerank_s"] = time.perf_counter() - rerank_start
        return results, raw_results, timings

    def build_sources(self, rows):
        blocks = []
        labels = []
        for row in rows:
            label = build_chunk_label(row)
            labels.append(label)
            text = row.get("content_markdown", "") or ""
            blocks.append(f"### Source: [{label}]\n\n{text}")
        return "\n\n---\n\n".join(blocks), labels

    def build_messages(self, query, sources_text, history=None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        user_prompt = f"Question: {query}\n\nSources:\n{sources_text}\n\nAnswer:"
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def chat_completion(
        self,
        endpoint,
        model_name,
        messages,
        max_tokens=256,
        temperature=0.2,
        top_p=1.0,
        enable_thinking=None,
        structured_outputs=None,
        timeout=60,
    ):
        url = endpoint.rstrip("/") + "/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        if structured_outputs is not None:
            payload["structured_outputs"] = structured_outputs
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"].get("content") or "").strip()

    def condense_query(self, history, query, endpoint=None, model_name=None, max_turns=5, log_prefix=""):
        if not history:
            return query
        endpoint = endpoint or self.config.llm_endpoint
        model_name = model_name or self.config.llm_model

        keep = max_turns * 2
        trimmed = history[-keep:]
        messages = self._build_condense_messages(trimmed, query)
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Standalone retrieval query rewritten from the latest user question "
                        "using conversation history to resolve references."
                    ),
                    "minLength": 5,
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }
        try:
            condensed = self.chat_completion(
                endpoint,
                model_name,
                messages,
                max_tokens=self.config.condense_max_tokens,
                temperature=self.config.condense_temperature,
                top_p=self.config.condense_top_p,
                enable_thinking=False,
                structured_outputs={"json": schema},
            )
            rewritten_query = self._extract_condensed_query(condensed, fallback=query)
            print(f"{log_prefix}[condense_query][raw_response] {condensed}")
            return rewritten_query
        except Exception:
            return query

    def _build_condense_messages(self, history, query):
        history_lines = []
        for item in history:
            role = (item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = (item.get("content") or "").strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            history_lines.append(f"{speaker}: {content}")

        history_block = "\n".join(history_lines) if history_lines else "(empty)"
        user_content = (
            "Conversation history:\n"
            f"{history_block}\n\n"
            "Latest user question:\n"
            f"{query}\n\n"
            'Return only JSON with one key exactly: {"query": "..."}'
        )
        return [
            {"role": "system", "content": CONDENSE_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _extract_condensed_query(self, raw_text, fallback):
        text = (raw_text or "").strip()
        if not text:
            return fallback

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return fallback
        candidate = payload.get("query", fallback).strip()

        return candidate

    def stream_chat_completion(
        self,
        endpoint,
        model_name,
        messages,
        max_tokens,
        temperature,
        top_p,
        enable_thinking=False,
        timeout=180,
    ):
        url = endpoint.rstrip("/") + "/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    yield {"usage": usage}

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if not isinstance(delta, dict):
                    continue

                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                content = delta.get("content")
                if reasoning:
                    yield {"reasoning": reasoning}
                if content:
                    yield {"content": content}
