import argparse
import hashlib
import json
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import pypdfium2 as pdfium
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel

DEFAULT_EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
EMBED_DIM = 2048


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index parsed JSONL records or files from a directory into LanceDB."
    )
    parser.add_argument("--jsonl", default=None, help="Path to parsed JSONL records.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing files to index directly (.md, .txt, .pdf).",
    )
    parser.add_argument(
        "--glob",
        default="*.md",
        help="Glob pattern for files inside --data-dir (examples: '*.md', '*.txt', '*.pdf').",
    )
    parser.add_argument("--db-dir", default="data/lancedb")
    parser.add_argument("--table", default="chunks")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        choices=["eager", "flash_attention_2", "kernels-community/flash-attn2"],
        help="Attention backend for transformers.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_pdf_records(path):
    source_id = sha256_file(path)
    doc = pdfium.PdfDocument(str(path))
    chunk_count = len(doc)
    try:
        for page_idx in range(chunk_count):
            page = doc.get_page(page_idx)
            textpage = None
            try:
                textpage = page.get_textpage()
                text = textpage.get_text_bounded() or ""
            finally:
                if textpage is not None:
                    textpage.close()
                page.close()

            yield {
                "source_id": source_id,
                "source_name": path.name,
                "chunk_index": page_idx,
                "chunk_count": chunk_count,
                "image_path": "",
                "parser_output": "",
                "content_markdown": text,
                "raw_text": text,
                "layout_classes": [],
                "layout_bboxes_abs": [],
                "layout_texts": [],
            }
    finally:
        doc.close()


def iter_data_records(data_dir, glob_pattern):
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")

    paths = sorted(path for path in base.glob(glob_pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No files found in {base} with glob {glob_pattern}")

    for path in paths:
        suffix = path.suffix.lower()
        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            source_id = sha256_text(str(path.resolve()))
            yield {
                "source_id": source_id,
                "source_name": path.name,
                "chunk_index": 0,
                "chunk_count": 1,
                "image_path": "",
                "parser_output": "",
                "content_markdown": text,
                "raw_text": text,
                "layout_classes": [],
                "layout_bboxes_abs": [],
                "layout_texts": [],
            }
            continue

        if suffix == ".pdf":
            yield from iter_pdf_records(path)
            continue

        print(f"Skipping unsupported file type: {path}")


def normalize_layout_elements(record):
    classes = record.get("layout_classes", []) or []
    bboxes = record.get("layout_bboxes_abs", []) or []
    texts = record.get("layout_texts", []) or []

    elements = []
    for cls, bbox, text in zip(classes, bboxes, texts):
        elements.append({"class": cls, "bbox_abs": bbox, "text": text})
    return json.dumps(elements, ensure_ascii=True)


def normalize_record(record):
    source_name = (record.get("source_name") or "document").strip() or "document"
    source_id = (record.get("source_id") or "").strip()
    chunk_index = int(record.get("chunk_index", 0))
    chunk_count = int(record.get("chunk_count", 1))

    content_markdown = record.get("content_markdown", "") or ""
    raw_text = record.get("raw_text", "") or content_markdown

    if not source_id:
        source_id = sha256_text(f"{source_name}:{chunk_index}:{content_markdown[:500]}")

    return {
        "id": f"{source_id}:{chunk_index}",
        "source_id": source_id,
        "source_name": source_name,
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "parser_output": record.get("parser_output", "") or "",
        "content_markdown": content_markdown,
        "text_clean": clean_text(content_markdown),
        "raw_text": raw_text,
        "layout_elements_json": normalize_layout_elements(record),
        "image_path": (record.get("image_path") or "").strip(),
    }


def build_schema():
    vector_type = pa.list_(pa.float32(), list_size=EMBED_DIM)
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("source_name", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("chunk_count", pa.int32()),
            pa.field("parser_output", pa.string()),
            pa.field("content_markdown", pa.string()),
            pa.field("text_clean", pa.string()),
            pa.field("raw_text", pa.string()),
            pa.field("layout_elements_json", pa.string()),
            pa.field("image_bytes", pa.binary()),
            pa.field("vector_image_text", vector_type),
            pa.field("vector_text", vector_type),
        ]
    )


def open_table(db, name, schema, overwrite):
    tables = db.list_tables()
    if hasattr(tables, "tables"):
        existing = set(tables.tables)
    else:
        existing = set(tables)
    if overwrite and name in existing:
        db.drop_table(name)
        existing.remove(name)
    if name in existing:
        return db.open_table(name)
    return db.create_table(name, schema=schema)


def load_existing_ids(table):
    existing = table.to_pandas(columns=["id"])
    return set(existing["id"].tolist())


def batch_iter(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_model(model_id, attn_impl):
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


def l2_normalize(tensor, eps=1e-12):
    return tensor / (tensor.norm(p=2, dim=-1, keepdim=True) + eps)


def embed_text_only(model, texts):
    model.processor.p_max_length = 8192
    with torch.inference_mode():
        embeds = model.encode_documents(texts=texts)
    return l2_normalize(embeds).float().cpu().numpy()


def embed_image_text(model, images, texts):
    model.processor.p_max_length = 10240
    with torch.inference_mode():
        embeds = model.encode_documents(images=images, texts=texts)
    return l2_normalize(embeds).float().cpu().numpy()


def iter_input_records(args):
    if not args.jsonl and not args.data_dir:
        raise SystemExit("Provide at least one input source: --jsonl and/or --data-dir")

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
        for record in iter_jsonl(jsonl_path):
            yield record

    if args.data_dir:
        for record in iter_data_records(args.data_dir, args.glob):
            yield record


def main():
    args = parse_args()

    start_time = time.perf_counter()
    db = lancedb.connect(args.db_dir)
    schema = build_schema()
    table = open_table(db, args.table, schema, args.overwrite)

    existing_ids = set()
    if not args.overwrite:
        existing_ids = load_existing_ids(table)

    model = load_model(args.embed_model, args.attn_impl)

    inserted = 0
    deduped = 0
    missing_images = 0
    progress = tqdm(unit="records")

    for batch in batch_iter(iter_input_records(args), args.batch_size):
        rows = []
        texts = []
        images = []

        for raw_record in batch:
            record = normalize_record(raw_record)
            if record["id"] in existing_ids:
                deduped += 1
                continue

            image_bytes = None
            image = None
            image_path = record.pop("image_path")
            if image_path:
                path = Path(image_path)
                if path.exists():
                    image_bytes = path.read_bytes()
                    with Image.open(path) as img:
                        image = img.convert("RGB")
                else:
                    missing_images += 1

            rows.append(
                {
                    "id": record["id"],
                    "source_id": record["source_id"],
                    "source_name": record["source_name"],
                    "chunk_index": record["chunk_index"],
                    "chunk_count": record["chunk_count"],
                    "parser_output": record["parser_output"],
                    "content_markdown": record["content_markdown"],
                    "text_clean": record["text_clean"],
                    "raw_text": record["raw_text"],
                    "layout_elements_json": record["layout_elements_json"],
                    "image_bytes": image_bytes,
                }
            )
            texts.append(record["content_markdown"])
            images.append(image)

        if not rows:
            progress.update(len(batch))
            continue

        vectors_text = embed_text_only(model, texts)
        vectors_image_text = np.array(vectors_text, copy=True)

        with_images_idx = [idx for idx, image in enumerate(images) if image is not None]
        if with_images_idx:
            image_batch = [images[idx] for idx in with_images_idx]
            text_batch = [texts[idx] for idx in with_images_idx]
            image_vectors = embed_image_text(model, image_batch, text_batch)
            for idx, vector in zip(with_images_idx, image_vectors):
                vectors_image_text[idx] = vector

        for row, vec_it, vec_t in zip(rows, vectors_image_text, vectors_text):
            row["vector_image_text"] = vec_it.tolist()
            row["vector_text"] = vec_t.tolist()

        table.add(rows)
        existing_ids.update(row["id"] for row in rows)
        inserted += len(rows)
        progress.update(len(batch))

    progress.close()

    try:
        table.create_fts_index("text_clean")
    except Exception as exc:
        print(f"FTS index not created: {exc}")

    elapsed = time.perf_counter() - start_time
    print(f"Completed indexing in {elapsed:.2f}s")
    print(f"Inserted: {inserted}")
    if deduped:
        print(f"Skipped existing records: {deduped}")
    if missing_images:
        print(f"Records with missing image paths (kept as text-only): {missing_images}")


if __name__ == "__main__":
    main()
