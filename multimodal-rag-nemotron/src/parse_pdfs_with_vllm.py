#!/usr/bin/env python
"""Parse PDF files with a vision-language model via vLLM and emit generic JSONL records."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from vllm import LLM, SamplingParams

from nemotron_parse_utils.postprocessing import (
    extract_classes_bboxes,
    postprocess_text,
    transform_bbox_to_original,
)

TASK_PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown>"
RENDER_DPI = 300
MIN_W = 1024
MIN_H = 1280
MAX_W = 1648
MAX_H = 2048


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bitmap_to_numpy(bitmap):
    mode = bitmap.mode
    arr = bitmap.to_numpy().copy()
    if mode in {"BGRA", "BGRX"}:
        arr = arr[..., [2, 1, 0, 3]]
    elif mode == "BGR":
        arr = arr[..., [2, 1, 0]]
    return arr


def ensure_rgb(image):
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def resize_to_range(image, min_w, min_h, max_w, max_h):
    width, height = image.size
    scale_down = min(max_w / width, max_h / height, 1.0)
    if scale_down < 1.0:
        new_w = max(1, int(round(width * scale_down)))
        new_h = max(1, int(round(height * scale_down)))
        image = image.resize((new_w, new_h), Image.LANCZOS)
        width, height = image.size

    pad_w = max(min_w - width, 0)
    pad_h = max(min_h - height, 0)
    if pad_w > 0 or pad_h > 0:
        new_w = max(width, min_w)
        new_h = max(height, min_h)
        canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        offset = ((new_w - width) // 2, (new_h - height) // 2)
        canvas.paste(image, offset)
        image = canvas

    return image


def render_page(doc, page_index, scale, output_dir):
    page = doc.get_page(page_index)
    textpage = None
    bitmap = None
    try:
        textpage = page.get_textpage()
        raw_text = textpage.get_text_bounded()

        bitmap = page.render(scale=scale)
        image = ensure_rgb(Image.fromarray(bitmap_to_numpy(bitmap)))
        image = resize_to_range(image, MIN_W, MIN_H, MAX_W, MAX_H)

        image_path = output_dir / f"chunk_{page_index:04d}.png"
        image.save(image_path)
        return {
            "chunk_index": page_index,
            "image_path": image_path,
            "raw_text": raw_text,
        }
    finally:
        if textpage is not None:
            textpage.close()
        if bitmap is not None:
            bitmap.close()
        page.close()


def render_pdf_chunks(pdf_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    chunk_infos = []
    scale = RENDER_DPI / 72

    try:
        for idx in range(len(doc)):
            chunk_infos.append(render_page(doc, idx, scale, output_dir))
    finally:
        doc.close()

    return chunk_infos


def load_images(paths):
    return [Image.open(path).convert("RGB") for path in paths]


def build_sampling_params():
    return SamplingParams(
        temperature=0,
        top_k=1,
        repetition_penalty=1.1,
        max_tokens=9000,
        skip_special_tokens=False,
    )


def build_prompts(images):
    return [
        {
            "prompt": TASK_PROMPT,
            "multi_modal_data": {"image": image},
        }
        for image in images
    ]


def parse_output(generated_text, image, table_format, text_format, blank_text_in_figures):
    classes, bboxes_norm, texts = extract_classes_bboxes(generated_text)
    bboxes_abs = [transform_bbox_to_original(bbox, image.width, image.height) for bbox in bboxes_norm]
    texts_post = [
        postprocess_text(
            text,
            cls=cls,
            table_format=table_format,
            text_format=text_format,
            blank_text_in_figures=blank_text_in_figures,
        )
        for text, cls in zip(texts, classes)
    ]
    return classes, bboxes_abs, texts_post


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse PDFs into generic chunk records with vLLM.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--glob", default="*.pdf", help="Glob for PDFs")
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-Parse-v1.1", help="Model ID or local path")
    parser.add_argument("--output", default="outputs/parse/annotations.jsonl", help="Output JSONL path")
    parser.add_argument("--image-dir", default="outputs/parse/images", help="Directory for chunk images")
    parser.add_argument("--table-format", default="markdown", choices=["latex", "HTML", "markdown"])
    parser.add_argument("--text-format", default="markdown", choices=["markdown", "plain"])
    parser.add_argument(
        "--blank-text-in-figures",
        action="store_true",
        help="Blank text for Picture elements (default: off).",
    )
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit number of chunks for a quick run")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="vLLM max_num_seqs setting")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    images_root = Path(args.image_dir)

    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory not found: {pdf_dir}")
    pdf_paths = sorted(pdf_dir.glob(args.glob))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {pdf_dir} with glob {args.glob}")

    llm = LLM(
        model=args.model,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling_params = build_sampling_params()

    records = []
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"Skipping missing PDF: {pdf_path}")
            continue

        source_id = sha256_file(pdf_path)
        chunk_infos = render_pdf_chunks(pdf_path, images_root / pdf_path.stem)
        if args.max_chunks:
            chunk_infos = chunk_infos[: args.max_chunks]

        chunk_count = len(chunk_infos)
        chunk_paths = [info["image_path"] for info in chunk_infos]
        images = load_images(chunk_paths)
        prompts = build_prompts(images)
        outputs = llm.generate(prompts, sampling_params)

        for info, path, image, output in zip(chunk_infos, chunk_paths, images, outputs):
            parser_output = output.outputs[0].text
            layout_classes, layout_bboxes_abs, layout_texts = parse_output(
                parser_output,
                image,
                args.table_format,
                args.text_format,
                args.blank_text_in_figures,
            )
            record = {
                "run_id": run_id,
                "source_name": pdf_path.name,
                "source_id": source_id,
                "chunk_index": info["chunk_index"],
                "chunk_count": chunk_count,
                "image_path": str(path),
                "image_width": image.width,
                "image_height": image.height,
                "parser_model": args.model,
                "parser_output": parser_output,
                "content_markdown": "\n\n".join(layout_texts),
                "raw_text": info.get("raw_text", ""),
                "layout_classes": layout_classes,
                "layout_bboxes_abs": layout_bboxes_abs,
                "layout_texts": layout_texts,
            }
            records.append(record)

    output_path = Path(args.output)
    write_jsonl(output_path, records)
    print(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
