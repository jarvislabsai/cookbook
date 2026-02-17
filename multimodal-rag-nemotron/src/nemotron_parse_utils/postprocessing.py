import re
from html import unescape

from .latex2html import convert_html_tables_to_markdown, latex_table_to_html

_CLASS_BBOX_RE = re.compile(
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)>(.*?)"
    r"<x_(\d+(?:\.\d+)?)><y_(\d+(?:\.\d+)?)><class_([^>]+)>",
    re.DOTALL,
)


def extract_classes_bboxes(text: str):
    classes = []
    bboxes = []
    texts = []
    for m in _CLASS_BBOX_RE.finditer(text):
        x1, y1, block_text, x2, y2, cls = m.groups()
        classes.append(cls)
        bboxes.append((float(x1), float(y1), float(x2), float(y2)))
        texts.append(block_text)

    classes = ["Formula" if cls == "Inline-formula" else cls for cls in classes]
    assert "Page-number" not in classes

    return classes, bboxes, texts


def transform_bbox_to_original(bbox, original_width, original_height, target_w=1648, target_h=2048):
    """Map normalized bbox from padded resize space back to original page pixels."""
    aspect_ratio = original_width / original_height
    new_width = original_width
    new_height = original_height

    if original_height > target_h:
        new_height = target_h
        new_width = int(new_height * aspect_ratio)

    if new_width > target_w:
        new_width = target_w
        new_height = int(new_width / aspect_ratio)

    resized_width = new_width
    resized_height = new_height

    pad_left = (target_w - resized_width) // 2
    pad_top = (target_h - resized_height) // 2

    left = ((bbox[0] * target_w) - pad_left) * original_width / resized_width
    right = ((bbox[2] * target_w) - pad_left) * original_width / resized_width

    top = ((bbox[1] * target_h) - pad_top) * original_height / resized_height
    bottom = ((bbox[3] * target_h) - pad_top) * original_height / resized_height

    return left, top, right, bottom


def postprocess_text(text, cls="Text", text_format="markdown", table_format="latex", blank_text_in_figures=False):
    assert text_format in ["markdown", "plain"], "Unknown text format. Supported: markdown | plain"
    assert table_format in ["latex", "HTML", "markdown"], "Unknown table format. Supported: latex | HTML | markdown"

    if cls == "Table" and table_format == "HTML":
        text = latex_table_to_html(text)
    elif cls == "Table" and table_format == "markdown":
        text = convert_html_tables_to_markdown(latex_table_to_html(text))
    elif cls != "Table" and text_format == "plain":
        text = convert_mmd_to_plain_text_ours(text)

    if text_format == "markdown":
        text = normalize_markdown_artifacts(text)

    if blank_text_in_figures and cls == "Picture":
        text = ""
    return text


def normalize_markdown_artifacts(text):
    if not text:
        return text
    text = re.sub(r"\\+\(\s*\\+(?:bullet|unknown)\s*\\+\)", "- ", text)
    text = re.sub(r"\\+(?:bullet|unknown)\b", "- ", text)
    text = re.sub(r"-\s+", "- ", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    return unescape(text)


def convert_mmd_to_plain_text_ours(mmd_text):
    mmd_text = re.sub(r"<sup>(.*?)</sup>", r"^{\\1}", mmd_text, flags=re.DOTALL)
    mmd_text = re.sub(r"<sub>(.*?)</sub>", r"_{\\1}", mmd_text, flags=re.DOTALL)
    mmd_text = mmd_text.replace("<br>", "\n")

    mmd_text = re.sub(r"#+\s", "", mmd_text)
    mmd_text = re.sub(r"\*\*(.*?)\*\*", r"\1", mmd_text)
    mmd_text = re.sub(r"\*(.*?)\*", r"\1", mmd_text)
    mmd_text = re.sub(r"(?<!\w)_([^_]+)_", r"\1", mmd_text)

    return mmd_text.strip()
