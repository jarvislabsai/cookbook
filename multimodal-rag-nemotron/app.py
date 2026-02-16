"""Chainlit app for generic multimodal RAG on user documents."""

import asyncio
import sys
import threading
import uuid
from pathlib import Path

import chainlit as cl

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pipeline as rag

CONFIG_PATH = ROOT_DIR / "config.toml"


@cl.cache
def load_runtime():
    runtime_config = rag.load_config(CONFIG_PATH)
    runtime_pipeline = rag.RAGPipeline(runtime_config)
    return runtime_config, runtime_pipeline


config, pipeline = load_runtime()

HISTORY_KEY = "history"
SOURCES_BY_KEY = "sources_by_key"
TURN_LOG_ID_KEY = "turn_log_id"
MAX_SOURCE_PAYLOADS = 30
GENERIC_ERROR_MSG = "Something went wrong. Please try again."
WORKING_STATUS_FRAMES = ("Working on it", "Working on it.", "Working on it..", "Working on it...")


def trim_history(history, max_turns):
    if not history:
        return []
    return history[-max_turns * 2 :]


def append_history(history, user_text, assistant_text, max_turns):
    updated = list(history)
    updated.append({"role": "user", "content": user_text})
    updated.append({"role": "assistant", "content": assistant_text})
    return trim_history(updated, max_turns)


def with_sources_payload(sources_by_key, source_key, sources_payload, max_entries=MAX_SOURCE_PAYLOADS):
    updated = dict(sources_by_key)
    updated[source_key] = sources_payload
    while len(updated) > max_entries:
        oldest_key = next(iter(updated))
        del updated[oldest_key]
    return updated


def build_sources_payload(top_rows):
    source_key = uuid.uuid4().hex
    actions = []
    sources_payload = []

    for idx, row in enumerate(top_rows, start=1):
        label = rag.build_chunk_label(row)
        display_label = rag.build_chunk_display_label(row)
        overlay = rag.render_overlay_bytes(
            row.get("image_bytes"),
            row.get("layout_elements_json"),
            show_boxes=config.show_bbox,
            show_labels=config.show_labels,
        )

        sources_payload.append(
            {
                "label": label,
                "display_label": display_label,
                "image": overlay,
                "text": row.get("content_markdown", "") or "",
            }
        )
        actions.append(
            cl.Action(
                name="open_source",
                payload={"source_key": source_key, "index": idx - 1},
                label=f"[{idx}] {display_label}",
            )
        )

    return source_key, sources_payload, actions


def store_sources_payload(source_key, sources_payload):
    sources_by_key = cl.user_session.get(SOURCES_BY_KEY) or {}
    cl.user_session.set(
        SOURCES_BY_KEY,
        with_sources_payload(sources_by_key, source_key, sources_payload),
    )


def get_sources_payload(source_key):
    sources_by_key = cl.user_session.get(SOURCES_BY_KEY) or {}
    return sources_by_key.get(source_key) or []


def condense_retrieval_query(history, query, max_turns, log_prefix):
    return pipeline.condense_query(
        history,
        query,
        endpoint=config.llm_endpoint,
        model_name=config.llm_model,
        max_turns=max_turns,
        log_prefix=log_prefix,
    )


def retrieve_rows(query):
    results, _, _ = pipeline.retrieve_and_rerank(
        query=query,
        vector_column=config.vector_column,
        search_mode=config.search_mode,
        top_k=config.top_k,
        rerank_top_n=config.rerank_top_n,
        rerank_modality=config.rerank_modality,
        rerank_batch_size=config.rerank_batch_size,
    )
    return results


def get_turn_log_prefix():
    turn_id = cl.user_session.get(TURN_LOG_ID_KEY)
    if not turn_id:
        turn_id = uuid.uuid4().hex[:8]
        cl.user_session.set(TURN_LOG_ID_KEY, turn_id)
        print("\n\n")
    return f"[turn:{turn_id}]"


async def stream_answer_tokens(answer_msg, messages, turn_prefix):
    queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            token_stream = pipeline.stream_chat_completion(
                config.llm_endpoint,
                config.llm_model,
                messages,
                config.max_tokens,
                config.temperature,
                config.top_p,
                enable_thinking=config.enable_thinking,
            )
            for token in token_stream:
                loop.call_soon_threadsafe(queue.put_nowait, {"token": token})
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, {"done": True})

    threading.Thread(target=worker, daemon=True).start()

    answer_text = ""
    usage = None
    while True:
        item = await queue.get()
        if item.get("done"):
            break
        if "error" in item:
            raise RuntimeError(item["error"])
        token = item["token"]
        if "usage" in token:
            usage = token["usage"]
            continue
        content = token.get("content")
        if content:
            answer_text += content
            await answer_msg.stream_token(content)
    await answer_msg.update()
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        print(
            f"{turn_prefix}[token_usage] prompt={prompt_tokens} "
            f"completion={completion_tokens} total={total_tokens}"
        )
    else:
        print(f"{turn_prefix}[token_usage] unavailable")
    return answer_text


async def run_generation(answer_msg, messages, turn_prefix):
    try:
        return await stream_answer_tokens(answer_msg, messages, turn_prefix)
    except Exception:
        answer_msg.content = GENERIC_ERROR_MSG
        await answer_msg.update()
        return answer_msg.content


async def animate_working_status(message, stop_event, interval=0.35):
    frame_idx = 0
    while not stop_event.is_set():
        message.content = WORKING_STATUS_FRAMES[frame_idx % len(WORKING_STATUS_FRAMES)]
        await message.update()
        frame_idx += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            continue


async def stop_working_status(stop_event, status_task):
    stop_event.set()
    try:
        await status_task
    except Exception:
        pass


@cl.on_message
async def main(message: cl.Message):
    query = (message.content or "").strip()
    if not query:
        await cl.Message(content="Please enter a question.").send()
        return
    turn_prefix = get_turn_log_prefix()
    print(f"{turn_prefix}[user_query] {query}")

    answer_msg = cl.Message(content=WORKING_STATUS_FRAMES[0])
    await answer_msg.send()
    stop_event = asyncio.Event()
    status_task = asyncio.create_task(animate_working_status(answer_msg, stop_event))

    try:
        max_turns = config.history_max_turns
        history = trim_history(cl.user_session.get(HISTORY_KEY) or [], max_turns)
        retrieval_query = await asyncio.to_thread(
            condense_retrieval_query,
            history,
            query,
            max_turns,
            turn_prefix,
        )
        if retrieval_query != query:
            print(f"{turn_prefix}[condensed_query] {retrieval_query}")
        results = await asyncio.to_thread(retrieve_rows, retrieval_query)
    except Exception:
        await stop_working_status(stop_event, status_task)
        answer_msg.content = GENERIC_ERROR_MSG
        await answer_msg.update()
        return
    except BaseException:
        await stop_working_status(stop_event, status_task)
        raise

    if results is None or results.empty:
        await stop_working_status(stop_event, status_task)
        answer_msg.content = GENERIC_ERROR_MSG
        await answer_msg.update()
        return

    rows = results.to_dict("records")
    context_count = min(len(rows), config.gen_top_n)
    print(f"{turn_prefix}[retrieved_sources] count={len(rows)} to_context={context_count}")
    top_rows = rows[: config.gen_top_n]
    sources_text, _ = pipeline.build_sources(top_rows)
    messages = pipeline.build_messages(query, sources_text, history=history)

    await stop_working_status(stop_event, status_task)
    answer_msg.content = ""
    await answer_msg.update()
    answer_text = await run_generation(answer_msg, messages, turn_prefix)

    source_key, sources_payload, actions = build_sources_payload(top_rows)
    store_sources_payload(source_key, sources_payload)
    answer_msg.content = f"{answer_text}\n\n**Sources** (click to open)".strip()
    answer_msg.actions = actions
    await answer_msg.update()

    history = append_history(history, query, answer_text, max_turns)
    cl.user_session.set(HISTORY_KEY, history)


@cl.action_callback("open_source")
async def on_open_source(action: cl.Action):
    payload = action.payload or {}
    source_key = payload.get("source_key")
    sources = get_sources_payload(source_key)
    try:
        index = int(payload.get("index", -1))
    except (TypeError, ValueError):
        return
    if index < 0 or index >= len(sources):
        return

    src = sources[index]
    if src.get("image"):
        image = cl.Image(
            name=f"source_{source_key}_{index + 1}",
            content=src["image"],
            mime="image/png",
            size="large",
        )
        title = f"Source [{index + 1}] {src['display_label']}"
        await cl.ElementSidebar.set_title(title)
        await cl.ElementSidebar.set_elements([image])
        return

    text = src.get("text") or "(No source text available.)"
    title = f"Source [{index + 1}] {src['display_label']}"
    preview = text if len(text) <= 5000 else text[:5000] + "\n\n... (truncated)"
    await cl.Message(content=f"### {title}\n\n{preview}").send()
