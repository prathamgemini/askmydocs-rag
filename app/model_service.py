"""
Unified Model Service:
- Cloud: Groq (GROQ_MODEL, default qwen/qwen3.8-27b) via the groq SDK
- Local: llama.cpp (bartowski/Ling-3.0-tiny-GGUF:Q4_K_M)
"""
import os
import re
import logging
import threading
from typing import Generator

from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_REASONING_EFFORT,
    LOCAL_MODEL_REPO,
    LOCAL_MODEL_FILE,
    LOCAL_MODELS_DIR,
    LOCAL_MODEL_PATH,
    LOCAL_MODEL_N_CTX,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("model_service")

LOCAL_PROVIDERS = ("llamacpp", "local")

# Reasoning tokens are billed against max_completion_tokens. Short helper calls
# (query rewrite, HyDE) would otherwise come back empty, so keep a floor.
GROQ_MIN_OUTPUT_TOKENS = 1024
GROQ_TIMEOUT_S = 60.0

_groq_client = None
_groq_client_lock = threading.Lock()

_llama_instance = None
_llama_load_lock = threading.Lock()
# llama.cpp contexts are not thread-safe: serialize all local inference.
_llama_infer_lock = threading.Lock()

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def resolve_provider(provider: str | None = None) -> str:
    """Normalize a requested provider to 'groq' or 'llamacpp'."""
    prov = (provider or LLM_PROVIDER or "groq").strip().lower()
    return "llamacpp" if prov in LOCAL_PROVIDERS else "groq"


def is_local_model_downloaded() -> bool:
    """Check if the GGUF model file is already present on local disk."""
    return LOCAL_MODEL_PATH.exists() and LOCAL_MODEL_PATH.stat().st_size > 100_000_000


# ══════════════════════════════════════════════════════════════════
# GROQ (cloud)
# ══════════════════════════════════════════════════════════════════

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        with _groq_client_lock:
            if _groq_client is None:
                api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY") or ""
                if not api_key:
                    raise RuntimeError(
                        "GROQ_API_KEY is not set. Add it to the .env file in the project root "
                        "(GROQ_API_KEY=your_key) and restart the backend."
                    )
                from groq import Groq
                # Fail fast instead of hanging forever if the model stalls
                _groq_client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT_S)
    return _groq_client


def _groq_messages(messages: list[dict]) -> list[dict]:
    """Keep only well-formed, non-empty chat messages."""
    cleaned = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content") or ""
        if not content.strip():
            continue
        if role not in ("system", "user", "assistant"):
            role = "assistant" if role in ("model", "bot") else "user"
        cleaned.append({"role": role, "content": content})
    if not cleaned:
        raise ValueError("No message to send to Groq")
    return cleaned


def _groq_request(messages: list[dict], temperature: float, max_tokens: int, top_p: float, stream: bool) -> dict:
    kwargs = dict(
        model=GROQ_MODEL,
        messages=_groq_messages(messages),
        temperature=temperature,
        max_completion_tokens=max(int(max_tokens), GROQ_MIN_OUTPUT_TOKENS),
        top_p=top_p,
        stream=stream,
        stop=None,
    )
    if GROQ_REASONING_EFFORT:
        kwargs["reasoning_effort"] = GROQ_REASONING_EFFORT
    return kwargs


def _strip_think_stream(deltas) -> Generator[str, None, None]:
    """
    Drop <think>...</think> reasoning blocks from a stream of text deltas,
    including tags that are split across chunk boundaries.
    """
    open_tag, close_tag = "<think>", "</think>"
    buffer = ""
    in_think = False
    for delta in deltas:
        buffer += delta
        while buffer:
            if in_think:
                end = buffer.lower().find(close_tag)
                if end == -1:
                    # Keep only a tail that could be the start of the closing tag
                    buffer = buffer[-(len(close_tag) - 1):]
                    break
                buffer = buffer[end + len(close_tag):].lstrip()
                in_think = False
            else:
                start = buffer.lower().find(open_tag)
                if start != -1:
                    if buffer[:start]:
                        yield buffer[:start]
                    buffer = buffer[start + len(open_tag):]
                    in_think = True
                    continue
                # Hold back a tail that could be the start of an opening tag
                hold = 0
                for n in range(min(len(open_tag) - 1, len(buffer)), 0, -1):
                    if open_tag.startswith(buffer[-n:].lower()):
                        hold = n
                        break
                emit = buffer[:len(buffer) - hold]
                if emit:
                    yield emit
                buffer = buffer[len(buffer) - hold:]
                break
    if buffer and not in_think:
        yield buffer


# ══════════════════════════════════════════════════════════════════
# LLAMA.CPP (local)
# ══════════════════════════════════════════════════════════════════

def get_llama_instance():
    global _llama_instance
    if _llama_instance is not None:
        return _llama_instance

    with _llama_load_lock:
        if _llama_instance is not None:
            return _llama_instance

        from llama_cpp import Llama
        LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if not is_local_model_downloaded():
            logger.info(f"Local model not found at {LOCAL_MODEL_PATH}. Downloading {LOCAL_MODEL_FILE} from HuggingFace ({LOCAL_MODEL_REPO})...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=LOCAL_MODEL_REPO,
                filename=LOCAL_MODEL_FILE,
                local_dir=LOCAL_MODELS_DIR,
            )

        logger.info(f"Loading local llama.cpp model from {LOCAL_MODEL_PATH}...")
        _llama_instance = Llama(
            model_path=str(LOCAL_MODEL_PATH),
            n_ctx=LOCAL_MODEL_N_CTX,
            n_threads=max(1, (os.cpu_count() or 4) - 1),
            verbose=False,
        )
        logger.info("Local llama.cpp Ling 3.0 Tiny model loaded successfully!")
        return _llama_instance


def _local_max_tokens(max_tokens: int) -> int:
    """Leave room for the prompt inside the local context window."""
    return max(64, min(int(max_tokens), LOCAL_MODEL_N_CTX // 4))


# ══════════════════════════════════════════════════════════════════
# UNIFIED API
# ══════════════════════════════════════════════════════════════════

def stream_chat_completion(
    messages: list[dict],
    provider: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    top_p: float = 0.95,
) -> Generator[str, None, None]:
    """
    Unified streaming generator yielding delta text tokens.
    """
    prov = resolve_provider(provider)

    if prov == "llamacpp":
        logger.info(f"Streaming response via Local llama.cpp ({LOCAL_MODEL_FILE})...")
        llm = get_llama_instance()
        with _llama_infer_lock:
            response = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=_local_max_tokens(max_tokens),
                top_p=top_p,
                stream=True,
            )
            for chunk in response:
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
    else:
        logger.info(f"Streaming response via Groq ({GROQ_MODEL})...")
        client = get_groq_client()
        completion = client.chat.completions.create(
            **_groq_request(messages, temperature, max_tokens, top_p, stream=True)
        )

        def deltas():
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

        yield from _strip_think_stream(deltas())


def generate_chat_text(
    messages: list[dict],
    provider: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 500,
    top_p: float = 0.95,
) -> str:
    """
    Unified non-streaming completion for query rewriting, HyDE, etc.
    """
    prov = resolve_provider(provider)

    if prov == "llamacpp":
        llm = get_llama_instance()
        with _llama_infer_lock:
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=_local_max_tokens(max_tokens),
                top_p=top_p,
                stream=False,
            )
        return (resp["choices"][0]["message"].get("content") or "").strip()
    else:
        client = get_groq_client()
        resp = client.chat.completions.create(
            **_groq_request(messages, temperature, max_tokens, top_p, stream=False)
        )
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        return _THINK_BLOCK.sub("", text).strip()
