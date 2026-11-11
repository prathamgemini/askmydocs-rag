"""
AskMyDocs — HyDE Generator Module
=========================================================
Generates hypothetical questions per chunk during document ingestion to boost
retrieval accuracy (matching question-to-question rather than question-to-paragraph).
"""

import time
import logging
from config import HYDE_QUESTIONS_PER_CHUNK, HYDE_PROMPT, LLM_PROVIDER, LOG_LEVEL, LOG_FORMAT
from model_service import generate_chat_text

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("hyde_generator")

MAX_CONSECUTIVE_FAILURES = 3


def generate_questions_for_chunk(chunk: dict) -> list[str]:
    """Generate 2-3 hypothetical user questions whose answer is found in the chunk."""
    content = chunk.get("content", "").strip()
    if len(content) < 80:
        return []

    doc_title = chunk.get("doc_title", "Document")
    section = chunk.get("section", doc_title)

    prompt = HYDE_PROMPT.format(
        doc_title=doc_title,
        section=section,
        content=content[:1500],
        n=HYDE_QUESTIONS_PER_CHUNK,
    )

    response = generate_chat_text(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    if not response:
        return []

    questions = []
    for line in response.splitlines():
        line = line.strip().lstrip("0123456789.-)*• ").strip()
        if len(line) > 10 and "?" in line:
            questions.append(line)

    return questions[:HYDE_QUESTIONS_PER_CHUNK]


def generate_hyde_for_chunks(chunks: list[dict]) -> list[dict]:
    """
    Enrich an in-memory list of document chunks with HyDE questions during ingestion.
    Gracefully skips on rate limits or API errors without failing ingestion.
    """
    count = 0
    consecutive_failures = 0
    for chunk in chunks:
        if chunk.get("chunk_type") in ("image", "section_index"):
            chunk["hyde_questions"] = []
            continue

        if chunk.get("hyde_questions"):
            continue

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            # LLM unavailable (missing key, quota, offline): index without HyDE
            chunk["hyde_questions"] = []
            continue

        try:
            chunk["hyde_questions"] = generate_questions_for_chunk(chunk)
            consecutive_failures = 0
        except Exception as e:
            chunk["hyde_questions"] = []
            consecutive_failures += 1
            logger.warning(f"HyDE generation failed for a chunk: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.warning("HyDE disabled for the rest of this document after repeated LLM failures")

        count += 1
        # Micro-throttle cloud API calls to stay under rate limits
        if LLM_PROVIDER == "groq" and count > 2:
            time.sleep(0.5)

    return chunks