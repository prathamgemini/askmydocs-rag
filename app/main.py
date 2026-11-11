"""
AskMyDocs — FastAPI Backend
===================================================
Provider: Local all-MiniLM-L6-v2 embeddings + Local ChromaDB + Groq LLM (or local llama.cpp)

Endpoints:
  POST /ask            → streaming SSE answer
  GET  /health         → check all services are up
  POST /ingest/file    → multi-modal document ingestion (streaming SSE)
  POST /ingest/url     → web page scraping ingestion (streaming SSE)
  GET  /documents      → list indexed documents
  DELETE /documents/{t}→ delete indexed document

Run:
  uvicorn main:app --reload --port 8000
"""

import sys
import os
import re
import json
import logging
import tempfile
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    LOCAL_MODEL_FILE,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    CONFIDENCE_THRESHOLD,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    LOG_LEVEL,
    LOG_FORMAT,
    UPLOAD_DIR,
    IMAGES_DIR,
    ALLOWED_EXTENSIONS,
)

from model_service import (
    stream_chat_completion,
    generate_chat_text,
    is_local_model_downloaded,
    resolve_provider,
)

from ingestion_service import ingest_file_stream, ingest_url_stream
from scraper import scrape_url
from chunker import chunk_document_content
from embedder import embed_query, embed_chunk_list
from uploader import (
    list_indexed_documents,
    delete_document_chunks,
    upload_chunks_batch,
    get_chroma_client,
    get_chroma_collection,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("main")

MAX_HISTORY = 6
RETRIEVAL_TOP_K_EXPANDED = RETRIEVAL_TOP_K
URL_PRIORITY_SCORE = 0.95


# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the AI Document Knowledge Assistant — an expert system for answering questions based on uploaded documents and web pages.

## YOUR KNOWLEDGE SOURCE
Answer ONLY from the context chunks provided below.
Never guess or make up policy details, numbers, dates, or specifications.

## HANDLING DIFFERENT QUESTION TYPES

OFF-TOPIC:
→ Politely redirect:
  "I am your Document AI Assistant. I can help answer questions based on your uploaded documents and web pages. What would you like to know?"

VAGUE QUESTION (context has partial answer):
→ Give what you found, then ask ONE specific follow-up.

SPECIFIC QUESTION:
→ Answer directly and completely from context.

FOLLOW-UP QUESTION:
→ Use conversation history to understand what "it", "that", "this" refers to.
→ Combine history + current question for a complete answer.

## FORMAT RULES
- Simple fact → 1-3 sentences, then detail if needed
- Criteria/Requirements → bullet list
- Process/Steps → numbered list
- Comparison → markdown table
- Use **bold** for key dates, metrics, numbers, and important terms
- Use `code` for codes, parameters, or technical identifiers
- Never pad with filler ("Great question!", "Certainly!", etc.)

## DIRECT ANSWER RULE
- Questions starting with "Can I", "Is there", "Does", "Will", "Am I" MUST start with YES or NO
- Never hedge with "it seems", "it appears", "however", "based on the context"
- If you know the answer from context → state it directly and confidently
- If context only partially answers → give what you know, then state exactly what is missing

## CITATIONS — MANDATORY SOURCE REQUIRED
End EVERY answer with source citation in this exact format:

📎 **Source:** [section name] — [document title] 🔗 [URL or filename]

RULES:
- Citation is MANDATORY — never omit it
- If context has a source_url → use it exactly
- If multiple sources → list each on its own line with its own source URL/file
- Cite ONLY the sources you actually used to write the answer — never list a source just because it was provided
- NEVER include internal labels like [SOURCE N: ...] or [CONTEXT N] in citations

## LINKS & IMAGES
If context contains reference links or image descriptions — include them inline where helpful.

## CANNOT ANSWER
If answer truly not in context:
"I could not find this information in the current indexed documents."
"""

# ══════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════

print(f"✅ Local Embeddings ready: {EMBEDDING_MODEL} (384d)")

print(f"Connecting to ChromaDB (local: {CHROMA_PERSIST_DIR.name})...")
chroma_client = get_chroma_client()
chroma_collection = get_chroma_collection(chroma_client)
print(f"✅ ChromaDB database ready: '{CHROMA_COLLECTION}' ({chroma_collection.count()} chunks)")

if GROQ_API_KEY:
    print(f"✅ Cloud LLM ready: Groq ({GROQ_MODEL})")
else:
    print("⚠️  GROQ_API_KEY not set — add it to .env in the project root to use the cloud model")
print(f"✅ Local LLM ready: llama.cpp ({LOCAL_MODEL_FILE}, downloaded={is_local_model_downloaded()})")
print("✅ Backend ready\n")

# ══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="AskMyDocs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    provider: str | None = None  # "groq" (default) or "llamacpp"

class UrlIngestRequest(BaseModel):
    url: str


# ══════════════════════════════════════════════════════════════════
# LLM CALLER (Groq / llama.cpp Ling 3.0 Tiny)
# ══════════════════════════════════════════════════════════════════

def llm_call(messages: list, max_tokens: int = 200, stream: bool = False, provider: str = None):
    if stream:
        return stream_chat_completion(
            messages=messages,
            provider=provider,
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=0.95,
        )
    else:
        return generate_chat_text(
            messages=messages,
            provider=provider,
            temperature=0.6,
            max_tokens=max_tokens,
            top_p=0.95,
        )


# ══════════════════════════════════════════════════════════════════
# QUERY CLASSIFIER & REWRITER
# ══════════════════════════════════════════════════════════════════

GREETINGS = ["hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you", "bye"]

def is_greeting(question: str) -> bool:
    q = re.sub(r"[^a-z\s]", "", question.lower()).strip()
    if not q or len(q.split()) > 3:
        # Longer messages are real questions even if they open with "hi"/"hey"
        return False
    return q in GREETINGS or any(q.startswith(g + " ") for g in GREETINGS)


def rewrite_query(question: str, history: list[ChatMessage], provider: str = None) -> str:
    """
    Rewrite user query for better retrieval using conversation context.
    Includes timeout protection to prevent hanging.
    """
    question = question.strip() if question else ""
    if not question:
        return question

    if len(question.split()) < 4 and not history:
        logger.info(f"Using original query (short): '{question}'")
        return question

    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in history[-2:]
        )

    try:
        prompt = (
            f"Rewrite this user question to be explicit, specific, and highly searchable "
            f"in a document knowledge vector index.\n\n"
            f"Rules:\n"
            f"- Expand abbreviations, pronouns, or vague references using conversation context\n"
            f"- Make implicit topics explicit\n"
            f"- Return ONLY the rewritten question string, nothing else\n\n"
        )
        if history_text:
            prompt += f"Conversation so far:\n{history_text}\n\n"
        prompt += f"Question to rewrite: {question}"

        rewritten = generate_chat_text(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            temperature=0.6,
            max_tokens=100,
            top_p=0.95,
        ).strip()
        final_query = rewritten if (rewritten and len(rewritten) > 3) else question
        logger.info(f"Rewritten: '{question}' → '{final_query}'")
        return final_query

    except Exception as e:
        logger.warning(f"Rewrite failed: {e}, using original query")
        return question


# ══════════════════════════════════════════════════════════════════
# KEYWORD & HYBRID RETRIEVAL (ChromaDB + TF/BM25)
# ══════════════════════════════════════════════════════════════════

def compute_keyword_similarity(query: str, doc_text: str) -> float:
    from collections import Counter

    q_words = re.findall(r'\b[a-z0-9]{2,}\b', query.lower())
    d_words = re.findall(r'\b[a-z0-9]{2,}\b', doc_text.lower())
    if not q_words or not d_words:
        return 0.0

    d_counts = Counter(d_words)
    total_d = len(d_words)

    matches = sum(d_counts[w] for w in q_words if w in d_counts)
    return min(1.0, (matches / total_d) * 5.0)


def _json_list(value) -> list:
    """Metadata lists are stored as JSON strings in ChromaDB — decode them safely."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def chunk_from_store(chunk_id: str, meta: dict, doc_text: str) -> dict:
    """Rebuild a chunk dict from a ChromaDB record."""
    chunk = dict(meta or {})
    chunk["chunk_id"] = chunk_id
    chunk["content"] = doc_text or ""
    chunk["references"] = _json_list(chunk.get("references"))
    chunk["hyde_questions"] = _json_list(chunk.get("hyde_questions"))
    return chunk


def retrieve_single_vector_and_keyword(query_text: str, top_k: int = RETRIEVAL_TOP_K_EXPANDED) -> list[dict]:
    query_text = query_text.strip()
    if not query_text:
        return []
    query_vec = embed_query(query_text)
    if not query_vec:
        logger.error(f"Query vectorization failed for '{query_text[:50]}'")
        return []

    collection = get_chroma_collection()
    total_docs = collection.count()
    if total_docs == 0:
        return []

    n_results = min(top_k, total_docs)
    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        return []

    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    scored = []
    for chunk_id, meta, doc_text, dist in zip(ids, metadatas, documents, distances):
        dense_sim = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
        meta = meta or {}
        doc_text = doc_text or ""
        content_for_kw = f"{doc_text} {meta.get('heading', '')} {meta.get('section', '')}"
        kw_sim = compute_keyword_similarity(query_text, content_for_kw)
        hybrid_score = (VECTOR_WEIGHT * dense_sim) + (BM25_WEIGHT * kw_sim)

        chunk_dict = chunk_from_store(chunk_id, meta, doc_text)
        chunk_dict["dense_score"] = dense_sim
        chunk_dict["keyword_score"] = kw_sim
        chunk_dict["rerank_score"] = hybrid_score
        scored.append(chunk_dict)
    return scored


def retrieve_chunks(question: str, rewritten_query: str = "", top_k: int = RETRIEVAL_TOP_K_EXPANDED, url_filter: list[str] = None) -> tuple[list[dict], float]:
    """
    Retrieve relevant chunks from ChromaDB using multi-query hybrid search (dense embeddings + BM25 keyword matching).
    Searches using both user question and the rewritten query, fusing results for maximum recall.
    If specific URLs are provided in url_filter, their chunks are given top priority.
    """
    question = question.strip() if question else ""
    rewritten_query = rewritten_query.strip() if rewritten_query else ""
    if not question and not rewritten_query and not url_filter:
        return [], 0.0

    candidates_map = {}

    # Priority: If specific URLs were referenced, include their chunks with top priority
    if url_filter:
        try:
            coll = get_chroma_collection()
            for target_u in url_filter:
                u_res = coll.get(where={"source_url": target_u}, include=["metadatas", "documents"])
                u_metas = u_res.get("metadatas", [])
                u_docs = u_res.get("documents", [])
                u_ids = u_res.get("ids", [])
                for cid, meta, doc in zip(u_ids, u_metas, u_docs):
                    if meta:
                        c_dict = chunk_from_store(cid, meta, doc)
                        c_dict["rerank_score"] = URL_PRIORITY_SCORE
                        c_dict["url_priority"] = True
                        candidates_map[cid] = c_dict
        except Exception as e:
            logger.warning(f"Failed to fetch chunks for target URL: {e}")

    def merge(candidates: list[dict], mutual_bonus: float = 0.0):
        for c in candidates:
            cid = c.get("chunk_id") or str(c.get("content", "")[:30])
            existing = candidates_map.get(cid)
            if existing is None:
                candidates_map[cid] = c
            elif existing.get("url_priority"):
                # Chunk of a URL named in the question: keep it on top, but rank
                # it against its siblings by actual relevance to the query.
                existing["rerank_score"] = max(existing["rerank_score"], URL_PRIORITY_SCORE + c["rerank_score"])
            else:
                existing["rerank_score"] = max(existing["rerank_score"], c["rerank_score"]) + mutual_bonus

    # Query 1: Original user question
    if question:
        merge(retrieve_single_vector_and_keyword(question, top_k))

    # Query 2: Rewritten query (if distinct)
    if rewritten_query and rewritten_query.lower() != question.lower():
        merge(retrieve_single_vector_and_keyword(rewritten_query, top_k), mutual_bonus=0.1)

    scored_candidates = list(candidates_map.values())
    if not scored_candidates:
        return [], 0.0

    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # ── Step 4: Cross-reference expansion ───────────────────────
    top_candidates = scored_candidates[:RERANK_TOP_K]
    ref_sections = set()
    for cand in top_candidates:
        refs = _json_list(cand.get("references", []))
        for ref in refs[:3]:
            if ref and ref not in [c.get("section") for c in top_candidates]:
                ref_sections.add(ref)

    if ref_sections:
        try:
            collection = get_chroma_collection()
            for section_name in list(ref_sections)[:3]:
                ref_res = collection.get(
                    where={"section": section_name},
                    include=["metadatas", "documents"],
                    limit=2
                )
                ref_ids = ref_res.get("ids", [])
                ref_metas = ref_res.get("metadatas", [])
                ref_docs = ref_res.get("documents", [])
                existing_ids = {c.get("chunk_id") for c in top_candidates}
                for r_id, r_meta, r_doc in zip(ref_ids, ref_metas, ref_docs):
                    if r_meta and r_id not in existing_ids:
                        r_chunk = chunk_from_store(r_id, r_meta, r_doc)
                        r_chunk["rerank_score"] = 0.5
                        top_candidates.append(r_chunk)
                        existing_ids.add(r_id)
        except Exception as e:
            logger.warning(f"Cross-reference expansion failed: {e}")

    top_score = scored_candidates[0]["rerank_score"] if scored_candidates else 0.0
    return top_candidates, top_score


# ══════════════════════════════════════════════════════════════════
# CONTEXT BUILDER & CITATIONS
# ══════════════════════════════════════════════════════════════════

def build_context(chunks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    context_parts = []
    sources       = []
    images        = []
    source_index  = {}   # "doc|section" -> source entry (several chunks can share one)

    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("chunk_type", "text")
        content    = chunk.get("content", "")
        doc_title  = chunk.get("doc_title", "")
        section    = chunk.get("section", "")
        source_url = chunk.get("source_url", "")
        rerank_score = chunk.get("rerank_score", 0)

        # Build clean snippet quote (~280 characters)
        clean_snippet = content.replace("\r", " ").replace("\n", " ").strip()
        clean_snippet = " ".join(clean_snippet.split())
        if len(clean_snippet) > 280:
            clean_snippet = clean_snippet[:280] + "..."

        noise_sections = [
            "this will be in effect",
            "important advisory",
        ]
        is_noise = any(n in section.lower() for n in noise_sections)

        if chunk_type in ("text", "table", "section_index"):
            ref = len(context_parts) + 1
            context_parts.append(
                f"[SOURCE {ref}: {doc_title} — {section} | URL: {source_url}]\n{content}"
            )
            source_key = f"{doc_title}|{section}"
            if source_key in source_index:
                source_index[source_key]["refs"].append(ref)
            elif not is_noise:
                sources.append({
                    "refs":    [ref],
                    "doc":     doc_title,
                    "section": section,
                    "url":     source_url,
                    "type":    "document",
                    "snippet": clean_snippet,
                    "score":   round(float(rerank_score), 3) if rerank_score else None,
                })
                source_index[source_key] = sources[-1]

        elif chunk_type == "image":
            image_content = chunk.get("image_content", "")
            ref = len(context_parts) + 1
            context_parts.append(
                f"[SOURCE {ref} (IMAGE): {section} | URL: {source_url}]\n"
                f"Image file: {chunk.get('image_file','')}\n"
                f"Content: {image_content or content}"
            )
            if chunk.get("image_file"):
                images.append({
                    "refs":    [ref],
                    "file":    chunk["image_file"],
                    "section": section,
                    "url":     source_url,
                })
            source_key = f"{doc_title}|{section}"
            if source_key in source_index:
                source_index[source_key]["refs"].append(ref)
            else:
                img_snippet = (image_content or content).replace("\r", " ").replace("\n", " ").strip()
                if len(img_snippet) > 280:
                    img_snippet = img_snippet[:280] + "..."
                sources.append({
                    "refs":    [ref],
                    "doc":     doc_title,
                    "section": section,
                    "url":     source_url,
                    "type":    "image",
                    "snippet": img_snippet,
                    "score":   round(float(rerank_score), 3) if rerank_score else None,
                })
                source_index[source_key] = sources[-1]

        elif chunk_type == "reference_link":
            when_to_refer    = chunk.get("when_to_refer", "")
            what_it_contains = chunk.get("what_it_contains", "")
            link_url         = chunk.get("link_url", source_url)
            ref = len(context_parts) + 1
            context_parts.append(
                f"[SOURCE {ref} (REFERENCE LINK): {section} | URL: {link_url}]\n"
                f"Title: {chunk.get('link_text','')}\n"
                f"URL: {link_url}\n"
                f"Contains: {what_it_contains}\n"
                f"Use when: {when_to_refer}"
            )
            sources.append({
                "refs":             [ref],
                "doc":              doc_title,
                "section":          section,
                "url":              link_url,
                "text":             chunk.get("link_text", ""),
                "what_it_contains": what_it_contains,
                "type":             "link",
                "is_link":          True,
                "snippet":          f"{what_it_contains} ({when_to_refer})",
                "score":            round(float(rerank_score), 3) if rerank_score else None,
            })

        elif chunk_type == "restricted_doc":
            link_url = chunk.get("link_url", source_url)
            ref = len(context_parts) + 1
            context_parts.append(
                f"[SOURCE {ref} (RESTRICTED DOCUMENT): {section} | URL: {link_url}]\n"
                f"URL: {link_url}\n"
                f"Note: {chunk.get('note', 'Requires login to access.')}\n"
                f"Context: {content}"
            )
            sources.append({
                "refs":    [ref],
                "doc":     doc_title,
                "section": section,
                "url":     link_url,
                "type":    "restricted",
                "note":    chunk.get("note", ""),
                "snippet": chunk.get("note", ""),
                "score":   round(float(rerank_score), 3) if rerank_score else None,
            })

    return "\n\n".join(context_parts), sources, images


# ══════════════════════════════════════════════════════════════════
# USED-SOURCE TRACKING
# ══════════════════════════════════════════════════════════════════
# The model ends its answer with a hidden marker such as "[[USED: 1, 3]]" naming
# the numbered context sources it actually relied on. The marker is removed from
# the streamed text and used to show only those sources as citations.

USED_MARKER = "[[USED"
USED_MARKER_RE = re.compile(r"\[\[USED\s*:?\s*([^\]]*)\]\]", re.IGNORECASE)
CANNOT_ANSWER_HINTS = (
    "could not find this information",
    "could not find relevant information",
)


def strip_used_marker(deltas):
    """
    Pass text deltas through, withholding the trailing used-sources marker
    (even when it is split across deltas). Returns the marker text via the
    generator's return value: `marker = yield from strip_used_marker(...)`.
    """
    buffer = ""
    marker_text = None
    for delta in deltas:
        if marker_text is not None:
            marker_text += delta
            continue
        buffer += delta
        idx = buffer.upper().find(USED_MARKER)
        if idx != -1:
            if buffer[:idx]:
                yield buffer[:idx]
            marker_text = buffer[idx:]
            buffer = ""
            continue
        # Hold back a tail that could be the beginning of the marker
        hold = 0
        for n in range(min(len(USED_MARKER) - 1, len(buffer)), 0, -1):
            if USED_MARKER.startswith(buffer[-n:].upper()):
                hold = n
                break
        emit = buffer[:len(buffer) - hold]
        if emit:
            yield emit
        buffer = buffer[len(buffer) - hold:]
    if buffer and marker_text is None:
        yield buffer
    return marker_text


def parse_used_refs(marker_text: str | None) -> set[int] | None:
    """Source numbers named in the marker; None when the model emitted no marker."""
    if not marker_text:
        return None
    match = USED_MARKER_RE.search(marker_text)
    body = match.group(1) if match else marker_text[len(USED_MARKER):]
    return {int(n) for n in re.findall(r"\d+", body)}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def select_used(items: list[dict], used_refs: set[int] | None, answer: str) -> list[dict]:
    """
    Keep only the sources/images the answer actually used, and drop the internal
    "refs" field. Falls back to matching names cited in the answer text, then to
    the single best-scoring source, if the model gave no usable marker.
    """
    if any(hint in answer.lower() for hint in CANNOT_ANSWER_HINTS) and not used_refs:
        selected = []
    elif used_refs:
        selected = [it for it in items if used_refs.intersection(it.get("refs", []))]
    else:
        selected = []

    if not selected and used_refs is None and items:
        answer_norm = _norm(answer)
        selected = [
            it for it in items
            if len(_norm(it.get("section", ""))) >= 4 and _norm(it.get("section", "")) in answer_norm
        ]
        if not selected and "score" in items[0]:
            selected = [max(items, key=lambda it: it.get("score") or 0)]

    return [{k: v for k, v in it.items() if k != "refs"} for it in selected]


# ══════════════════════════════════════════════════════════════════
# SSE HELPER
# ══════════════════════════════════════════════════════════════════

def sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


# ══════════════════════════════════════════════════════════════════
# INGESTION ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    # Strip any client-supplied directory components (path traversal guard)
    safe_name = Path((file.filename or "").replace("\\", "/")).name
    ext = Path(safe_name).suffix.lower()
    if not safe_name or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="upload_")
        os.close(temp_fd)

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        # Also save to persistent UPLOAD_DIR for citations & document download
        try:
            with open(UPLOAD_DIR / safe_name, "wb") as buffer:
                buffer.write(content)
        except Exception as e:
            logger.warning(f"Could not write copy to UPLOAD_DIR: {e}")

        def stream_and_cleanup(path: str = temp_path):
            try:
                yield from ingest_file_stream(Path(path), safe_name)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        return StreamingResponse(
            stream_and_cleanup(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            }
        )
    except HTTPException:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/files/{filename:path}")
def get_uploaded_file(filename: str):
    """Serve uploaded documents and images for citation previews and downloads."""
    clean_name = Path(filename.replace("\\", "/")).name
    if not clean_name:
        raise HTTPException(status_code=404, detail="File not found")
    candidates = [
        UPLOAD_DIR / clean_name,
        IMAGES_DIR / clean_name,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return FileResponse(p, filename=clean_name)
    raise HTTPException(status_code=404, detail=f"File '{clean_name}' not found")


@app.post("/ingest/url")
async def ingest_url(req: UrlIngestRequest):
    url = req.url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    return StreamingResponse(
        ingest_url_stream(url),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.get("/documents")
def get_documents():
    return {"documents": list_indexed_documents()}


@app.delete("/documents/{doc_title:path}")
def delete_document(doc_title: str):
    # ":path" so titles containing "/" (e.g. web page titles or URLs) can be deleted
    success = delete_document_chunks(doc_title)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete document '{doc_title}'")
    # Remove the stored copy of an uploaded file, if any
    stored_name = Path(doc_title.replace("\\", "/")).name
    try:
        if stored_name and (UPLOAD_DIR / stored_name).is_file():
            (UPLOAD_DIR / stored_name).unlink()
    except OSError as e:
        logger.warning(f"Could not remove stored file for '{doc_title}': {e}")
    return {"status": "ok", "deleted": doc_title}


# ══════════════════════════════════════════════════════════════════
# ASK ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.post("/ask")
async def ask(req: AskRequest):

    # Plain (sync) generator on purpose: Starlette iterates it in a worker thread,
    # so the blocking embedding / scraping / LLM calls don't freeze the event loop.
    def stream():
        try:
            question = req.question.strip()

            if is_greeting(question):
                yield sse("token", {"text": (
                    "Hello! 👋 I am your AI Document Knowledge Assistant.\n\n"
                    "You can upload files (PDF, DOCX, TXT, MD, Images) or paste web URLs into the Knowledge Base, "
                    "or even include a URL directly in your question, and I will answer directly from your content!"
                )})
                yield sse("done", {})
                return

            # Live URL detection: If question contains a URL, fetch & index it automatically on the fly
            URL_REGEX = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+'
            raw_urls = re.findall(URL_REGEX, question)
            detected_urls = []
            if raw_urls:
                for raw_url in raw_urls[:2]:
                    clean_url = raw_url.rstrip(".,?!;:)'\"]")
                    target_url = clean_url if clean_url.startswith("http") else f"https://{clean_url}"
                    detected_urls.append(target_url)
                    try:
                        coll = get_chroma_collection()
                        existing = coll.get(where={"source_url": target_url}, limit=1)
                        if not existing or not existing.get("ids"):
                            yield sse("status", {"text": f"Reading web content from {target_url}..."})
                            doc_data = scrape_url(target_url)
                            if doc_data.get("content"):
                                u_chunks = chunk_document_content(doc_data["content"], doc_data["doc_title"], target_url)
                                if u_chunks:
                                    u_chunks = embed_chunk_list(u_chunks)
                                    upload_chunks_batch(u_chunks)
                                    logger.info(f"Auto-ingested {len(u_chunks)} chunks for URL: {target_url}")
                    except Exception as e:
                        logger.warning(f"On-the-fly URL reading failed for {target_url}: {e}")

            selected_provider = resolve_provider(req.provider)

            yield sse("status", {"text": "Understanding your question..."})
            rewritten = rewrite_query(question, req.history, provider=selected_provider)

            yield sse("status", {"text": "Searching document database..."})
            chunks, top_score = retrieve_chunks(question, rewritten_query=rewritten, url_filter=detected_urls)

            if top_score < CONFIDENCE_THRESHOLD and chunks:
                logger.info(f"Low confidence ({top_score:.2f}), broadening search...")
                yield sse("status", {"text": "Searching deeper..."})
                chunks2, score2 = retrieve_chunks(question, top_k=35, url_filter=detected_urls)
                if score2 > top_score:
                    chunks    = chunks2
                    top_score = score2

            if not chunks:
                yield sse("token", {"text": (
                    "I could not find relevant information in the currently indexed documents.\n"
                    "Please upload a document or paste a webpage URL using the Knowledge Base tool above."
                )})
                yield sse("done", {})
                return

            context_text, sources, images = build_context(chunks)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for msg in req.history[-MAX_HISTORY:]:
                role = "assistant" if msg.role.lower() in ("assistant", "model", "bot") else "user"
                if msg.content and msg.content.strip():
                    messages.append({"role": role, "content": msg.content})

            messages.append({
                "role": "user",
                "content": (
                    f"CONTEXT FROM INDEXED DOCUMENTS:\n"
                    f"{context_text}\n\n"
                    f"USER QUESTION: {question}\n\n"
                    f"Instructions:\n"
                    f"- Answer directly and confidently from context above\n"
                    f"- If question is yes/no → start with YES or NO\n"
                    f"- Include ALL relevant URLs/filenames from context inline\n"
                    f"- End with source citation — Source URL/Filename is MANDATORY\n"
                    f"- Cite ONLY sources whose content you actually used; ignore irrelevant ones\n"
                    f"- Never show internal labels like [SOURCE N: ...] in your answer\n"
                    f"- After the citation, finish with one final line listing the numbers of the "
                    f"sources you used, in exactly this form: [[USED: 1, 3]] "
                    f"(use [[USED: none]] if the context did not contain the answer)"
                )
            })

            if selected_provider == "llamacpp":
                if not is_local_model_downloaded():
                    yield sse("status", {"text": "Preparing local Ling-3.0-tiny model (downloading GGUF, ~4.6 GB)..."})
                else:
                    yield sse("status", {"text": "Generating answer via Local llama.cpp (Ling 3.0 Tiny)..."})
            else:
                yield sse("status", {"text": f"Generating answer via Groq ({GROQ_MODEL})..."})

            answer_parts = []

            def visible_tokens():
                # Hide the trailing [[USED: ...]] marker from the user
                marker = yield from strip_used_marker(
                    stream_chat_completion(
                        messages=messages,
                        provider=selected_provider,
                        temperature=0.6,
                        max_tokens=4096,
                        top_p=0.95,
                    )
                )
                return marker

            token_stream = visible_tokens()
            marker_text = None
            while True:
                try:
                    delta_text = next(token_stream)
                except StopIteration as stop:
                    marker_text = stop.value
                    break
                if delta_text:
                    answer_parts.append(delta_text)
                    yield sse("token", {"text": delta_text})

            # Only show citations for sources the answer actually used
            answer_text = "".join(answer_parts)
            used_refs = parse_used_refs(marker_text)
            used_sources = select_used(sources, used_refs, answer_text)
            used_images = select_used(images, used_refs, answer_text) if used_refs else []
            logger.info(f"Sources used: {sorted(used_refs) if used_refs is not None else 'no marker'} "
                        f"→ {len(used_sources)}/{len(sources)} citations")

            yield sse("sources", {"sources": used_sources})
            if used_images:
                yield sse("images", {"images": used_images})
            yield sse("done", {})

        except Exception as e:
            logger.exception(f"Stream error: {e}")
            yield sse("error", {"text": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        }
    )


# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════

@app.get("/models")
def get_models():
    return {
        "current_provider": resolve_provider(LLM_PROVIDER),
        "models": [
            {
                "id": "groq",
                "name": GROQ_MODEL,
                "badge": "Cloud (Groq)",
                "description": "Cloud inference via the Groq API",
                "ready": bool(GROQ_API_KEY),
            },
            {
                "id": "llamacpp",
                "name": "Ling 3.0 Tiny",
                "badge": "Local (llama.cpp)",
                "description": "100% offline GGUF (Q4_K_M) running directly on CPU",
                "ready": is_local_model_downloaded(),
            },
        ]
    }


# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    try:
        coll = get_chroma_collection()
        count = coll.count()
        chroma_ok = f"ok — {count} points"
    except Exception as e:
        chroma_ok = f"error: {e}"

    return {
        "status":               "ok",
        "llm_provider":         LLM_PROVIDER,
        "cloud_model":          GROQ_MODEL,
        "cloud_model_ready":    bool(GROQ_API_KEY),
        "local_model":          LOCAL_MODEL_FILE,
        "local_model_ready":    is_local_model_downloaded(),
        "embed_model":          EMBEDDING_MODEL,
        "embed_provider":       "local (all-MiniLM-L6-v2)",
        "chromadb":             chroma_ok,
        "collection":           CHROMA_COLLECTION,
        "hybrid_search":        f"vector({VECTOR_WEIGHT}) + bm25({BM25_WEIGHT})",
        "retrieval_top_k":      RETRIEVAL_TOP_K_EXPANDED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)