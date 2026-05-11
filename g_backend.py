import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

load_dotenv()

app = FastAPI(title="GEN AI Based Framework for Identifying Manipulated News Content API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add this to a .env file in the same folder:
# GROQ_API_KEY="your_groq_api_key_here"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
KB_PATH = Path(__file__).with_name("simple_knowledge_base.json")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 160
TOP_K = 4

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def clean_json(text: str) -> Dict[str, Any]:
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse JSON from response: {text[:300]}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9']+", text.lower())
    stop_words = {
        "the", "and", "for", "are", "with", "that", "this", "from", "was",
        "were", "has", "have", "had", "not", "but", "you", "your", "their",
        "they", "them", "its", "our", "his", "her", "she", "him", "who",
        "what", "when", "where", "why", "how", "will", "would", "could",
        "should", "about", "into", "over",
    }
    return [word for word in words if len(word) > 2 and word not in stop_words]


def chunk_text(text: str, source: str) -> list[Dict[str, Any]]:
    text = normalize_text(text)
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "source": source,
                    "text": chunk,
                    "tokens": tokenize(chunk),
                }
            )
        if end == len(text):
            break
        start = max(0, end - CHUNK_OVERLAP)

    return chunks


def load_kb() -> list[Dict[str, Any]]:
    if not KB_PATH.exists():
        return []
    try:
        data = json.loads(KB_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def save_kb(chunks: list[Dict[str, Any]]) -> None:
    KB_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")


def retrieve_context(query: str) -> list[Dict[str, Any]]:
    kb = load_kb()
    query_tokens = tokenize(query)
    if not kb:
        return []
    if not query_tokens:
        return [
            {
                "source": item.get("source", "Unknown source"),
                "text": item.get("text", ""),
                "score": 0.0,
                "match_type": "available_source",
            }
            for item in kb[:TOP_K]
        ]

    query_tf = {token: query_tokens.count(token) for token in set(query_tokens)}
    doc_freq: Dict[str, int] = {}

    for item in kb:
        for token in set(item.get("tokens", [])):
            doc_freq[token] = doc_freq.get(token, 0) + 1

    scored = []
    for item in kb:
        tokens = item.get("tokens", [])
        if not tokens:
            continue

        token_counts = {token: tokens.count(token) for token in set(tokens)}
        score = 0.0
        for token, q_count in query_tf.items():
            if token in token_counts:
                idf = math.log((1 + len(kb)) / (1 + doc_freq.get(token, 0))) + 1
                score += q_count * token_counts[token] * idf

        norm = math.sqrt(len(tokens)) * math.sqrt(len(query_tokens))
        score = score / norm if norm else 0.0
        if score > 0:
            scored.append(
                {
                    "source": item.get("source", "Unknown source"),
                    "text": item.get("text", ""),
                    "score": round(score, 4),
                    "match_type": "keyword_match",
                }
            )

    scored.sort(key=lambda item: item["score"], reverse=True)
    if scored:
        return scored[:TOP_K]

    return [
        {
            "source": item.get("source", "Unknown source"),
            "text": item.get("text", ""),
            "score": 0.0,
            "match_type": "low_keyword_match",
        }
        for item in kb[:TOP_K]
    ]


def format_evidence(evidence: list[Dict[str, Any]]) -> str:
    if not evidence:
        return "No trusted RAG evidence was found."

    return "\n\n".join(
        f"Evidence {index} | Source: {item['source']} | Score: {item['score']}\n{item['text']}"
        for index, item in enumerate(evidence, start=1)
    )


def analyze_core_features(text: str, evidence: list[Dict[str, Any]]) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    prompt = f"""
You are a news manipulation, propaganda, and AI-generated content analyst.
Analyze the following news content and return ONLY a valid JSON object.
Do not include markdown, explanation, or extra text.
Use the trusted RAG evidence if it is relevant.
If evidence is missing or unrelated, mark factual support as unverified.

Return exactly this JSON structure:
{{
  "bias_score": <integer 0-100>,
  "manipulation_tactics": ["tactic1", "tactic2"],
  "emotions": {{"Anger": <0.0-1.0>, "Fear": <0.0-1.0>, "Trust": <0.0-1.0>, "Joy": <0.0-1.0>, "Sadness": <0.0-1.0>}},
  "ai_detection": <integer 0-100>,
  "propaganda_era": "<era name>",
  "neutral_rewrite": "<objective rewrite of the text>",
  "rag_assessment": "<how the evidence supports, contradicts, or fails to verify the news>",
  "evidence_summary": ["short evidence point 1", "short evidence point 2"],
  "verdict": "<REAL or FAKE or MANIPULATED or UNVERIFIED>"
}}

Trusted RAG evidence:
{format_evidence(evidence)}

News content:
{text[:3000]}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a JSON-only response bot. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return clean_json(response.choices[0].message.content)


@app.post("/ingest")
async def ingest(
    text: Optional[str] = Form(None),
    source: Optional[str] = Form("Manual source"),
):
    source_text = text or ""
    source_name = source or "Manual source"

    if not source_text.strip():
        raise HTTPException(status_code=400, detail="No trusted source text provided.")

    chunks = chunk_text(source_text, source_name)
    kb = load_kb()
    kb.extend(chunks)
    save_kb(kb)

    return {
        "message": "Trusted source added to RAG knowledge base.",
        "source": source_name,
        "chunks_added": len(chunks),
        "total_chunks": len(kb),
    }


@app.get("/knowledge-base")
async def knowledge_base_status():
    kb = load_kb()
    sources = sorted({item.get("source", "Unknown source") for item in kb})
    return {"chunks": len(kb), "sources": sources}


@app.post("/analyze")
async def analyze(text: Optional[str] = Form(None)):
    article_text = text or ""

    if not article_text.strip():
        raise HTTPException(status_code=400, detail="No text content to analyze.")

    evidence = retrieve_context(article_text)

    try:
        analysis = analyze_core_features(article_text, evidence)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse model response: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {e}")

    analysis["rag_evidence"] = evidence
    analysis["mutation_tracker"] = "RAG evidence checked against the submitted news content."
    return analysis


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
