import asyncio
from typing import List, Optional
from ..llm_clients import call_hf
from ..Schema import ContractState, ClassifiedClause

LEGAL_BERT_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
CANDIDATE_LABELS = ["Confidentiality", "Liability", "Termination", "Other"]

def _safe_page(value: Optional[object], default: int = 1) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default

def _safe_section(value: Optional[object], default: str = "Unknown") -> str:
    if isinstance(value, str):
        s = value.strip()
        return s if s else default
    return default

async def classify_clause_bert(text: str, candidate_labels: Optional[List[str]] = None) -> dict:
    if candidate_labels is None:
        candidate_labels = CANDIDATE_LABELS
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    data = await call_hf(LEGAL_BERT_URL, payload)

    # Inference Providers format: list of {label, score}
    if isinstance(data, list) and data and isinstance(data[0], dict) and "label" in data[0]:
        best = max(data, key=lambda x: x.get("score", 0.0))
        return {"type": best.get("label", "Other"), "confidence": float(best.get("score", 0.0))}
    # Pipeline-like format: dict with labels/scores arrays
    if isinstance(data, dict) and "labels" in data and "scores" in data and data["labels"] and data["scores"]:
        return {"type": data["labels"][0], "confidence": float(data["scores"][0])}
    # Fallback
    return {"type": "Other", "confidence": 0.5}

async def classification_agent(state: ContractState) -> ContractState:
    classified_dicts = []

    async def classify_single_clause(clause: dict):
        result = await classify_clause_bert(clause["text"])
        page = _safe_page(clause.get("page"))
        section = _safe_section(clause.get("section"))
        classified_clause = ClassifiedClause(
            clause_id=clause["clause_id"],
            text=clause["text"],
            page=page,
            section=section,
            cls_type=result.get("type", "Other"),
            confidence=float(result.get("confidence", 0.0)),
        )
        classified_dicts.append(classified_clause.model_dump())

    await asyncio.gather(*(classify_single_clause(c) for c in state.clauses))
    state.classified = classified_dicts
    return state
