import asyncio
from typing import List
from ..llm_clients import call_hf
from ..Schema import ContractState, ClassifiedClause

LEGAL_BERT_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
CANDIDATE_LABELS = ["Confidentiality", "Liability", "Termination", "Other"]


async def classify_clause_bert(text: str, candidate_labels: List[str] = None) -> dict:
    """
    Calls HF BART zero-shot API safely and returns top label with confidence.
    """
    if candidate_labels is None:
        candidate_labels = CANDIDATE_LABELS

    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": candidate_labels}
    }

    data = await call_hf(LEGAL_BERT_URL, payload)

    # HF zero-shot returns dict with 'labels' and 'scores'
    if isinstance(data, dict) and "labels" in data and "scores" in data:
        return {"type": data["labels"][0], "confidence": data["scores"][0]}

    # Fallback
    return {"type": "Other", "confidence": 0.5}


async def classification_agent(state: ContractState) -> ContractState:
    """
    Classifies each clause in the ContractState using HF zero-shot and updates state.classified.
    """
    classified_dicts = []

    async def classify_single_clause(clause: dict):
        result = await classify_clause_bert(clause["text"])
        page = clause.get("page", -1)
        section = clause.get("section", "Unknown")

        classified_clause = ClassifiedClause(
            clause_id=clause["clause_id"],
            text=clause["text"],
            page=page,
            section=section,
            cls_type=result["type"],
            confidence=result["confidence"]
        )

        # Convert to dict for Pydantic
        classified_dicts.append(classified_clause.dict())

    # Run all clause classifications concurrently for speed
    await asyncio.gather(*(classify_single_clause(c) for c in state.clauses))

    state.classified = classified_dicts
    return state
