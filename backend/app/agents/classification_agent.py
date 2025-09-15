from ..llm_clients import classify_clause_bert

async def classification_agent(state):
    """
    Classification Agent:
    - Uses Legal-BERT for clause classification
    """
    out = []
    for c in state["clauses"]:
        result = await classify_clause_bert(c["text"])
        out.append({**c, "cls_type": result["type"], "confidence": result["confidence"]})
    state["classified"] = out
    return state
