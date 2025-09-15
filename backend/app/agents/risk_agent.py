from ..llm_clients import analyze_risk_t5

async def risk_agent(state):
    """
    Risk Agent:
    - Uses Flan-T5 for compliance risk analysis
    - Works independently (no references/logging)
    """
    out = []
    for c in state["classified"]:
        result = await analyze_risk_t5(c["text"], c["cls_type"])
        result["clause_id"] = c["clause_id"]
        out.append(result)
    state["risks"] = out
    state["overall_score"] = sum(r.get("score", 0.5) for r in out) / len(out)
    return state
