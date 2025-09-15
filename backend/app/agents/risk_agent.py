import json
from ..llm_clients import call_hf

async def risk_agent(state):
    """
    Risk Agent:
    - Evaluates compliance risks of classified clauses
    """
    out = []
    for c in state["classified"]:
        prompt = f"""Evaluate compliance risk of this clause against IRDAI, GDPR, HIPAA.
                        Return JSON: {{
                        "clause_id": "{c['clause_id']}",
                        "risk": "Low|Medium|High",
                        "framework": "<IRDAI|GDPR|HIPAA>",
                        "status": "Aligned|Partial|Gap",
                        "reason": "...",
                        "score": 0.xx
                        }}
Clause: {c['text']}
Type: {c['cls_type']}"""

        text = await call_hf(prompt)

        try:
            result = json.loads(text)
        except:
            result = {
                "clause_id": c["clause_id"],
                "risk": "Medium",
                "framework": "IRDAI",
                "status": "Partial",
                "reason": "Fallback parse",
                "score": 0.5
            }
        out.append(result)

    state["risks"] = out
    state["overall_score"] = sum(r.get("score", 0.5) for r in out) / len(out)
    return state
