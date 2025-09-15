import re
from ..llm_clients import call_hf

async def classification_agent(state):
    """
    Classification Agent:
    - Calls HF/NVIDIA API to classify clauses
    """
    out = []
    for c in state["clauses"]:
        prompt = f"""Classify this clause into one of:
Coverage, Exclusion, Claims Obligation, Premium Adjustment, Privacy/Data, Other.
Return JSON: {{ "type": "...", "confidence": 0.xx }}
Clause: {c['text']}"""
        
        text = await call_hf(prompt)

        cls_type = "Other"
        conf = 0.6
        m = re.search(r'"type"\s*:\s*"([^"]+)"', text)
        if m: cls_type = m.group(1)
        m2 = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if m2: conf = float(m2.group(1))

        out.append({**c, "cls_type": cls_type, "confidence": conf})

    state["classified"] = out
    return state
