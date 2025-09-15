from ..llm_clients import call_gemini

async def advisory_agent(state):
    """
    Advisory Agent:
    - Uses Gemini API to produce executive summaries and recommendations
    """
    risks_text = "\n".join([f"{r['clause_id']}: {r['risk']} - {r['reason']}" for r in state["risks"]])
    prompt = f"Generate a compliance summary with recommendations:\n{risks_text}"
    summary = await call_gemini(prompt)
    state["advisory"] = {"executive_summary": summary, "recommendations": summary.split("\n")}
    return state
