from ..llm_clients import call_gemini

async def advisory_agent(state):
    """
    Advisory Agent:
    - Uses Gemini API to generate human-readable summary
    """
    risks_text = "\n".join(
        [f"{r['clause_id']} - {r['risk']} - {r['reason']}" for r in state["risks"]]
    )

    prompt = f"""Generate an executive summary of the following risks with top 3 recommendations:
{risks_text}"""

    text = await call_gemini(prompt)
    recs = [line.strip("-* ") for line in text.splitlines() if len(line.strip()) > 10]

    state["advisory"] = {"executive_summary": text, "recommendations": recs[:5]}
    return state
