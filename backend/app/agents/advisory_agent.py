from ..llm_clients import call_gemini
from ..Schema import ContractState, AdvisoryResult

async def advisory_agent(state: ContractState) -> ContractState:
    # Build human-readable risk summary
    risks_summary = "\n".join(
        [f"- {r['clause_id']}: {r['risk']} ({r['framework']}, {r['status']})" for r in state.risks]
    )

    prompt = f"""
    You are a compliance advisor. Based on the following risk assessment:

    {risks_summary}

    Provide:
    1. Executive summary
    2. Top 3 recommendations
    """

    response = await call_gemini(prompt)

    # Parse response safely
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    executive_summary = lines[0] if lines else "Automated compliance summary."
    top_recommendations = lines[1:4] if len(lines) > 1 else ["Review contract clauses for compliance."]

    state.advisory = AdvisoryResult(
        executive_summary=executive_summary[:500],  # Limit to 500 chars
        recommendations=top_recommendations
    )
    return state
