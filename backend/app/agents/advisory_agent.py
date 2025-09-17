from ..llm_clients import call_gemini
from ..workflow import ContractState, AdvisoryResult

async def advisory_agent(state: ContractState) -> ContractState:
    risks_summary = "\n".join(
        [f"- {r.clause_id}: {r.risk} ({r.framework}, {r.status})" for r in state.risks]
    )

    prompt = f"""
    You are a compliance advisor. Based on the following risk assessment:

    {risks_summary}

    Provide:
    1. Executive summary
    2. Top 3 recommendations
    """

    response = await call_gemini(prompt)

    state.advisory = AdvisoryResult(
        executive_summary=response[:500],  # Simplified parsing
        recommendations=response.split("\n")[:3],
    )
    return state
