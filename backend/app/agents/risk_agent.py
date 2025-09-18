# file: agents/risk_agent.py
import asyncio
from typing import List
from ..llm_clients import analyze_risk_t5
from ..Schema import ContractState, RiskResult


async def risk_agent(state: ContractState) -> ContractState:
    """
    For each classified clause in ContractState, evaluate risks using FLAN-T5.
    Updates state.risks with RiskResult dictionaries.
    """
    risks_list = []

    async def analyze_single_clause(clause):
        try:
            result = await analyze_risk_t5(clause["text"], clause["cls_type"])
        except Exception as e:
            print(f"Risk analysis failed for clause {clause['clause_id']}: {e}")
            result = {
                "risk": "Medium",
                "framework": "IRDAI",
                "status": "Partial",
                "reason": "Fallback",
                "score": 0.5
            }

        risk_obj = RiskResult(
            clause_id=clause["clause_id"],
            risk=result.get("risk", "Medium"),
            framework=result.get("framework", "IRDAI"),
            status=result.get("status", "Partial"),
            reason=result.get("reason", "Fallback"),
            score=result.get("score", 0.5)
        )

        # Convert to dict for Pydantic
        risks_list.append(risk_obj.dict())

    # Run risk analysis concurrently
    await asyncio.gather(*(analyze_single_clause(c) for c in state.classified))

    state.risks = risks_list
    return state
