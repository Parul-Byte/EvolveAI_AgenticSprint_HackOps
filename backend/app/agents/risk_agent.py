from ..llm_clients import analyze_risk_t5
from ..workflow import ContractState, RiskResult

async def risk_agent(state: ContractState) -> ContractState:
    risks = []
    for clause in state.classified:
        result = await analyze_risk_t5(clause.text, clause.cls_type)
        risks.append(
            RiskResult(
                clause_id=clause.clause_id,
                risk=result["risk"],
                framework=result["framework"],
                status=result["status"],
                reason=result["reason"],
                score=result.get("score", 0.5),
            )
        )
    state.risks = risks
    state.overall_score = sum(r.score for r in risks) / len(risks) if risks else 0.0
    return state
