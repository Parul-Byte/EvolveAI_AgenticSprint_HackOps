from ..llm_clients import classify_clause_bert
from ..workflow import ContractState, ClassifiedClause

async def classification_agent(state: ContractState) -> ContractState:
    classified = []
    for clause in state.clauses:
        result = await classify_clause_bert(clause["text"])
        classified.append(
            ClassifiedClause(
                clause_id=clause["clause_id"],
                text=clause["text"],
                page=clause.get("page"),
                section=clause.get("section"),
                cls_type=result["type"],
                confidence=result["confidence"],
            )
        )
    state.classified = classified
    return state