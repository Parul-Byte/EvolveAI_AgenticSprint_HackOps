from langgraph.graph import StateGraph, END
from typing import TypedDict
from .agents.ingestion_agent import ingestion_agent
from .agents.classification_agent import classification_agent
from .agents.risk_agent import risk_agent
from .agents.advisory_agent import advisory_agent

class ContractState(TypedDict):
    file_path: str
    clauses: list
    classified: list
    risks: list
    advisory: dict
    overall_score: float

workflow = StateGraph(ContractState)

# Add nodes
workflow.add_node("ingestion", ingestion_agent)
workflow.add_node("classification", classification_agent)
workflow.add_node("risk", risk_agent)
workflow.add_node("advisory", advisory_agent)

# Define workflow
workflow.set_entry_point("ingestion")
workflow.add_edge("ingestion", "classification")
workflow.add_edge("classification", "risk")
workflow.add_edge("risk", "advisory")
workflow.add_edge("advisory", END)

compiled_workflow = workflow.compile()
