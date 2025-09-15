from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from .agents.ingestion_agent import ingestion_agent
from .agents.classification_agent import classification_agent
from .agents.risk_agent import risk_agent
from .agents.advisory_agent import advisory_agent

class ContractState(BaseModel):
    file_path: str
    clauses: List[Dict[str, Any]] = Field(default_factory=list)
    classified: List[Dict[str, Any]] = Field(default_factory=list)
    risks: List[Dict[str, Any]] = Field(default_factory=list)
    advisory: Dict[str, Any] = Field(default_factory=dict)
    overall_score: float = 0.0

workflow = StateGraph(ContractState)

workflow.add_node("ingestion_agent", ingestion_agent)
workflow.add_node("classification_agent", classification_agent)
workflow.add_node("risk_agent", risk_agent)
workflow.add_node("advisory_agent", advisory_agent)

workflow.set_entry_point("ingestion_agent")
workflow.add_edge("ingestion_agent", "classification_agent")
workflow.add_edge("classification_agent", "risk_agent")
workflow.add_edge("risk_agent", "advisory_agent")

compiled_workflow = workflow.compile()
