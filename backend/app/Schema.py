from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ContractState(BaseModel):
    file_path: str
    clauses: List[Dict[str, Any]] = Field(default_factory=list)
    classified: List[Dict[str, Any]] = Field(default_factory=list)
    risks: List[Dict[str, Any]] = Field(default_factory=list)
    advisory: Dict[str, Any] = Field(default_factory=dict)
    overall_score: float = 0.0

class ClassifiedClause(BaseModel):
    clause_id: str
    text: str
    page: int = None
    section: str = None
    cls_type: str
    confidence: float

class RiskResult(BaseModel):
    clause_id: str
    risk: str
    framework: str
    status: str
    reason: str
    score: float

class AdvisoryResult(BaseModel):
    executive_summary: str
    recommendations: List[str]
