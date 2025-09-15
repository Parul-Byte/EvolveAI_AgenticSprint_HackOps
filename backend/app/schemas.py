from typing import List, Optional
from pydantic import BaseModel

# -----------------------------
# Clauses and Parsing
# -----------------------------
class Clause(BaseModel):
    clause_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None

class ParseResponse(BaseModel):
    file_name: str
    clauses: List[Clause]

# -----------------------------
# Classification
# -----------------------------
class ClassifiedClause(Clause):
    cls_type: str
    confidence: float

class ClassificationResponse(BaseModel):
    classified_clauses: List[ClassifiedClause]

# -----------------------------
# Risk Analysis
# -----------------------------
class RiskResult(BaseModel):
    clause_id: str
    risk: str   # Low / Medium / High
    framework: str
    status: str # Aligned / Partial / Gap
    reason: str
    score: Optional[float] = None

class AnalyzeResponse(BaseModel):
    risks: List[RiskResult]
    overall_score: float

# -----------------------------
# Advisory / Summary
# -----------------------------
class AdvisoryResult(BaseModel):
    executive_summary: str
    recommendations: List[str]

class SummaryResponse(BaseModel):
    advisory: AdvisoryResult

# -----------------------------
# Full Workflow
# -----------------------------
class WorkflowOutput(BaseModel):
    clauses: List[Clause]
    classified: List[ClassifiedClause]
    risks: List[RiskResult]
    advisory: AdvisoryResult
    overall_score: float
