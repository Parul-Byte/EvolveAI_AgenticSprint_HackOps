from ..docling_parser import extract_clauses

async def ingestion_agent(state):
    """
    Ingestion Agent:
    - Uses Docling to parse contract and extract clauses
    """
    state["clauses"] = extract_clauses(state["file_path"])
    return state
