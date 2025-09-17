from ..docling_parser import extract_clauses

async def ingestion_agent(state):
    """
    Ingestion Agent:
    - Uses Docling parser for clause extraction
    """
    state.clauses = extract_clauses(state.file_path)
    return state
