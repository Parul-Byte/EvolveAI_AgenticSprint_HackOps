import re
import spacy
from docling.document_converter import DocumentConverter  # Updated import for correct API
from sentence_transformers import SentenceTransformer, util

# Load NLP + embeddings with fallback
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model 'en_core_web_sm' loaded successfully")
except OSError:
    print("⚠️  spaCy model 'en_core_web_sm' not found. Please run:")
    print("   uv run python -m spacy download en_core_web_sm")
    print("   Using basic sentence splitting as fallback...")
    nlp = None

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Clause Splitting
# -----------------------------
def _split_regex(text: str):
    """
    Split text based on legal numbering patterns like:
    1., 1.1, Section 3, Clause 7, Article IV, (a), (i)
    """
    pattern = re.compile(
        r"(?=(\d+(\.\d+)*\s|Section\s+\d+|Clause\s+\d+|Article\s+\w+|\([a-z]\)|\([ivx]+\)))",
        re.IGNORECASE
    )
    return [p.strip() for p in pattern.split(text) if p and p.strip()]

def _split_nlp(text: str):
    """Fallback: sentence segmentation with spaCy"""
    if nlp is None:
        # Fallback to simple sentence splitting if spaCy model not available
        import re
        sentences = re.split(r'[.!?]+', text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 20]

    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]

# -----------------------------
# Semantic Merging (optimized)
# -----------------------------
def _merge_semantic(sentences, threshold=0.8):
    """
    Merge consecutive sentences if they are semantically very similar
    to avoid splitting one clause into fragments.
    Adaptive threshold based on sentence length.
    """
    if len(sentences) == 1:
        return sentences

    embeddings = embedder.encode(sentences, convert_to_tensor=True, batch_size=16)
    merged, buffer = [], sentences[0]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
        # Adaptive threshold: stricter for short sentences
        adaptive_th = threshold + 0.05 if len(sentences[i]) < 50 else threshold
        if sim > adaptive_th:
            buffer += " " + sentences[i]
        else:
            merged.append(buffer)
            buffer = sentences[i]
    merged.append(buffer)
    return merged

# -----------------------------
# Clause Type Heuristics
# -----------------------------
def _guess_clause_type(text: str) -> str:
    """Heuristic to add hints for clause types"""
    lowered = text.lower()
    if "termination" in lowered:
        return "Termination"
    if "confidential" in lowered or "non-disclosure" in lowered:
        return "Confidentiality"
    if "governing law" in lowered or "jurisdiction" in lowered:
        return "Governing Law"
    if "indemnify" in lowered or "liability" in lowered:
        return "Liability"
    if "payment" in lowered or "fees" in lowered:
        return "Payment"
    if "privacy" in lowered or "data protection" in lowered:
        return "Data Protection"
    return "General"

# -----------------------------
# Extract Clauses
# -----------------------------
def extract_clauses(file_path: str):
    """
    Extract structured clauses from PDF/DOCX:
    - Handles numbered sections, bullets, tables
    - Adds heuristic clause_type hints
    """
    # Updated to use DocumentConverter (replaces old loaders)
    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc = result.document

    clauses, cid = [], 0

    for block in doc.body:  # Corrected from doc.blocks to doc.body
        text = (block.text or "").strip()
        if not text:
            continue

        # Handle tables
        if block.type == "table":
            table_text = " ".join([" | ".join(row) for row in block.cells])
            text = f"[TABLE] {table_text}"

        # Handle bullets (simple regex)
        if re.match(r"^[-•·]\s+", text):
            text = f"[BULLET] {text}"

        # Split and merge
        parts = _split_regex(text) or _split_nlp(text)
        parts = _merge_semantic(parts) if len(parts) > 1 else parts

        for chunk in parts:
            if len(chunk) < 20:
                continue
            cid += 1
            clauses.append({
                "clause_id": f"C{cid}",
                "text": chunk,
                "page": getattr(block, "page", None),
                "section": getattr(block, "heading", None),
                "hint_type": _guess_clause_type(chunk),  # NEW
            })

    return clauses