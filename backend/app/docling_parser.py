import re
from typing import List
from langchain_docling import DoclingLoader
from langchain.schema import Document as LC_Document
import nltk

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

# Lazy load embeddings model
embedder = None

def _get_embedder():
    """Lazy load the sentence transformer model"""
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder


def _split_regex(text: str) -> List[str]:
    """Split text using legal-style markers like Section, Clause, Article, etc."""
    pattern = re.compile(
        r"(?=(\d+(\.\d+)*\s|Section\s+\d+|Clause\s+\d+|Article\s+\w+|\([a-z]\)|\([ivx]+\)))",
        re.IGNORECASE,
    )
    return [p.strip() for p in pattern.split(text) if p and p.strip()]


def _split_sentences(text: str) -> List[str]:
    """Fallback: split text into sentences using NLTK."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _split_markdown_sections(text: str) -> List[str]:
    """Split markdown text by headers and sections."""
    import re
    # Split by markdown headers (# ## ###) and other section markers
    sections = re.split(r'(?=^#{1,6}\s|\n#{1,6}\s|\n\n[A-Z][^.!?]*:)', text, flags=re.MULTILINE)
    return [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]


def _merge_semantic(sentences: List[str], threshold=0.8) -> List[str]:
    """Merge semantically similar consecutive chunks."""
    if len(sentences) <= 1:
        return sentences

    from sentence_transformers import util
    embedder = _get_embedder()
    embeddings = embedder.encode(sentences, convert_to_tensor=True, batch_size=16)
    merged, buffer = [], sentences[0]
    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i - 1], embeddings[i]).item()
        adaptive_th = threshold + 0.05 if len(sentences[i]) < 50 else threshold
        if sim > adaptive_th:
            buffer += " " + sentences[i]
        else:
            merged.append(buffer)
            buffer = sentences[i]
    merged.append(buffer)
    return merged


def extract_clauses(file_path: str, use_semantic_merge: bool = False):
    """
    Extract clauses from contracts using DoclingLoader with aggressive splitting.

    Args:
        file_path: Path to the PDF file
        use_semantic_merge: Whether to use semantic merging (slower but more accurate)
    """
    loader = DoclingLoader(file_path=file_path, export_type="markdown")
    docs: List[LC_Document] = loader.load()

    clauses, cid = [], 0
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        # For markdown export, split by headers and sections
        parts = _split_markdown_sections(text)
        if not parts or len(parts) == 1:
            parts = _split_regex(text)
        if not parts or len(parts) == 1:
            parts = [p for p in text.split("\n\n") if p.strip()]
        if not parts or len(parts) == 1:
            parts = _split_sentences(text)

        # Only use semantic merging if explicitly requested (it's slow)
        if use_semantic_merge and len(parts) > 1:
            parts = _merge_semantic(parts)

        for chunk in parts:
            if len(chunk) < 20:
                continue
            cid += 1
            clauses.append(
                {
                    "clause_id": f"C{cid}",
                    "text": chunk,
                    "page": doc.metadata.get("page", None),
                    "section": doc.metadata.get("section", None),
                }
            )

    return clauses
