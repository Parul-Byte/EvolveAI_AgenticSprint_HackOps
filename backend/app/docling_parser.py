import re
from typing import List
from langchain_docling import DoclingLoader
from langchain.schema import Document as LC_Document
from sentence_transformers import SentenceTransformer, util
import nltk

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)

# Load embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")


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


def _merge_semantic(sentences: List[str], threshold=0.8) -> List[str]:
    """Merge semantically similar consecutive chunks."""
    if len(sentences) <= 1:
        return sentences
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


def extract_clauses(file_path: str):
    """
    Extract clauses from contracts using DoclingLoader with aggressive splitting.
    """
    loader = DoclingLoader(file_path=file_path, export_type="DOC_CHUNKS")
    docs: List[LC_Document] = loader.load()

    clauses, cid = [], 0
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        # Try regex → then paragraphs → then sentences
        parts = _split_regex(text)
        if not parts or len(parts) == 1:
            parts = [p for p in text.split("\n\n") if p.strip()]
        if not parts or len(parts) == 1:
            parts = _split_sentences(text)

        parts = _merge_semantic(parts) if len(parts) > 1 else parts

        for chunk in parts:
            if len(chunk) < 20:
                continue
            cid += 1
            clauses.append(
                {
                    "clause_id": f"C{cid}",
                    "text": chunk,
                    "metadata": doc.metadata,  # page, section info
                }
            )

    return clauses
