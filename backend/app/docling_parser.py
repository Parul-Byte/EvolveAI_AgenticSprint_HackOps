from docling.document_converter import DocumentConverter
import re

def extract_clauses(file_path: str):
    """
    Extracts clauses from a PDF or DOCX file using Docling.
    Returns a list of dicts with clause_id, text, page, section.
    Improved: Uses capturing regex to include clause headers, groups by sections from headings,
    and handles more clause patterns.
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        doc = result.document

        clauses = []
        cid = 0
        current_section = None

        for block in doc.blocks:
            # Update section if block is a heading
            if hasattr(block, 'item_type') and block.item_type == 'heading':
                current_section = (block.text or "").strip()
                continue  # Skip headings as clauses unless they contain sub-clauses

            text = (block.text or "").strip()
            if not text:
                continue

            clause_pattern = r"(\d+\.\d+|\bClause\s+\d+|\bSection\s+\d+|\bArticle\s+\d+|\bSubsection\s+\d+)"
            parts = re.split(clause_pattern, text)

            for i in range(0, len(parts), 2):
                body = parts[i].strip()
                header = parts[i+1].strip() if i+1 < len(parts) else ""
                if body or header:
                    cid += 1
                    full_text = f"{header} {body}".strip()
                    clauses.append({
                        "clause_id": f"C{cid}",
                        "text": full_text,
                        "page": getattr(block, "page_no", getattr(block, "page", None)),
                        "section": current_section
                    })

        return clauses
    except Exception as e:
        raise RuntimeError(f"Error parsing document {file_path}: {str(e)}")
