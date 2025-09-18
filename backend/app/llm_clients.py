import os
import json
import httpx
import logging
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not found in .env")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not found, Gemini will fallback")

# Hugging Face endpoints
LEGAL_BERT_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
FLAN_T5_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Gemini client (new Google Gen AI SDK)
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except Exception as _:
    GEMINI_CLIENT = None
    GEMINI_AVAILABLE = False

async def call_hf(model_url: str, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    # Set longer timeout for slow API responses
    timeout = httpx.Timeout(timeout=120.0)  # 2 minutes total timeout
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(model_url, headers=headers, json=payload)
            data = r.json()
            
        if isinstance(data, dict) and "error" in data:
            logger.error(f"HuggingFace API error: {data['error']}")
            
        return data
        
    except httpx.ReadTimeout:
        logger.error(f"HuggingFace API timeout for {model_url}")
        return {"error": "API timeout"}
    except httpx.TimeoutException:
        logger.error(f"HuggingFace API timeout (general) for {model_url}")
        return {"error": "API timeout"}
    except Exception as e:
        logger.error(f"HuggingFace API error: {e}")
        return {"error": f"API error: {str(e)}"}

async def classify_clause_bert(text: str, candidate_labels=None) -> dict:
    if candidate_labels is None:
        candidate_labels = ["Confidentiality", "Liability", "Termination", "Other"]
    payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
    data = await call_hf(LEGAL_BERT_URL, payload)

    # Handle timeout/error responses
    if isinstance(data, dict) and "error" in data:
        logger.warning(f"Classification failed, using fallback: {data['error']}")
        return {"type": "Other", "confidence": 0.5}

    # Inference Providers format
    if isinstance(data, list) and data and isinstance(data[0], dict) and "label" in data[0]:
        best = max(data, key=lambda x: x.get("score", 0.0))
        return {"type": best.get("label", "Other"), "confidence": float(best.get("score", 0.0))}
    # Pipeline-like format
    if isinstance(data, dict) and "labels" in data and "scores" in data and data["labels"] and data["scores"]:
        return {"type": data["labels"][0], "confidence": float(data["scores"][0])}
    return {"type": "Other", "confidence": 0.5}

async def analyze_risk_t5(text: str, cls_type: str) -> dict:
    prompt = f"""You are a compliance analyst.
Evaluate this clause for {cls_type} risks under GDPR, HIPAA, IRDAI.
Clause: {text}

Return JSON:
{{
  "risk": "Low|Medium|High",
  "framework": "<IRDAI|GDPR|HIPAA>",
  "status": "Aligned|Partial|Gap",
  "reason": "...",
  "score": 0.xx
}}"""
    data = await call_hf(FLAN_T5_URL, {"inputs": prompt, "parameters": {"max_new_tokens": 200}})

    # Guard HF error (e.g., 404 for FLAN or timeout)
    if isinstance(data, dict) and "error" in data:
        logger.warning(f"Risk analysis failed, using fallback: {data['error']}")
        return {"risk": "Medium", "framework": "IRDAI", "status": "Partial", "reason": "HF error/404/timeout fallback", "score": 0.5}

    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        try:
            return json.loads(data[0]["generated_text"])
        except Exception as e:
            logger.error(f"Risk parsing error: {e}")

    return {"risk": "Medium", "framework": "IRDAI", "status": "Partial", "reason": "Fallback", "score": 0.5}

async def call_gemini(prompt: str, max_tokens: int = 512) -> str:
    if not GEMINI_AVAILABLE or GEMINI_CLIENT is None:
        return _get_fallback_response(prompt)
    try:
        cfg = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.3,
            top_p=0.8,
            top_k=40,
        )
        resp = GEMINI_CLIENT.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=cfg,
        )
        # Prefer resp.text; fallback to candidates if needed
        text = getattr(resp, "text", None)
        if not text and getattr(resp, "candidates", None):
            cand = resp.candidates[0]
            if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                part0 = cand.content.parts[0]
                text = getattr(part0, "text", None)
        return text or _get_fallback_response(prompt)
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return _get_fallback_response(prompt)

def _get_fallback_response(prompt: str) -> str:
    return """# Compliance Advisory Report

## Executive Summary
Automated analysis detected compliance considerations requiring attention.

## Key Recommendations
1. High Priority: Review against GDPR, HIPAA, IRDAI.
2. Medium Priority: Optimize processes and monitoring.
3. Ongoing: Enhance staff training and workflows.

*This advisory is AI-generated. Please consult legal experts for final decisions.*"""
