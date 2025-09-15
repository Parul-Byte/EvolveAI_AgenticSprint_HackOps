import os
import httpx
import json
from dotenv import load_dotenv
from typing import Dict, Any

# Load .env file
load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate keys early
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY not found in .env or environment")

if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not found. Gemini calls will use fallback.")

# Hugging Face models
LEGAL_BERT_URL = "https://api-inference.huggingface.co/models/nlpaueb/legal-bert-base-uncased"
FLAN_T5_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Import Google Gemini client
try:
    from google import genai
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: google-genai not installed. Install with: pip install google-genai")
    GEMINI_CLIENT = None
    GEMINI_AVAILABLE = False


# -----------------------------
# Hugging Face API Calls
# -----------------------------
async def call_hf(model_url: str, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(model_url, headers=headers, json=payload)
        try:
            data = r.json()
        except Exception:
            return {"error": "Invalid HF response"}
        
        if isinstance(data, dict) and "error" in data:
            print(f"HF API error: {data['error']}")
        return data


async def classify_clause_bert(text: str) -> dict:
    """Classify clause type using Legal-BERT"""
    data = await call_hf(LEGAL_BERT_URL, {"inputs": text})
    
    # Standard HF classifier output
    if isinstance(data, list) and data and isinstance(data[0], list):
        best = max(data[0], key=lambda x: x["score"])
        return {"type": best["label"], "confidence": best["score"]}
    
    # Alternative HF JSON output
    if isinstance(data, dict) and "labels" in data:
        return {"type": data["labels"][0], "confidence": data["scores"][0]}
    
    return {"type": "Other", "confidence": 0.5}


async def analyze_risk_t5(text: str, cls_type: str) -> dict:
    """Analyze risk using Flan-T5 with structured JSON output"""
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

    data = await call_hf(
        FLAN_T5_URL,
        {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    )

    if isinstance(data, list) and data and "generated_text" in data[0]:
        try:
            return json.loads(data[0]["generated_text"])
        except Exception as e:
            print(f"⚠️ Risk parsing error: {e}")

    return {
        "risk": "Medium",
        "framework": "IRDAI",
        "status": "Partial",
        "reason": "Fallback used",
        "score": 0.5
    }


# -----------------------------
# Gemini API Calls
# -----------------------------
async def call_gemini(prompt: str, max_tokens: int = 512) -> str:
    """
    Calls Google Gemini API using the official client.
    Returns a generated text response, or fallback if unavailable.
    """
    if not GEMINI_AVAILABLE or GEMINI_CLIENT is None:
        print("⚠️ Gemini client not available, using fallback response")
        return _get_fallback_response(prompt)

    try:
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-1.5-flash",  # Stable fast model
            contents=prompt,
            config=genai.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
            )
        )

        # Prefer structured candidates
        if hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
        
        return getattr(response, "text", "")

    except Exception as e:
        print(f"⚠️ Error calling Gemini API: {str(e)}")
        return _get_fallback_response(prompt)


def _get_fallback_response(prompt: str) -> str:
    """Generate a fallback compliance advisory if Gemini is unavailable"""
    return """# Compliance Advisory Report

## Executive Summary
Automated analysis detected compliance considerations requiring attention. 
Areas of risk exist where current practices may not fully align with regulations.

## Key Recommendations
1. **High Priority**: Conduct a compliance framework review against GDPR, HIPAA, IRDAI.
2. **Medium Priority**: Optimize internal processes and monitoring.
3. **Ongoing**: Enhance staff training and implement clear workflows.

## Implementation
- Phase 1 (0-3 months): Immediate compliance review and policy updates
- Phase 2 (3-6 months): Process automation and training
- Phase 3 (6-12 months): Full monitoring + quality controls

*This advisory is AI-generated. Please validate with legal experts.*"""
