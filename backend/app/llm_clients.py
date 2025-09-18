import os
import httpx
import json
from dotenv import load_dotenv

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

# Gemini client
try:
    from google import genai
    GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_CLIENT = None
    GEMINI_AVAILABLE = False


async def call_hf(model_url: str, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(model_url, headers=headers, json=payload)
        try:
            data = r.json()
        except Exception:
            return {"error": "Invalid HF response"}
        if "error" in data:
            print(f"HuggingFace API error: {data['error']}")
        return data


# async def classify_clause_bert(text: str, candidate_labels=None) -> dict:
    if candidate_labels is None:
        candidate_labels = ["Confidentiality", "Liability", "Termination", "Other"]

    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": candidate_labels}
    }

    data = await call_hf(LEGAL_BERT_URL, payload)

    # Hugging Face returns 'labels' and 'scores'
    if isinstance(data, dict) and "labels" in data and "scores" in data:
        return {"type": data["labels"][0], "confidence": data["scores"][0]}

    # fallback
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
    if isinstance(data, list) and "generated_text" in data[0]:
        try:
            return json.loads(data[0]["generated_text"])
        except Exception as e:
            print(f"Risk parsing error: {e}")
    return {"risk": "Medium", "framework": "IRDAI", "status": "Partial", "reason": "Fallback", "score": 0.5}


async def call_gemini(prompt: str, max_tokens: int = 512) -> str:
    if not GEMINI_AVAILABLE or GEMINI_CLIENT is None:
        return _get_fallback_response(prompt)
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=genai.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
                top_p=0.8,
                top_k=40,
            )
        )
        if hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text
        return getattr(response, "text", "")
    except Exception as e:
        print(f"Gemini API error: {e}")
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
