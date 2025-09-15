import os
import httpx
import json

# Load API keys
HF_API_KEY = os.getenv("HF_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Hugging Face model endpoints
LEGAL_BERT_URL = "https://api-inference.huggingface.co/models/nlpaueb/legal-bert-base-uncased"
FLAN_T5_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# ----------------------------
# Hugging Face client (general)
# ----------------------------
async def call_hf(model_url: str, payload: dict) -> dict:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(model_url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Hugging Face API error {resp.status_code}: {resp.text}")
        return resp.json()

# ----------------------------
# Clause classification (Legal-BERT)
# ----------------------------
async def classify_clause_bert(clause_text: str) -> dict:
    """
    Use Legal-BERT for classification.
    Hugging Face will return logits/scores, so we map them manually.
    """
    payload = {"inputs": clause_text}
    data = await call_hf(LEGAL_BERT_URL, payload)

    # Expected output: [{"label": "LABEL_0", "score": 0.95}, ...]
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        best = max(data[0], key=lambda x: x["score"])
        return {"type": best["label"], "confidence": best["score"]}

    # Fallback
    return {"type": "Other", "confidence": 0.5}

# ----------------------------
# Risk analysis (Flan-T5)
# ----------------------------
async def analyze_risk_t5(clause_text: str, cls_type: str) -> dict:
    """
    Use Flan-T5 for reasoning about compliance risks.
    Returns JSON-like output.
    """
    prompt = f"""You are a compliance analyst. Evaluate the following clause for compliance with IRDAI, GDPR, HIPAA.

Clause: {clause_text}
Classification: {cls_type}

Return JSON:
{{
  "risk": "Low|Medium|High",
  "framework": "<IRDAI|GDPR|HIPAA>",
  "status": "Aligned|Partial|Gap",
  "reason": "...",
  "score": 0.xx
}}"""

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
    data = await call_hf(FLAN_T5_URL, payload)

    if isinstance(data, list) and "generated_text" in data[0]:
        text = data[0]["generated_text"]
        try:
            return json.loads(text)
        except:
            return {
                "risk": "Medium",
                "framework": "IRDAI",
                "status": "Partial",
                "reason": "Fallback",
                "score": 0.5
            }

    return {"risk": "Medium", "framework": "IRDAI", "status": "Partial", "reason": "Fallback", "score": 0.5}

# ----------------------------
# Gemini client (Advisory)
# ----------------------------
async def call_gemini(prompt: str, max_tokens: int = 512) -> str:
    """
    Calls Gemini API for summaries and recommendations.
    Replace URL with actual Gemini endpoint.
    """
    url = "https://api.gemini.example/v1/generate"
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_output_tokens": max_tokens, "temperature": 0.2}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data.get("text") or data.get("output") or json.dumps(data)
