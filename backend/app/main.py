from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import os
import base64
import traceback
import logging
import aiofiles
import tempfile
import json

from .workflow import compiled_workflow
from .Schema import ContractState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Recursively convert Pydantic models to dicts"""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, 'dict'):
        return obj.dict()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

app = FastAPI(title="Compliance Review Backend", debug=True)

@app.get("/")
async def root():
    return {"message": "AgenticSpark Compliance Analyzer Backend", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "backend": "running"}

@app.post("/analyze")
async def analyze(file_data: str = Form(...), filename: str = Form(...)):
    logger.info(f"Received analyze request for file: {filename}")

    # Decode base64 file data
    try:
        file_bytes = base64.b64decode(file_data)
        logger.info(f"Successfully decoded file data, size: {len(file_bytes)} bytes")
    except Exception as e:
        logger.error(f"Failed to decode file data: {str(e)}")
        return JSONResponse(status_code=400, content={"error": f"Invalid file data: {str(e)}"})

    temp_path = None

    # Save to a temporary file asynchronously
    try:
        with tempfile.NamedTemporaryFile(delete=False, prefix="tmp_", suffix=f"_{filename}") as tmp_file:
            temp_path = tmp_file.name

        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(file_bytes)

        logger.info(f"File saved to: {temp_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Failed to save file: {str(e)}"})

    # Run workflow
    try:
        logger.info("Starting workflow execution...")
        state = ContractState(file_path=temp_path)
        result = await compiled_workflow.ainvoke(state)
        logger.info("Workflow completed successfully")

        # Convert nested Pydantic models recursively
        payload = convert_to_serializable(result)
        logger.info(f"Final payload type after conversion: {type(payload)}")
        logger.info("Payload conversion successful")
        
        return JSONResponse(content=payload)
        
    except Exception as e:
        error_msg = f"Workflow error: {str(e)}"
        full_traceback = traceback.format_exc()
        logger.error(f"{error_msg}\nFull traceback:\n{full_traceback}")
        return JSONResponse(status_code=500, content={"error": error_msg, "traceback": full_traceback})
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")
