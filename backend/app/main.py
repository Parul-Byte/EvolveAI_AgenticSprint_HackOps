from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import os
import base64
import traceback
import logging

from .workflow import compiled_workflow, ContractState

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Compliance Review Backend")

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

    temp_path = f"temp_{filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"File saved to: {temp_path}")
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Failed to save file: {str(e)}"})

    try:
        logger.info("Starting workflow execution...")
        state = ContractState(file_path=temp_path)
        result = await compiled_workflow.ainvoke(state)
        logger.info("Workflow completed successfully")
        return JSONResponse(content=result.dict())
    except Exception as e:
        error_msg = f"Workflow error: {str(e)}"
        logger.error(f"{error_msg}\nTraceback: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": error_msg, "traceback": traceback.format_exc()})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temp file: {temp_path}")
