from fastapi import FastAPI, UploadFile, File
import os, shutil
from .workflow import compiled_workflow, ContractState

app = FastAPI(title="Compliance Review Agent (LangGraph)")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/workflow")
async def run_workflow(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    state = ContractState(file_path=file_path)
    result = await compiled_workflow.ainvoke(state)
    return result
