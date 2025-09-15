from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import os
import base64

from .workflow import compiled_workflow, ContractState

app = FastAPI(title="Compliance Review Backend")

@app.post("/analyze")
async def analyze(file_data: str = Form(...), filename: str = Form(...)):
    # Decode base64 file data
    try:
        file_bytes = base64.b64decode(file_data)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid file data: {str(e)}"})

    temp_path = f"temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        state = ContractState(file_path=temp_path)
        result = await compiled_workflow.ainvoke(state)
        return JSONResponse(content=result.dict())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
