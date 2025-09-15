# Backend for EvolveAI AgenticSprint HackOps

This backend provides a FastAPI-based service for contract analysis using multi-agent workflows powered by LangGraph. It processes uploaded documents, extracts clauses, classifies them, assesses risks, and provides advisory recommendations.

## Features

- **Document Ingestion**: Parses PDF/DOCX files using Docling
- **Clause Classification**: Categorizes contract clauses using AI
- **Risk Assessment**: Identifies potential risks in contracts
- **Advisory Generation**: Provides recommendations based on analysis
- **Multi-Agent Workflow**: Orchestrates agents using LangGraph

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── schemas.py           # Pydantic models
│   ├── docling_parser.py    # Document parsing logic
│   ├── llm_clients.py       # API clients for Gemini and HF
│   ├── workflow.py          # LangGraph workflow definition
│   └── agents/
│       ├── ingestion_agent.py
│       ├── classification_agent.py
│       ├── risk_agent.py
│       └── advisory_agent.py
├── uploads/                 # Uploaded files 
└── requirements.txt         # Python dependencies
```

## Setup

1. **Clone the repository** (if not already done):
   ```
   git clone https://github.com/Parul-Byte/EvolveAI_AgenticSprint_HackOps.git
   cd EvolveAI_AgenticSprint_HackOps/backend
   ```

2. **Create a virtual environment**:
   ```
   uv venv
   ```

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. **Install dependencies**:
   ```
   uv pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   - Copy `.env.example` from the root to `.env`
   - Fill in your API keys for Gemini, Hugging Face, etc.

## Running the Application

1. **Start the FastAPI server**:
   ```
   uvicorn app.main:app --reload
   ```

2. **Run Streamlit frontend**:
   ```
   streamlit run backend/frontend/frontend.py
   ```
## API Endpoints

- `POST /upload`: Upload a contract document for analysis
- `GET /status/{task_id}`: Check analysis status
- `GET /results/{task_id}`: Get analysis results

## Technologies Used

- **FastAPI**: Web framework
- **LangGraph**: Multi-agent orchestration
- **Docling**: Document parsing
- **Pydantic**: Data validation
- **Google Gemini**: AI model for classification/risk assessment
- **Hugging Face/NVIDIA**: Alternative LLM providers

## License

This project is licensed under the MIT License - see the LICENSE file in the root directory for details.
