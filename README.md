# AI YAML Specification Assistant

An autonomous AI agent that interprets user queries, evaluates YAML specifications, and returns structured API documentation and document guidance. The included dataset is a generic, synthetic example for demonstrating YAML-driven tool use.

## Overview

The assistant helps a user understand which document combinations can satisfy a missing certification requirement. It combines deterministic YAML lookup with an LLM-powered conversational layer, keeping the document rules in a file that can be reviewed and updated independently of the application code.

## Architecture / Pipeline

```text
Browser chat UI
	|
	v
FastAPI (/chat, /badges, /alternatives/{badge_name})
	|
	v
LangChain + LangGraph agent ----> custom YAML alternative-documents tool
	|
	v
Groq-hosted LLM response
```

1. The static frontend sends a user message and compact conversation history to FastAPI.
2. The agent identifies the relevant specification and invokes the custom tool when a required document is missing.
3. The tool loads `data/alternative_docs.yaml`, ranks alternatives, and formats their conditions and recovery guidance.
4. LangChain/LangGraph and a Groq model turn the grounded result into a concise chat response.

## Project Structure

- `api/` FastAPI application and HTTP routes
- `agent/` LangChain/LangGraph orchestration and response polishing
- `tools/` YAML-backed alternative-document tool
- `data/` generic synthetic specification rules
- `frontend/` static HTML chat client

## Prerequisites

- Python 3.10+
- A Groq API key

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
API_PORT=8000
```

## Run The App

Open two terminals from the project root.

Start the backend API:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start the frontend server:

```powershell
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Open `http://127.0.0.1:5500` for the chat UI or `http://127.0.0.1:8000/docs` for the API documentation.

## Quick Test Prompt

Try: `The supplier doesn't have an FSSC 22000 certificate`

The assistant returns alternative document combinations in confidence order, including required fields, conditions, and common recovery actions.

## Configuration Notes

- Edit `data/alternative_docs.yaml` to add or revise synthetic specifications.
- The frontend expects the API at `http://127.0.0.1:8000`; update the `API` constant in `frontend/index.html` for another local port.
- Never commit `.env` or real supplier documents. Use `.env.example` as the credential template.