"""Centralized application configuration with Pydantic validation."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Validates all environment configuration at startup."""

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    api_port: int = 8000
    api_host: str = "127.0.0.1"
    chroma_dir: Path = Path("data/chroma_db")
    yaml_path: Path = Path("data/alternative_docs.yaml")
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_temperature: float = 0.0
    max_agent_iterations: int = 5
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
