"""Centralized configuration for DocuQuery."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

TOP_K: int = 3
MIN_RELEVANCE_SCORE: float = 0.2
RETRIEVAL_CANDIDATES: int = 50
RRF_K: int = 60
QUERY_EXPANSION: bool = True
QUERY_EXPANSION_TIMEOUT_SECONDS: float = 10.0
CONTEXT_NEIGHBORS: int = 1

DEFAULT_LLM_MODEL: str = os.getenv("DOCUQUERY_LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE: float = 0.0
LLM_TIMEOUT_SECONDS: float = 30.0
LLM_MAX_RETRIES: int = 1

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()