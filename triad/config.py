"""Paths and environment for TRIAD-RAG. Everything heavy lives on D: (C: is full)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
CACHE_DIR = ROOT / ".cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"
INDEX_DIR = CACHE_DIR / "index"
KEYS_FILE = ROOT / "secrets" / "groq_keys.txt"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Keep model downloads (Contriever, bge) off the nearly-full C: drive.
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))
