"""
Configuración del LLM (mismo esquema de Lecture11/part01: un cliente OpenAI
apuntando a cualquier backend compatible).

La API key NUNCA va en el código: se lee del archivo .env (ver .env.example).
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

from datos import ROOT

load_dotenv(ROOT / ".env")

BACKEND = os.getenv("LLM_BACKEND", "nvidia")   # "nvidia" | "groq" | "ollama"

CONFIGS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b"),
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "model": os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": os.getenv("LLM_MODEL", "qwen2.5:7b"),
    },
}

# Los embeddings siempre se calculan con NVIDIA NIM (índice construido con este modelo)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b")
NVIDIA_BASE_URL = CONFIGS["nvidia"]["base_url"]

cfg = CONFIGS[BACKEND]
MODEL = cfg["model"]


def get_client() -> OpenAI:
    """Cliente para embeddings (NVIDIA NIM)."""
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        raise ValueError("Falta NVIDIA_API_KEY en el archivo .env (https://build.nvidia.com/)")
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=key)


def get_chat_client() -> OpenAI:
    """Cliente para el LLM del agente (backend configurable)."""
    if not cfg["api_key"]:
        raise ValueError(f"Falta la API key del backend '{BACKEND}' en el archivo .env")
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])
