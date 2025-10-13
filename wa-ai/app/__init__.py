"""
Romstal Assistant App Package

This package contains the core functionality for the Romstal WhatsApp assistant:
- settings: Configuration management
- supa: Supabase database operations
- llm: OpenAI LLM integration with tools
- ocr: Mistral OCR functionality
- outbound: N8N webhook integration
"""

from .settings import settings
from .supa import supabase_client
from .llm import llm_client
from .ocr import ocr_client
from .outbound import n8n_client

__all__ = [
    "settings",
    "supabase_client",
    "llm_client",
    "ocr_client",
    "n8n_client"
]