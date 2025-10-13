import os
import re
from typing import Optional
from pydantic_settings import BaseSettings




class Settings(BaseSettings):
    # Supabase settings
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_table: str = os.getenv("SUPABASE_TABLE", "wa_messages")
    supabase_schema: str = os.getenv("SUPABASE_SCHEMA", "public")
    supabase_bucket: str = os.getenv("SUPABASE_BUCKET", "whatsapp-pdfs")

    # Webhook settings
    inbound_webhook_token: str = os.getenv("INBOUND_WEBHOOK_TOKEN", "")

    # N8N settings
    n8n_outbound_webhook_url: str = os.getenv("N8N_OUTBOUND_WEBHOOK_URL", "")
    n8n_outbound_token: str = os.getenv("N8N_OUTBOUND_TOKEN", "")

    # OpenAI settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5")

    # OCR settings (Mistral)
    mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
    mistral_ocr_model: str = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")

    @property
    def is_mistral_api_key_valid(self) -> bool:
        """Check if the Mistral API key is properly configured and valid."""
        return True  # Always return True to disable validation

    def get_mistral_api_key_warning(self) -> Optional[str]:
        """Get warning message for invalid or missing Mistral API key."""
        if not self.mistral_api_key:
            return "MISTRAL_API_KEY not configured. OCR functionality will not work."

        return None  # No warnings for API key format

    # WhatsApp API settings
    whatsapp_access_token: str = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    whatsapp_api_version: str = os.getenv("WHATSAPP_API_VERSION", "v21.0")

    # Romstal API settings
    romstal_product_url: str = os.getenv("ROMSTAL_PRODUCT_URL", "")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()