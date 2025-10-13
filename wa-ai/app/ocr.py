import logging
import httpx
import base64
import asyncio
import time
from typing import Optional, Dict, Any, List
from .settings import settings
from .correlation import generate_correlation_id, get_correlation_id, set_correlation_id, CorrelationContext

logger = logging.getLogger(__name__)


class MistralOCRClient:
    def __init__(self):
        self.api_key = settings.mistral_api_key
        self.model = settings.mistral_ocr_model
        self.base_url = "https://api.mistral.ai/v1"
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

        # Log warning if API key is not properly configured
        if warning := settings.get_mistral_api_key_warning():
            logger.warning(f"Mistral OCR Client: {warning}")

    async def _make_ocr_request_with_retry(self, ocr_request: Dict[str, Any], headers: Dict[str, str], correlation_id: str) -> Dict[str, Any]:
        """Make OCR request with retry logic and comprehensive error handling."""
        with CorrelationContext(correlation_id):
            last_exception = Exception("No attempts made")

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"[MISTRAL-OCR] [{correlation_id}] Attempt {attempt + 1}/{self.max_retries} for OCR request")

                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.post(
                            f"{self.base_url}/ocr",
                            json=ocr_request,
                            headers=headers
                        )
                        response.raise_for_status()
                        result = response.json()

                    logger.info(f"[MISTRAL-OCR] [{correlation_id}] OCR request completed successfully")
                    return result

                except httpx.TimeoutException as e:
                    last_exception = e
                    logger.warning(f"[MISTRAL-OCR] [{correlation_id}] Timeout on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue

                except httpx.ConnectError as e:
                    last_exception = e
                    logger.error(f"[MISTRAL-OCR] [{correlation_id}] Connection error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue

                except httpx.NetworkError as e:
                    last_exception = e
                    logger.error(f"[MISTRAL-OCR] [{correlation_id}] Network error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue

                except Exception as e:
                    last_exception = e
                    logger.exception(f"[MISTRAL-OCR] [{correlation_id}] Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                    # Don't retry on unexpected errors like API key issues
                    break

            # All retries failed
            logger.error(f"[MISTRAL-OCR] [{correlation_id}] OCR request failed after {self.max_retries} attempts. Last error: {type(last_exception).__name__}: {last_exception}")
            raise last_exception

    async def process_url(self, image_url: str, correlation_id: Optional[str] = None) -> Optional[str]:
        """
        Process an image from URL using Mistral OCR with enhanced error handling.
        Returns extracted text or None if failed.
        """
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            logger.info(f"[MISTRAL-OCR] [{correlation_id}] Starting OCR processing for URL: {image_url[:50]}...")

            if not self.api_key:
                logger.warning(f"[MISTRAL-OCR] [{correlation_id}] Mistral API key not configured")
                return None

            try:
                # First, download the image with retry logic
                image_data = await self._download_image_with_retry(image_url, correlation_id)
                if not image_data:
                    logger.error(f"[MISTRAL-OCR] [{correlation_id}] Failed to download image from URL")
                    return None

                # Convert to base64
                base64_image = base64.b64encode(image_data).decode('utf-8')
                logger.info(f"[MISTRAL-OCR] [{correlation_id}] Image downloaded and converted to base64, size: {len(base64_image)}")

                # Prepare OCR request
                ocr_request = {
                    "model": self.model,
                    "document": {
                        "type": "image_url",
                        "image_url": image_url
                    }
                }

                # Make OCR API call with retry logic
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                result = await self._make_ocr_request_with_retry(ocr_request, headers, correlation_id)

                # Extract text from response
                extracted_text = ""
                if "pages" in result:
                    for page in result["pages"]:
                        if "markdown" in page:
                            extracted_text += page["markdown"] + "\n"

                final_text = extracted_text.strip() if extracted_text else None
                logger.info(f"[MISTRAL-OCR] [{correlation_id}] OCR processing completed, extracted text length: {len(final_text) if final_text else 0}")
                return final_text

            except Exception as e:
                logger.exception(f"[MISTRAL-OCR] [{correlation_id}] Error processing image URL: {type(e).__name__}: {e}")
                return None

    async def _download_image_with_retry(self, image_url: str, correlation_id: str) -> Optional[bytes]:
        """Download image with retry logic."""
        with CorrelationContext(correlation_id):
            last_exception = None

            for attempt in range(3):  # Fewer retries for image download
                try:
                    logger.info(f"[MISTRAL-OCR] [{correlation_id}] Download attempt {attempt + 1}/3 for image")

                    async with httpx.AsyncClient(timeout=30.0) as client:
                        response = await client.get(image_url)
                        response.raise_for_status()
                        image_data = response.content

                    logger.info(f"[MISTRAL-OCR] [{correlation_id}] Image downloaded successfully, size: {len(image_data)} bytes")
                    return image_data

                except Exception as e:
                    last_exception = e
                    logger.warning(f"[MISTRAL-OCR] [{correlation_id}] Download attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                    if attempt < 2:  # Less than 3 attempts
                        await asyncio.sleep(1.0 * (attempt + 1))

            logger.error(f"[MISTRAL-OCR] [{correlation_id}] Image download failed after 3 attempts: {type(last_exception).__name__}: {last_exception}")
            return None

    async def process_base64(self, base64_image: str, mime_type: str = "image/jpeg") -> Optional[str]:
        """
        Process a base64 encoded image using Mistral OCR.
        Returns extracted text or None if failed.
        """
        if not self.api_key:
            logger.warning("Mistral API key not configured. Please set MISTRAL_API_KEY in your .env file.")
            return None

        try:
            # Prepare OCR request with base64 image
            ocr_request = {
                "model": self.model,
                "document": {
                    "type": "base64",
                    "data": base64_image,
                    "mime_type": mime_type
                }
            }

            # Make OCR API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/ocr",
                    json=ocr_request,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()

            # Extract text from response
            extracted_text = ""
            if "pages" in result:
                for page in result["pages"]:
                    if "markdown" in page:
                        extracted_text += page["markdown"] + "\n"

            return extracted_text.strip() if extracted_text else None

        except Exception as e:
            logger.error(f"Error processing base64 image with Mistral OCR: {e}")
            return None

    async def run_for_row(self, row: Dict[str, Any]) -> Optional[str]:
        """
        Process OCR for a database row that might contain image URLs or data.
        This is a flexible method that can handle different row structures.
        """
        # Look for image URLs in common fields
        image_fields = ["image_url", "image", "photo", "picture", "attachment", "media_url"]

        for field in image_fields:
            if field in row and row[field]:
                image_url = row[field]
                if isinstance(image_url, str) and (image_url.startswith('http') or image_url.startswith('data:')):
                    logger.info(f"Processing OCR for image field: {field}")
                    return await self.process_url(image_url)

        # Look for base64 image data
        for field in image_fields:
            if field in row and row[field]:
                image_data = row[field]
                if isinstance(image_data, str) and image_data.startswith('data:image'):
                    logger.info(f"Processing OCR for base64 image field: {field}")
                    # Extract base64 data from data URL
                    try:
                        header, base64_data = image_data.split(',', 1)
                        mime_type = header.split(':')[1].split(';')[0]
                        return await self.process_base64(base64_data, mime_type)
                    except Exception as e:
                        logger.error(f"Error parsing base64 image data: {e}")
                        continue

        logger.warning(f"No processable image found in row: {list(row.keys())}")
        return None


# Global OCR client instance
ocr_client = MistralOCRClient()