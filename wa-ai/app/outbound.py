import logging
import httpx
import asyncio
from typing import Dict, Any, Optional
from .settings import settings
from .correlation import generate_correlation_id, get_correlation_id, set_correlation_id, CorrelationContext

logger = logging.getLogger(__name__)


class N8NOutboundClient:
    def __init__(self):
        self.webhook_url = settings.n8n_outbound_webhook_url
        self.token = settings.n8n_outbound_token
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    async def send_to_n8n(self, phone_number: str, message: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Trimite mesajul către N8N pentru livrare pe WhatsApp cu retry și error handling îmbunătățit.
        Returns: Dict cu status_code, text, sau error details
        """
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            logger.info(f"[N8N] Starting message send to {phone_number[:10]}... with message length: {len(message)}")

            if not self.webhook_url:
                logger.warning(f"[N8N] [{correlation_id}] N8N_OUTBOUND_WEBHOOK_URL not configured")
                return {
                    "success": False,
                    "skipped": True,
                    "reason": "N8N_OUTBOUND_WEBHOOK_URL missing",
                    "correlation_id": correlation_id
                }

            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["X-Auth-Token"] = self.token

            # Prepare request payload with correlation ID
            payload = {
                "phone_number": phone_number,
                "message": message,
                "correlation_id": correlation_id,
                "timestamp": asyncio.get_event_loop().time()
            }

            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"[N8N] [{correlation_id}] Attempt {attempt + 1}/{self.max_retries} to send message")

                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.post(
                            self.webhook_url,
                            json=payload,
                            headers=headers,
                        )

                        logger.info(f"[N8N] [{correlation_id}] Response status: {response.status_code}")

                        # Check for successful response
                        if response.status_code < 400:
                            logger.info(f"[N8N] [{correlation_id}] Message sent successfully")
                            return {
                                "success": True,
                                "status_code": response.status_code,
                                "text": response.text,
                                "correlation_id": correlation_id
                            }
                        else:
                            logger.warning(f"[N8N] [{correlation_id}] HTTP error {response.status_code}: {response.text}")

                            if attempt == self.max_retries - 1:  # Last attempt
                                return {
                                    "success": False,
                                    "status_code": response.status_code,
                                    "text": response.text,
                                    "error": f"HTTP {response.status_code}: {response.text}",
                                    "correlation_id": correlation_id
                                }

                except httpx.TimeoutException as e:
                    last_exception = e
                    logger.warning(f"[N8N] [{correlation_id}] Timeout on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

                except httpx.ConnectError as e:
                    last_exception = e
                    logger.error(f"[N8N] [{correlation_id}] Connection error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

                except httpx.NetworkError as e:
                    last_exception = e
                    logger.error(f"[N8N] [{correlation_id}] Network error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

                except Exception as e:
                    last_exception = e
                    logger.exception(f"[N8N] [{correlation_id}] Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

            # All retries failed
            logger.error(f"[N8N] [{correlation_id}] All {self.max_retries} attempts failed. Last error: {type(last_exception).__name__}: {last_exception}")
            return {
                "success": False,
                "error": f"Failed after {self.max_retries} attempts. Last error: {str(last_exception)}",
                "correlation_id": correlation_id,
                "attempts": self.max_retries
            }

    async def send_message(self, phone_number: str, message: str) -> Dict[str, Any]:
        """
        Alias for send_to_n8n for backward compatibility.
        """
        return await self.send_to_n8n(phone_number, message)

    def is_configured(self) -> bool:
        """
        Check if N8N outbound is properly configured.
        """
        return bool(self.webhook_url)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the N8N webhook.
        """
        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            logger.info(f"[N8N] [{correlation_id}] Starting health check")

            if not self.webhook_url:
                return {
                    "healthy": False,
                    "error": "N8N_OUTBOUND_WEBHOOK_URL not configured",
                    "correlation_id": correlation_id
                }

            try:
                # Send a simple health check payload
                health_payload = {
                    "phone_number": "health_check",
                    "message": "health_check",
                    "correlation_id": correlation_id,
                    "is_health_check": True
                }

                headers = {"Content-Type": "application/json"}
                if self.token:
                    headers["X-Auth-Token"] = self.token

                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        self.webhook_url,
                        json=health_payload,
                        headers=headers,
                    )

                    if response.status_code < 500:  # Accept 4xx as healthy (client errors)
                        logger.info(f"[N8N] [{correlation_id}] Health check passed with status {response.status_code}")
                        return {
                            "healthy": True,
                            "status_code": response.status_code,
                            "correlation_id": correlation_id
                        }
                    else:
                        logger.error(f"[N8N] [{correlation_id}] Health check failed with status {response.status_code}")
                        return {
                            "healthy": False,
                            "status_code": response.status_code,
                            "error": response.text,
                            "correlation_id": correlation_id
                        }

            except Exception as e:
                logger.exception(f"[N8N] [{correlation_id}] Health check failed with exception: {type(e).__name__}: {e}")
                return {
                    "healthy": False,
                    "error": str(e),
                    "correlation_id": correlation_id
                }


# Global N8N outbound client instance
n8n_client = N8NOutboundClient()