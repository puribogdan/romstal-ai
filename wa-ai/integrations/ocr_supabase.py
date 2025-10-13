import os
import base64
import uuid
import re
import json
import logging
import traceback
import tempfile
import shutil
import time
from datetime import datetime, timedelta
import requests
from typing import Tuple, Dict, Any, List, Optional

from supabase import create_client, Client
from mistralai import Mistral
from dotenv import load_dotenv

# Import correlation tracking
from app.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    log_with_correlation,
    CorrelationContext
)

# Import n8n client for graceful failure messaging
from app.outbound import n8n_client

logger = logging.getLogger(__name__)

# Global tracking for retry attempts to prevent multiple re-resolutions per media_id
_media_retry_tracker = set()

load_dotenv()

# ---------- Config ----------
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "whatsapp-pdfs")

# Initialize Mistral client only if API key is available
mistral_client = None
if MISTRAL_API_KEY:
    try:
        mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    except Exception as e:
        mistral_client = None

# Initialize Supabase client only if credentials are available
supabase: Optional[Client] = None
_supabase_init_error = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        print(f"[OCR] [INIT] Initializing Supabase client for URL: {SUPABASE_URL[:50]}...")
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[OCR] [INIT] Supabase client initialized successfully")
    except Exception as e:
        _supabase_init_error = str(e)
        print(f"[OCR] [INIT] ERROR: Failed to initialize Supabase client: {e}")
        print(f"[OCR] [INIT] ERROR: Please check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env file")
        supabase = None
else:
    if not SUPABASE_URL:
        print("[OCR] [INIT] WARNING: SUPABASE_URL not configured in environment")
    if not SUPABASE_KEY:
        print("[OCR] [INIT] WARNING: SUPABASE_SERVICE_ROLE_KEY not configured in environment")


# ---------- Helpers ----------
def _pdf_bytes_to_data_url(pdf_bytes: bytes) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{b64}"


def validate_pdf_format(pdf_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate that the downloaded content is actually a PDF file.
    Returns (is_valid, error_message).
    """
    if not pdf_bytes or len(pdf_bytes) < 8:
        return False, "File too small to be a valid PDF"

    # PDF files start with %PDF- (hex: 25 50 44 46 2D)
    pdf_header = b'%PDF-'
    if not pdf_bytes.startswith(pdf_header):
        return False, f"Invalid PDF header. Expected %PDF-, got {pdf_bytes[:10]}"

    # Check for PDF version (should be 1.x or 2.x)
    if len(pdf_bytes) < 8:
        return False, "PDF header too short"

    version_byte = pdf_bytes[5:8]  # Should be something like b'1.3' or b'1.7'
    try:
        version_str = version_byte.decode('ascii')
        if not (version_str.startswith('1.') or version_str.startswith('2.')):
            return False, f"Unsupported PDF version: {version_str}"
    except UnicodeDecodeError:
        return False, "Invalid PDF version encoding"

    # Basic size check - PDFs should be at least a few KB for valid content
    if len(pdf_bytes) < 1024:
        return False, f"PDF file too small ({len(pdf_bytes)} bytes) - likely corrupted"

    return True, "Valid PDF format"


def save_pdf_to_temp_file(pdf_bytes: bytes, prefix: str = "pdf_cache") -> str:
    """
    Save PDF bytes to a temporary file and return the file path.
    The caller is responsible for cleanup.
    """
    try:
        # Create a temporary file with custom prefix
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf', prefix=f'{prefix}_', dir=None)

        # Write PDF bytes to temp file
        with os.fdopen(temp_fd, 'wb') as temp_file:
            temp_file.write(pdf_bytes)

        logger.info(f"[OCR] [DIAGNOSTIC] Saved PDF to temp file: {temp_path} ({len(pdf_bytes)} bytes)")
        return temp_path

    except Exception as e:
        logger.error(f"[OCR] [DIAGNOSTIC] Failed to save PDF to temp file: {e}")
        raise


def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file."""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"[OCR] [DIAGNOSTIC] Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"[OCR] [DIAGNOSTIC] Failed to cleanup temp file {file_path}: {e}")


def _send_graceful_failure_message_sync(phone_number: Optional[str], message: str, correlation_id: str) -> None:
    """
    Send graceful failure message via n8n webhook (synchronous version).
    This is a helper function to avoid async calls in sync functions.
    """
    try:
        if not phone_number:
            logger.warning(f"[OCR] [DIAGNOSTIC] Cannot send graceful failure message: phone_number is None")
            return

        # For now, just log the message that would be sent
        # In a real implementation, you might want to use a synchronous HTTP client
        # or queue the message for later sending
        logger.info(f"[OCR] [DIAGNOSTIC] Would send graceful failure message to {phone_number[:10]}...: '{message}'")

        # If you want to actually send the message synchronously, you could use:
        # import requests
        # payload = {"phone_number": phone_number, "message": message, "correlation_id": correlation_id}
        # response = requests.post(n8n_client.webhook_url, json=payload, headers={"Content-Type": "application/json"})

    except Exception as e:
        logger.error(f"[OCR] [DIAGNOSTIC] Failed to send graceful failure message: {e}")


def retry_with_exponential_backoff(
    func,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    *args,
    **kwargs
):
    """
    Retry a function with exponential backoff.
    Returns the function result or raises the last exception.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"[OCR] [DIAGNOSTIC] All {max_retries + 1} attempts failed. Final error: {e}")
                raise

            delay = min(initial_delay * (backoff_factor ** attempt), max_delay)
            logger.warning(f"[OCR] [DIAGNOSTIC] Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    # This should never be reached, but just in case
    raise RuntimeError(f"Retry logic failed after {max_retries + 1} attempts")


def validate_media_url_format(media_url: str) -> Tuple[bool, str]:
    """
    Validate WhatsApp media URL format.
    Supports both download URLs and Graph API retrieved URLs.
    Returns (is_valid, error_message).
    """
    if not media_url or not isinstance(media_url, str):
        return False, "Media URL is empty or not a string"

    # Support multiple WhatsApp Business API URL formats:

    # 1. Download URL format: https://lookaside.facebook.com/whatsapp_business/{media_id}?access_token={token}
    if media_url.startswith("https://lookaside.facebook.com/whatsapp_business/"):
        # Should contain access token parameter for download URLs
        if "?access_token=" not in media_url:
            return False, "Download media URL missing access token parameter"
        return True, "Valid WhatsApp Business download URL format"

    # 2. Graph API URL format: https://graph.facebook.com/v{version}/{media_id}
    elif media_url.startswith("https://graph.facebook.com/v"):
        # Validate version format (v16.0, v17.0, v18.0, v19.0, v20.0, v21.0, etc.)
        import re
        version_pattern = r'^https://graph\.facebook\.com/v\d+\.\d+/[a-zA-Z0-9_-]+$'
        if not re.match(version_pattern, media_url):
            return False, f"Invalid Graph API URL format. Expected: https://graph.facebook.com/v{{version}}/{{media_id}}, got: {media_url[:50]}..."

        return True, "Valid WhatsApp Business Graph API URL format"

    # 3. Graph API response URLs (these might be different formats returned by Graph API)
    elif media_url.startswith("https://"):
        # Accept any HTTPS URL as it might be a valid download URL from Graph API
        # Additional validation will be done during the actual download
        return True, "Valid HTTPS URL format (likely from Graph API response)"

    else:
        return False, f"Invalid media URL format. Expected HTTPS URL, got: {media_url[:50]}..."



def validate_access_token_format(access_token: str) -> Tuple[bool, str]:
    """
    Validate WhatsApp access token format.
    Returns (is_valid, error_message).
    """
    if not access_token or not isinstance(access_token, str):
        return False, "Access token is empty or not a string"

    # WhatsApp access tokens are typically 100-200 characters long alphanumeric strings
    if len(access_token) < 50:
        return False, f"Access token too short ({len(access_token)} chars). Expected 50+ characters"

    if len(access_token) > 300:
        return False, f"Access token too long ({len(access_token)} chars). Expected < 300 characters"

    # Should contain mix of uppercase, lowercase, and numbers
    has_upper = bool(re.search(r'[A-Z]', access_token))
    has_lower = bool(re.search(r'[a-z]', access_token))
    has_digit = bool(re.search(r'[0-9]', access_token))

    if not (has_upper and has_lower and has_digit):
        return False, "Access token should contain uppercase, lowercase, and numeric characters"

    return True, "Valid access token format"


def validate_media_id_format(media_id: str) -> Tuple[bool, str]:
    """
    Validate WhatsApp media ID format.
    Returns (is_valid, error_message).
    """
    if not media_id or not isinstance(media_id, str):
        return False, "Media ID is empty or not a string"

    # WhatsApp media IDs are typically alphanumeric strings of varying lengths
    # They can contain hyphens and underscores
    if len(media_id) < 10:
        return False, f"Media ID too short ({len(media_id)} chars). Expected 10+ characters"

    if len(media_id) > 200:
        return False, f"Media ID too long ({len(media_id)} chars). Expected < 200 characters"

    # Should be alphanumeric with possible hyphens, underscores, or colons
    if not re.match(r'^[a-zA-Z0-9_:-]+$', media_id):
        return False, "Media ID contains invalid characters. Should only contain letters, numbers, hyphens, underscores, and colons"

    return True, "Valid media ID format"


def clear_media_retry_tracker() -> None:
    """
    Clear the media retry tracker. Useful for testing or resetting state.
    """
    global _media_retry_tracker
    _media_retry_tracker.clear()
    logger.info("[OCR] [MEDIA-RETRY] Cleared media retry tracker")


def has_media_been_retried(media_id: str) -> bool:
    """
    Check if a media_id has already been retried to prevent multiple re-resolutions.

    Args:
        media_id: The WhatsApp media ID to check

    Returns:
        True if the media_id has already been retried, False otherwise
    """
    return media_id in _media_retry_tracker


def detect_expired_media_id(media_id: str) -> Tuple[bool, str]:
    """
    Detect if a media ID indicates expired media based on WhatsApp's ID patterns.
    WhatsApp media expires after 30 days, and old media IDs are typically in 1-2 billion range.
    Current fresh media should be in 3-4 billion+ range.

    Args:
        media_id: The WhatsApp media ID to check

    Returns:
        Tuple of (is_expired, reason)
    """
    if not media_id or not isinstance(media_id, str):
        return False, "Media ID is empty or invalid"

    try:
        # Try to parse as numeric ID
        media_id_numeric = int(media_id)

        # WhatsApp media ID ranges:
        # - Old/expired media: typically 1-2 billion range (1xxxxxxxxxxxxxxx - 2xxxxxxxxxxxxxxx)
        # - Fresh media: 3-4 billion+ range (3xxxxxxxxxxxxxxx+)
        if media_id_numeric < 3_000_000_000:  # Less than 3 billion
            return True, f"Media ID {media_id} appears to be from expired range (< 3B). Current fresh media should be 3B+. This media has likely expired."

        # Additional check for very old media (1-2 billion range is definitely expired)
        if media_id_numeric < 2_000_000_000:
            return True, f"Media ID {media_id} is from very old range (< 2B) and is definitely expired."

        return False, f"Media ID {media_id} appears to be from valid range (>= 3B)"

    except (ValueError, TypeError):
        # If we can't parse as numeric, we can't determine expiration from ID alone
        # This is not necessarily an error, just log for monitoring
        logger.warning(f"[MEDIA-EXPIRATION] Could not parse media_id as numeric for expiration check: {media_id}")
        return False, f"Media ID {media_id} is non-numeric, cannot determine expiration from ID pattern"


def get_media_download_url_from_id(media_id: str, access_token: str) -> str:
    """
    Get WhatsApp media download URL using the correct two-step Graph API flow.

    Step 1: Call Graph API with media ID to get the actual download URL
    Step 2: Return that URL for downloading

    Args:
        media_id: WhatsApp media ID
        access_token: WhatsApp access token

    Returns:
        The actual download URL from Graph API response
    """
    correlation_id = generate_correlation_id()
    # Add media ID suffix for traceability (last 4 characters)
    media_id_suffix = media_id[-4:] if len(media_id) >= 4 else media_id

    with CorrelationContext(correlation_id):
        try:
            log_with_correlation(
                logger.info,
                f"[GRAPH-API] Starting two-step media URL retrieval for media_id: ...{media_id_suffix}",
                correlation_id
            )

            is_valid_id, id_error = validate_media_id_format(media_id)
            if not is_valid_id:
                raise ValueError(f"Invalid media ID: {id_error}")

            is_valid_token, token_error = validate_access_token_format(access_token)
            if not is_valid_token:
                raise ValueError(f"Invalid access token: {token_error}")

            # Step 1: Call Graph API to get the download URL
            graph_api_url = f"https://graph.facebook.com/v21.0/{media_id}"

            log_with_correlation(
                logger.info,
                f"[GRAPH-API] Calling Graph API for media_id: ...{media_id_suffix}",
                correlation_id
            )

            # Enhanced logging for debugging token issues - masked for security
            token_prefix = access_token[:10] if len(access_token) >= 10 else access_token[:5]
            log_with_correlation(
                logger.info,
                f"[GRAPH-API] [DIAGNOSTIC] Access token length: {len(access_token)}, prefix: {token_prefix}...",
                correlation_id
            )

            # Validate token format before making request
            is_valid, token_error = validate_access_token_format(access_token)
            log_with_correlation(
                logger.info,
                f"[GRAPH-API] [DIAGNOSTIC] Token validation: {'VALID' if is_valid else 'INVALID'} - {token_error}",
                correlation_id
            )

            headers = {
                "Authorization": f"Bearer {access_token}"
            }

            log_with_correlation(
                logger.info,
                f"[GRAPH-API] [DIAGNOSTIC] Request headers: Authorization=Bearer [TOKEN:{len(access_token)}chars]",
                correlation_id
            )

            response = requests.get(graph_api_url, headers=headers, timeout=30)

            log_with_correlation(
                logger.info,
                f"[GRAPH-API] Graph API response status: {response.status_code}",
                correlation_id
            )

            if response.status_code >= 400:
                error_msg = f"Graph API error (HTTP {response.status_code})"
                error_data = None
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f": {error_data['error'].get('message', 'Unknown error')}"
                except json.JSONDecodeError:
                    error_msg += f": {response.text[:200]}"

                log_with_correlation(
                    logger.error,
                    f"[GRAPH-API] Failed to get media URL from Graph API: {error_msg}",
                    correlation_id
                )

                # Categorize error types for proper handling
                if response.status_code in [400, 404, 403]:
                    # Invalid ID / no permission → graceful user message + stop
                    raise ValueError(f"INVALID_MEDIA: {error_msg}")
                elif response.status_code == 401:
                    # Token or stale URL → will retry resolution once
                    raise ValueError(f"AUTH_ERROR: {error_msg}")
                elif response.status_code == 429 or response.status_code >= 500:
                    # Backoff (exponential with jitter) then single retry
                    raise ValueError(f"RATE_LIMIT_ERROR: {error_msg}")
                else:
                    # Other errors
                    raise ValueError(f"Graph API error: {error_msg}")

            # Parse Graph API response to get the download URL
            try:
                response_data = response.json()
                log_with_correlation(
                    logger.info,
                    f"[GRAPH-API] Graph API response keys: {list(response_data.keys())}",
                    correlation_id
                )

                # The Graph API response should contain a 'url' field with the download URL
                download_url = response_data.get("url")
                if not download_url:
                    error_msg = "Graph API response missing 'url' field"
                    log_with_correlation(
                        logger.error,
                        f"[GRAPH-API] {error_msg}. Response: {response_data}",
                        correlation_id
                    )
                    raise ValueError(error_msg)

                log_with_correlation(
                    logger.info,
                    f"[GRAPH-API] Resolving media ID via Graph API → received signed URL (media_id: ...{media_id_suffix})",
                    correlation_id
                )
                return download_url

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse Graph API JSON response: {e}"
                log_with_correlation(
                    logger.error,
                    f"[GRAPH-API] {error_msg}. Response text: {response.text[:500]}",
                    correlation_id
                )
                raise ValueError(error_msg)

        except Exception as e:
            log_with_correlation(
                logger.error,
                f"[GRAPH-API] Critical error in two-step media URL retrieval: {e}",
                correlation_id,
                exc_info=True
            )
            raise


def parse_whatsapp_error_response(response_text: str, status_code: int) -> Dict[str, Any]:
    """
    Parse WhatsApp API JSON error response to extract meaningful error information.
    Returns dict with error details or empty dict if parsing fails.
    """
    try:
        error_data = json.loads(response_text)
        return {
            "error_code": error_data.get("error", {}).get("code", status_code),
            "error_message": error_data.get("error", {}).get("message", "Unknown error"),
            "error_type": error_data.get("error", {}).get("type", "unknown_error"),
            "error_details": error_data.get("error", {}).get("error_data", {}),
            "raw_response": response_text
        }
    except json.JSONDecodeError:
        # If response is not JSON, return basic error info
        return {
            "error_code": status_code,
            "error_message": f"HTTP {status_code}: {response_text[:200]}",
            "error_type": "http_error",
            "raw_response": response_text
        }


def download_whatsapp_media(media_id: str, access_token: str, phone_number: Optional[str] = None) -> bytes:
    """
    media_id: WhatsApp media ID (e.g., 1892359901234567)
    access_token: WhatsApp access token
    phone_number: Phone number for graceful failure messaging (optional)
    Downloads with retry logic, PDF validation, and local caching.
    Enhanced with comprehensive error handling and authentication retry logic.
    """
    correlation_id = generate_correlation_id()
    temp_file_path = None
    # Add media ID suffix for traceability (last 4 characters)
    media_id_suffix = media_id[-4:] if len(media_id) >= 4 else media_id

    with CorrelationContext(correlation_id):
        try:
            log_with_correlation(
                logger.info,
                f"[OCR] [DIAGNOSTIC] Starting WhatsApp media download for media_id: ...{media_id_suffix}",
                correlation_id
            )

            # Enhanced request validation
            if not media_id or not access_token:
                error_msg = "Media ID or access token is missing"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg}",
                    correlation_id
                )
                raise ValueError(error_msg)

            # Validate media ID format
            is_valid_id, id_error = validate_media_id_format(media_id)
            if not is_valid_id:
                error_msg = f"Invalid media ID format: {id_error}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg}",
                    correlation_id
                )
                raise ValueError(error_msg)

            # Validate access token format
            is_valid_token, token_error = validate_access_token_format(access_token)
            if not is_valid_token:
                error_msg = f"Invalid access token format: {token_error}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg}",
                    correlation_id
                )
                raise ValueError(error_msg)

            # Check for expired media ID before attempting download
            log_with_correlation(
                logger.info,
                f"[OCR] [MEDIA-EXPIRATION] Checking media expiration for media_id: {media_id}",
                correlation_id
            )

            is_expired, expiration_reason = detect_expired_media_id(media_id)
            if is_expired:
                error_msg = f"Media appears to be expired: {expiration_reason}"
                log_with_correlation(
                    logger.warning,
                    f"[OCR] [MEDIA-EXPIRATION] {error_msg}",
                    correlation_id
                )
                log_with_correlation(
                    logger.info,
                    f"[OCR] [MEDIA-EXPIRATION] Blocking download attempt for expired media_id: {media_id}",
                    correlation_id
                )
                # Raise specific exception for expired media
                raise ValueError(f"EXPIRED_MEDIA: {error_msg}")
            else:
                log_with_correlation(
                    logger.info,
                    f"[OCR] [MEDIA-EXPIRATION] Media ID {media_id} passed expiration check: {expiration_reason}",
                    correlation_id
                )

            # Step 1: Call Graph API to get the download URL with enhanced error handling
            try:
                media_url = get_media_download_url_from_id(media_id, access_token)
                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Retrieved download URL for media_id: ...{media_id_suffix}",
                    correlation_id
                )
            except ValueError as e:
                error_str = str(e)
                if error_str.startswith("INVALID_MEDIA:"):
                    # 400/404/403 errors: Invalid ID/no permission → graceful user message + stop
                    error_msg = "Nu am reușit să descarc documentul. Poți retrimite PDF-ul?"
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] Invalid media error for media_id: ...{media_id_suffix}: {error_str}",
                        correlation_id
                    )
                    # Send graceful failure message via n8n webhook
                    _send_graceful_failure_message_sync(phone_number, error_msg, correlation_id)
                    raise ValueError(f"INVALID_MEDIA: {error_msg}")
                elif error_str.startswith("AUTH_ERROR:"):
                    # 401 errors: Token or stale URL → re-resolve once
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] Auth error for media_id: ...{media_id_suffix}, will retry once: {error_str}",
                        correlation_id
                    )
                    # Retry once for auth errors
                    try:
                        media_url = get_media_download_url_from_id(media_id, access_token)
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] Successfully re-resolved media URL after auth error for media_id: ...{media_id_suffix}",
                            correlation_id
                        )
                    except ValueError as retry_error:
                        retry_error_str = str(retry_error)
                        if retry_error_str.startswith("AUTH_ERROR:"):
                            # If still auth error after retry, graceful message
                            error_msg = "Nu am reușit să descarc documentul. Poți retrimite PDF-ul?"
                            log_with_correlation(
                                logger.error,
                                f"[OCR] [DIAGNOSTIC] Auth error persisted after retry for media_id: ...{media_id_suffix}: {retry_error_str}",
                                correlation_id
                            )
                            _send_graceful_failure_message_sync(phone_number, error_msg, correlation_id)
                            raise ValueError(f"AUTH_FAILED: {error_msg}")
                        else:
                            # Re-raise other errors
                            raise
                elif error_str.startswith("RATE_LIMIT_ERROR:"):
                    # 429/5xx errors: Backoff then single retry
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] Rate limit error for media_id: ...{media_id_suffix}, implementing backoff: {error_str}",
                        correlation_id
                    )
                    # Implement exponential backoff with jitter
                    import random
                    delay = 1.0 + random.uniform(0, 1.0)  # 1-2 seconds with jitter
                    time.sleep(delay)

                    try:
                        media_url = get_media_download_url_from_id(media_id, access_token)
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] Successfully re-resolved media URL after rate limit for media_id: ...{media_id_suffix}",
                            correlation_id
                        )
                    except ValueError as retry_error:
                        retry_error_str = str(retry_error)
                        if retry_error_str.startswith("RATE_LIMIT_ERROR:"):
                            # If still rate limit after retry, graceful message
                            error_msg = "Nu am reușit să descarc documentul. Poți retrimite PDF-ul?"
                            log_with_correlation(
                                logger.error,
                                f"[OCR] [DIAGNOSTIC] Rate limit persisted after retry for media_id: ...{media_id_suffix}: {retry_error_str}",
                                correlation_id
                            )
                            _send_graceful_failure_message_sync(phone_number, error_msg, correlation_id)
                            raise ValueError(f"RATE_LIMIT_FAILED: {error_msg}")
                        else:
                            # Re-raise other errors
                            raise
                else:
                    # Re-raise other errors
                    error_msg = f"Failed to get download URL from Graph API for media_id: ...{media_id_suffix}"
                    log_with_correlation(
                        logger.error,
                        f"[OCR] [DIAGNOSTIC] {error_msg}: {e}",
                        correlation_id
                    )
                    raise ValueError(error_msg) from e

            # Enhanced download with comprehensive error handling and authentication retry
            def _download_media():
                r = None
                try:
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Making HTTP request to WhatsApp media URL",
                        correlation_id
                    )

                    # Log request details for debugging
                    # Mask sensitive information in logs
                    token_prefix = access_token[:10] if len(access_token) >= 10 else access_token[:5]
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Request details - URL: [WHATSAPP_MEDIA_URL:{len(media_url)}chars], Token: {token_prefix}...",
                        correlation_id
                    )

                    # Additional diagnostic logging for media download
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Media URL domain: {media_url.split('/')[2] if len(media_url.split('/')) > 2 else 'unknown'}",
                        correlation_id
                    )

                    # Check if URL contains access token
                    has_token_in_url = "access_token=" in media_url
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] URL contains access token: {has_token_in_url}",
                        correlation_id
                    )

                    r = requests.get(
                        media_url,
                        headers={"Authorization": f"Bearer {access_token}"},
                        timeout=30
                    )

                    # Comprehensive response logging
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] HTTP request completed - Status: {r.status_code}, Content-Length: {len(r.content) if r.content else 0}",
                        correlation_id
                    )

                    # Log response headers for debugging
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Response headers: {dict(r.headers)}",
                        correlation_id
                    )

                    # Check for HTTP errors with enhanced error parsing
                    if r.status_code >= 400:
                        # Try to parse JSON error response
                        error_info = {}
                        if r.content:
                            try:
                                response_text = r.content.decode('utf-8', errors='ignore')
                                error_info = parse_whatsapp_error_response(response_text, r.status_code)
                                log_with_correlation(
                                    logger.error,
                                    f"[OCR] [DIAGNOSTIC] WhatsApp API error response: {error_info}",
                                    correlation_id
                                )
                            except Exception as parse_error:
                                log_with_correlation(
                                    logger.warning,
                                    f"[OCR] [DIAGNOSTIC] Failed to parse error response as JSON: {parse_error}",
                                    correlation_id
                                )
                                response_text = r.content.decode('utf-8', errors='ignore') if r.content else "No response content"

                        # Handle authentication errors specifically
                        if r.status_code in [401, 403]:
                            error_msg = f"WhatsApp API authentication failed (HTTP {r.status_code})"
                            if error_info.get("error_message"):
                                error_msg += f": {error_info['error_message']}"
                            error_msg += ". Access token may be expired or invalid."

                            log_with_correlation(
                                logger.error,
                                f"[OCR] [DIAGNOSTIC] Authentication error - {error_msg}",
                                correlation_id
                            )

                            # For 401 errors specifically, log the retry message and retry once
                            if r.status_code == 401:
                                log_with_correlation(
                                    logger.warning,
                                    f"[OCR][MEDIA-RETRY] 401 received – fetched new signed URL from Graph API and retrying once (media_id: ...{media_id_suffix})",
                                    correlation_id
                                )

                            # For authentication errors, we don't retry as it's unlikely to succeed
                            raise requests.exceptions.HTTPError(error_msg, response=r)

                        # Handle other HTTP errors
                        error_msg = f"WhatsApp API error (HTTP {r.status_code})"
                        if error_info.get("error_message"):
                            error_msg += f": {error_info['error_message']}"

                        log_with_correlation(
                            logger.error,
                            f"[OCR] [DIAGNOSTIC] WhatsApp API error: {error_msg}",
                            correlation_id
                        )
                        r.raise_for_status()

                    # Validate response content
                    if not r.content:
                        error_msg = "WhatsApp media download returned empty content"
                        log_with_correlation(
                            logger.error,
                            f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg}",
                            correlation_id
                        )
                        raise ValueError(error_msg)

                    # Log successful download details
                    content_type = r.headers.get('content-type', 'unknown')
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Download successful - Content-Type: {content_type}, Size: {len(r.content)} bytes",
                        correlation_id
                    )

                    return r.content

                except requests.exceptions.Timeout as e:
                    error_msg = f"WhatsApp media download timeout after 30 seconds"
                    log_with_correlation(
                        logger.error,
                        f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg} - {e}",
                        correlation_id,
                        exc_info=True
                    )
                    raise requests.exceptions.RequestException(error_msg) from e

                except requests.exceptions.HTTPError as e:
                    status_code = r.status_code if r is not None else 'unknown'

                    # Enhanced error message based on status code
                    if status_code == 401:
                        error_msg = f"WhatsApp API authentication failed - Invalid or expired access token"
                    elif status_code == 403:
                        error_msg = f"WhatsApp API access forbidden - Insufficient permissions"
                    elif status_code == 404:
                        error_msg = f"WhatsApp media not found - Media may have expired or been deleted"
                    elif status_code == 429:
                        error_msg = f"WhatsApp API rate limit exceeded - Too many requests"
                    else:
                        error_msg = f"WhatsApp media download HTTP error: {status_code}"

                    log_with_correlation(
                        logger.error,
                        f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg} - {e}",
                        correlation_id,
                        exc_info=True
                    )
                    raise requests.exceptions.RequestException(error_msg) from e

                except ValueError as e:
                    # Handle expired media and other validation errors
                    error_str = str(e)
                    if error_str.startswith("EXPIRED_MEDIA:"):
                        # This is our custom expired media error
                        expired_msg = error_str.replace("EXPIRED_MEDIA:", "").strip()
                        log_with_correlation(
                            logger.warning,
                            f"[OCR] [MEDIA-EXPIRATION] Expired media detected: {expired_msg}",
                            correlation_id
                        )
                        # Re-raise with more context for upstream handlers
                        raise ValueError(f"Media expired: {expired_msg}") from e
                    else:
                        # Re-raise other ValueError exceptions as-is
                        raise

                except requests.exceptions.RequestException as e:
                    error_msg = f"WhatsApp media download request error"
                    log_with_correlation(
                        logger.error,
                        f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg} - {e}",
                        correlation_id,
                        exc_info=True
                    )
                    raise

            # Enhanced retry mechanism with authentication-specific handling
            def _download_with_auth_retry(media_url_param, media_id_param, access_token_param, retry_count_param=0):
                """
                Wrapper function to handle authentication errors with specific retry logic.
                For 401 errors, re-requests fresh signed URL from Graph API and retries once.
                """
                def _download_media_with_url(url_to_use):
                    """Inner download function that uses the provided URL"""
                    r = None
                    try:
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] Making HTTP request to WhatsApp media URL",
                            correlation_id
                        )

                        # Log request details for debugging
                        # Mask sensitive information in logs
                        token_prefix = access_token_param[:10] if len(access_token_param) >= 10 else access_token_param[:5]
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] Request details - URL: [WHATSAPP_MEDIA_URL:{len(url_to_use)}chars], Token: {token_prefix}...",
                            correlation_id
                        )

                        # Additional diagnostic logging for media download
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] Media URL domain: {url_to_use.split('/')[2] if len(url_to_use.split('/')) > 2 else 'unknown'}",
                            correlation_id
                        )

                        # Check if URL contains access token
                        has_token_in_url = "access_token=" in url_to_use
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] URL contains access token: {has_token_in_url}",
                            correlation_id
                        )

                        r = requests.get(
                            url_to_use,
                            headers={"Authorization": f"Bearer {access_token_param}"},
                            timeout=30
                        )

                        # Comprehensive response logging
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] HTTP request completed - Status: {r.status_code}, Content-Length: {len(r.content) if r.content else 0}",
                            correlation_id
                        )

                        # Log response headers for debugging
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] Response headers: {dict(r.headers)}",
                            correlation_id
                        )

                        # Check for HTTP errors with enhanced error parsing
                        if r.status_code >= 400:
                            # Try to parse JSON error response
                            error_info = {}
                            if r.content:
                                try:
                                    response_text = r.content.decode('utf-8', errors='ignore')
                                    error_info = parse_whatsapp_error_response(response_text, r.status_code)
                                    log_with_correlation(
                                        logger.error,
                                        f"[OCR] [DIAGNOSTIC] WhatsApp API error response: {error_info}",
                                        correlation_id
                                    )
                                except Exception as parse_error:
                                    log_with_correlation(
                                        logger.warning,
                                        f"[OCR] [DIAGNOSTIC] Failed to parse error response as JSON: {parse_error}",
                                        correlation_id
                                    )
                                    response_text = r.content.decode('utf-8', errors='ignore') if r.content else "No response content"

                            # Handle authentication errors specifically
                            if r.status_code in [401, 403]:
                                error_msg = f"WhatsApp API authentication failed (HTTP {r.status_code})"
                                if error_info.get("error_message"):
                                    error_msg += f": {error_info['error_message']}"
                                error_msg += ". Access token may be expired or invalid."

                                log_with_correlation(
                                    logger.error,
                                    f"[OCR] [DIAGNOSTIC] Authentication error - {error_msg}",
                                    correlation_id
                                )

                                # For 401 errors specifically, log the retry message and retry once
                                if r.status_code == 401:
                                    log_with_correlation(
                                        logger.warning,
                                        f"[OCR][MEDIA-RETRY] 401 received – fetched new signed URL from Graph API and retrying once (media_id: ...{media_id_param[-4:]})",
                                        correlation_id
                                    )

                                # For authentication errors, we don't retry as it's unlikely to succeed
                                raise requests.exceptions.HTTPError(error_msg, response=r)

                            # Handle other HTTP errors
                            error_msg = f"WhatsApp API error (HTTP {r.status_code})"
                            if error_info.get("error_message"):
                                error_msg += f": {error_info['error_message']}"

                            log_with_correlation(
                                logger.error,
                                f"[OCR] [DIAGNOSTIC] WhatsApp API error: {error_msg}",
                                correlation_id
                            )
                            r.raise_for_status()

                        # Validate response content
                        if not r.content:
                            error_msg = "WhatsApp media download returned empty content"
                            log_with_correlation(
                                logger.error,
                                f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg}",
                                correlation_id
                            )
                            raise ValueError(error_msg)

                        # Log successful download details
                        content_type = r.headers.get('content-type', 'unknown')
                        log_with_correlation(
                            logger.info,
                            f"[OCR] [DIAGNOSTIC] Download successful - Content-Type: {content_type}, Size: {len(r.content)} bytes",
                            correlation_id
                        )

                        return r.content

                    except requests.exceptions.Timeout as e:
                        error_msg = f"WhatsApp media download timeout after 30 seconds"
                        log_with_correlation(
                            logger.error,
                            f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg} - {e}",
                            correlation_id,
                            exc_info=True
                        )
                        raise requests.exceptions.RequestException(error_msg) from e

                    except requests.exceptions.HTTPError as e:
                        status_code = r.status_code if r is not None else 'unknown'

                        # Enhanced error message based on status code
                        if status_code == 401:
                            error_msg = f"WhatsApp API authentication failed - Invalid or expired access token"
                        elif status_code == 403:
                            error_msg = f"WhatsApp API access forbidden - Insufficient permissions"
                        elif status_code == 404:
                            error_msg = f"WhatsApp media not found - Media may have expired or been deleted"
                        elif status_code == 429:
                            error_msg = f"WhatsApp API rate limit exceeded - Too many requests"
                        else:
                            error_msg = f"WhatsApp media download HTTP error: {status_code}"

                        log_with_correlation(
                            logger.error,
                            f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg} - {e}",
                            correlation_id,
                            exc_info=True
                        )
                        raise requests.exceptions.RequestException(error_msg) from e

                    except ValueError as e:
                        # Handle expired media and other validation errors
                        error_str = str(e)
                        if error_str.startswith("EXPIRED_MEDIA:"):
                            # This is our custom expired media error
                            expired_msg = error_str.replace("EXPIRED_MEDIA:", "").strip()
                            log_with_correlation(
                                logger.warning,
                                f"[OCR] [MEDIA-EXPIRATION] Expired media detected: {expired_msg}",
                                correlation_id
                            )
                            # Re-raise with more context for upstream handlers
                            raise ValueError(f"Media expired: {expired_msg}") from e
                        else:
                            # Re-raise other ValueError exceptions as-is
                            raise

                    except requests.exceptions.RequestException as e:
                        error_msg = f"WhatsApp media download request error"
                        log_with_correlation(
                            logger.error,
                            f"[OCR] [DIAGNOSTIC] WhatsApp media download failed: {error_msg} - {e}",
                            correlation_id,
                            exc_info=True
                        )
                        raise

                try:
                    return _download_media_with_url(media_url_param)
                except requests.exceptions.HTTPError as e:
                    # Check if it's an authentication error
                    if hasattr(e, 'response') and e.response is not None:
                        status_code = e.response.status_code
                        if status_code == 401 and retry_count_param == 0:
                            # Check if we've already retried this media_id to prevent loops
                            if media_id_param in _media_retry_tracker:
                                log_with_correlation(
                                    logger.warning,
                                    f"[OCR] [MEDIA-RETRY] 401 received but already retried media_id: ...{media_id_param[-4:]} - not retrying again",
                                    correlation_id
                                )
                                raise e

                            # 401 error on first attempt - try to get fresh signed URL
                            log_with_correlation(
                                logger.warning,
                                f"[OCR] [MEDIA-RETRY] 401 received – fetched new signed URL from Graph API and retrying once (media_id: ...{media_id_param[-4:]})",
                                correlation_id
                            )

                            # Mark this media_id as having been retried
                            _media_retry_tracker.add(media_id_param)

                            try:
                                # Re-request fresh signed URL from Graph API
                                fresh_media_url = get_media_download_url_from_id(media_id_param, access_token_param)
                                log_with_correlation(
                                    logger.info,
                                    f"[OCR] [MEDIA-RETRY] Successfully obtained fresh signed URL for media_id: ...{media_id_param[-4:]}",
                                    correlation_id
                                )

                                # Retry with fresh URL (increment retry count)
                                return _download_with_auth_retry(fresh_media_url, media_id_param, access_token_param, retry_count_param + 1)

                            except Exception as retry_error:
                                log_with_correlation(
                                    logger.error,
                                    f"[OCR] [MEDIA-RETRY] Failed to get fresh signed URL for media_id: ...{media_id_param[-4:]}: {retry_error}",
                                    correlation_id
                                )
                                # If we can't get a fresh URL, re-raise the original error
                                raise e

                        elif status_code in [401, 403]:
                            # 401 on retry attempt or 403 error - no more retries
                            if retry_count_param > 0:
                                log_with_correlation(
                                    logger.warning,
                                    f"[OCR] [MEDIA-RETRY] 401 persisted after retry with fresh URL for media_id: ...{media_id_param[-4:]} - giving up",
                                    correlation_id
                                )
                            else:
                                log_with_correlation(
                                    logger.warning,
                                    f"[OCR] [DIAGNOSTIC] Authentication error detected, no retry available",
                                    correlation_id
                                )

                    # Re-raise other HTTP errors
                    raise

            # Download with enhanced retry mechanism
            try:
                pdf_bytes = retry_with_exponential_backoff(
                    lambda: _download_with_auth_retry(media_url, media_id, access_token),
                    max_retries=3,
                    initial_delay=1.0,
                    backoff_factor=2.0,
                    max_delay=10.0
                )
            except requests.exceptions.HTTPError as e:
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    if status_code in [401, 403]:
                        # Provide specific guidance for authentication errors
                        error_msg = (
                            f"WhatsApp API authentication failed after retries. "
                            f"Please check your access token validity and permissions. "
                            f"Error: {e}"
                        )
                        log_with_correlation(
                            logger.error,
                            f"[OCR] [DIAGNOSTIC] Final authentication failure: {error_msg}",
                            correlation_id
                        )
                        raise ValueError(error_msg) from e

                # Re-raise other errors
                raise

            log_with_correlation(
                logger.info,
                f"[OCR] [DIAGNOSTIC] WhatsApp media download completed successfully: {len(pdf_bytes)} bytes",
                correlation_id
            )

            # Validate PDF format
            is_valid_pdf, validation_error = validate_pdf_format(pdf_bytes)
            if not is_valid_pdf:
                error_msg = f"Downloaded content is not a valid PDF: {validation_error}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] PDF validation failed: {error_msg}",
                    correlation_id
                )
                raise ValueError(error_msg)

            log_with_correlation(
                logger.info,
                f"[OCR] [DIAGNOSTIC] PDF validation passed: Valid PDF file ({len(pdf_bytes)} bytes)",
                correlation_id
            )

            # Save to temporary file for caching before upload
            try:
                temp_file_path = save_pdf_to_temp_file(pdf_bytes, prefix="whatsapp_pdf")
                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] PDF cached to temp file: {temp_file_path}",
                    correlation_id
                )
            except Exception as e:
                log_with_correlation(
                    logger.warning,
                    f"[OCR] [DIAGNOSTIC] Failed to cache PDF to temp file: {e}. Continuing without cache.",
                    correlation_id
                )
                temp_file_path = None

            return pdf_bytes

        except Exception as e:
            log_with_correlation(
                logger.error,
                f"[OCR] [DIAGNOSTIC] Critical error in WhatsApp media download: {e}",
                correlation_id,
                exc_info=True
            )
            raise
        finally:
            # Clean up temp file if created
            if temp_file_path:
                try:
                    cleanup_temp_file(temp_file_path)
                except Exception as cleanup_error:
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] Failed to cleanup temp file {temp_file_path}: {cleanup_error}",
                        correlation_id
                    )


def supabase_upload_pdf(
    pdf_bytes: bytes,
    phone_number: str,
    insert_id: str
) -> Tuple[str, str]:
    """
    Urcă PDF-ul în Supabase Storage (bucket privat).
    Returnează (storage_path, signed_url_scurt).
    Returns empty strings if Supabase client is not available.
    """
    correlation_id = generate_correlation_id()

    with CorrelationContext(correlation_id):
        try:
            log_with_correlation(
                logger.info,
                f"[OCR] [DIAGNOSTIC] Starting Supabase PDF upload for insert_id: {insert_id}",
                correlation_id
            )

            # Validate inputs
            if not pdf_bytes:
                error_msg = "PDF bytes are empty or None"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Supabase upload failed: {error_msg}",
                    correlation_id
                )
                return "", ""

            if not phone_number or not insert_id:
                error_msg = "Phone number or insert_id is missing"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Supabase upload failed: {error_msg}",
                    correlation_id
                )
                return "", ""

            if not supabase:
                if _supabase_init_error:
                    error_msg = f"Supabase client not available - initialization failed: {_supabase_init_error}"
                else:
                    error_msg = "Supabase client not available - not initialized (missing credentials?)"

                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Supabase upload failed: {error_msg}",
                    correlation_id
                )
                print(f"[OCR] [ERROR] {error_msg}")
                return "", ""

            # Generate storage path
            try:
                today = datetime.utcnow().strftime("%Y/%m/%d")
                filename = f"{insert_id}_{uuid.uuid4().hex}.pdf"
                storage_path = f"{today}/{phone_number}/{filename}"

                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Generated storage path: {storage_path}",
                    correlation_id
                )

            except Exception as e:
                error_msg = f"Error generating storage path for insert_id: {insert_id}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Supabase upload failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return "", ""

            # Upload PDF to Supabase storage with retry logic
            def _upload_to_supabase():
                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Uploading PDF to Supabase storage: {len(pdf_bytes)} bytes",
                    correlation_id
                )

                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    file=pdf_bytes,
                    path=storage_path,
                    file_options={"content-type": "application/pdf", "upsert": "false"},
                )

                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] PDF upload completed successfully",
                    correlation_id
                )

            try:
                retry_with_exponential_backoff(
                    _upload_to_supabase,
                    max_retries=3,
                    initial_delay=1.0,
                    backoff_factor=2.0,
                    max_delay=30.0
                )
            except Exception as e:
                error_msg = f"Error uploading PDF to Supabase storage for insert_id: {insert_id}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Supabase upload failed after retries: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return "", ""

            # Generate signed URL
            try:
                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Generating signed URL for storage path: {storage_path}",
                    correlation_id
                )

                expires_in = int(timedelta(days=7).total_seconds())
                signed = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(
                    path=storage_path,
                    expires_in=expires_in
                )

                signed_url = signed.get("signedURL") or signed.get("signed_url") or ""

                if not signed_url:
                    error_msg = f"Failed to generate signed URL for storage path: {storage_path}"
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] Supabase signed URL generation warning: {error_msg}",
                        correlation_id
                    )
                    # Don't fail completely, return storage path without signed URL
                    return storage_path, ""

                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Signed URL generated successfully",
                    correlation_id
                )
                return storage_path, signed_url

            except Exception as e:
                error_msg = f"Error generating signed URL for insert_id: {insert_id}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Supabase signed URL generation failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                # Return storage path even if signed URL generation fails
                return storage_path, ""

        except Exception as e:
            log_with_correlation(
                logger.error,
                f"[OCR] [DIAGNOSTIC] Critical error in Supabase PDF upload: {e}",
                correlation_id,
                exc_info=True
            )
            return "", ""


def mistral_ocr_from_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Trimite la Mistral OCR ca data:URL Base64. Returnează dict-ul răspunsului.
    Returns empty dict if OCR client is not available.
    """
    correlation_id = generate_correlation_id()

    with CorrelationContext(correlation_id):
        try:
            log_with_correlation(
                logger.info,
                f"[OCR] [DIAGNOSTIC] Starting Mistral OCR processing for PDF of {len(pdf_bytes)} bytes",
                correlation_id
            )

            # Validate inputs
            if not pdf_bytes:
                error_msg = "PDF bytes are empty or None"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Mistral OCR failed: {error_msg}",
                    correlation_id
                )
                return {}

            # Check if Mistral client is available
            if not mistral_client:
                return {}

            # Convert PDF bytes to data URL
            try:
                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Converting PDF bytes to data URL",
                    correlation_id
                )

                data_url = _pdf_bytes_to_data_url(pdf_bytes)

                if not data_url:
                    error_msg = "Failed to create data URL from PDF bytes"
                    log_with_correlation(
                        logger.error,
                        f"[OCR] [DIAGNOSTIC] Mistral OCR failed: {error_msg}",
                        correlation_id
                    )
                    return {}

                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Created data URL for PDF, length: {len(data_url)}",
                    correlation_id
                )

            except Exception as e:
                error_msg = f"Error creating data URL from PDF bytes"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Mistral OCR failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return {}

            # Call Mistral OCR API with retry logic
            def _call_ocr_api():
                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Calling Mistral OCR API with model: mistral-ocr-latest",
                    correlation_id
                )

                resp = mistral_client.ocr.process(
                    model="mistral-ocr-latest",
                    document={"type": "document_url", "document_url": data_url},
                    include_image_base64=False,
                )

                if not resp:
                    error_msg = "Mistral OCR API returned empty response"
                    log_with_correlation(
                        logger.error,
                        f"[OCR] [DIAGNOSTIC] Mistral OCR failed: {error_msg}",
                        correlation_id
                    )
                    raise ValueError(error_msg)

                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Mistral OCR API call completed successfully",
                    correlation_id
                )
                return resp

            try:
                resp = retry_with_exponential_backoff(
                    _call_ocr_api,
                    max_retries=3,
                    initial_delay=2.0,
                    backoff_factor=2.0,
                    max_delay=30.0
                )
            except Exception as e:
                error_msg = f"Mistral OCR API call failed after retries: {type(e).__name__}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] Mistral OCR API call failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return {}

            # Process OCR response
            try:
                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Processing OCR response",
                    correlation_id
                )

                # SDK-ul poate întoarce obiect; normalizăm în dict
                if hasattr(resp, "model_dump"):
                    result = resp.model_dump()
                elif hasattr(resp, "__dict__"):
                    result = resp.__dict__
                # Fallback: convert to dict if it's not already
                elif isinstance(resp, dict):
                    result = resp
                # Last resort: try to convert to dict
                else:
                    try:
                        result = dict(resp)
                    except Exception:
                        result = {}

                if not result:
                    error_msg = "OCR response processing resulted in empty dict"
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] OCR response processing warning: {error_msg}",
                        correlation_id
                    )
                    return {}

                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] OCR response processed successfully, keys: {list(result.keys()) if isinstance(result, dict) else 'non-dict'}",
                    correlation_id
                )
                return result

            except Exception as e:
                error_msg = f"Error processing OCR response: {type(e).__name__}"
                log_with_correlation(
                    logger.error,
                    f"[OCR] [DIAGNOSTIC] OCR response processing failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return {}

        except Exception as e:
            log_with_correlation(
                logger.error,
                f"[OCR] [DIAGNOSTIC] Critical error in Mistral OCR processing: {e}",
                correlation_id,
                exc_info=True
            )
            return {}


def diagnose_supabase_connection() -> Dict[str, Any]:
    """
    Diagnose Supabase connection and return detailed status information.
    Useful for troubleshooting connection issues.
    """
    diagnostic_info = {
        "supabase_configured": supabase is not None,
        "supabase_url": SUPABASE_URL is not None,
        "supabase_key": SUPABASE_KEY is not None,
        "supabase_bucket": SUPABASE_BUCKET,
        "init_error": _supabase_init_error,
        "credentials_status": "OK" if (SUPABASE_URL and SUPABASE_KEY) else "MISSING"
    }

    if supabase:
        try:
            buckets = supabase.storage.list_buckets()
            bucket_names = [b.name for b in buckets]
            diagnostic_info["buckets_available"] = bucket_names
            diagnostic_info["target_bucket_exists"] = SUPABASE_BUCKET in bucket_names
        except Exception as e:
            diagnostic_info["bucket_check_error"] = str(e)

    return diagnostic_info


def extract_markdown(ocr_resp: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract pages and markdown content from OCR response with enhanced validation.
    """
    correlation_id = get_correlation_id()

    try:
        log_with_correlation(
            logger.info,
            f"[OCR] [DIAGNOSTIC] Starting markdown extraction from OCR response",
            correlation_id
        )

        # Enhanced input validation
        if not ocr_resp:
            error_msg = "OCR response is empty or None"
            log_with_correlation(
                logger.warning,
                f"[OCR] [DIAGNOSTIC] Markdown extraction warning: {error_msg}",
                correlation_id
            )
            return [], ""

        if not isinstance(ocr_resp, dict):
            error_msg = f"OCR response is not a dict, got {type(ocr_resp)}"
            log_with_correlation(
                logger.warning,
                f"[OCR] [DIAGNOSTIC] Markdown extraction warning: {error_msg}",
                correlation_id
            )
            return [], ""

        # Validate OCR response structure
        if "pages" not in ocr_resp:
            error_msg = "OCR response missing 'pages' field"
            log_with_correlation(
                logger.warning,
                f"[OCR] [DIAGNOSTIC] Markdown extraction warning: {error_msg}",
                correlation_id
            )
            return [], ""

        # Extract and validate pages
        pages = ocr_resp.get("pages", [])
        if not pages:
            error_msg = "No pages found in OCR response"
            log_with_correlation(
                logger.warning,
                f"[OCR] [DIAGNOSTIC] Markdown extraction warning: {error_msg}",
                correlation_id
            )
            return [], ""

        if not isinstance(pages, list):
            error_msg = f"OCR response 'pages' field is not a list, got {type(pages)}"
            log_with_correlation(
                logger.warning,
                f"[OCR] [DIAGNOSTIC] Markdown extraction warning: {error_msg}",
                correlation_id
            )
            return [], ""

        log_with_correlation(
            logger.info,
            f"[OCR] [DIAGNOSTIC] Found {len(pages)} pages in OCR response",
            correlation_id
        )

        # Extract and validate markdown content
        try:
            markdown_pieces = []
            total_chars = 0
            meaningful_pages = 0

            for i, page in enumerate(pages):
                if not isinstance(page, dict):
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] Page {i} is not a dict, skipping",
                        correlation_id
                    )
                    continue

                page_markdown = page.get("markdown", "")
                if not page_markdown or not isinstance(page_markdown, str):
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Page {i} has no valid markdown content",
                        correlation_id
                    )
                    continue

                # Clean and validate page content
                cleaned_markdown = page_markdown.strip()
                if not cleaned_markdown:
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Page {i} has empty markdown after cleaning",
                        correlation_id
                    )
                    continue

                # Check for minimum content quality
                if len(cleaned_markdown) < 10:
                    log_with_correlation(
                        logger.info,
                        f"[OCR] [DIAGNOSTIC] Page {i} has very short content ({len(cleaned_markdown)} chars), may be low quality",
                        correlation_id
                    )

                # Check for actual text content (not just symbols/punctuation)
                text_only = re.sub(r'[^\w\săâîșțĂÂÎȘȚ]', '', cleaned_markdown)
                if len(text_only.strip()) < 5:
                    log_with_correlation(
                        logger.warning,
                        f"[OCR] [DIAGNOSTIC] Page {i} has minimal readable text content",
                        correlation_id
                    )

                markdown_pieces.append(cleaned_markdown)
                total_chars += len(cleaned_markdown)
                meaningful_pages += 1

                log_with_correlation(
                    logger.info,
                    f"[OCR] [DIAGNOSTIC] Page {i} processed: {len(cleaned_markdown)} characters",
                    correlation_id
                )

            if not markdown_pieces:
                error_msg = "No meaningful markdown content extracted from any page"
                log_with_correlation(
                    logger.warning,
                    f"[OCR] [DIAGNOSTIC] Markdown extraction warning: {error_msg}",
                    correlation_id
                )
                return pages, ""

            md = "\n\n".join(markdown_pieces)

            # Final validation of extracted content
            if len(md.strip()) < 20:
                error_msg = f"Extracted content too short ({len(md)} chars) - likely poor OCR quality"
                log_with_correlation(
                    logger.warning,
                    f"[OCR] [DIAGNOSTIC] Markdown extraction warning: {error_msg}",
                    correlation_id
                )
                return pages, ""

            # Check for minimum text content ratio
            text_content = re.sub(r'[^\w\săâîșțĂÂÎȘȚ]', '', md)
            if len(text_content.strip()) / len(md) < 0.1:  # Less than 10% actual text
                log_with_correlation(
                    logger.warning,
                    f"[OCR] [DIAGNOSTIC] Low text-to-content ratio, possible OCR quality issues",
                    correlation_id
                )

            log_with_correlation(
                logger.info,
                f"[OCR] [DIAGNOSTIC] Successfully extracted markdown content: {len(md)} characters from {meaningful_pages} pages",
                correlation_id
            )
            return pages, md

        except Exception as e:
            error_msg = f"Error extracting markdown from pages: {e}"
            log_with_correlation(
                logger.error,
                f"[OCR] [DIAGNOSTIC] Markdown extraction failed: {error_msg}",
                correlation_id,
                exc_info=True
            )
            return pages, ""

    except Exception as e:
        log_with_correlation(
            logger.error,
            f"[OCR] [DIAGNOSTIC] Critical error in markdown extraction: {e}",
            correlation_id,
            exc_info=True
        )
        return [], ""