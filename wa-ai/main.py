import logging
import json
import uuid
import functools
import asyncio
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our modular components
from app.settings import settings
from app.supa import supabase_client
from app.llm import llm_client
from app.outbound import n8n_client
from app.ocr import ocr_client

# Import PDF handler
from handlers.handle_pdf_message import pdf_handler

# Import the correct two-step media URL function
from integrations.ocr_supabase import get_media_download_url_from_id

load_dotenv()

# Configure enhanced logging for full tracebacks and structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('romstal_assistant.log', mode='a', encoding='utf-8')
    ]
)

# Create formatters for different log levels
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(funcName)s:%(lineno)d - %(message)s'
)

# Add detailed formatter for ERROR and CRITICAL levels
class CorrelationFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = getattr(record, '_correlation_id', 'NO-CORR-ID')
        return True

# Apply correlation filter to all handlers
for handler in logging.root.handlers:
    handler.addFilter(CorrelationFilter())

# --- Enhanced Logging Setup ---
logger = logging.getLogger(__name__)

# Import correlation ID management functions
from app.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    log_with_correlation,
    log_exceptions,
    CorrelationContext
)

# Simple connection test
print("Initializing Romstal Assistant...")
try:
    # Just test if we can create the clients (don't execute query yet)
    print(f"Supabase client created for table: {settings.supabase_table}")
    print(f"OpenAI client configured: {llm_client.client is not None}")
    print(f"N8N client configured: {n8n_client.is_configured()}")
except Exception as e:
    print(f"Failed to create clients: {e}")

app = FastAPI(title="Romstal Assistant", version="1.0")

# Export the FastAPI app for testing
fastapi_app = app


class NewMessage(BaseModel):
    id: Optional[int] = None
    wa_message_id: Optional[str] = None


def build_short_history(messages: List[Dict[str, Any]], max_items: int = 5) -> List[Dict[str, str]]:
    """Ia ultimele N mesaje și le compactează pt prompt."""
    tail = messages[-max_items:] if len(messages) > max_items else messages
    return [{"from": m["from"], "text": m["text"]} for m in tail]


async def detect_pdf_attachment(message_data: Dict[str, Any], phone_number: str) -> Optional[Dict[str, Any]]:
    """
    Detect if a message contains a PDF attachment using WhatsApp's actual webhook structure.
    Includes media age validation to prevent processing expired media IDs.

    Args:
        message_data: The raw message data from database

    Returns:
        Dict with media_url, access_token, mime_type, and filename if PDF detected, None otherwise
    """
    try:
        raw_data = message_data.get("raw", {})

        # Check if this is a document/media message
        if not isinstance(raw_data, dict):
            logger.debug(f"[PDF-DETECT] Raw data is not a dict: {type(raw_data)}")
            return None

        # Enhanced logging for debugging
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Full message data structure for debugging:")
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Message keys: {list(message_data.keys())}")
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Raw data keys: {list(raw_data.keys())}")
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Raw data type field: {raw_data.get('type')}")
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Full raw data: {json.dumps(raw_data, indent=2, default=str)}")

        # NEW: Handle WhatsApp's actual webhook structure with messages[0].document format
        messages = raw_data.get("messages", [])
        if not messages or not isinstance(messages, list) or len(messages) == 0:
            logger.debug(f"[PDF-DETECT] No messages array found in raw data")
            return None

        message = messages[0]
        if not isinstance(message, dict):
            logger.debug(f"[PDF-DETECT] First message is not a dict: {type(message)}")
            return None

        # Check for document in the message
        document = message.get("document")
        if not document or not isinstance(document, dict):
            logger.debug(f"[PDF-DETECT] No document found in first message")
            return None

        # Enhanced logging for document structure
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Document keys: {list(document.keys())}")
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Full document structure: {json.dumps(document, indent=2, default=str)}")

        # Check if it's a PDF by mime type or filename
        mime_type = document.get("mime_type", "")
        filename = document.get("filename", "")

        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] MIME type: '{mime_type}', filename: '{filename}'")

        if (mime_type == "application/pdf" or
            filename.lower().endswith('.pdf')):

            # Get media ID from document
            media_id = document.get("id")
            if not media_id:
                logger.warning(f"[PDF-DETECT] PDF detected but no media_id found in document")
                return None

            # MEDIA AGE VALIDATION: Check if media ID indicates expiration
            logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] Checking media age for media_id: {media_id}")

            # Check media ID range - WhatsApp media IDs are large numbers (trillion+ range for fresh media)
            try:
                media_id_numeric = int(media_id)
                logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] Media ID numeric value: {media_id_numeric}")

                # Enhanced media ID validation with detailed logging
                logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] DEBUG: media_id_numeric={media_id_numeric}, type={type(media_id_numeric)}")

                # WhatsApp media ID ranges (these are very large numbers):
                # - Very old/expired media: typically < 1 trillion (1,000,000,000,000)
                # - Old media: 1-2 trillion range
                # - Fresh media: 3+ trillion range (current fresh media should be 3T+)

                TRILLION = 1_000_000_000_000

                if media_id_numeric < TRILLION:  # Less than 1 trillion indicates very old/expired media
                    logger.warning(f"[PDF-DETECT] [MEDIA-VALIDATION] Media ID {media_id} appears to be from very old range (< 1T). Current fresh media should be 3T+. This is likely expired media.")
                    logger.warning(f"[PDF-DETECT] [MEDIA-VALIDATION] Rejecting expired media ID: {media_id}")
                    logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] Media validation failed - expired media detected. Media ID: {media_id}, Range: < 1T")
                    return None
                elif media_id_numeric < 2 * TRILLION:  # 1-2 trillion range (old but might still work)
                    logger.warning(f"[PDF-DETECT] [MEDIA-VALIDATION] Media ID {media_id} is in 1-2T range (older media) - may be expired")
                elif media_id_numeric < 3 * TRILLION:  # 2-3 trillion range (moderate age)
                    logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] Media ID {media_id} is in 2-3T range (moderate age)")
                else:  # 3T+ range (fresh media)
                    logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] Media ID {media_id} is in 3T+ range (fresh media)")

                logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] Media ID validation passed for {media_id}")

            except (ValueError, TypeError) as e:
                logger.warning(f"[PDF-DETECT] [MEDIA-VALIDATION] Could not parse media_id as numeric: {media_id}, error: {e}")
                logger.info(f"[PDF-DETECT] [MEDIA-VALIDATION] Continuing with non-numeric media ID: {media_id}")
                # Continue processing as we can't determine age from non-numeric ID

            # Get access token from environment (not from webhook payload)
            access_token = settings.whatsapp_access_token
            if not access_token:
                logger.warning(f"[PDF-DETECT] PDF detected but no WhatsApp access token configured")
                return None

            # Use the correct two-step Graph API flow to get the actual download URL
            try:
                media_url = get_media_download_url_from_id(media_id, access_token)
                logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Retrieved download URL via Graph API: '{media_url[:80]}...'")
            except Exception as e:
                logger.error(f"[PDF-DETECT] [DIAGNOSTIC] Failed to get download URL via Graph API: {e}")
                # Graceful failure: send user-friendly message and return None
                correlation_id = generate_correlation_id()
                with CorrelationContext(correlation_id):
                    logger.error(f"[PDF-DETECT] [{correlation_id}] Failed to retrieve PDF download URL for media_id: {media_id}. Sending graceful failure message to user.")

                    # Send graceful failure message to user
                    failure_message = "Nu am reușit să descarc documentul. Poți retrimite PDF-ul?"
                    try:
                        await n8n_client.send_to_n8n(phone_number, failure_message, correlation_id)
                        logger.info(f"[PDF-DETECT] [{correlation_id}] Graceful failure message sent to user {phone_number}")
                    except Exception as msg_error:
                        logger.error(f"[PDF-DETECT] [{correlation_id}] Failed to send graceful failure message: {msg_error}")

                return None

            logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Using access_token from environment: '{access_token[:20]}...'")

            logger.info(f"[PDF-DETECT] PDF attachment detected: {filename} (MIME: {mime_type})")
            return {
                "media_url": media_url,
                "access_token": access_token,
                "mime_type": mime_type,
                "filename": filename
            }

        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] No PDF attachment detected in message")
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Final assessment - has messages: {'messages' in raw_data}, message count: {len(messages) if isinstance(messages, list) else 'not list'}")
        return None

    except Exception as e:
        logger.error(f"[PDF-DETECT] Error detecting PDF attachment: {e}", exc_info=True)
        logger.info(f"[PDF-DETECT] [DIAGNOSTIC] Message data structure during error: {json.dumps(message_data, indent=2, default=str)}")
        return None


async def process_pdf_message(message_data: Dict[str, Any], insert_id: int) -> Optional[str]:
    """
    Process a PDF message using the PDF handler.

    Args:
        message_data: The raw message data from database
        insert_id: Database insert ID of the message

    Returns:
        AI response text if successful, None if failed
    """
    try:
        logger.info(f"[PDF] [DIAGNOSTIC] Starting PDF processing for insert_id: {insert_id}")

        # Detect PDF attachment
        logger.info(f"[PDF] [DIAGNOSTIC] Detecting PDF attachment for insert_id: {insert_id}")
        # Extract phone number from message_data for graceful failure handling
        phone_number = message_data.get("wa_from", "")
        pdf_info = await detect_pdf_attachment(message_data, phone_number)
        if not pdf_info:
            logger.error(f"[PDF] [DIAGNOSTIC] No PDF attachment detected for insert_id: {insert_id}")
            return None

        logger.info(f"[PDF] [DIAGNOSTIC] Processing PDF attachment: {pdf_info.get('filename', 'unknown')} for insert_id: {insert_id}")

        # Call PDF handler
        logger.info(f"[PDF] [DIAGNOSTIC] Calling PDF handler for insert_id: {insert_id}")
        result = await pdf_handler.handle_pdf_message(
            insert_id=insert_id
        )

        if result.get("success"):
            logger.info(f"[PDF] [DIAGNOSTIC] Successfully processed PDF message for insert_id: {insert_id}, response length: {len(result.get('ai_response', ''))}")
            return result["ai_response"]
        else:
            logger.error(f"[PDF] [DIAGNOSTIC] PDF processing failed for insert_id: {insert_id}: {result.get('error')}")
            return None

    except Exception as e:
        logger.error(f"[PDF] [DIAGNOSTIC] Exception during PDF processing for insert_id: {insert_id}: {type(e).__name__}: {e}")
        return None


# --- Health Checks ---
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/health/detailed")
async def detailed_health():
    """Comprehensive health check for all external services."""
    correlation_id = generate_correlation_id()

    with CorrelationContext(correlation_id):
        logger.info(f"[HEALTH] [{correlation_id}] Starting comprehensive health check")

        health_results = {
            "overall": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id,
            "services": {}
        }

        # Check N8N service
        try:
            n8n_health = await n8n_client.health_check()
            health_results["services"]["n8n"] = n8n_health
            if not n8n_health.get("healthy", False):
                health_results["overall"] = "degraded"
        except Exception as e:
            logger.exception(f"[HEALTH] [{correlation_id}] N8N health check failed: {type(e).__name__}: {e}")
            health_results["services"]["n8n"] = {
                "healthy": False,
                "error": str(e),
                "correlation_id": correlation_id
            }
            health_results["overall"] = "unhealthy"

        # Check Supabase service
        try:
            supabase_healthy = True
            try:
                # Try a simple query to test database connectivity
                test_result = supabase_client.client.table("wa_messages").select("id").limit(1).execute()
                health_results["services"]["supabase"] = {
                    "healthy": True,
                    "message": "Database connection successful",
                    "correlation_id": correlation_id
                }
            except Exception as db_error:
                supabase_healthy = False
                logger.error(f"[HEALTH] [{correlation_id}] Supabase health check failed: {type(db_error).__name__}: {db_error}")
                health_results["services"]["supabase"] = {
                    "healthy": False,
                    "error": str(db_error),
                    "correlation_id": correlation_id
                }
                health_results["overall"] = "unhealthy"
        except Exception as e:
            logger.exception(f"[HEALTH] [{correlation_id}] Supabase health check error: {type(e).__name__}: {e}")
            health_results["services"]["supabase"] = {
                "healthy": False,
                "error": str(e),
                "correlation_id": correlation_id
            }
            health_results["overall"] = "unhealthy"

        # Check OpenAI service
        try:
            if llm_client.client:
                health_results["services"]["openai"] = {
                    "healthy": True,
                    "message": "OpenAI client initialized",
                    "correlation_id": correlation_id
                }
            else:
                health_results["services"]["openai"] = {
                    "healthy": False,
                    "error": "OpenAI client not initialized",
                    "correlation_id": correlation_id
                }
                health_results["overall"] = "degraded"
        except Exception as e:
            logger.exception(f"[HEALTH] [{correlation_id}] OpenAI health check failed: {type(e).__name__}: {e}")
            health_results["services"]["openai"] = {
                "healthy": False,
                "error": str(e),
                "correlation_id": correlation_id
            }
            health_results["overall"] = "unhealthy"

        # Check Mistral OCR service
        try:
            if ocr_client.api_key and settings.is_mistral_api_key_valid:
                health_results["services"]["mistral_ocr"] = {
                    "healthy": True,
                    "message": "Mistral OCR client configured",
                    "correlation_id": correlation_id
                }
            else:
                health_results["services"]["mistral_ocr"] = {
                    "healthy": False,
                    "error": "Mistral OCR not properly configured",
                    "correlation_id": correlation_id
                }
                health_results["overall"] = "degraded"
        except Exception as e:
            logger.exception(f"[HEALTH] [{correlation_id}] Mistral OCR health check failed: {type(e).__name__}: {e}")
            health_results["services"]["mistral_ocr"] = {
                "healthy": False,
                "error": str(e),
                "correlation_id": correlation_id
            }
            health_results["overall"] = "unhealthy"

        logger.info(f"[HEALTH] [{correlation_id}] Health check completed: {health_results['overall']}")
        return health_results

# --- Endpoints ---


@app.post("/new-message")
async def new_message(payload: NewMessage, x_webhook_token: str = Header(None)):
    if settings.inbound_webhook_token and x_webhook_token != settings.inbound_webhook_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1) Găsește rândul inserat
    print(f"[DEBUG] Looking for message with payload: {payload}")
    row = None

    if payload.id is not None:
        print(f"[DEBUG] Searching by ID: {payload.id}")
        try:
            row = supabase_client.get_message_by_id(payload.id)
            print(f"[DEBUG] Query by ID returned: {type(row)} - {row is not None}")
        except Exception as e:
            print(f"[ERROR] Exception during get_message_by_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    elif payload.wa_message_id:
        print(f"[DEBUG] Searching by WA message ID: {payload.wa_message_id}")
        try:
            row = supabase_client.get_message_by_wa_id(payload.wa_message_id)
            print(f"[DEBUG] Query by WA ID returned: {type(row)} - {row is not None}")
        except Exception as e:
            print(f"[ERROR] Exception during get_message_by_wa_id: {e}")
            raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
    else:
        print("[ERROR] No ID or WA message ID provided in payload")
        raise HTTPException(status_code=400, detail="Either id or wa_message_id must be provided")

    print(f"[DEBUG] Row check - row is None: {row is None}, bool(row): {bool(row)}, type(row): {type(row)}")
    if not row or not isinstance(row, dict):
        print(f"[ERROR] Message not found in database - row is: {row}, type: {type(row)}")
        raise HTTPException(status_code=404, detail="Mesajul nu a fost găsit în wa_messages")
    else:
        wa_text = row.get('wa_text') or ''; print(f"[SUCCESS] Found message: {wa_text[:50]}...")

    # Defensive check for row being None (additional safety)
    if row is None:
        print(f"[ERROR] Row became None after initial check")
        raise HTTPException(status_code=500, detail="Internal error: message data became unavailable")

    # Enhanced phone extraction with defensive error handling
    phone = None
    try:
        # Log detailed information about the row object for debugging
        logger.info(f"[PHONE-EXTRACTION] [DIAGNOSTIC] Row type: {type(row)}")
        logger.info(f"[PHONE-EXTRACTION] [DIAGNOSTIC] Row is dict: {isinstance(row, dict)}")
        logger.info(f"[PHONE-EXTRACTION] [DIAGNOSTIC] Row has get method: {hasattr(row, 'get')}")
        logger.info(f"[PHONE-EXTRACTION] [DIAGNOSTIC] Row keys type: {type(list(row.keys()) if hasattr(row, 'keys') else 'no keys method')}")

        # Additional data validation
        if not hasattr(row, 'get'):
            logger.error(f"[PHONE-EXTRACTION] [ERROR] Row object does not have 'get' method. Row type: {type(row)}")
            logger.error(f"[PHONE-EXTRACTION] [ERROR] Row object methods: {[m for m in dir(row) if not m.startswith('_')]}")
            raise HTTPException(status_code=500, detail="Invalid row data structure: missing get method")

        # Try to extract phone with multiple fallback methods
        try:
            phone = row.get("wa_from")
            logger.info(f"[PHONE-EXTRACTION] [SUCCESS] Successfully extracted phone using .get(): {phone}")
        except Exception as get_error:
            logger.error(f"[PHONE-EXTRACTION] [ERROR] Failed to call .get() on row: {get_error}")
            logger.error(f"[PHONE-EXTRACTION] [ERROR] Row object repr: {repr(row)[:500]}")
            raise HTTPException(status_code=500, detail=f"Failed to extract phone number: {str(get_error)}")

        # Validate the extracted phone value
        if phone is None:
            logger.error(f"[PHONE-EXTRACTION] [ERROR] Phone number is None in database record")
            logger.error(f"[PHONE-EXTRACTION] [ERROR] Available row keys: {list(row.keys()) if hasattr(row, 'keys') else 'unable to get keys'}")
            logger.error(f"[PHONE-EXTRACTION] [ERROR] Row content preview: {str(row)[:1000]}")
            raise HTTPException(status_code=400, detail="wa_from lipsește pe rândul inserat")
        elif not isinstance(phone, str):
            logger.warning(f"[PHONE-EXTRACTION] [WARNING] Phone number is not a string, converting. Type: {type(phone)}")
            phone = str(phone)
            logger.info(f"[PHONE-EXTRACTION] [SUCCESS] Converted phone to string: {phone}")
        else:
            logger.info(f"[PHONE-EXTRACTION] [SUCCESS] Phone number extracted and validated: {phone}")

    except HTTPException:
        # Re-raise HTTP exceptions as they have proper error messages
        raise
    except Exception as e:
        # Catch-all for any unexpected errors during phone extraction
        logger.exception(f"[PHONE-EXTRACTION] [CRITICAL] Unexpected error during phone extraction: {type(e).__name__}: {e}")
        logger.error(f"[PHONE-EXTRACTION] [CRITICAL] Row object details - Type: {type(row)}, Has get: {hasattr(row, 'get')}")
        if hasattr(row, 'keys'):
            try:
                logger.error(f"[PHONE-EXTRACTION] [CRITICAL] Row keys: {list(row.keys())}")
            except Exception as keys_error:
                logger.error(f"[PHONE-EXTRACTION] [CRITICAL] Failed to get row keys: {keys_error}")
        logger.error(f"[PHONE-EXTRACTION] [CRITICAL] Row object string representation: {str(row)[:500]}")
        raise HTTPException(status_code=500, detail=f"Critical error extracting phone number: {str(e)}")

    print(f"[PHONE] Phone number: {phone}")

    # Get ALL messages in chronological order (oldest first) using inserted_at
    print(f"[FETCH] Fetching all messages for phone: {phone}")
    history = supabase_client.get_message_history(phone)
    print(f"[HISTORY] Found {len(history)} total messages for {phone}")

    # Convert all messages to array (oldest first)
    all_messages = []
    for h in history:
        text = (h.get("wa_text") or "").strip()
        if text:  # Only skip completely empty messages
            role = "agent" if h.get("direction") == "outbound" else "user"
            all_messages.append({
                "from": role,
                "text": text,
                "ts": h.get("wa_timestamp") or h.get("inserted_at")
            })

    # Take the LAST 10 messages (most recent 10)
    recent_messages = all_messages[-10:] if len(all_messages) >= 10 else all_messages

    # Get the current message that triggered this request
    current_message = row.get("wa_text", "")

    # Add current message if not already in recent messages
    current_in_context = any(m["text"] == current_message for m in recent_messages)
    if current_message and not current_in_context:
        # Defensive check for row being None (additional safety)
        if row is None:
            print(f"[ERROR] Row became None when trying to get timestamp")
            raise HTTPException(status_code=500, detail="Internal error: message data became unavailable")
        current_timestamp = row.get("wa_timestamp") or row.get("inserted_at")
        recent_messages.append({
            "from": "user",
            "text": current_message,
            "ts": current_timestamp
        })

    # Check for PDF attachment first
    pdf_response = await process_pdf_message(row, payload.id or 0)

    if pdf_response:
        # PDF was successfully processed by the PDF handler, which already sent the response
        # The PDF handler manages its own N8N sending and database saving, so we just return
        logger.info(f"[PDF] PDF message processed successfully by PDF handler, skipping duplicate response")

        return {
            "ok": True,
            "conversation_count": len(recent_messages),
            "ai_reply": pdf_response,
            "processed_as": "pdf",
            "note": "Response already sent by PDF handler"
        }

    # Use the current message for AI response (regular text processing)
    user_message_to_respond = current_message

    if not user_message_to_respond:
        reply_text = "Salut! Sunt asistentul Romstal. Cu ce te pot ajuta?"
        function_call_details = None
    else:
        # Get or create conversation session FIRST to access function call history
        print(f"[SESSION] Getting or creating conversation session for {phone}")
        session = supabase_client.get_or_create_conversation_session(phone, "")

        if session:
            print(f"[SUCCESS] Session found/created: {session['id']}")
        else:
            print(f"[ERROR] Failed to get/create conversation session")

        # Build context string for AI, including function call history from session
        hist_lines = []

        # Check session for function call history
        if session:
            full_conversation = session.get("full_conversation", [])
            # Look for recent function calls in the conversation history
            for msg in full_conversation[-5:]:  # Last 5 messages
                if msg.get("role") == "function_calls":
                    # Add function call context
                    func_details = msg.get("details", [])
                    for func in func_details:
                        func_name = func.get("function", "unknown")
                        func_args = func.get("args", {})
                        func_result = func.get("result", {})
                        hist_lines.append(f"- [Function: {func_name}({func_args}) -> {func_result.get('ok', False)}]")

        # Add regular message history
        for m in recent_messages:
            hist_lines.append(f"- {m['from']}: {m['text']}")

        hist_context = "\n".join(hist_lines)

        # Debug logging
        print(f"[DEBUG] Current user message: {current_message}")
        print(f"[DEBUG] Context messages: {len(recent_messages)}")
        for i, msg in enumerate(recent_messages):
            safe_text = msg['text'][:50].encode('ascii', 'ignore').decode('ascii')
            print(f"   {i+1}. {msg['from']}: {safe_text}...")

        system_prompt = (
            "Ești un asistent Romstal prietenos și util pe WhatsApp.\n"
            "Răspunde în română, natural și conversațional.\n"
            "Poți purta discuții casual și răspunde la întrebări personale simple, dar rolul tău principal este să ajuți utilizatorii cu informații despre Romstal, produse, servicii, program, livrare și alte detalii utile.\n"
            "Fii prietenos, natural și adaptabil — dacă cineva te întreabă „ce faci\" sau „cum ești\", răspunde firesc, ca un prieten.\n"
            "Dacă nu știi sigur un detaliu despre Romstal, spune că vei verifica și vei reveni cu informațiile corecte.\n"
            "Important:\n"
            "- Nu propune acțiuni precum adăugarea produselor în stoc, efectuarea comenzilor, programări sau alte procese operative.\n"
            "- Nu poti sa cauti produse doar daca primesti un cod de produs specific"
            "- Nu face follow-up pentru a oferi servicii sau a iniția alte conversații.\n"
            "- Poți face follow-up doar despre produsul sau subiectul discutat (ex: recomandări similare, specificații, întreținere, garanție etc.).\n"
            "- Menține un ton profesionist, empatic și prietenos, ca un consultant Romstal care vorbește relaxat, dar informat.\n"
        )

        # Update session with correct system prompt if it was created empty
        if session and not session.get("system_prompt"):
            supabase_client.update_session_system_prompt(int(session["id"]), system_prompt)

        user_prompt = (
            f"Context conversație (include apeluri funcții anterioare):\n{hist_context}\n\n"
            f"Mesajul utilizatorului: {user_message_to_respond}\n\n"
            "Generează un răspuns helpful și natural în română, folosind contextul complet de mai sus."
        )

        # Call LLM with tools and capture any function call details
        correlation_id = generate_correlation_id()
        reply_text, function_call_details = await llm_client.call_llm_with_tools(system_prompt, user_prompt, correlation_id)

        # Update conversation session with user message, function calls (if any), and AI response
        if session:
            print(f"[SAVE] Updating conversation session {session['id']} with new messages")
            message_timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00')

            # Use the correct user message that triggered this request
            user_message_content = user_message_to_respond

            user_message = {
                "role": "user",
                "content": user_message_content,
                "timestamp": message_timestamp,
                "context_used": recent_messages
            }

            # Prepare messages list
            messages_to_save = [user_message]

            # Add function call details if any functions were called
            if function_call_details:
                messages_to_save.append({
                    "role": "function_calls",
                    "content": json.dumps(function_call_details, ensure_ascii=False),
                    "timestamp": message_timestamp,
                    "details": function_call_details
                })

            # Add final assistant response
            assistant_message = {
                "role": "assistant",
                "content": reply_text,
                "timestamp": message_timestamp,
                "context_used": recent_messages
            }
            messages_to_save.append(assistant_message)

            # Update session with all messages
            success = supabase_client.update_conversation_session(int(session["id"]), messages_to_save, recent_messages)

            if success:
                print(f"[SUCCESS] Conversation session updated successfully with both messages")
            else:
                print(f"[ERROR] Failed to update conversation session")

    # Trimite răspunsul către WhatsApp prin N8N
    n8n_result = await n8n_client.send_to_n8n(phone_number=phone, message=reply_text)

    # Salvează răspunsul în baza de date
    try:
        print(f"[SAVE] Attempting to save outbound message for {phone}")
        safe_reply = reply_text[:50].encode('ascii', 'ignore').decode('ascii')
        print(f"[MESSAGE] Message: {safe_reply}{'...' if len(reply_text) > 50 else ''}")

        supabase_client.insert_outbound_message(phone, reply_text, {"source": "romstal_assistant", "n8n_result": n8n_result})
        safe_reply = reply_text.encode('ascii', 'ignore').decode('ascii')
        print(f"[SUCCESS] Outbound message saved successfully for {phone}")
        logger.info(f"[SUCCESS] Outbound message saved successfully for {phone}: {len(safe_reply)} chars")

    except Exception as e:
        safe_phone = phone.encode('ascii', 'ignore').decode('ascii')
        print(f"[ERROR] Failed to save outbound message for {safe_phone}: Unicode encoding issue in logging")
        logger.error(f"[ERROR] Failed to log outbound message for {safe_phone}: Unicode encoding issue in logging")

    return {
        "ok": True,
        "conversation_count": len(recent_messages),
        "ai_reply": reply_text,
        "n8n_result": n8n_result
    }

# Architecture successfully applied!
