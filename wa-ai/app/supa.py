import logging
import asyncio
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from .settings import settings
from .correlation import generate_correlation_id, get_correlation_id, set_correlation_id, CorrelationContext

logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        self._client: Optional[Client] = None
        self._table: str = ""
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds
        self._cleanup_counter = 0  # Counter for periodic cleanup tasks

    @property
    def client(self) -> Client:
        if self._client is None:
            # Reload environment variables in case we're in a subprocess
            from dotenv import load_dotenv
            import os
            load_dotenv()

            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

            if not supabase_url or not supabase_key:
                error_msg = "Supabase URL și Key trebuie setate în .env"
                logger.error(f"[SUPABASE] {error_msg}")
                raise RuntimeError(error_msg)

            try:
                self._client = create_client(supabase_url, supabase_key)
                logger.info("[SUPABASE] Client initialized successfully")
            except Exception as e:
                logger.exception(f"[SUPABASE] Failed to initialize client: {type(e).__name__}: {e}")
                raise
        return self._client

    def _get_table_name(self) -> str:
        """Get table name directly without using property"""
        from dotenv import load_dotenv
        import os
        load_dotenv()
        return os.getenv("SUPABASE_TABLE", "wa_messages")

    async def _execute_with_retry(self, operation_name: str, operation_func, correlation_id: Optional[str] = None, **kwargs):
        """Execute a database operation with retry logic and comprehensive error handling."""
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            last_exception = Exception("No attempts made")

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"[SUPABASE] [{correlation_id}] Attempt {attempt + 1}/{self.max_retries} for {operation_name}")

                    # Ensure client is available
                    client = self.client

                    # Execute the operation
                    result = await operation_func(client, **kwargs)

                    logger.info(f"[SUPABASE] [{correlation_id}] {operation_name} completed successfully")
                    return result

                except Exception as e:
                    last_exception = e
                    error_type = type(e).__name__

                    if "connection" in str(e).lower() or "timeout" in str(e).lower():
                        logger.warning(f"[SUPABASE] [{correlation_id}] {error_type} on attempt {attempt + 1}: {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            continue
                    else:
                        # Non-retryable error
                        logger.error(f"[SUPABASE] [{correlation_id}] Non-retryable {error_type} on attempt {attempt + 1}: {e}")
                        break

            # All retries failed
            logger.error(f"[SUPABASE] [{correlation_id}] {operation_name} failed after {self.max_retries} attempts. Last error: {type(last_exception).__name__}: {last_exception}")
            raise last_exception

    def get_message_by_id(self, message_id: int, correlation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a message by its ID with enhanced error handling."""
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            last_exception = Exception("No attempts made")

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"[SUPABASE] [{correlation_id}] Attempt {attempt + 1}/{self.max_retries} for get_message_by_id({message_id})")

                    # Ensure client is available
                    client = self.client
                    table_name = self._get_table_name()

                    logger.info(f"[SUPABASE] [{correlation_id}] Fetching message by ID {message_id} from table {table_name}")

                    result = client.table(table_name)\
                        .select("*")\
                        .eq("id", message_id)\
                        .execute()

                    logger.debug(f"[SUPABASE] [{correlation_id}] Query result: {len(result.data) if result.data else 0} rows")
                    logger.info(f"[SUPABASE] [{correlation_id}] get_message_by_id({message_id}) completed successfully")
                    return (result.data or [None])[0]

                except Exception as e:
                    last_exception = e
                    error_type = type(e).__name__

                    if "connection" in str(e).lower() or "timeout" in str(e).lower():
                        logger.warning(f"[SUPABASE] [{correlation_id}] {error_type} on attempt {attempt + 1}: {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                    else:
                        # Non-retryable error
                        logger.error(f"[SUPABASE] [{correlation_id}] Non-retryable {error_type} on attempt {attempt + 1}: {e}")
                        break

            # All retries failed
            logger.error(f"[SUPABASE] [{correlation_id}] get_message_by_id({message_id}) failed after {self.max_retries} attempts. Last error: {type(last_exception).__name__}: {last_exception}")
            raise last_exception

    def get_message_by_wa_id(self, wa_message_id: str) -> Optional[Dict[str, Any]]:
        """Get a message by its WhatsApp message ID."""
        try:
            logger.info(f"[FETCH] Fetching message by WA message ID {wa_message_id}")
            result = self.client.table(self._get_table_name())\
                .select("*")\
                .eq("wa_message_id", wa_message_id)\
                .execute()
            logger.debug(f"Supabase response: {result.data}")
            return (result.data or [None])[0]
        except Exception as e:
            logger.exception("[ERROR] Error querying message by WA ID")
            raise

    def get_message_history(self, phone: str) -> List[Dict[str, Any]]:
        """Get all messages for a phone number in chronological order."""
        try:
            logger.info(f"[FETCH] Fetching message history for phone {phone}")
            result = self.client.table(self._get_table_name())\
                .select("wa_from, wa_text, wa_timestamp, inserted_at, direction")\
                .eq("wa_from", phone)\
                .order("inserted_at", desc=False)\
                .execute()
            logger.debug(f"Supabase history response: {len(result.data) if result.data else 0} messages found")
            return result.data or []
        except Exception as e:
            logger.exception("[ERROR] Error querying message history")
            raise

    def insert_outbound_message(self, phone: str, message: str, raw_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Insert an outbound message into the database."""
        from datetime import datetime
        import uuid

        current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00')
        outbound_message_id = f"outbound_{uuid.uuid4().hex[:16]}"

        outbound_data = {
            "wa_from": phone,
            "wa_type": "text",
            "wa_text": message,
            "wa_timestamp": current_time,
            "inserted_at": current_time,
            "wa_message_id": outbound_message_id,
            "raw": raw_data or {"source": "romstal_assistant"},
            "direction": "outbound"
        }

        try:
            logger.info(f"[FETCH] Inserting outbound message for phone {phone}")
            result = self.client.table(self._get_table_name()).insert(outbound_data).execute()
            logger.debug(f"Supabase outbound message insert result: {result.data}")
            return result.data[0] if result.data else {}
        except Exception as e:
            logger.exception("[ERROR] Error inserting outbound message")
            raise

    def get_or_create_conversation_session(self, phone: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Get existing session or create new one"""
        try:
            # Look for recent session (last 24 hours)
            twenty_four_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)

            try:
                logger.info(f"[FETCH] Looking for recent conversation session for phone {phone}")
                result = self.client.table("conversation_sessions")\
                    .select("*")\
                    .eq("phone_number", phone)\
                    .gte("last_activity", twenty_four_hours_ago)\
                    .order("last_activity", desc=True)\
                    .limit(1)\
                    .execute()
                logger.debug(f"Supabase conversation session query result: {len(result.data) if result.data else 0} sessions found")
            except Exception:
                logger.exception("[ERROR] Error querying conversation sessions")
                raise

            if result.data:
                return result.data[0]
            else:
                # Create new session
                new_session = {
                    "phone_number": phone,
                    "system_prompt": system_prompt,
                    "full_conversation": [],
                    "context_messages": [],
                    "message_count": 0
                }
                try:
                    logger.info(f"[FETCH] Creating new conversation session for phone {phone}")
                    result = self.client.table("conversation_sessions").insert(new_session).execute()
                    logger.debug(f"Supabase conversation session insert result: {result.data}")
                except Exception:
                    logger.exception("[ERROR] Error creating conversation session")
                    raise
                return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error managing conversation session: {e}")
            return None

    def update_conversation_session(self, session_id: int, new_messages: List[Dict], context_messages: List[Dict]) -> bool:
        """Update session with new messages and context"""
        try:
            # Get current session
            try:
                logger.info(f"[FETCH] Fetching conversation session by ID {session_id}")
                result = self.client.table("conversation_sessions")\
                    .select("full_conversation, message_count")\
                    .eq("id", session_id)\
                    .execute()
                logger.debug(f"Supabase conversation session fetch result: {result.data}")
            except Exception:
                logger.exception("[ERROR] Error fetching conversation session")
                raise

            if not result.data:
                return False

            current_session = result.data[0]
            full_conversation = current_session.get("full_conversation", [])

            # Add all new messages to the conversation
            for message in new_messages:
                full_conversation.append(message)

            # Update session
            update_data = {
                "full_conversation": full_conversation,
                "context_messages": context_messages,
                "message_count": current_session.get("message_count", 0) + len(new_messages),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            try:
                logger.info(f"[FETCH] Updating conversation session {session_id}")
                self.client.table("conversation_sessions")\
                    .update(update_data)\
                    .eq("id", session_id)\
                    .execute()
                logger.debug(f"Supabase conversation session update completed for session {session_id}")
            except Exception:
                logger.exception("[ERROR] Error updating conversation session")
                raise

            return True
        except Exception as e:
            logger.error(f"Error updating conversation session: {e}")
            return False

    def update_session_system_prompt(self, session_id: int, system_prompt: str) -> bool:
        """Update the system prompt for a conversation session"""
        try:
            self.client.table("conversation_sessions")\
                .update({"system_prompt": system_prompt})\
                .eq("id", session_id)\
                .execute()
            return True
        except Exception as e:
            logger.error(f"Failed to update session system_prompt: {e}")
            return False

    def get_media_download_info_by_insert_id(self, insert_id: int) -> Optional[Dict[str, str]]:
        """
        Get media download information for a message by its insert_id using the proper two-step flow.

        Args:
            insert_id: Database insert ID of the message

        Returns:
            Dict with media_url, access_token, mime_type, and filename if PDF detected, None otherwise
        """
        try:
            logger.info(f"[MEDIA] Getting media info for insert_id: {insert_id}")

            # Get message by insert_id
            message = self.get_message_by_id(insert_id)
            if not message:
                logger.error(f"[MEDIA] Message not found for insert_id: {insert_id}")
                return None

            # Extract raw message data
            raw_data = message.get("raw", {})
            if not isinstance(raw_data, dict):
                logger.error(f"[MEDIA] Raw data is not a dict for insert_id: {insert_id}")
                return None

            # Handle WhatsApp's webhook structure with messages[0].document format
            messages = raw_data.get("messages", [])
            if not messages or not isinstance(messages, list) or len(messages) == 0:
                logger.error(f"[MEDIA] No messages array found in raw data for insert_id: {insert_id}")
                return None

            message_data = messages[0]
            if not isinstance(message_data, dict):
                logger.error(f"[MEDIA] First message is not a dict for insert_id: {insert_id}")
                return None

            # Check for document in the message
            document = message_data.get("document")
            if not document or not isinstance(document, dict):
                logger.error(f"[MEDIA] No document found in message for insert_id: {insert_id}")
                return None

            # Check if it's a PDF by mime type or filename
            mime_type = document.get("mime_type", "")
            filename = document.get("filename", "")

            if not (mime_type == "application/pdf" or filename.lower().endswith('.pdf')):
                logger.error(f"[MEDIA] Document is not a PDF for insert_id: {insert_id}")
                return None

            # Get media ID from document
            media_id = document.get("id")
            if not media_id:
                logger.error(f"[MEDIA] PDF detected but no media_id found for insert_id: {insert_id}")
                return None

            # Get access token from settings
            from .settings import settings
            access_token = settings.whatsapp_access_token
            if not access_token:
                logger.error(f"[MEDIA] No WhatsApp access token configured for insert_id: {insert_id}")
                return None

            # Enhanced logging for token debugging
            logger.info(f"[MEDIA] [DIAGNOSTIC] Retrieved access token for insert_id: {insert_id}, length: {len(access_token)}, starts with: {access_token[:20]}...")

            # Use the correct two-step Graph API flow to get the actual download URL
            try:
                from integrations.ocr_supabase import get_media_download_url_from_id
                media_url = get_media_download_url_from_id(media_id, access_token)
                logger.info(f"[MEDIA] Successfully retrieved download URL for insert_id: {insert_id}")
            except Exception as e:
                logger.error(f"[MEDIA] Failed to get download URL via Graph API for insert_id: {insert_id}: {e}")
                return None

            logger.info(f"[MEDIA] PDF attachment detected for insert_id: {insert_id}: {filename} (MIME: {mime_type})")
            return {
                "media_url": media_url,
                "access_token": access_token,
                "mime_type": mime_type,
                "filename": filename
            }

        except Exception as e:
            logger.error(f"[MEDIA] Error getting media info for insert_id: {insert_id}: {e}")
            return None

    def get_media_id_and_token_by_insert_id(self, insert_id: int) -> Optional[Dict[str, str]]:
        """
        Get media ID and access token for a message by its insert_id.

        This is a minimal helper that only extracts the raw media_id and access_token
        without calling the Graph API or constructing download URLs.

        Args:
            insert_id: Database insert ID of the message

        Returns:
            Dict with media_id and access_token if found, None otherwise
        """
        try:
            logger.info(f"[MEDIA] Getting media ID and token for insert_id: {insert_id}")

            # Get message by insert_id
            message = self.get_message_by_id(insert_id)
            if not message:
                logger.error(f"[MEDIA] Message not found for insert_id: {insert_id}")
                return None

            # Extract raw message data
            raw_data = message.get("raw", {})
            if not isinstance(raw_data, dict):
                logger.error(f"[MEDIA] Raw data is not a dict for insert_id: {insert_id}")
                return None

            # Handle WhatsApp's webhook structure with messages[0].document format
            messages = raw_data.get("messages", [])
            if not messages or not isinstance(messages, list) or len(messages) == 0:
                logger.error(f"[MEDIA] No messages array found in raw data for insert_id: {insert_id}")
                return None

            message_data = messages[0]
            if not isinstance(message_data, dict):
                logger.error(f"[MEDIA] First message is not a dict for insert_id: {insert_id}")
                return None

            # Check for document in the message
            document = message_data.get("document")
            if not document or not isinstance(document, dict):
                logger.error(f"[MEDIA] No document found in message for insert_id: {insert_id}")
                return None

            # Get media ID from document
            media_id = document.get("id")
            if not media_id:
                logger.error(f"[MEDIA] No media_id found in document for insert_id: {insert_id}")
                return None

            # Get access token from settings
            from .settings import settings
            access_token = settings.whatsapp_access_token
            if not access_token:
                logger.error(f"[MEDIA] No WhatsApp access token configured for insert_id: {insert_id}")
                return None

            logger.info(f"[MEDIA] Successfully retrieved media_id: {media_id[:20]}... for insert_id: {insert_id}")
            return {
                "media_id": media_id,
                "access_token": access_token
            }

        except Exception as e:
            logger.error(f"[MEDIA] Error getting media ID and token for insert_id: {insert_id}: {e}")
            return None

    def insert_ocr_result(self, message_insert_id: int, phone_number: str, supabase_path: str,
                         pages_json: List[Dict[str, Any]], full_markdown: str,
                         created_at: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Insert OCR result into the ocr_results table.

        Args:
            message_insert_id: Database insert ID of the original message
            phone_number: Phone number associated with the message
            supabase_path: Path in Supabase storage where OCR files are stored
            pages_json: List of page objects from OCR processing
            full_markdown: Complete markdown content from OCR
            created_at: Creation timestamp (defaults to current UTC time)

        Returns:
            Dict containing the inserted record data

        Raises:
            Exception: If the insert operation fails
        """
        if created_at is None:
            created_at = datetime.now(timezone.utc)

        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            try:
                logger.info(f"[OCR] [{correlation_id}] Inserting OCR result for message_insert_id: {message_insert_id}")

                # Prepare the data for insertion
                ocr_data = {
                    "message_insert_id": message_insert_id,
                    "phone_number": phone_number,
                    "supabase_path": supabase_path,
                    "pages_json": pages_json,
                    "full_markdown": full_markdown,
                    "created_at": created_at.isoformat()
                }

                # Perform the insert operation
                result = self.client.table("ocr_results").insert(ocr_data).execute()

                if not result.data:
                    error_msg = f"No data returned after inserting OCR result for message_insert_id: {message_insert_id}"
                    logger.error(f"[OCR] [{correlation_id}] {error_msg}")
                    raise RuntimeError(error_msg)

                logger.info(f"[OCR] [{correlation_id}] Successfully inserted OCR result for message_insert_id: {message_insert_id}")
                logger.debug(f"[OCR] [{correlation_id}] Inserted record ID: {result.data[0].get('id')}")

                return result.data[0]

            except Exception as e:
                error_msg = f"Failed to insert OCR result for message_insert_id: {message_insert_id}"
                logger.error(f"[OCR] [{correlation_id}] {error_msg}: {type(e).__name__}: {e}")
                raise

    def store_product_link_context(self, session_id: int, phone_number: str, product_links: List[Dict[str, Any]], message_insert_id: int) -> bool:
        """
        Store product link context for a conversation session.

        Args:
            session_id: Conversation session ID
            phone_number: Phone number for the conversation
            product_links: List of product link data dictionaries
            message_insert_id: Database insert ID of the message where links were displayed

        Returns:
            bool: True if successful, False otherwise
        """
        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            try:
                logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Storing {len(product_links)} product links for session {session_id}")

                # Calculate expiry time (24 hours from now)
                expires_at = datetime.now(timezone.utc) + timedelta(hours=24)

                # Prepare context records for insertion
                context_records = []
                for link_data in product_links:
                    context_records.append({
                        "conversation_session_id": session_id,
                        "phone_number": phone_number,
                        "product_url": link_data.get("url", ""),
                        "product_code": link_data.get("code", ""),
                        "product_name": link_data.get("name", ""),
                        "displayed_at": datetime.now(timezone.utc).isoformat(),
                        "message_insert_id": message_insert_id,
                        "context_expires_at": expires_at.isoformat()
                    })

                if context_records:
                    # Insert context records
                    result = self.client.table("product_link_context").insert(context_records).execute()
                    logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Successfully stored {len(context_records)} product link contexts")
                    return True
                else:
                    logger.warning(f"[PRODUCT-CONTEXT] [{correlation_id}] No valid product links to store")
                    return True

            except Exception as e:
                logger.error(f"[PRODUCT-CONTEXT] [{correlation_id}] Failed to store product link context: {type(e).__name__}: {e}")
                return False

    def get_product_link_context(self, phone_number: str, session_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve active product link context for a conversation.

        Args:
            phone_number: Phone number for the conversation
            session_id: Optional session ID to filter by

        Returns:
            List of active product link context records
        """
        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            try:
                logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Retrieving product link context for phone: {phone_number}")

                # Build query for active context (not expired)
                query = self.client.table("product_link_context")\
                    .select("*")\
                    .eq("phone_number", phone_number)\
                    .gte("context_expires_at", datetime.now(timezone.utc).isoformat())\
                    .order("displayed_at", desc=True)

                # Add session filter if provided
                if session_id:
                    query = query.eq("conversation_session_id", session_id)

                result = query.execute()

                context_records = result.data or []
                logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Retrieved {len(context_records)} active product link contexts")

                return context_records

            except Exception as e:
                logger.error(f"[PRODUCT-CONTEXT] [{correlation_id}] Failed to retrieve product link context: {type(e).__name__}: {e}")
                return []

    def cleanup_expired_context(self, phone_number: Optional[str] = None) -> int:
        """
        Clean up expired product link context records.

        Args:
            phone_number: Optional phone number to clean up context for

        Returns:
            int: Number of records cleaned up
        """
        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            try:
                logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Cleaning up expired product link context")

                # First get expired records to count them
                query = self.client.table("product_link_context")\
                    .select("id")\
                    .lt("context_expires_at", datetime.now(timezone.utc).isoformat())

                # Add phone filter if provided
                if phone_number:
                    query = query.eq("phone_number", phone_number)

                expired_result = query.execute()
                expired_records = expired_result.data or []

                if not expired_records:
                    logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] No expired records found")
                    return 0

                # Delete expired records
                delete_query = self.client.table("product_link_context")\
                    .delete()\
                    .lt("context_expires_at", datetime.now(timezone.utc).isoformat())

                # Add phone filter if provided
                if phone_number:
                    delete_query = delete_query.eq("phone_number", phone_number)

                delete_result = delete_query.execute()

                deleted_count = len(expired_records)
                logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Cleaned up {deleted_count} expired product link contexts")

                return deleted_count

            except Exception as e:
                logger.error(f"[PRODUCT-CONTEXT] [{correlation_id}] Failed to cleanup expired context: {type(e).__name__}: {e}")
                return 0

    def format_product_context_for_prompt(self, context_records: List[Dict[str, Any]]) -> str:
        """
        Format product link context for inclusion in LLM prompts.

        Args:
            context_records: List of product link context records

        Returns:
            str: Formatted context string for LLM prompt
        """
        if not context_records:
            return ""

        context_lines = []
        context_lines.append("Produse discutate anterior în conversație:")

        for i, record in enumerate(context_records[:5], 1):  # Limit to last 5 products
            product_name = record.get("product_name", "Produs necunoscut")
            product_url = record.get("product_url", "")
            product_code = record.get("product_code", "")

            context_line = f"{i}. {product_name}"
            if product_code:
                context_line += f" (cod: {product_code})"
            if product_url:
                context_line += f" - {product_url}"

            context_lines.append(context_line)

        context_lines.append("")  # Add spacing
        return "\n".join(context_lines)

    def reset_conversation_context(self, phone_number: str, session_id: Optional[int] = None) -> bool:
        """
        Reset/clear product link context for a conversation.
        This is useful when conversations are reset or need a fresh start.

        Args:
            phone_number: Phone number for the conversation
            session_id: Optional specific session ID to reset

        Returns:
            bool: True if successful, False otherwise
        """
        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            try:
                logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Resetting product link context for phone: {phone_number}")

                # Delete context records for this conversation
                query = self.client.table("product_link_context").delete()

                # Add filters
                if session_id:
                    query = query.eq("conversation_session_id", session_id)
                else:
                    query = query.eq("phone_number", phone_number)

                result = query.execute()

                # Count deleted records (we'll estimate based on phone filter)
                deleted_count = len(result.data or []) if result.data else 0
                logger.info(f"[PRODUCT-CONTEXT] [{correlation_id}] Reset {deleted_count} product link contexts for phone: {phone_number}")

                return True

            except Exception as e:
                logger.error(f"[PRODUCT-CONTEXT] [{correlation_id}] Failed to reset conversation context: {type(e).__name__}: {e}")
                return False


# Global Supabase client instance
supabase_client = SupabaseClient()