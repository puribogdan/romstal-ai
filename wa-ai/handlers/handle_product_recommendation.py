import logging
import json
import traceback
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone

# Import existing database functions
from app.supa import supabase_client

# Import AI integration
from app.llm import llm_client

# Import settings and outbound client
from app.settings import settings
from app.outbound import n8n_client

# Import centralized prompt management
from app.prompts import PromptManager

# Import correlation tracking
from app.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    log_with_correlation,
    CorrelationContext
)

logger = logging.getLogger(__name__)


class ProductRecommendationHandler:
    """Handler for processing product recommendation requests using LLM tools."""

    def __init__(self):
        self.db_client = supabase_client
        self.llm_client = llm_client
        self.n8n_client = n8n_client

    async def handle_product_recommendation(
        self,
        insert_id: int,
        phone_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main handler for product recommendation requests.

        Args:
            insert_id: Database insert ID of the message
            phone_number: Optional phone number override

        Returns:
            Dict with processing results and status
        """
        # Generate correlation ID for this request
        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            try:
                log_with_correlation(
                    logger.info,
                    f"Starting product recommendation processing for insert_id: {insert_id}",
                    correlation_id
                )

                # Step 1: Extract phone number from message using insert_id
                log_with_correlation(
                    logger.info,
                    f"Step 1: Extracting phone number for insert_id: {insert_id}",
                    correlation_id
                )
                phone = await self._extract_phone_number(insert_id, phone_number)
                if not phone:
                    error_msg = "Could not extract phone number from message"
                    log_with_correlation(
                        logger.error,
                        f"Step 1 failed: {error_msg} for insert_id: {insert_id}",
                        correlation_id
                    )
                    return {
                        "success": False,
                        "error": error_msg,
                        "correlation_id": correlation_id,
                        "step": "phone_extraction"
                    }

                log_with_correlation(
                    logger.info,
                    f"Step 1 completed: Extracted phone number {phone} for insert_id: {insert_id}",
                    correlation_id
                )

                # Step 2: Get message content and context
                log_with_correlation(
                    logger.info,
                    f"Step 2: Getting message content and context for insert_id: {insert_id}",
                    correlation_id
                )

                message_data = await self._get_message_and_context(insert_id, phone)
                if not message_data:
                    error_msg = "Could not get message content or context"
                    log_with_correlation(
                        logger.error,
                        f"Step 2 failed: {error_msg} for insert_id: {insert_id}",
                        correlation_id
                    )
                    return {
                        "success": False,
                        "error": error_msg,
                        "correlation_id": correlation_id,
                        "step": "message_context_retrieval"
                    }

                current_message = message_data["current_message"]
                conversation_context = message_data["conversation_context"]

                log_with_correlation(
                    logger.info,
                    f"Step 2 completed: Retrieved message content for insert_id: {insert_id}",
                    correlation_id
                )

                # Step 3: Generate AI response with product recommendations
                log_with_correlation(
                    logger.info,
                    f"Step 3: Generating AI response with product recommendations for phone: {phone}",
                    correlation_id
                )

                ai_response, function_calls = await self._generate_ai_response_with_tools(
                    phone, current_message, conversation_context
                )

                if not ai_response:
                    error_msg = "Failed to generate AI response"
                    log_with_correlation(
                        logger.error,
                        f"Step 3 failed: {error_msg} for phone: {phone}",
                        correlation_id
                    )
                    return {
                        "success": False,
                        "error": error_msg,
                        "correlation_id": correlation_id,
                        "step": "ai_response_generation"
                    }

                log_with_correlation(
                    logger.info,
                    f"Step 3 completed: Generated AI response for phone: {phone}",
                    correlation_id
                )

                # Step 4: Send response via N8N webhook
                log_with_correlation(
                    logger.info,
                    f"Step 4: Sending response via N8N for phone: {phone}",
                    correlation_id
                )
                n8n_result = await self._send_n8n_response(phone, ai_response)
                log_with_correlation(
                    logger.info,
                    f"Step 4 completed: N8N response result received for phone: {phone}",
                    correlation_id
                )

                # Step 5: Extract and store product link context
                log_with_correlation(
                    logger.info,
                    f"Step 5: Extracting and storing product link context for phone: {phone}",
                    correlation_id
                )

                # Extract product links from the AI response
                product_links = self._extract_product_links_from_response(ai_response, function_calls)

                # Store product link context if we have a valid session and product links
                if product_links:
                    try:
                        # Get or create conversation session for context storage
                        session = self.db_client.get_or_create_conversation_session(phone, "")
                        if session:
                            context_stored = self.db_client.store_product_link_context(
                                session_id=int(session["id"]),
                                phone_number=phone,
                                product_links=product_links,
                                message_insert_id=insert_id
                            )
                            if context_stored:
                                log_with_correlation(
                                    logger.info,
                                    f"Step 5: Successfully stored {len(product_links)} product link contexts",
                                    correlation_id
                                )
                            else:
                                log_with_correlation(
                                    logger.warning,
                                    f"Step 5: Failed to store product link contexts",
                                    correlation_id
                                )
                        else:
                            log_with_correlation(
                                logger.warning,
                                f"Step 5: Could not get/create conversation session for context storage",
                                correlation_id
                            )
                    except Exception as e:
                        log_with_correlation(
                            logger.error,
                            f"Step 5: Error storing product link context: {e}",
                            correlation_id
                        )

                log_with_correlation(
                    logger.info,
                    f"Step 5 completed: Product link context processed for phone: {phone}",
                    correlation_id
                )

                # Step 6: Save outbound message to database
                log_with_correlation(
                    logger.info,
                    f"Step 6: Saving outbound message for phone: {phone}",
                    correlation_id
                )
                db_result = await self._save_outbound_message(phone, ai_response, {
                    "source": "product_recommendation_handler",
                    "insert_id": insert_id,
                    "n8n_result": n8n_result,
                    "function_calls": function_calls,
                    "correlation_id": correlation_id,
                    "product_links_count": len(product_links) if product_links else 0
                })
                log_with_correlation(
                    logger.info,
                    f"Step 6 completed: Successfully saved outbound message for phone: {phone}",
                    correlation_id
                )

                log_with_correlation(
                    logger.info,
                    f"Successfully processed product recommendation request for {phone} with correlation_id: {correlation_id}",
                    correlation_id
                )

                return {
                    "success": True,
                    "phone": phone,
                    "ai_response": ai_response,
                    "n8n_result": n8n_result,
                    "db_result": db_result,
                    "function_calls": function_calls,
                    "correlation_id": correlation_id
                }

            except Exception as e:
                log_with_correlation(
                    logger.error,
                    f"Critical error processing product recommendation: {e}",
                    correlation_id,
                    exc_info=True
                )
                return {
                    "success": False,
                    "error": str(e),
                    "correlation_id": correlation_id,
                    "step": "critical_error",
                    "traceback": traceback.format_exc()
                }

    async def _extract_phone_number(self, insert_id: int, phone_override: Optional[str] = None) -> Optional[str]:
        """Extract phone number from message using insert_id."""
        try:
            if phone_override:
                return phone_override

            # Get message by insert_id
            message = self.db_client.get_message_by_id(insert_id)
            if not message:
                logger.error(f"[PRODUCT-REC] Message not found for insert_id: {insert_id}")
                return None

            phone = message.get("wa_from")
            if not phone:
                logger.error(f"[PRODUCT-REC] Phone number missing in message {insert_id}")
                return None

            logger.info(f"[PRODUCT-REC] Extracted phone number: {phone}")
            return phone

        except Exception as e:
            logger.error(f"[PRODUCT-REC] Error extracting phone number: {e}")
            return None

    async def _get_message_and_context(self, insert_id: int, phone: str) -> Optional[Dict[str, Any]]:
        """Get current message and conversation context."""
        try:
            # Get the current message
            message = self.db_client.get_message_by_id(insert_id)
            if not message:
                logger.error(f"[PRODUCT-REC] Current message not found for insert_id: {insert_id}")
                return None

            current_message = message.get("wa_text", "").strip()
            if not current_message:
                logger.error(f"[PRODUCT-REC] Current message is empty for insert_id: {insert_id}")
                return None

            # Get conversation history for context
            history = self.db_client.get_message_history(phone)
            recent_messages = []

            # Process last 10 messages for context
            for msg in history[-10:]:
                text = (msg.get("wa_text") or "").strip()
                if text:
                    role = "agent" if msg.get("direction") == "outbound" else "user"
                    recent_messages.append({
                        "role": role,
                        "content": text,
                        "timestamp": msg.get("wa_timestamp") or msg.get("inserted_at")
                    })

            # Add current message if not already in recent messages
            current_in_context = any(msg["content"] == current_message for msg in recent_messages)
            if not current_in_context:
                current_timestamp = message.get("wa_timestamp") or message.get("inserted_at")
                recent_messages.append({
                    "role": "user",
                    "content": current_message,
                    "timestamp": current_timestamp
                })

            return {
                "current_message": current_message,
                "conversation_context": recent_messages
            }

        except Exception as e:
            logger.error(f"[PRODUCT-REC] Error getting message and context: {e}")
            return None

    async def _generate_ai_response_with_tools(
        self,
        phone: str,
        current_message: str,
        conversation_context: list
    ) -> Tuple[Optional[str], Optional[list]]:
        """Generate AI response using LLM with tools for product recommendations."""
        try:
            logger.info(f"[PRODUCT-REC] Starting AI response generation for phone: {phone}")

            # Build conversation context for the prompt
            context_lines = []
            for msg in conversation_context[-8:]:  # Last 8 messages for context
                context_lines.append(f"{msg['role']}: {msg['content']}")

            hist_context = "\n".join(context_lines)

            # Use the unified system prompt
            system_prompt = PromptManager.get_unified_prompt()

            # Create user prompt with context
            user_prompt = (
                f"Context conversație recentă:\n{hist_context}\n\n"
                f"Mesajul utilizatorului: {current_message}\n\n"
                "Bazat pe mesajul utilizatorului și contextul conversației, generează un răspuns helpful și natural în română. "
                "Dacă utilizatorul cere recomandări de produse sau informații despre categorii specifice, folosește tool-urile disponibile pentru a obține informații actualizate."
            )

            logger.info(f"[PRODUCT-REC] Calling LLM with tools - prompt length: {len(user_prompt)}")

            # Generate AI response using existing LLM client with tools
            ai_response, function_calls = await self.llm_client.call_llm_with_tools(
                system_prompt, user_prompt, get_correlation_id()
            )

            if not ai_response:
                logger.error(f"[PRODUCT-REC] LLM returned empty response for phone: {phone}")
                return None, None

            logger.info(f"[PRODUCT-REC] Generated AI response: {len(ai_response)} characters for phone: {phone}")
            return ai_response, function_calls

        except Exception as e:
            logger.error(f"[PRODUCT-REC] Error generating AI response: {e}")
            return None, None

    async def _send_n8n_response(self, phone: str, message: str) -> Dict[str, Any]:
        """Send response via N8N webhook."""
        try:
            logger.info(f"[PRODUCT-REC] Sending response via N8N for phone: {phone}")

            # Use existing N8N client
            result = await self.n8n_client.send_to_n8n(phone, message)

            logger.info(f"[PRODUCT-REC] N8N response result: {result}")
            return result

        except Exception as e:
            logger.error(f"[PRODUCT-REC] Error sending N8N response: {e}")
            return {"error": str(e)}

    def _extract_product_links_from_response(self, ai_response: str, function_calls: Optional[List] = None) -> List[Dict[str, Any]]:
        """Extract product links and information from AI response and function calls."""
        product_links = []

        try:
            # Extract URLs from the AI response text
            import re
            url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
            urls = re.findall(url_pattern, ai_response)

            # Filter for Romstal URLs and extract product information
            for url in urls:
                if 'romstal.ro' in url:
                    # Try to extract product code from URL if present
                    product_code = self._extract_product_code_from_url(url)

                    # Try to get product name from function calls if available
                    product_name = self._extract_product_name_from_function_calls(function_calls, product_code)

                    product_links.append({
                        "url": url,
                        "code": product_code,
                        "name": product_name
                    })

            logger.info(f"[PRODUCT-REC] Extracted {len(product_links)} product links from response")
            return product_links

        except Exception as e:
            logger.error(f"[PRODUCT-REC] Error extracting product links: {e}")
            return []

    def _extract_product_code_from_url(self, url: str) -> str:
        """Extract product code from Romstal URL if present."""
        try:
            # Look for patterns like /product/code or code=CODE in URL
            import re

            # Pattern 1: /product/code format
            product_pattern = r'/product/([^/?]+)'
            match = re.search(product_pattern, url)
            if match:
                return match.group(1)

            # Pattern 2: code=CODE parameter
            code_pattern = r'[?&]code=([^&]+)'
            match = re.search(code_pattern, url)
            if match:
                return match.group(1)

            return ""
        except Exception:
            return ""

    def _extract_product_name_from_function_calls(self, function_calls: Optional[List], product_code: str) -> str:
        """Extract product name from function call results."""
        if not function_calls or not product_code:
            return ""

        try:
            for call in function_calls:
                if call.get("function") == "fetch_product_details":
                    result = call.get("result", {})
                    if result.get("ok") and result.get("code") == product_code:
                        data = result.get("data", {})
                        if isinstance(data, dict):
                            info = data.get("info", {})
                            return info.get("product", "")

                elif call.get("function") == "search_products_romstal":
                    result = call.get("result", {})
                    if result.get("ok"):
                        products = result.get("products", [])
                        for product in products:
                            if product.get("url", "").find(product_code) != -1:
                                return product.get("name", "")

            return ""
        except Exception:
            return ""

    async def _save_outbound_message(
        self,
        phone: str,
        message: str,
        raw_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Save outbound message to database."""
        try:
            logger.info(f"[PRODUCT-REC] Saving outbound message for phone: {phone}")

            # Use existing database function
            result = self.db_client.insert_outbound_message(phone, message, raw_data)

            logger.info(f"[PRODUCT-REC] Successfully saved outbound message")
            return result

        except Exception as e:
            logger.error(f"[PRODUCT-REC] Error saving outbound message: {e}")
            return {"error": str(e)}


# Global product recommendation handler instance
product_recommendation_handler = ProductRecommendationHandler()