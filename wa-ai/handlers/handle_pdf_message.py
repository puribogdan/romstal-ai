import logging
import json
import traceback
import requests
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone

# Import existing database functions
from app.supa import supabase_client

# Import OCR functions
from integrations.ocr_supabase import (
    download_whatsapp_media,
    supabase_upload_pdf,
    mistral_ocr_from_pdf_bytes,
    extract_markdown
)

# Import AI integration
from app.llm import llm_client

# Import settings and outbound client
from app.settings import settings
from app.outbound import n8n_client

# Import correlation tracking
from app.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    log_with_correlation,
    CorrelationContext
)

logger = logging.getLogger(__name__)


class PDFMessageHandler:
    """Handler for processing PDF messages through OCR and AI response generation."""

    def __init__(self):
        self.db_client = supabase_client
        self.llm_client = llm_client
        self.n8n_client = n8n_client

    async def handle_pdf_message(
        self,
        insert_id: int,
        phone_number: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main handler for PDF messages.

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
                    f"Starting PDF message processing for insert_id: {insert_id}",
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

                # Step 1.5: Send immediate acknowledgment to user
                log_with_correlation(
                    logger.info,
                    f"Step 1.5: Sending immediate acknowledgment to {phone}",
                    correlation_id
                )
                try:
                    await self.n8n_client.send_to_n8n(phone, "Am primit documentul, îl citesc…")
                    log_with_correlation(
                        logger.info,
                        f"Step 1.5 completed: Sent immediate acknowledgment to {phone}",
                        correlation_id
                    )
                except Exception as e:
                    log_with_correlation(
                        logger.warning,
                        f"Step 1.5 failed: Failed to send acknowledgment to {phone}: {e}",
                        correlation_id
                    )
                    # Continue processing even if acknowledgment fails

                # Step 2: Get media ID and access token
                log_with_correlation(
                    logger.info,
                    f"Step 2: Getting media ID and access token for insert_id: {insert_id}",
                    correlation_id
                )

                # Get media ID and token using the minimal database function
                media_info = self.db_client.get_media_id_and_token_by_insert_id(insert_id)
                if not media_info:
                    error_msg = "Could not get media ID and token from database"
                    log_with_correlation(
                        logger.error,
                        f"Step 2 failed: {error_msg} for insert_id: {insert_id}",
                        correlation_id
                    )
                    return {
                        "success": False,
                        "error": error_msg,
                        "correlation_id": correlation_id,
                        "step": "media_info_retrieval"
                    }

                media_id = media_info["media_id"]
                access_token = media_info["access_token"]

                log_with_correlation(
                    logger.info,
                    f"Step 2 completed: Retrieved media_id for insert_id: {insert_id}",
                    correlation_id
                )

                # Step 3: Download and process PDF using existing OCR pipeline
                log_with_correlation(
                    logger.info,
                    f"Step 3: Starting PDF download and processing for insert_id: {insert_id}",
                    correlation_id
                )

                pdf_content = await self._download_and_process_pdf(media_id, access_token, phone, insert_id)

                if not pdf_content or pdf_content.strip() == "":
                    log_with_correlation(
                        logger.warning,
                        f"Step 2 warning: PDF processing returned empty content for {phone}, using fallback message",
                        correlation_id
                    )
                    # Use a more informative fallback message
                    pdf_content = "PDF document received but content extraction failed. This could be due to:\n- Unsupported file format\n- Corrupted file\n- OCR processing errors\n- Expired media (file no longer available on WhatsApp servers)\n\nPlease provide the information as text, upload a different file, or describe what you need help with regarding Romstal products or services."

                # Validate that we have meaningful content
                if len(pdf_content.strip()) < 50:
                    log_with_correlation(
                        logger.warning,
                        f"Step 2 warning: PDF processing returned very short content ({len(pdf_content)} chars) for {phone}",
                        correlation_id
                    )

                # Check if the response indicates expired media and log appropriately
                if "expired" in pdf_content.lower() or "no longer available" in pdf_content.lower():
                    log_with_correlation(
                        logger.info,
                        f"Step 2 completed: PDF processing detected expired media for {phone}, providing user-friendly message",
                        correlation_id
                    )
                    log_with_correlation(
                        logger.info,
                        f"[PDF] [EXPIRED-MEDIA] Successfully handled expired media scenario for insert_id: {insert_id}, phone: {phone}",
                        correlation_id
                    )
                else:
                    log_with_correlation(
                        logger.info,
                        f"Step 2 completed: PDF processing successful for insert_id: {insert_id}, content length: {len(pdf_content)} characters",
                        correlation_id
                    )

                log_with_correlation(
                    logger.info,
                    f"Step 2 completed: PDF processing finished for insert_id: {insert_id}, content length: {len(pdf_content)} characters",
                    correlation_id
                )

                # Step 3: Generate AI response based on PDF content
                log_with_correlation(
                    logger.info,
                    f"Step 3: Generating AI response for phone: {phone}",
                    correlation_id
                )
                ai_response = await self._generate_ai_response(phone, pdf_content)
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

                # Step 5: Save outbound message to database
                log_with_correlation(
                    logger.info,
                    f"Step 5: Saving outbound message for phone: {phone}",
                    correlation_id
                )
                db_result = await self._save_outbound_message(phone, ai_response, {
                    "source": "pdf_handler",
                    "insert_id": insert_id,
                    "n8n_result": n8n_result,
                    "correlation_id": correlation_id
                })
                log_with_correlation(
                    logger.info,
                    f"Step 5 completed: Successfully saved outbound message for phone: {phone}",
                    correlation_id
                )

                log_with_correlation(
                    logger.info,
                    f"Successfully processed PDF message for {phone} with correlation_id: {correlation_id}",
                    correlation_id
                )

                return {
                    "success": True,
                    "phone": phone,
                    "ai_response": ai_response,
                    "n8n_result": n8n_result,
                    "db_result": db_result,
                    "pdf_content_length": len(pdf_content),
                    "correlation_id": correlation_id
                }

            except Exception as e:
                log_with_correlation(
                    logger.error,
                    f"Critical error processing PDF message: {e}",
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
                logger.error(f"[PDF] Message not found for insert_id: {insert_id}")
                return None

            phone = message.get("wa_from")
            if not phone:
                logger.error(f"[PDF] Phone number missing in message {insert_id}")
                return None

            logger.info(f"[PDF] Extracted phone number: {phone}")
            return phone

        except Exception as e:
            logger.error(f"[PDF] Error extracting phone number: {e}")
            return None

    async def _download_and_process_pdf(
        self,
        media_id: str,
        access_token: str,
        phone: str,
        insert_id: int
    ) -> Optional[str]:
        """Download PDF and process through OCR pipeline."""
        correlation_id = get_correlation_id()

        try:
            log_with_correlation(
                logger.info,
                f"[DIAGNOSTIC] Starting PDF download for insert_id: {insert_id}, media_id: {media_id[:20]}...",
                correlation_id
            )

            # Step 2.1: Download PDF bytes from WhatsApp
            log_with_correlation(
                logger.info,
                f"[DIAGNOSTIC] Step 2.1: Downloading media for media_id: {media_id[:20]}... for insert_id: {insert_id}",
                correlation_id
            )

            try:
                pdf_bytes = download_whatsapp_media(media_id, access_token)
                if not pdf_bytes:
                    error_msg = f"Failed to download PDF from WhatsApp - no data received for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.error,
                        f"[DIAGNOSTIC] Step 2.1 failed: {error_msg}",
                        correlation_id
                    )
                    return "PDF document received but download failed. Please try uploading the file again or provide the information as text."

                log_with_correlation(
                    logger.info,
                    f"[DIAGNOSTIC] Step 2.1 completed: Downloaded PDF: {len(pdf_bytes)} bytes for insert_id: {insert_id}",
                    correlation_id
                )

            except ValueError as e:
                # Handle different types of ValueError including expired media
                error_str = str(e)

                # Check if this is an expired media error
                if "Media expired:" in error_str or "expired media" in error_str.lower():
                    error_msg = f"PDF processing failed due to expired media for insert_id: {insert_id}: {e}"
                    log_with_correlation(
                        logger.warning,
                        f"[DIAGNOSTIC] Step 2.1 failed: {error_msg} - Media has expired and is no longer available",
                        correlation_id,
                        exc_info=True
                    )
                    return "PDF document received but the media has expired and is no longer available on WhatsApp servers. Please upload the file again or share a fresh copy of the document."
                else:
                    # PDF validation errors (invalid format, corrupted file, etc.)
                    error_msg = f"PDF validation failed for insert_id: {insert_id}: {e}"
                    log_with_correlation(
                        logger.error,
                        f"[DIAGNOSTIC] Step 2.1 failed: {error_msg}",
                        correlation_id,
                        exc_info=True
                    )
                    return f"PDF document received but file validation failed: {str(e)}. Please ensure you're uploading a valid PDF file."

            except requests.exceptions.Timeout as e:
                error_msg = f"WhatsApp media download timeout for insert_id: {insert_id}"
                log_with_correlation(
                    logger.error,
                    f"[DIAGNOSTIC] Step 2.1 failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return "PDF document received but download timed out. Please try uploading the file again."

            except requests.exceptions.HTTPError as e:
                status_code = getattr(e.response, 'status_code', 'unknown') if hasattr(e, 'response') else 'unknown'
                error_msg = f"WhatsApp media download HTTP error (status: {status_code}) for insert_id: {insert_id}"
                log_with_correlation(
                    logger.error,
                    f"[DIAGNOSTIC] Step 2.1 failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return f"PDF document received but download failed with HTTP error {status_code}. Please try again or contact support if the issue persists."

            except requests.exceptions.RequestException as e:
                error_msg = f"WhatsApp media download request error for insert_id: {insert_id}"
                log_with_correlation(
                    logger.error,
                    f"[DIAGNOSTIC] Step 2.1 failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return "PDF document received but download failed due to network issues. Please try uploading the file again."

            except Exception as e:
                error_msg = f"Unexpected error during WhatsApp media download for insert_id: {insert_id}"
                log_with_correlation(
                    logger.error,
                    f"[DIAGNOSTIC] Step 2.1 failed: {error_msg} - {e}",
                    correlation_id,
                    exc_info=True
                )
                return "PDF document received but download encountered an unexpected error. Please try again or provide the information as text."

            # Step 2.2: Upload PDF to Supabase storage
            log_with_correlation(
                logger.info,
                f"[DIAGNOSTIC] Step 2.2: Uploading PDF to Supabase storage for insert_id: {insert_id}",
                correlation_id
            )

            try:
                storage_path, signed_url = supabase_upload_pdf(pdf_bytes, phone, str(insert_id))

                # Check if upload was successful
                if not storage_path:
                    error_msg = f"Supabase upload failed - no storage path returned for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.error,
                        f"[DIAGNOSTIC] Step 2.2 failed: {error_msg}",
                        correlation_id
                    )
                    return "PDF document processed but storage upload failed. Content will still be analyzed from local cache."

                log_with_correlation(
                    logger.info,
                    f"[DIAGNOSTIC] Step 2.2 completed: Uploaded PDF to storage: {storage_path} for insert_id: {insert_id}",
                    correlation_id
                )

                # Log signed URL status (it might be empty if generation failed, but that's not critical)
                if signed_url:
                    log_with_correlation(
                        logger.info,
                        f"[DIAGNOSTIC] Step 2.2: Signed URL generated successfully for insert_id: {insert_id}",
                        correlation_id
                    )
                else:
                    log_with_correlation(
                        logger.warning,
                        f"[DIAGNOSTIC] Step 2.2: Signed URL generation failed for insert_id: {insert_id}, but upload succeeded",
                        correlation_id
                    )

            except Exception as e:
                error_msg = f"Supabase storage upload error for insert_id: {insert_id}: {e}"
                log_with_correlation(
                    logger.error,
                    f"[DIAGNOSTIC] Step 2.2 failed: {error_msg}",
                    correlation_id,
                    exc_info=True
                )
                return "PDF document processed but storage upload failed. Content will still be analyzed from local cache."

            # Step 2.3: Process PDF through Mistral OCR
            log_with_correlation(
                logger.info,
                f"[DIAGNOSTIC] Step 2.3: Starting Mistral OCR processing for insert_id: {insert_id}",
                correlation_id
            )

            try:
                ocr_result = mistral_ocr_from_pdf_bytes(pdf_bytes)

                if not ocr_result:
                    error_msg = f"OCR processing failed or returned empty result for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.error,
                        f"[DIAGNOSTIC] Step 2.3 failed: {error_msg}",
                        correlation_id
                    )
                    return "PDF document received but OCR processing failed to extract content. This could be due to:\n- Complex document layout\n- Poor image quality\n- Unsupported content type\n\nPlease describe the content or provide it as text for better assistance."

                # Validate OCR result structure
                if not isinstance(ocr_result, dict):
                    error_msg = f"OCR processing returned invalid format (expected dict, got {type(ocr_result)}) for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.error,
                        f"[DIAGNOSTIC] Step 2.3 failed: {error_msg}",
                        correlation_id
                    )
                    return "PDF document received but OCR processing returned invalid data format. Please try again or provide the content as text."

                # Check for pages in OCR result
                if "pages" not in ocr_result:
                    error_msg = f"OCR processing missing 'pages' field in result for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.error,
                        f"[DIAGNOSTIC] Step 2.3 failed: {error_msg}",
                        correlation_id
                    )
                    return "PDF document received but OCR processing failed to identify pages. Please try again or provide the content as text."

                pages = ocr_result.get("pages", [])
                if not pages or len(pages) == 0:
                    error_msg = f"OCR processing found no pages in document for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.error,
                        f"[DIAGNOSTIC] Step 2.3 failed: {error_msg}",
                        correlation_id
                    )
                    return "PDF document received but appears to be empty or unreadable. Please ensure the PDF contains visible content and try again."

                log_with_correlation(
                    logger.info,
                    f"[DIAGNOSTIC] Step 2.3 completed: OCR processing completed for insert_id: {insert_id}, found {len(pages)} pages",
                    correlation_id
                )

            except Exception as e:
                error_msg = f"Mistral OCR processing error for insert_id: {insert_id}: {e}"
                log_with_correlation(
                    logger.error,
                    f"[DIAGNOSTIC] Step 2.3 failed: {error_msg}",
                    correlation_id,
                    exc_info=True
                )
                return "PDF document received but OCR processing encountered an error. Please describe the content or provide it as text for assistance with Romstal products or services."

            # Step 2.4: Extract markdown content from OCR result
            log_with_correlation(
                logger.info,
                f"[DIAGNOSTIC] Step 2.4: Extracting markdown content for insert_id: {insert_id}",
                correlation_id
            )

            try:
                pages, markdown_content = extract_markdown(ocr_result)

                if not markdown_content or not markdown_content.strip():
                    error_msg = f"OCR returned empty or invalid content for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.warning,
                        f"[DIAGNOSTIC] Step 2.4 warning: {error_msg}",
                        correlation_id
                    )
                    return "PDF document received but no readable content could be extracted. This might be due to:\n- Scanned images without text\n- Very small or illegible text\n- Encrypted or password-protected PDF\n\nPlease ensure the PDF contains selectable text, or describe what you need help with regarding Romstal products or services."

                # Validate content quality
                if len(markdown_content.strip()) < 20:
                    error_msg = f"OCR extracted very little content ({len(markdown_content)} chars) for insert_id: {insert_id}"
                    log_with_correlation(
                        logger.warning,
                        f"[DIAGNOSTIC] Step 2.4 warning: {error_msg}",
                        correlation_id
                    )
                    return f"PDF document processed but only minimal content extracted. Please review and provide additional details if needed:\n\n{markdown_content}"

                # Check for meaningful text content (not just symbols/punctuation)
                import re
                text_only = re.sub(r'[^\w\săâîșțĂÂÎȘȚ]', '', markdown_content)
                if len(text_only.strip()) < 10:
                    log_with_correlation(
                        logger.warning,
                        f"[DIAGNOSTIC] Step 2.4 warning: Extracted content has very little readable text for insert_id: {insert_id}",
                        correlation_id
                    )
                    return f"PDF document processed but content appears to be mostly non-text elements. Extracted content:\n\n{markdown_content}\n\nIf this doesn't contain the information you need, please describe what you're looking for."

                log_with_correlation(
                    logger.info,
                    f"[DIAGNOSTIC] Step 2.4 completed: Extracted markdown content: {len(markdown_content)} characters, {len(pages)} pages for insert_id: {insert_id}",
                    correlation_id
                )

                # Step 2.5: Persist OCR results to database
                log_with_correlation(
                    logger.info,
                    f"[DIAGNOSTIC] Step 2.5: Persisting OCR results for insert_id: {insert_id}",
                    correlation_id
                )

                try:
                    # Import the insert function from supabase client
                    from app.supa import supabase_client

                    # Prepare OCR data for persistence
                    ocr_result_data = {
                        "message_insert_id": insert_id,
                        "phone_number": phone,
                        "supabase_path": storage_path,
                        "pages_json": pages,
                        "full_markdown": markdown_content
                    }

                    # Insert OCR result into database
                    db_result = supabase_client.insert_ocr_result(**ocr_result_data)

                    log_with_correlation(
                        logger.info,
                        f"[DIAGNOSTIC] Step 2.5 completed: Successfully persisted OCR results for insert_id: {insert_id}, record ID: {db_result.get('id')}",
                        correlation_id
                    )

                except Exception as e:
                    log_with_correlation(
                        logger.warning,
                        f"[DIAGNOSTIC] Step 2.5 failed: Failed to persist OCR results for insert_id: {insert_id}: {e}",
                        correlation_id
                    )
                    # Continue processing even if OCR persistence fails

                return markdown_content

            except Exception as e:
                error_msg = f"Error extracting markdown content for insert_id: {insert_id}: {e}"
                log_with_correlation(
                    logger.error,
                    f"[DIAGNOSTIC] Step 2.4 failed: {error_msg}",
                    correlation_id,
                    exc_info=True
                )
                return "PDF document received but content extraction encountered an error. Please describe what you need help with or provide the information as text."

        except Exception as e:
            log_with_correlation(
                logger.error,
                f"[DIAGNOSTIC] Critical error in PDF download/processing: {e}",
                correlation_id,
                exc_info=True
            )
            return None

    async def _generate_ai_response(self, phone: str, pdf_content: str) -> Optional[str]:
        """Generate AI response based on PDF content."""
        try:
            logger.info(f"[PDF] [DIAGNOSTIC] Starting AI response generation for phone: {phone}")

            # Get message history for context
            logger.info(f"[PDF] [DIAGNOSTIC] Fetching message history for phone: {phone}")
            history = self.db_client.get_message_history(phone)
            recent_messages = []
            for msg in history[-5:]:  # Last 5 messages for context
                text = (msg.get("wa_text") or "").strip()
                if text:
                    role = "agent" if msg.get("direction") == "outbound" else "user"
                    recent_messages.append(f"{role}: {text}")

            logger.info(f"[PDF] [DIAGNOSTIC] Found {len(recent_messages)} recent messages for context")

            # Build context with PDF content
            context = "Conținut PDF extras:\n" + pdf_content + "\n\n"
            if recent_messages:
                context += "Context conversație recentă:\n" + "\n".join(recent_messages) + "\n\n"

            # Create system prompt for PDF processing
            system_prompt = (
                "Ești un asistent Romstal prietenos și util pe WhatsApp.\n"
                "Utilizatorul ți-a trimis un PDF. Analizează conținutul și oferă informații relevante.\n"
                "Răspunde în română, natural și conversațional.\n"
                "Poți purta discuții casual și răspunde la întrebări personale simple, dar rolul tău principal este să ajuți utilizatorii cu informații despre Romstal, produse, servicii, program, livrare și alte detalii utile.\n"
                "Fii prietenos, natural și adaptabil.\n"
                "Important:\n"
                "- Nu propune acțiuni precum adăugarea produselor în stoc, efectuarea comenzilor, programări sau alte procese operative.\n"
                "- Nu face follow-up pentru a oferi servicii sau a iniția alte conversații.\n"
                "- Poți face follow-up doar despre produsul sau subiectul discutat (ex: recomandări similare, specificații, întreținere, garanție etc.).\n"
                "- Menține un ton profesionist, empatic și prietenos, ca un consultant Romstal care vorbește relaxat, dar informat.\n"
            )

            # Generate AI response using existing LLM client
            user_prompt = f"{context}\nBazat pe conținutul PDF și contextul conversației, generează un răspuns helpful și natural în română."

            logger.info(f"[PDF] [DIAGNOSTIC] Calling LLM with system_prompt length: {len(system_prompt)}, user_prompt length: {len(user_prompt)}")
            ai_response, _ = await self.llm_client.call_llm_with_tools(system_prompt, user_prompt)

            if not ai_response:
                logger.error(f"[PDF] [DIAGNOSTIC] LLM returned empty response for phone: {phone}")
                return None

            logger.info(f"[PDF] [DIAGNOSTIC] Generated AI response: {len(ai_response)} characters for phone: {phone}")
            return ai_response

        except Exception as e:
            logger.error(f"[PDF] Error generating AI response: {e}")
            return None

    async def _send_n8n_response(self, phone: str, message: str) -> Dict[str, Any]:
        """Send response via N8N webhook."""
        try:
            logger.info(f"[PDF] Sending response via N8N for phone: {phone}")

            # Use existing N8N client
            result = await self.n8n_client.send_to_n8n(phone, message)

            logger.info(f"[PDF] N8N response result: {result}")
            return result

        except Exception as e:
            logger.error(f"[PDF] Error sending N8N response: {e}")
            return {"error": str(e)}

    async def _save_outbound_message(
        self,
        phone: str,
        message: str,
        raw_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Save outbound message to database."""
        try:
            logger.info(f"[PDF] Saving outbound message for phone: {phone}")

            # Use existing database function
            result = self.db_client.insert_outbound_message(phone, message, raw_data)

            logger.info(f"[PDF] Successfully saved outbound message")
            return result

        except Exception as e:
            logger.error(f"[PDF] Error saving outbound message: {e}")
            return {"error": str(e)}


# Global PDF message handler instance
pdf_handler = PDFMessageHandler()