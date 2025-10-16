import logging
import json
from typing import Optional, List, Tuple
from .settings import settings
from .correlation import generate_correlation_id, CorrelationContext
from .prompts import PromptManager

logger = logging.getLogger(__name__)


class LLMClient:
    """OpenAI LLM client with tools support using Responses API."""

    # OpenAI Tool schema (Responses API) - CORRECTED FOR GPT-5
    OPENAI_TOOLS = [
        {
            "type": "function",
            "name": "fetch_product_details",
            "description": (
                "Use ONLY when the user provides a clear Romstal product code "
                "(ex: 64px9822). If the message lacks a code, do NOT call this; "
                "ask for the code or consider web search if they want docs/info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Exact code string provided by the user (no extra text)."
                    }
                },
                "required": ["code"],
                "additionalProperties": False
            }
        },
        {
            "type": "web_search", 
             "user_location": {  # Correct parameter name
            "type": "approximate",  # Required field
            "city": "Bucharest",
            "country": "RO",         # Required (ISO country code)
            "timezone": "Europe/Bucharest"  # Optional
        },
             
            # Optional: can add filters if needed
             "filters": {"allowed_domains": ["www.romstal.ro", "romstal.ro", "romstalpartener.ro" , "www.romstalpartener.ro", "shop.romstal.ro", "romstal.md"]},
            #"filters": {"allowed_domains": ["emag.ro", "www.emag.ro"]},
            "search_context_size": "medium"
        }
    ]

    def _extract_text_from_responses(self, resp) -> Optional[str]:
        """
        Extract text from OpenAI Responses API response.
        For web_search, the structure is:
        1. message (initial reasoning)
        2. web_search_call (status: completed)
        3. message (final answer with citations)
        We want the LAST message content.
        """
        logger.info("[DEBUG] ===== TEXT EXTRACTION DEBUG =====")

        # 1) Try direct output_text first (fastest path)
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            logger.info(f"[DEBUG] Found output_text directly: {len(txt)} chars")
            logger.info("[DEBUG] ===== TEXT EXTRACTION END (DIRECT) =====")
            return txt.strip()

        # 2) Extract from output array - get LAST message item
        output = getattr(resp, "output", None)
        if isinstance(output, list) and len(output) > 0:
            logger.info(f"[DEBUG] Processing {len(output)} output items")
            
            # Find all message items (type == "message")
            message_items = []
            for i, item in enumerate(output):
                item_type = getattr(item, "type", None)
                logger.info(f"[DEBUG] Item {i}: type={item_type}, id={getattr(item, 'id', 'N/A')}")
                
                if item_type == "message":
                    message_items.append((i, item))
            
            # Get the LAST message item (contains final answer after web search)
            if message_items:
                last_idx, last_msg = message_items[-1]
                logger.info(f"[DEBUG] Using LAST message item at index {last_idx}")
                
                content = getattr(last_msg, "content", None)
                if isinstance(content, list):
                    # Collect all text parts from this message
                    text_parts = []
                    for j, content_item in enumerate(content):
                        content_type = getattr(content_item, "type", None)
                        
                        if content_type in ["output_text", "text"]:
                            text = getattr(content_item, "text", None)
                            if isinstance(text, str) and text.strip():
                                logger.info(f"[DEBUG] Found text in content item {j}: {len(text)} chars")
                                text_parts.append(text.strip())
                    
                    if text_parts:
                        result = "\n\n".join(text_parts)
                        logger.info(f"[DEBUG] Extracted {len(text_parts)} text parts, total {len(result)} chars")
                        logger.info(f"[DEBUG] Text preview: {result[:200]}...")
                        logger.info("[DEBUG] ===== TEXT EXTRACTION END (SUCCESS) =====")
                        return result

        # 3) Fallback: search entire response structure
        logger.warning("[DEBUG] Standard extraction failed, trying fallback...")
        try:
            payload = resp.model_dump()
            
            def find_all_texts(obj, path=""):
                """Recursively find all text fields."""
                texts = []
                
                if isinstance(obj, dict):
                    # Check for text field
                    if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip():
                        texts.append(obj["text"].strip())
                    
                    # Recurse into all values
                    for key, value in obj.items():
                        texts.extend(find_all_texts(value, f"{path}.{key}" if path else key))
                        
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        texts.extend(find_all_texts(item, f"{path}[{i}]"))
                
                return texts
            
            all_texts = find_all_texts(payload)
            if all_texts:
                # Return the LAST text found (most likely the final answer)
                result = all_texts[-1]
                logger.info(f"[DEBUG] Fallback found {len(all_texts)} texts, using last one: {len(result)} chars")
                logger.info("[DEBUG] ===== TEXT EXTRACTION END (FALLBACK) =====")
                return result
            
            logger.error("[OpenAI] No text found in response")
            logger.error(f"[DEBUG] Response keys: {list(payload.keys())}")
            
        except Exception as e:
            logger.exception(f"[OpenAI] Fallback extraction failed: {e}")
        
        logger.info("[DEBUG] ===== TEXT EXTRACTION END (FAILED) =====")
        return None

    def _extract_tool_calls_from_response(self, resp):
        """
        Extracts all tool calls from Responses API output.
        Returns list of dicts: {"id": <call_id>, "type": <call_type>, "name": <function_name>, "args": <dict>}
        """
        calls = []
        output = getattr(resp, "output", []) or []

        logger.info(f"[DEBUG] Extracting tool calls from {len(output)} output items")

        for i, item in enumerate(output):
            item_type = getattr(item, "type", None)
            item_id = getattr(item, "id", None)

            # Handle function calls
            if item_type == "function_call":
                name = getattr(item, "name", None)
                if name:
                    raw_args = getattr(item, "arguments", {}) or {}
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception as e:
                        logger.warning(f"[DEBUG] Failed to parse function args: {e}")
                        args = {}

                    logger.info(f"[DEBUG] Found function call: {name} with args: {args}")
                    calls.append({
                        "id": getattr(item, "call_id", item_id),
                        "type": "function_call",
                        "name": name,
                        "args": args
                    })

            # Handle web search calls
            elif item_type == "web_search_call":
                logger.info(f"[DEBUG] Found web_search_call with status: {getattr(item, 'status', 'unknown')}")
                
                # Web search is handled internally by OpenAI
                # We just log it for tracking
                calls.append({
                    "id": item_id,
                    "type": "web_search_call",
                    "name": "web_search",
                    "args": {
                        "status": getattr(item, "status", "unknown")
                    }
                })

        logger.info(f"[DEBUG] Extracted {len(calls)} total tool calls")
        return calls

    async def call_llm_with_tools(self, system_prompt: str, user_prompt: str, correlation_id: Optional[str] = None) -> Tuple[str, Optional[List]]:
        """
        Call LLM with tools support using Responses API.
        Returns: (reply_text, function_call_details)
        """
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            logger.info(f"[OpenAI] [{correlation_id}] Starting LLM call with tools")

            if not self.client:
                logger.warning(f"[OpenAI] [{correlation_id}] Client not initialized")
                return "Salut! Sunt asistentul Romstal. Cu ce te pot ajuta?", None

            try:
                # Make API call with tools
                logger.info(f"[OpenAI] [{correlation_id}] Making Responses API call")
                
                response = self.client.responses.create(
                    model=settings.openai_model,
                    input=[
                        {"role": "system", "content": PromptManager.get_unified_prompt()},
                        {"role": "user", "content": user_prompt}
                    ],
                    tools=self.OPENAI_TOOLS,
                    tool_choice="auto",
                    max_output_tokens=16000,
                    reasoning={"effort": "low"}
                )

                logger.info(f"[DEBUG] Response ID: {response.id}")
                logger.info(f"[DEBUG] Response status: {getattr(response, 'status', 'unknown')}")

                # Log full response structure for debugging
                try:
                    resp_dump = response.model_dump()
                    logger.info(f"[DEBUG] Response output items: {len(resp_dump.get('output', []))}")
                    for i, item in enumerate(resp_dump.get('output', [])):
                        logger.info(f"[DEBUG] Output[{i}]: type={item.get('type')}, id={item.get('id')}")
                except Exception as e:
                    logger.warning(f"[DEBUG] Could not dump response: {e}")

                # Extract tool calls
                tool_calls = self._extract_tool_calls_from_response(response)
                function_call_history = []

                # Check for function calls that need execution
                has_function_calls = any(tc["type"] == "function_call" for tc in tool_calls)
                has_web_search = any(tc["type"] == "web_search_call" for tc in tool_calls)

                if has_function_calls:
                    logger.info(f"[OpenAI] [{correlation_id}] Found function calls, executing...")
                    
                    # Execute function calls
                    tool_outputs = []
                    for call in tool_calls:
                        if call["type"] == "function_call":
                            function_name = call["name"]
                            args = call["args"]
                            
                            logger.info(f"[OpenAI] Executing {function_name} with args: {args}")
                            
                            if function_name == "fetch_product_details":
                                result = await self.tool_fetch_product_details(args.get("code", ""))
                                tool_outputs.append({
                                    "tool_call_id": call["id"],
                                    "output": result
                                })
                                function_call_history.append({
                                    "function": function_name,
                                    "args": args,
                                    "result": result
                                })

                    # Make follow-up call with function results
                    if tool_outputs:
                        logger.info(f"[OpenAI] [{correlation_id}] Making follow-up call with function results")
                        
                        follow_up_input = [
                            {"role": "system", "content": PromptManager.get_unified_prompt()},
                            {"role": "user", "content": user_prompt}
                        ]
                        
                        for output in tool_outputs:
                            follow_up_input.append({
                                "type": "function_call_output",
                                "call_id": output["tool_call_id"],
                                "output": json.dumps(output["output"], ensure_ascii=False)
                            })
                        
                        follow_up_response = self.client.responses.create(
                            model=settings.openai_model,
                            input=follow_up_input,  # type: ignore
                            tools=self.OPENAI_TOOLS,
                            previous_response_id=response.id,
                            max_output_tokens=16000,
                            reasoning={"effort": "low"}
                        )
                        
                        # Extract text from follow-up response
                        final_text = self._extract_text_from_responses(follow_up_response)
                        if final_text:
                            return final_text, function_call_history
                
                # For web_search or no tool calls, extract text from initial response
                logger.info(f"[OpenAI] [{correlation_id}] Extracting text from response (web_search: {has_web_search})")
                final_text = self._extract_text_from_responses(response)
                
                if final_text:
                    return final_text, function_call_history if function_call_history else None
                
                # If no text extracted, return error
                logger.error(f"[OpenAI] [{correlation_id}] Failed to extract any text from response")
                return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou.", None

            except Exception as e:
                logger.exception(f"[OpenAI] [{correlation_id}] Exception in call_llm_with_tools: {e}")
                return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou.", None

    def __init__(self):
        """Initialize the LLM client."""
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            # Reload environment variables in case we're in a subprocess
            from dotenv import load_dotenv
            import os
            load_dotenv()

            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.error("[OpenAI] OpenAI API key not found in environment")
                return

            self.client = OpenAI(api_key=api_key)
            logger.info("[OpenAI] Client initialized successfully")
        except ImportError:
            logger.error("[OpenAI] OpenAI package not installed")
        except Exception as e:
            logger.exception(f"[OpenAI] Failed to initialize client: {e}")

    async def tool_fetch_product_details(self, code: str) -> str:
        """
        Fetch product details for a given Romstal product code.

        Args:
            code: The product code to look up

        Returns:
            String containing product information or error message
        """
        if not code or not code.strip():
            return "Cod produs invalid sau lipsă."

        correlation_id = generate_correlation_id()
        with CorrelationContext(correlation_id):
            try:
                logger.info(f"[PRODUCT] [{correlation_id}] Fetching details for code: {code}")

                # Here you would implement the actual product lookup logic
                # For now, return a placeholder response
                return f"Am găsit produsul cu codul {code}. Aceasta este o implementare placeholder - conectarea la baza de date Romstal va fi implementată separat."

            except Exception as e:
                logger.exception(f"[PRODUCT] [{correlation_id}] Error fetching product {code}: {e}")
                return f"Eroare la căutarea produsului {code}. Te rog să încerci din nou."


# Global LLM client instance
llm_client = LLMClient()