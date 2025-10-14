import json
import logging
import os
import asyncio
import time
import re
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
from .settings import settings
from .correlation import generate_correlation_id, get_correlation_id, set_correlation_id, CorrelationContext

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.client = None
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

        if settings.openai_api_key:
            try:
                self.client = OpenAI(api_key=settings.openai_api_key)
                logger.info("[OpenAI] Client initialized successfully")
            except Exception as e:
                logger.exception(f"[OpenAI] Failed to initialize client: {type(e).__name__}: {e}")
                self.client = None

    # OpenAI Tool schema (Responses API)
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
            # Built-in tool: do NOT add name/description/parameters here.
            "user_location": {"type": "approximate", "country": "RO", "city": "București"},
            "filters": {"allowed_domains": ["romstal.ro"]}
        }
    ]

    def _log_openai_error(self, context: str, error: Exception) -> None:
        """Helper function to log OpenAI errors consistently."""
        logger.error(f"[OpenAI] {context} failed: {type(error).__name__}: {error}")

    def _validate_tools_schema(self, tools: list) -> bool:
        """Validate that tools schema matches OpenAI Responses API requirements."""
        if not tools:
            logger.warning("[OpenAI] No tools provided in schema")
            return True

        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                logger.error(f"[OpenAI] Tool {i} is not a dictionary: {type(tool)}")
                return False

            # Check required fields for function tools
            if tool.get("type") == "function":
                required_fields = ["name", "description", "parameters"]
                missing_fields = [field for field in required_fields if field not in tool]

                if missing_fields:
                    logger.error(f"[OpenAI] Tool {i} missing required fields: {missing_fields}")
                    logger.error(f"[OpenAI] Tool {i} structure: {tool}")
                    return False

                # Validate parameters structure
                params = tool.get("parameters", {})
                if not isinstance(params, dict):
                    logger.error(f"[OpenAI] Tool {i} parameters is not a dictionary: {type(params)}")
                    return False

                if params.get("type") != "object":
                    logger.error(f"[OpenAI] Tool {i} parameters type should be 'object': {params.get('type')}")
                    return False

            logger.debug(f"[OpenAI] Tool {i} validation passed: {tool.get('name', 'unnamed')}")

        logger.info(f"[OpenAI] All {len(tools)} tools validated successfully")
        return True

    def _extract_text_from_responses(self, resp) -> Optional[str]:
        """Extract text from OpenAI response."""
        # 1) helper direct (dacă e prezent în SDK)
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        # 2) items -> content -> text (updated for Responses API)
        items = getattr(resp, "output", None)
        if isinstance(items, list):
            for item in items:
                # Skip reasoning items, but check function calls and web search calls for text content
                item_type = getattr(item, "type", None)
                if item_type in ["reasoning"]:
                    continue

                # Check if item has direct text content
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        t = getattr(c, "text", None)
                        if isinstance(t, str) and t.strip():
                            return t.strip()

                # For web_search_call items, check if there's any associated text
                if item_type in ["function_call", "web_search_call"]:
                    # Look for any text in the item that might be the response
                    item_text = getattr(item, "output", None) or getattr(item, "result", None)
                    if isinstance(item_text, str) and item_text.strip():
                        return item_text.strip()

        # 3) fallback: caută „text” oriunde
        try:
            payload = resp.model_dump()
            def find_text(obj):
                if isinstance(obj, dict):
                    if isinstance(obj.get("text"), str) and obj["text"].strip():
                        return obj["text"].strip()
                    for v in obj.values():
                        ft = find_text(v)
                        if ft:
                            return ft
                elif isinstance(obj, list):
                    for v in obj:
                        ft = find_text(v)
                        if ft:
                            return ft
                return None
            any_txt = find_text(payload)
            if any_txt:
                return any_txt
            logger.error(f"[OpenAI] No extractable text in response (first 1k): {json.dumps(payload)[:1000]}")
        except Exception:
            logger.exception("[OpenAI] Failed to parse response payload")
        return None

    def _extract_tool_calls_from_response(self, resp):
        """
        Extracts all tool calls (function calls and web search calls) emitted by the Responses API.
        Returns a list of dicts like:
          {"id": <call_id>, "type": <call_type>, "name": <function_name>, "args": <dict>}
        """
        calls = []
        output = getattr(resp, "output", []) or []

        for item in output:
            item_type = getattr(item, "type", None)

            # Handle function calls
            if item_type == "function_call" and getattr(item, "name", None):
                raw_args = getattr(item, "arguments", {}) or {}
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                calls.append({
                    "id": getattr(item, "call_id", None),
                    "type": "function_call",
                    "name": item.name,
                    "args": args
                })

            # Handle web search calls
            elif item_type == "web_search_call":
                action = getattr(item, "action", None)
                if action:
                    # Handle ActionSearch object attributes
                    query = getattr(action, "query", "")
                    sources = getattr(action, "sources", None)
                else:
                    query = ""
                    sources = None

                calls.append({
                    "id": getattr(item, "id", None),
                    "type": "web_search_call",
                    "name": "web_search",
                    "args": {
                        "query": query,
                        "sources": sources
                    }
                })

        return calls

    def _extract_web_search_results(self, response, call_id: str) -> List[Dict]:
        """
        Extract web search results from the response for a specific call_id.
        Returns a list of search result dictionaries.
        """
        results = []
        output = getattr(response, "output", []) or []

        for item in output:
            item_type = getattr(item, "type", None)
            item_id = getattr(item, "id", None)

            # Look for web search results that correspond to our call
            if (item_type == "web_search_call" and item_id == call_id and
                getattr(item, "status", None) == "completed"):

                # Extract search results if available
                # The results might be in a subsequent response or in the item itself
                # For now, we'll return basic info about the completed search
                action = getattr(item, "action", None)
                if action:
                    # Handle ActionSearch object attributes
                    query = getattr(action, "query", "")
                else:
                    query = ""

                results.append({
                    "query": query,
                    "status": "completed",
                    "call_id": call_id
                })

        return results


    async def tool_fetch_product_details(self, code: str) -> dict:
        """Handler pentru OpenAI tool - cheamă API-ul Romstal și returnează răspunsul complet."""
        import httpx

        url = f"https://www.romstalpartener.ro/cs-includes/evolio/product.php?code={code}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()  # API-ul tău returnează JSON

            return {"ok": True, "code": code, "data": data}

        except Exception as e:
            logger.error(f"[tool_fetch_product_details] {e}")
            return {"ok": False, "code": code, "error": str(e)}







    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Simple LLM call without tools."""
        if not self.client:
            return "Salut! Sunt asistentul Romstal. Cu ce te pot ajuta astăzi?"

        model_main = settings.openai_model
        model_fallback = "gpt-5-mini"

        def _responses_call(model: str, max_tokens: int):
            return self.client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_output_tokens=max_tokens,
                reasoning={"effort": "minimal"},
            )

        # 1) Prima încercare cu un buget rezonabil
        try:
            resp = _responses_call(model_main, 700)
            txt = resp.output_text
            if txt and txt.strip():
                return txt.strip()
            logger.warning("[OpenAI] Responses API returned no text; trying fallback model")
        except Exception as e:
            self._log_openai_error(f"Responses API ({model_main})", e)

        # 2) Fallback: același setup dar cu gpt-5-mini (mai „economic” în reasoning)
        try:
            resp = _responses_call(model_fallback, 600)
            txt = resp.output_text
            if txt and txt.strip():
                logger.info("[OpenAI] Used fallback model gpt-5-mini")
                return txt.strip()
        except Exception as e:
            self._log_openai_error(f"Responses API ({model_fallback})", e)

        # 3) Ultimul fallback: gpt-4o-mini
        try:
            resp = _responses_call("gpt-4o-mini", 300)
            txt = resp.output_text
            if txt and txt.strip():
                logger.info("[OpenAI] Fallback via gpt-4o-mini")
                return txt.strip()
        except Exception as e:
            self._log_openai_error("Responses API (gpt-4o-mini)", e)

        return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu."

    async def call_llm_with_tools(self, system_prompt: str, user_prompt: str, correlation_id: Optional[str] = None) -> Tuple[str, Optional[List]]:
        """
        Call LLM with tools support and enhanced error handling.
        Returns: (reply_text, function_call_details)
        - reply_text: The final text response
        - function_call_details: List of function calls and results if any were made, None otherwise
        """
        if correlation_id is None:
            correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            logger.info(f"[OpenAI] [{correlation_id}] Starting LLM call with tools, prompt length: {len(user_prompt)}")

            if not self.client:
                logger.warning(f"[OpenAI] [{correlation_id}] Client not initialized")
                return "Salut! Sunt asistentul Romstal. Cu ce te pot ajuta?", None

            try:
                logger.info("[DEBUG] ===== Starting call_llm_with_tools =====")
                logger.info(f"[DEBUG] User prompt length: {len(user_prompt)}")
                logger.info(f"[DEBUG] System prompt length: {len(system_prompt)}")

                # Step 1: Make initial API call using Responses API (GPT-5 style)
                logger.info("[DEBUG] Making initial API call with tools...")

                # Check if web_search tool is being used
                has_web_search = any(tool.get("type") == "web_search" for tool in self.OPENAI_TOOLS)

                # Use different reasoning effort based on tools
                reasoning_effort = "medium" if has_web_search else "minimal"

                if has_web_search:
                    logger.info(f"[OpenAI] [{correlation_id}] Web search tool available, using medium reasoning effort")

                response = self.client.responses.create(
                    model=settings.openai_model,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "Ești asistent Romstal pe WhatsApp. "
                                "Dacă utilizatorul furnizează clar un cod de produs (ex: 64px9822), "
                                "apelează funcția `fetch_product_details`. "
                                "Dacă cere recomandări de produse sau ce să cumpere într-un anumit scenariu, apeleaza funcția `web_search`.  "
                                "După orice căutare, sintetizează un răspuns final concis pentru utilizator. Arată cele mai relevante linkuri. "
                                "Nu modifica URL-urile sau alte date. "
                                "Răspunde prietenos, în română.\n\n"
                                + system_prompt
                            )
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    tools=self.OPENAI_TOOLS,
                    tool_choice="auto",
                    max_output_tokens=3500,  # Increased from 700 to allow for reasoning + response
                    reasoning={"effort": reasoning_effort}
                )

                logger.info(f"[DEBUG] Initial response ID: {response.id}")
                logger.info(f"[DEBUG] Initial response output type: {type(response.output)}")

                # Step 2: Extract tool calls using the existing helper function
                tool_calls = self._extract_tool_calls_from_response(response)
                logger.info(f"[DEBUG] Extracted tool calls: {len(tool_calls)}")

                if tool_calls:
                    logger.info(f"[OpenAI] Found {len(tool_calls)} function call(s), executing...")
                    logger.info(f"[DEBUG] Tool calls details: {json.dumps([{'name': tc['name'], 'id': tc['id']} for tc in tool_calls], indent=2)}")

                    # Step 3: Execute function calls
                    tool_outputs = []
                    function_call_history = []  # Track function calls and results

                    for call in tool_calls:
                        function_name = call["name"]
                        call_id = call["id"]

                        try:
                            # Arguments are already parsed as dict from the helper function
                            args = call["args"]
                        except Exception:
                            args = {}

                        logger.info(f"[OpenAI] Executing function {function_name} with args: {args}")

                        # Execute the function based on name
                        if function_name == "fetch_product_details":
                            result = await self.tool_fetch_product_details(args.get("code", ""))
                            tool_outputs.append({
                                "tool_call_id": call_id,
                                "output": result
                            })
                            # Store the function call history
                            function_call_history.append({
                                "function": function_name,
                                "args": args,
                                "result": result
                            })
                        elif function_name == "web_search":
                            # Web search is handled internally by OpenAI - no custom output needed
                            # Just log the search execution for tracking
                            logger.info(f"[OpenAI] Web search executed: {args.get('query', 'No query')}")
                            function_call_history.append({
                                "function": function_name,
                                "args": args,
                                "result": {"ok": True, "note": "Web search handled internally by OpenAI"}
                            })
                        else:
                            logger.warning(f"[OpenAI] Unknown function: {function_name}")
                            error_result = {"ok": False, "error": f"Unknown function: {function_name}"}
                            tool_outputs.append({
                                "tool_call_id": call_id,
                                "output": error_result
                            })
                            function_call_history.append({
                                "function": function_name,
                                "args": args,
                                "result": error_result
                            })

                    # Step 4: Make follow-up API call with function results using Responses API
                    # Only make follow-up call if we have actual function call outputs (not web search)
                    function_outputs = [output for output in tool_outputs if output["output"].get("ok") is not None]
                    if function_outputs:
                        logger.info("[OpenAI] Making follow-up API call with function results")
                        logger.info(f"[DEBUG] Function outputs count: {len(function_outputs)}")
                        logger.info(f"[DEBUG] Function outputs structure: {json.dumps([{'id': fo['tool_call_id'], 'ok': fo['output'].get('ok')} for fo in function_outputs], indent=2)}")

                        # Build input with function call outputs
                        follow_up_input = [
                            {
                                "role": "system",
                                "content": (
                                    "Ești asistent Romstal pe WhatsApp. "
                                    "Folosește detaliile din tool_result pentru a răspunde. "
                                    "Dacă `ok=False` sau lipsesc date, spune politicos că momentan nu poți accesa detaliile produsului și cere un alt cod sau mai multe informații. "
                                    "Răspunde scurt, prietenos, în română.\n\n"
                                    + system_prompt
                                )
                            },
                            {"role": "user", "content": user_prompt},
                        ]

                        # Add function call outputs to input (web search is handled internally)
                        for tool_output in function_outputs:
                            follow_up_input.append({
                                "type": "function_call_output",
                                "call_id": tool_output["tool_call_id"],
                                "output": json.dumps(tool_output["output"], ensure_ascii=False)
                            })

                        logger.info(f"[DEBUG] Follow-up input length: {len(follow_up_input)} items")
                        logger.info(f"[DEBUG] Using previous_response_id: {response.id}")

                        # Use appropriate reasoning effort for follow-up call too
                        follow_up_reasoning_effort = "medium" if has_web_search else "minimal"

                        if has_web_search:
                            logger.info(f"[OpenAI] [{correlation_id}] Web search tool available for follow-up call")

                        follow_up_response = self.client.responses.create(
                            model=settings.openai_model,
                            input=follow_up_input,
                            tools=self.OPENAI_TOOLS,
                            previous_response_id=response.id,  # Thread the conversation
                            max_output_tokens=2500,  # Ensure enough tokens for response after function call
                            reasoning={"effort": follow_up_reasoning_effort}
                        )

                        logger.info(f"[DEBUG] Follow-up response ID: {follow_up_response.id}")
                        logger.info(f"[DEBUG] Follow-up response output_text present: {hasattr(follow_up_response, 'output_text') and follow_up_response.output_text is not None}")

                        # Extract final response text from follow-up call
                        final_text = follow_up_response.output_text
                        if final_text:
                            logger.info(f"[DEBUG] ===== Function call completed successfully, returning text length: {len(final_text)} =====")
                            return final_text.strip(), function_call_history
                        else:
                            logger.error("[DEBUG] ===== Follow-up response has NO output_text! =====")
                            logger.error(f"[DEBUG] Follow-up response dump: {follow_up_response.model_dump()}")

                else:
                    # No function calls, return the direct response
                    logger.info("[OpenAI] No function calls found, returning direct response")
                    if has_web_search:
                        logger.info(f"[OpenAI] [{correlation_id}] Direct response suggests web_search may have been used internally by OpenAI")
                    logger.info(f"[DEBUG] Direct response output_text present: {hasattr(response, 'output_text') and response.output_text is not None}")
                    final_text = response.output_text
                    logger.info(f"[DEBUG] output_text value: '{final_text}' (type: {type(final_text)}, len: {len(final_text) if final_text else 'N/A'})")
                    logger.info(f"[DEBUG] output_text repr: {repr(final_text)}")

                    if final_text and final_text.strip():
                        logger.info(f"[DEBUG] ===== Direct response completed, text length: {len(final_text)} =====")
                        return final_text.strip(), None
                    else:
                        logger.error(f"[DEBUG] output_text is empty or whitespace-only")
                        # Try alternative extraction
                        alt_text = self._extract_text_from_responses(response)
                        if alt_text:
                            logger.info(f"[DEBUG] Alternative extraction succeeded, length: {len(alt_text)}")
                            return alt_text, None

                        # Log the full response for debugging
                        try:
                            response_dump = response.model_dump()
                            logger.error(f"[DEBUG] Full response dump (first 2000 chars): {json.dumps(response_dump, ensure_ascii=False)[:2000]}")
                        except Exception as e:
                            logger.error(f"[DEBUG] Could not dump response: {e}")

                logger.warning("[OpenAI] No text extracted from response")
                logger.warning("[DEBUG] ===== RETURNING GENERIC ERROR MESSAGE =====")
                return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu.", None

            except Exception as e:
                logger.exception(f"[OpenAI] [{correlation_id}] Exception in call_llm_with_tools: {type(e).__name__}: {e}")
                return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu.", None


# Global LLM client instance
llm_client = LLMClient()