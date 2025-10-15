import json
import logging
import asyncio
from copy import deepcopy
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
from .settings import settings
from .correlation import generate_correlation_id, CorrelationContext

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        self.client = None
        self.max_retries = 2
        self.retry_delay = 0.8  # seconds

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
            "filters": {"allowed_domains": ["romstal.ro"]},
            "user_location": {"type": "approximate", "country": "RO", "city": "București"},
            # Optional; supported in many environments. Safe to keep, but we guard updates.
            "search_context_size": "low"
        }
    ]

    # --- helpers -------------------------------------------------------------

    def _log_openai_error(self, context: str, error: Exception) -> None:
        logger.error(f"[OpenAI] {context} failed: {type(error).__name__}: {error}")

    def _extract_text_from_responses(self, resp) -> Optional[str]:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        items = getattr(resp, "output", None)
        if isinstance(items, list):
            all_texts = []
            for item in items:
                item_type = getattr(item, "type", None)
                if item_type in ["message", "text"]:
                    content = getattr(item, "content", None)
                    if isinstance(content, list):
                        for c in content:
                            t = getattr(c, "text", None)
                            if isinstance(t, str) and t.strip():
                                all_texts.append(t.strip())
                elif item_type in ["reasoning"]:
                    continue
                else:
                    content = getattr(item, "content", None)
                    if isinstance(content, list):
                        for c in content:
                            t = getattr(c, "text", None)
                            if isinstance(t, str) and t.strip():
                                all_texts.append(t.strip())
                    if item_type in ["function_call", "web_search_call"]:
                        item_text = getattr(item, "output", None) or getattr(item, "result", None)
                        if isinstance(item_text, str) and item_text.strip():
                            all_texts.append(item_text.strip())
            if all_texts:
                return " ".join(all_texts).strip()

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
            logger.error(f"[OpenAI] No extractable text in response (first 1k): {json.dumps(payload, ensure_ascii=False)[:1000]}")
        except Exception:
            logger.exception("[OpenAI] Failed to parse response payload")
        return None

    def _extract_tool_calls_from_response(self, resp):
        calls = []
        output = getattr(resp, "output", []) or []
        for item in output:
            item_type = getattr(item, "type", None)
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
            elif item_type == "web_search_call":
                action = getattr(item, "action", None)
                query = getattr(action, "query", None) if action else None
                sources = getattr(action, "sources", None) if action else None
                if query:
                    calls.append({
                        "id": getattr(item, "id", None),
                        "type": "web_search_call",
                        "name": "web_search",
                        "args": {"query": query, "sources": sources}
                    })
        return calls

    async def tool_fetch_product_details(self, code: str) -> dict:
        import httpx
        url = f"https://www.romstalpartener.ro/cs-includes/evolio/product.php?code={code}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()
            return {"ok": True, "code": code, "data": data}
        except Exception as e:
            logger.error(f"[tool_fetch_product_details] {e}")
            return {"ok": False, "code": code, "error": str(e)}

    # Structured Outputs schema (compact fallback)
    @staticmethod
    def _compact_response_format():
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "romstal_reco",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "picks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "why": {"type": "string"}
                                },
                                "required": ["title", "url"]
                            }
                        },
                        "note": {"type": "string"}
                    },
                    "required": ["picks"]
                }
            }
        }

    # --- plain call ----------------------------------------------------------

    def call_llm(self, system_prompt: str, user_prompt: str) -> str:
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

        try:
            resp = _responses_call(model_main, 700)
            txt = resp.output_text
            if txt and txt.strip():
                return txt.strip()
            logger.warning("[OpenAI] Responses API returned no text; trying fallback model")
        except Exception as e:
            self._log_openai_error(f"Responses API ({model_main})", e)

        try:
            resp = _responses_call(model_fallback, 600)
            txt = resp.output_text
            if txt and txt.strip():
                logger.info("[OpenAI] Used fallback model gpt-5-mini")
                return txt.strip()
        except Exception as e:
            self._log_openai_error(f"Responses API ({model_fallback})", e)

        try:
            resp = _responses_call("gpt-4o-mini", 300)
            txt = resp.output_text
            if txt and txt.strip():
                logger.info("[OpenAI] Fallback via gpt-4o-mini")
                return txt.strip()
        except Exception as e:
            self._log_openai_error("Responses API (gpt-4o-mini)", e)

        return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu."

    # --- tools call ----------------------------------------------------------

    async def call_llm_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        correlation_id: Optional[str] = None
    ) -> Tuple[str, Optional[List]]:

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

                has_web_search = any(tool.get("type") == "web_search" for tool in self.OPENAI_TOOLS)
                reasoning_effort = "medium" if has_web_search else "minimal"

                # Retry once with a compact output profile if we hit token ceiling
                response = None
                used_compact_profile = False

                for attempt in range(self.max_retries):
                    current_tools = deepcopy(self.OPENAI_TOOLS)

                    # Optionally bump search_context_size from low -> medium on retry (if key exists)
                    if has_web_search and attempt == 1:
                        for tool in current_tools:
                            if tool.get("type") == "web_search" and "search_context_size" in tool:
                                tool["search_context_size"] = "medium"

                    max_tokens = 1000 if attempt == 0 else 900  # smaller for compact profile
                    kwargs = {
                        "model": settings.openai_model,
                        "input": [
                            {
                                "role": "system",
                                "content": (
                                    "Ești asistent Romstal pe WhatsApp. "
                                    "Dacă utilizatorul furnizează clar un cod de produs (ex: 64px9822), "
                                    "apelează funcția `fetch_product_details`. "
                                    "Dacă cere recomandări de produse sau ce să cumpere într-un anumit scenariu, apelează `web_search`. "
                                    "După căutare, oferă un răspuns concis cu linkuri relevante, în română."
                                    "\n\n" + system_prompt
                                )
                            },
                            {"role": "user", "content": user_prompt},
                        ],
                        "tools": current_tools,
                        "tool_choice": "auto",
                        "parallel_tool_calls": False,  # serialize tool calls to save tokens
                        "include": ["web_search_call.action.sources"],  # return sources for rendering
                        "max_output_tokens": max_tokens,
                        "reasoning": {"effort": reasoning_effort}
                    }

                    # On retry, force compact structured output to guarantee completion
                    if attempt == 1:
                        kwargs["response_format"] = self._compact_response_format()
                        used_compact_profile = True

                    logger.info(
                        f"[OpenAI] [{correlation_id}] Attempt {attempt + 1}/{self.max_retries} "
                        f"(max_tokens={max_tokens}, compact={used_compact_profile})"
                    )

                    response = self.client.responses.create(**kwargs)

                    # Detect truncation safely whether dict-like or attr-like
                    incomplete = getattr(response, "incomplete_details", None)
                    reason = None
                    if incomplete:
                        # could be dict or object with .reason
                        if isinstance(incomplete, dict):
                            reason = incomplete.get("reason")
                        else:
                            reason = getattr(incomplete, "reason", None)

                    if reason == "max_output_tokens":
                        logger.warning(f"[OpenAI] [{correlation_id}] Truncated (max_output_tokens) on attempt {attempt + 1}")
                        # If first attempt truncated, loop and retry with compact JSON profile
                        await asyncio.sleep(self.retry_delay)
                        continue

                    # Success path
                    break

                # If nothing came back (very unlikely)
                if response is None:
                    logger.error(f"[OpenAI] [{correlation_id}] No response obtained after retries")
                    return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu.", None

                logger.info(f"[DEBUG] Response ID: {response.id}")
                tool_calls = self._extract_tool_calls_from_response(response)
                logger.info(f"[DEBUG] Extracted tool calls: {len(tool_calls)}")

                function_call_history: List[Dict[str, Any]] = []

                if tool_calls:
                    logger.info(f"[OpenAI] Found {len(tool_calls)} tool call(s), executing...")
                    tool_outputs = []

                    for call in tool_calls:
                        name = call["name"]
                        call_id = call["id"]
                        args = call.get("args", {}) or {}

                        logger.info(f"[OpenAI] Executing function {name} with args: {args}")

                        if name == "fetch_product_details":
                            result = await self.tool_fetch_product_details(args.get("code", ""))
                            tool_outputs.append({"tool_call_id": call_id, "output": result})
                            function_call_history.append({"function": name, "args": args, "result": result})

                        elif name == "web_search":
                            # Built-in; we don't execute anything here
                            logger.info(f"[OpenAI] Web search executed: {args.get('query')}")
                            function_call_history.append({
                                "function": name,
                                "args": args,
                                "result": {"ok": True, "note": "Web search handled by OpenAI"}
                            })

                        else:
                            logger.warning(f"[OpenAI] Unknown function: {name}")
                            err = {"ok": False, "error": f"Unknown function: {name}"}
                            tool_outputs.append({"tool_call_id": call_id, "output": err})
                            function_call_history.append({"function": name, "args": args, "result": err})

                    # If we actually called our own function(s), do a follow-up pass to let the model read their outputs.
                    function_outputs = [o for o in tool_outputs if isinstance(o.get("output"), dict)]
                    if function_outputs:
                        follow_up_input = [
                            {
                                "role": "system",
                                "content": (
                                    "Ești asistent Romstal pe WhatsApp. "
                                    "Folosește detaliile primite din tool_result pentru a răspunde pe scurt, în română."
                                    "\n\n" + system_prompt
                                )
                            },
                            {"role": "user", "content": user_prompt},
                        ]
                        for fo in function_outputs:
                            follow_up_input.append({
                                "type": "function_call_output",
                                "call_id": fo["tool_call_id"],
                                "output": json.dumps(fo["output"], ensure_ascii=False)
                            })

                        follow_up = self.client.responses.create(
                            model=settings.openai_model,
                            input=follow_up_input,
                            tools=self.OPENAI_TOOLS,
                            previous_response_id=response.id,
                            parallel_tool_calls=False,
                            include=["web_search_call.action.sources"],
                            max_output_tokens=900,
                            reasoning={"effort": reasoning_effort},
                            response_format=self._compact_response_format()  # Keep compact on follow-up
                        )

                        final_text = getattr(follow_up, "output_text", None)
                        if final_text and final_text.strip():
                            return final_text.strip(), function_call_history

                        alt = self._extract_text_from_responses(follow_up)
                        if alt:
                            return alt, function_call_history

                        try:
                            logger.error(f"[DEBUG] Follow-up dump (first 2000): {json.dumps(follow_up.model_dump(), ensure_ascii=False)[:2000]}")
                        except Exception:
                            logger.exception("[DEBUG] Could not dump follow-up response")

                        return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu.", function_call_history

                # If only built-in tools were used (or none), return from this response
                final_text = getattr(response, "output_text", None)
                if final_text and final_text.strip():
                    return final_text.strip(), function_call_history

                alt = self._extract_text_from_responses(response)
                if alt:
                    return alt, function_call_history

                try:
                    logger.error(f"[DEBUG] No text; dump (first 2000): {json.dumps(response.model_dump(), ensure_ascii=False)[:2000]}")
                except Exception:
                    logger.exception("[DEBUG] Could not dump response")

                return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu.", function_call_history

            except Exception as e:
                logger.exception(f"[OpenAI] [{correlation_id}] Exception in call_llm_with_tools: {type(e).__name__}: {e}")
                return "Îmi pare rău, momentan nu pot procesa cererea. Te rog să încerci din nou mai târziu.", None


# Global LLM client instance
llm_client = LLMClient()
