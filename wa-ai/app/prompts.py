"""
Centralized prompt management for Romstal Assistant.
Single source of truth for all system prompts and tool usage policies.
"""

from typing import Dict, Any


class PromptManager:
    """Centralized management of all system prompts with unified tool decision policies."""

    # Single unified system prompt with embedded tool decision policies
    UNIFIED_ROMSTAL_PROMPT = """
Ești asistentul Romstal.

POLITICA DE DECIZIE A TOOL-URILOR
1) Saluturi / small talk / întrebări generale (ex: "Hi", "Bună", "Ce faci?"):
   - Răspunde prietenos, concis.
   - NU apela niciun tool.

2) Cod de produs CLAR (ex: 64px9822, fără text în plus):
   - Apelează funcția `fetch_product_details` cu exact acel cod.
   - După tool_result: afișează nume, preț (dacă există), disponibilitate, linkul oficial.
   - Dacă `ok=False` sau codul nu e valid → spune politicos și cere un cod valid.

3) Căutare / recomandări / "ai link", "ce produs?", specificații, compatibilități:
   - Dacă informația nu e evidentă din context, FOLOSEȘTE `web_search` (romstal.ro).
   - Returnează 3–6 rezultate relevante: **Nume** — motiv scurt (max 1 propoziție)
     pe linia următoare URL direct. Nu inventa linkuri.
   - Dacă nu găsești produse exacte, arată și pagini de categorie și spune că sunt generale.
   - Dacă nu există rezultate utile: spune explicit că nu ai găsit pagini relevante pe romstal.ro.

4) Dacă cererea este ambiguă:
   - Pune o singură clarificare scurtă înainte să folosești tool-uri.

REGULI GENERALE
- Răspunde în română, prietenos și concis.
- Nu modifica URL-urile.
- Nu folosi tool-uri când nu aduc valoare (ex: saluturi, întrebări simple).
- Nu repeta aceleași informații.

EXEMPLE GHID:
- "Hi" → răspuns scurt, fără tool.
- "Cod 64px9822" → `fetch_product_details`.
- "Îmi recomanzi o baterie de bucătărie?" → `web_search`.
- "Am nevoie de detalii, dar nu știu exact modelul" → întreabă o clarificare, apoi decide.

IMPORTANT: Aplică această politică pentru fiecare mesaj. Dacă mesajul este un simplu salut sau small talk, răspunde direct fără tool-uri. Dacă necesită informații despre produse, folosește tool-urile disponibile în mod inteligent.
"""

    @staticmethod
    def get_unified_prompt() -> str:
        """Get the single unified system prompt with embedded tool policies."""
        return PromptManager.UNIFIED_ROMSTAL_PROMPT.strip()

    @staticmethod
    def get_prompt_with_tools() -> str:
        """Get the unified prompt formatted for LLM with tools."""
        return PromptManager.get_unified_prompt()

    @staticmethod
    def get_prompt_without_tools() -> str:
        """Get the unified prompt formatted for simple LLM calls."""
        return PromptManager.get_unified_prompt()

    @staticmethod
    def should_use_tools(message_text: str, context: str = "") -> bool:
        """
        Helper function to determine if tools should be used based on message content.
        This implements the decision logic from the unified prompt.
        """
        if not message_text:
            return False

        message_lower = message_text.lower().strip()
        context_lower = context.lower() if context else ""

        # 1. Simple greetings and small talk - NO tools
        simple_phrases = {
            "hi", "hello", "hey", "salut", "bună", "buna", "ce faci", "cum ești", "cum esti",
            "mulțumesc", "mersi", "thanks", "pa", "bye", "la revedere", "ne revedem"
        }

        if message_lower in simple_phrases:
            return False

        # Check if message starts with simple greeting patterns
        greeting_patterns = ["hi", "hello", "hey", "salut", "bună", "buna"]
        if any(message_lower.startswith(pattern) for pattern in greeting_patterns):
            return False

        # 2. Clear product codes - YES tools (fetch_product_details)
        # Look for patterns like "cod 64px9822" or just "64px9822"
        import re
        product_code_pattern = r'\b\d+[a-zA-Z]*\d+[a-zA-Z]*\d+\b'
        if re.search(product_code_pattern, message_text):
            # Check if it's a standalone code or preceded by "cod"
            words = message_text.split()
            for word in words:
                if re.match(product_code_pattern, word):
                    # If it's just the code or preceded by "cod", use tools
                    if len(words) <= 2 and (len(words) == 1 or words[0].lower() == "cod"):
                        return True

        # 3. Product-related queries - YES tools (web_search)
        product_keywords = {
            "produs", "produse", "recomand", "recomanda", "recomandă", "caut", "vreau",
            "cumpar", "cumpăr", "pret", "preț", "disponibil", "stoc", "link", "specificatii",
            "compatibil", "baterie", "chiuveta", "cada", "dus", "robinet", "teava", "țeavă",
            "centrala", "centrală", "radiator", "parchet", "gresie", "faianta", "faianță"
        }

        # Check both message and context for product keywords
        combined_text = f"{message_lower} {context_lower}"
        if any(keyword in combined_text for keyword in product_keywords):
            return True

        # 4. Ambiguous cases - NO tools (let AI ask for clarification)
        return False

    @staticmethod
    def analyze_message_type(message_text: str) -> Dict[str, Any]:
        """
        Analyze message to determine type and recommended action.
        Returns dict with analysis results.
        """
        if not message_text:
            return {"type": "empty", "use_tools": False, "action": "simple_response"}

        message_lower = message_text.lower().strip()

        # Simple greetings
        simple_phrases = {"hi", "hello", "hey", "salut", "bună", "buna", "ce faci", "cum ești", "cum esti"}
        if message_lower in simple_phrases:
            return {"type": "greeting", "use_tools": False, "action": "simple_response"}

        # Product codes
        import re
        product_code_pattern = r'\b\d+[a-zA-Z]*\d+[a-zA-Z]*\d+\b'
        if re.search(product_code_pattern, message_text):
            words = message_text.split()
            for word in words:
                if re.match(product_code_pattern, word):
                    if len(words) <= 2 and (len(words) == 1 or words[0].lower() == "cod"):
                        return {"type": "product_code", "use_tools": True, "tool": "fetch_product_details", "code": word}

        # Product-related queries
        product_keywords = {"produs", "produse", "recomand", "recomanda", "recomandă", "caut", "vreau", "cumpar", "cumpăr"}
        if any(keyword in message_lower for keyword in product_keywords):
            return {"type": "product_search", "use_tools": True, "tool": "web_search"}

        # Default case
        return {"type": "general", "use_tools": False, "action": "simple_response"}