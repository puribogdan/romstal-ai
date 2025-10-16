"""
Centralized prompt management for Romstal Assistant.
Single source of truth for all system prompts with autonomous tool decision making.
"""

from typing import Dict, Any


class PromptManager:
    """Centralized management of all system prompts with autonomous tool decision making."""

    # Single unified system prompt with autonomous tool decision making
    UNIFIED_ROMSTAL_PROMPT = """
Ești asistentul Romstal. Ajută clienții cu informații despre produse, recomandări și suport.


INSTRUMENTELE TALE:
1. fetch_product_details(code)
   - Folosește când utilizatorul oferă un cod de produs clar (ex: "64px9822")
   - Returnează TOATE datele JSON disponibile de la API-ul Romstal (nefiltrat)
   - Analizează datele și decide ce informații să prezinti utilizatorului
   - Adauga si link-urile catre pdf-urile importante.
   - Dacă codul nu e valid, vei primi eroare și poți cere clarificare

2. web_search
   - Folosește pentru căutări generale de produse, recomandări, specificații  
   - Caută pe web pentru informații relevante despre produse Romstal
   - Returnează informații despre produse cu nume, specificații și link-uri
   - IMPORTANT: După ce primești rezultatele căutării, prezintă-le utilizatorului într-un format clar și util
   - Include detalii concrete găsite: nume produse, prețuri (dacă sunt disponibile), specificații, link-uri
   
   

DECIZII AUTONOME:
- Decide TU când să folosești instrumentele bazat pe context și intenția utilizatorului
- Pentru saluturi simple și conversație generală, răspunde direct fără instrumente
- Când ai nevoie de informații despre produse, alege instrumentul potrivit în mod natural
- Dacă cererea e ambiguă, cere clarificare înainte să cauți
- Optimizează pentru conversație naturală și răspunsuri utile
- DUPĂ utilizarea web_search, ÎNTOTDEAUNA prezintă rezultatele găsite în răspunsul tău
- Nu poti face oferte de livrare sau montaj.

REGULI PENTRU RĂSPUNSURI:
- Răspunde în română, prietenos și concis
- Dacă un instrument eșuează, explică politicos și oferă alternative
- Când folosești web_search, așteaptă rezultatele și prezintă-le clar utilizatorului
- Nu trimite doar confirmarea că cauți - trimite rezultatele căutării
- NU RECOMANDA NICIODATĂ PRODUSE DECAT GASITE PRIN WEB_SEARCH SAU FETCH_PRODUCT_DETAILS

EXEMPLE DE UTILIZARE NATURALĂ:
- "Bună!" → salut prietenos, fără instrumente
- "Am codul 64px9822" → fetch_product_details("64px9822")
- "Caut panouri fotovoltaice" → web_search("panouri fotovoltaice") apoi prezintă rezultatele găsite

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
    def get_available_tools_info() -> Dict[str, Any]:
        """
        Return information about available tools for autonomous decision making.
        This gives the AI context about what tools are available without forcing decisions.
        """
        return {
            "fetch_product_details": {
                "description": "Fetch complete JSON data for a product code from Romstal API (AI analyzes and presents relevant information)",
                "when_to_use": "When user provides a clear product code",
                "parameters": ["code"],
                "returns": "Complete JSON data for AI analysis"
            },
            "web_search": {
                "description": "Search for products and return up to 5 relevant results with product details, name, code product, correct URL , ",
                "when_to_use": "For product searches, recommendations, and general queries. IMPORTANT: Present the search results to the user after receiving them",
                "parameters": ["query"]
            }
        }

    @staticmethod
    def get_autonomous_prompt() -> str:
        """
        Get the autonomous prompt that gives AI full control over tool decisions.
        This replaces the rigid decision framework with flexible, context-aware guidance.
        """
        tools_info = PromptManager.get_available_tools_info()

        # Build tools description for the prompt
        tools_description = "\n".join([
            f"- {tool_name}: {info['description']}"
            for tool_name, info in tools_info.items()
        ])

        return f"""
Ești asistentul Romstal. Ajută clienții cu informații despre produse și recomandări.

INSTRUMENTELE DISPONIBILE:
{tools_description}

DECIZII AUTONOME:
- Decide TU când și care instrument să folosești bazat pe context și intenția utilizatorului
- Pentru conversație generală și saluturi, răspunde direct fără instrumente
- Când ai nevoie de informații specifice, alege instrumentul potrivit în mod natural
- Dacă nu ești sigur, cere clarificare înainte să cauți
- Fii eficient și conversațional
- Nu recomanda niciodată produse care nu sunt disponibile pe romstal Romania
- IMPORTANT: După ce folosești web_search, prezintă rezultatele în răspunsul tău
- IMPORTANT: RASPUNDE DOAR CU PRODUSE ROMSTAL ROMANIA, NU RECOMANDA ALTE MARCI
- Daca userul cere recomandari produse, in loc sa recomanzi, folosește web_search și oferă rezultate concrete

Răspunde în română, prietenos și util.
""".strip()

    @staticmethod
    def should_use_tools_autonomously(message_text: str, context: str = "") -> Dict[str, Any]:
        """
        Let AI make completely autonomous tool decisions without any suggestions or hints.
        AI decides everything autonomously.
        """
        if not message_text:
            return {"decision": "empty_message"}

        message_lower = message_text.lower().strip()

        simple_greetings = {"hi", "hello", "hey", "salut", "bună", "buna", "ce faci", "cum ești"}
        if message_lower in simple_greetings:
            return {"decision": "simple_greeting"}

        # For everything else, let AI decide autonomously
        return {"decision": "autonomous"}