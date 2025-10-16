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
   - Returnează detalii complete despre produs: nume, preț, disponibilitate, link oficial
   - Dacă codul nu e valid, vei primi eroare și poți cere clarificare

2. web_search(query)
   - Folosește pentru căutări generale de produse, recomandări, specificații  
   - Returnează maxim 5 rezultate relevante cu nume produs, cod și link
   - Dacă nu găsești produse specifice, arată și pagini de categorie generale

DECIZII AUTONOME:
- Decide TU când să folosești instrumentele bazat pe context și intenția utilizatorului
- Pentru saluturi simple și conversație generală, răspunde direct fără instrumente
- Când ai nevoie de informații despre produse, alege instrumentul potrivit în mod natural
- Dacă cererea e ambiguă, cere clarificare înainte să cauți
- Optimizează pentru conversație naturală și răspunsuri utile

REGULI GENERALE:
- Răspunde în română, prietenos și concis
- Dacă un instrument eșuează, explică politicos și oferă alternative

EXEMPLE DE UTILIZARE NATURALĂ:
- "Bună!" → salut prietenos, fără instrumente
- "Am codul 64px9822" → fetch_product_details("64px9822")
- "Caut o baterie de bucătărie" → web_search("baterie bucătărie romstal")

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
                "description": "Lookup specific product details by code (e.g., '64px9822')",
                "when_to_use": "When user provides a clear product code",
                "parameters": ["code"]
            },
            "web_search": {
                "description": "Search for products, recommendations, and general information. Return up to 5 relevant results with product name, code, and URL for that product",
                "when_to_use": "For product searches, recommendations, and general queries",
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
- Cand sunt cerute recomandari de produse, foloseste web_search si raspunde doar cu rezultate pe care le gasesti
- Nu recomanda niciodata produse care nu sunt disponibile pe romstal.ro
- Nu recomanda produse sau link-uri din catalog search, doar din rezultatele web_search

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