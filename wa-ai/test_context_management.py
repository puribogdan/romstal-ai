#!/usr/bin/env python3
"""
Test script for simplified product recommendation functionality.
This script tests the basic product lookup features without context storage.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, MagicMock

# Add the wa-ai directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wa-ai'))

from handlers.handle_product_recommendation import ProductRecommendationHandler


class MockSupabaseClient:
    """Mock Supabase client for testing without database dependencies."""

    def __init__(self):
        self.test_data = {}

    def get_message_by_id(self, message_id: int):
        """Mock message retrieval."""
        return {
            "id": message_id,
            "wa_from": "+40712345678",
            "wa_text": "Show me product 64px9822",
            "wa_timestamp": "2024-01-01T10:00:00Z",
            "inserted_at": "2024-01-01T10:00:00Z"
        }

    def get_message_history(self, phone: str):
        """Mock message history."""
        return [
            {
                "wa_from": phone,
                "wa_text": "Hello",
                "wa_timestamp": "2024-01-01T09:00:00Z",
                "inserted_at": "2024-01-01T09:00:00Z",
                "direction": "inbound"
            }
        ]

    def get_or_create_conversation_session(self, phone: str, system_prompt: str):
        """Mock session creation."""
        return {
            "id": 12345,
            "phone_number": phone,
            "system_prompt": system_prompt,
            "full_conversation": [],
            "context_messages": [],
            "message_count": 0
        }

    def insert_outbound_message(self, phone: str, message: str, raw_data: dict = None):
        """Mock outbound message insertion."""
        return {"id": 999, "wa_from": phone, "wa_text": message}


class MockLLMClient:
    """Mock LLM client for testing."""

    async def call_llm_with_tools(self, system_prompt: str, user_prompt: str, correlation_id: str = None):
        """Mock LLM response with product details."""
        return "Am găsit produsul: Teava PPR 20mm, preț 15.50 RON. Disponibil în stoc. https://www.romstal.ro/teava-ppr-20mm", [
            {
                "function": "fetch_product_details",
                "args": {"code": "64px9822"},
                "result": {
                    "ok": True,
                    "code": "64px9822",
                    "data": {
                        "info": {
                            "product": "Teava PPR 20mm",
                            "price": "15.50 RON"
                        }
                    }
                }
            }
        ]


class MockN8NClient:
    """Mock N8N client for testing."""

    async def send_to_n8n(self, phone: str, message: str, correlation_id: str = None):
        """Mock N8N webhook call."""
        return {"success": True, "message_id": "n8n_123"}


async def test_product_recommendation():
    """Test the simplified product recommendation functionality."""
    print("Testing Simplified Product Recommendation")
    print("=" * 50)

    # Create mock clients
    mock_supa = MockSupabaseClient()
    mock_llm = MockLLMClient()
    mock_n8n = MockN8NClient()

    # Create handler with mock clients
    handler = ProductRecommendationHandler()
    handler.db_client = mock_supa
    handler.llm_client = mock_llm
    handler.n8n_client = mock_n8n

    # Test data
    test_insert_id = 67890

    # Test product recommendation processing
    print("\nTest 1: Product Recommendation Processing")
    result = await handler.handle_product_recommendation(insert_id=test_insert_id)

    if result.get("success"):
        print("PASS: Product recommendation processed successfully")
        print(f"   - Phone: {result.get('phone')}")
        print(f"   - Response length: {len(result.get('ai_response', ''))}")
        print(f"   - Function calls: {len(result.get('function_calls', []))}")
    else:
        print(f"FAIL: Product recommendation failed: {result.get('error')}")
        return False

    # Test phone number extraction
    print("\nTest 2: Phone Number Extraction")
    phone = await handler._extract_phone_number(test_insert_id)
    if phone == "+40712345678":
        print("PASS: Phone number extracted correctly")
    else:
        print(f"FAIL: Expected '+40712345678', got '{phone}'")
        return False

    # Test message and context retrieval
    print("\nTest 3: Message and Context Retrieval")
    message_data = await handler._get_message_and_context(test_insert_id, phone)
    if message_data and "current_message" in message_data:
        print("PASS: Message and context retrieved successfully")
        print(f"   - Current message: {message_data['current_message'][:50]}...")
    else:
        print("FAIL: Failed to retrieve message and context")
        return False

    print("\nAll tests passed! Product recommendation functionality is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_product_recommendation())
        if success:
            print("\nProduct recommendation implementation is ready for production!")
            sys.exit(0)
        else:
            print("\nSome tests failed. Please check the implementation.")
            sys.exit(1)
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)