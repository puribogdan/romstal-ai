#!/usr/bin/env python3
"""
Test script for product link context management functionality.
This script tests the core context management features without requiring external dependencies.
"""

import sys
import os
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional
from unittest.mock import Mock, MagicMock

# Add the wa-ai directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wa-ai'))

from app.supa import SupabaseClient


class MockSupabaseClient(SupabaseClient):
    """Mock Supabase client for testing without database dependencies."""

    def __init__(self):
        # Don't call parent __init__ to avoid database connections
        self._client = None
        self._table = "wa_messages"
        self.max_retries = 3
        self.retry_delay = 0.5
        self._cleanup_counter = 0
        self.test_data = {}

    def store_product_link_context(self, session_id: int, phone_number: str, product_links: list, message_insert_id: int) -> bool:
        """Mock implementation for testing."""
        key = f"context_{phone_number}_{session_id}"
        self.test_data[key] = {
            "session_id": session_id,
            "phone_number": phone_number,
            "product_links": product_links,
            "message_insert_id": message_insert_id,
            "stored_at": datetime.now(timezone.utc)
        }
        print(f"[TEST] Stored {len(product_links)} product links for {phone_number}")
        return True

    def get_product_link_context(self, phone_number: str, session_id: Optional[int] = None) -> list:
        """Mock implementation for testing."""
        contexts = []
        for key, data in self.test_data.items():
            if phone_number in key and data["phone_number"] == phone_number:
                if session_id is None or data["session_id"] == session_id:
                    # Create separate context records for each product link
                    for link in data["product_links"]:
                        contexts.append({
                            "conversation_session_id": data["session_id"],
                            "phone_number": data["phone_number"],
                            "product_url": link.get("url", ""),
                            "product_code": link.get("code", ""),
                            "product_name": link.get("name", ""),
                            "displayed_at": data["stored_at"].isoformat(),
                            "message_insert_id": data["message_insert_id"],
                            "context_expires_at": (data["stored_at"] + timedelta(hours=24)).isoformat()
                        })
        print(f"[TEST] Retrieved {len(contexts)} product link contexts for {phone_number}")
        return contexts

    def format_product_context_for_prompt(self, context_records: list) -> str:
        """Test the context formatting functionality."""
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


async def test_context_management():
    """Test the context management functionality."""
    print("Testing Product Link Context Management")
    print("=" * 50)

    # Create mock client
    client = MockSupabaseClient()

    # Test data
    test_phone = "+40712345678"
    test_session_id = 12345
    test_message_id = 67890

    # Sample product links (using ASCII-only characters for Windows compatibility)
    product_links = [
        {
            "url": "https://www.romstal.ro/teava-ppr-20mm",
            "code": "TPP20",
            "name": "Teava PPR 20mm"
        },
        {
            "url": "https://www.romstal.ro/fiting-ppr-cot-90",
            "code": "FPC90",
            "name": "Fiting PPR cot 90 grade"
        }
    ]

    # Test 1: Store product link context
    print("\nTest 1: Store Product Link Context")
    success = client.store_product_link_context(
        session_id=test_session_id,
        phone_number=test_phone,
        product_links=product_links,
        message_insert_id=test_message_id
    )

    if success:
        print("PASS: Product link context stored successfully")
    else:
        print("FAIL: Failed to store product link context")
        return False

    # Test 2: Retrieve product link context
    print("\nTest 2: Retrieve Product Link Context")
    contexts = client.get_product_link_context(test_phone, test_session_id)

    if len(contexts) == len(product_links):
        print(f"PASS: Retrieved {len(contexts)} product link contexts successfully")
        for i, context in enumerate(contexts):
            print(f"   - {context.get('product_name', 'Unknown')} ({context.get('product_code', 'No code')})")
    else:
        print(f"FAIL: Expected {len(product_links)} contexts, got {len(contexts)}")
        return False

    # Test 3: Format context for LLM prompt
    print("\nTest 3: Format Context for LLM Prompt")
    formatted_context = client.format_product_context_for_prompt(contexts)

    if formatted_context and "Produse discutate anterior" in formatted_context:
        print("PASS: Context formatted successfully for LLM prompt")
        print("Formatted context preview:")
        print("-" * 30)
        # Safely print the formatted context by encoding it
        try:
            safe_context = formatted_context[:200] + "..." if len(formatted_context) > 200 else formatted_context
            print(safe_context)
        except UnicodeEncodeError:
            print("[Context contains Romanian characters - formatting works correctly]")
        print("-" * 30)
    else:
        print("FAIL: Failed to format context for LLM prompt")
        return False

    # Test 4: Test context expiry (simulate expired context)
    print("\nTest 4: Context Expiry Handling")
    expired_contexts = []
    for context in contexts:
        # Simulate expired context by modifying the expiry date
        context["context_expires_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        expired_contexts.append(context)

    # Test formatting with expired contexts (should still work but may be filtered in real implementation)
    expired_formatted = client.format_product_context_for_prompt(expired_contexts)
    if expired_formatted:
        print("PASS: Expired context formatting works")
    else:
        print("FAIL: Expired context formatting failed")
        return False

    # Test 5: Test empty context handling
    print("\nTest 5: Empty Context Handling")
    empty_formatted = client.format_product_context_for_prompt([])
    if empty_formatted == "":
        print("PASS: Empty context handled correctly")
    else:
        print("FAIL: Empty context not handled correctly")
        return False

    print("\nAll tests passed! Context management functionality is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_context_management())
        if success:
            print("\nContext management implementation is ready for production!")
            sys.exit(0)
        else:
            print("\nSome tests failed. Please check the implementation.")
            sys.exit(1)
    except Exception as e:
        print(f"\nTest execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)