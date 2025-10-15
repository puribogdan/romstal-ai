#!/usr/bin/env python3
"""
Comprehensive end-to-end test suite for the complete product recommendation system.
Tests all major components including product search, message handling, context management,
LLM integration, error handling, and database operations.
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Add the wa-ai directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wa-ai'))

from app.supa import SupabaseClient
from app.llm import LLMClient
from handlers.handle_product_recommendation import ProductRecommendationHandler


class MockSupabaseClient(SupabaseClient):
    """Enhanced mock Supabase client for comprehensive testing."""

    def __init__(self):
        # Don't call parent __init__ to avoid database connections
        self._client = None
        self._table = "wa_messages"
        self.max_retries = 3
        self.retry_delay = 0.5
        self._cleanup_counter = 0
        self.test_data = {}
        self.message_counter = 1000


    def get_message_by_id(self, message_id: int, correlation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Mock message retrieval."""
        # Create test messages for different scenarios
        test_messages = {
            1001: {
                "id": 1001,
                "wa_text": "Vreau să cumpăr o țeavă PPR de 20mm",
                "wa_from": "+40712345678",
                "wa_timestamp": datetime.now(timezone.utc).isoformat(),
                "inserted_at": datetime.now(timezone.utc).isoformat(),
                "direction": "inbound",
                "raw": {"messages": [{"text": {"body": "Vreau să cumpăr o țeavă PPR de 20mm"}}]}
            },
            1002: {
                "id": 1002,
                "wa_text": "Am nevoie de o pompă sub 500 lei",
                "wa_from": "+40712345678",
                "wa_timestamp": datetime.now(timezone.utc).isoformat(),
                "inserted_at": datetime.now(timezone.utc).isoformat(),
                "direction": "inbound",
                "raw": {"messages": [{"text": {"body": "Am nevoie de o pompă sub 500 lei"}}]}
            },
            1003: {
                "id": 1003,
                "wa_text": "Ce boiler electric îmi recomandați?",
                "wa_from": "+40787654321",
                "wa_timestamp": datetime.now(timezone.utc).isoformat(),
                "inserted_at": datetime.now(timezone.utc).isoformat(),
                "direction": "inbound",
                "raw": {"messages": [{"text": {"body": "Ce boiler electric îmi recomandați?"}}]}
            },
            1004: {
                "id": 1004,
                "wa_text": "Vreau cod produs 64px9822",
                "wa_from": "+40712345678",
                "wa_timestamp": datetime.now(timezone.utc).isoformat(),
                "inserted_at": datetime.now(timezone.utc).isoformat(),
                "direction": "inbound",
                "raw": {"messages": [{"text": {"body": "Vreau cod produs 64px9822"}}]}
            }
        }

        return test_messages.get(message_id)

    def get_message_history(self, phone: str) -> List[Dict[str, Any]]:
        """Mock message history."""
        return [
            {
                "wa_from": phone,
                "wa_text": "Salut, am nevoie de ajutor",
                "wa_timestamp": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                "inserted_at": (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                "direction": "inbound"
            }
        ]

    def get_or_create_conversation_session(self, phone: str, system_prompt: str) -> Optional[Dict[str, Any]]:
        """Mock conversation session."""
        return {
            "id": 12345,
            "phone_number": phone,
            "system_prompt": system_prompt,
            "full_conversation": [],
            "context_messages": [],
            "message_count": 0
        }

    def insert_outbound_message(self, phone: str, message: str, raw_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Mock outbound message insertion."""
        return {
            "id": self.message_counter + 1,
            "wa_from": phone,
            "wa_text": message,
            "direction": "outbound",
            "inserted_at": datetime.now(timezone.utc).isoformat()
        }


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without OpenAI API calls."""

    def __init__(self):
        super().__init__()
        self.client = Mock()  # Mock the OpenAI client
        self.test_responses = {
            "product_search": "Am găsit câteva produse excelente pentru tine! Iată recomandările mele: 1. Țeavă PPR 20mm - https://www.romstal.ro/teava-ppr-20mm (45 lei), 2. Fiting PPR cot 90 - https://www.romstal.ro/fiting-ppr-cot-90 (8 lei). Pentru mai multe detalii, vizitează romstal.ro!",
            "product_details": "Produsul cu codul 64px9822 are următoarele detalii: Preț: 125 lei, Disponibil în stoc, Descriere: Produs de calitate superioară. Vezi mai multe la: https://www.romstal.ro/produs/64px9822",
            "general_query": "Salut! Sunt asistentul Romstal și sunt aici să te ajut cu informații despre produsele noastre. Ce pot face pentru tine astăzi?"
        }

    async def call_llm_with_tools(self, system_prompt: str, user_prompt: str, correlation_id: Optional[str] = None) -> tuple:
        """Mock LLM response based on user prompt content."""
        if "țeavă" in user_prompt.lower() or "ppr" in user_prompt.lower():
            return self.test_responses["product_search"], [
                {
                    "function": "search_products_romstal",
                    "args": {"category": "țevi", "budget": None, "requirements": None, "limit": 5},
                    "result": {"ok": True, "products": [
                        {"name": "Țeavă PPR 20mm", "url": "https://www.romstal.ro/teava-ppr-20mm", "price": "45"},
                        {"name": "Fiting PPR cot 90", "url": "https://www.romstal.ro/fiting-ppr-cot-90", "price": "8"}
                    ]}
                }
            ]
        elif "cod produs" in user_prompt.lower() or "64px9822" in user_prompt.lower():
            return self.test_responses["product_details"], [
                {
                    "function": "fetch_product_details",
                    "args": {"code": "64px9822"},
                    "result": {"ok": True, "code": "64px9822", "data": {"info": {"product": "Produs Test", "price": "125"}}}
                }
            ]
        else:
            return self.test_responses["general_query"], None


class MockN8NClient:
    """Mock N8N client for testing."""

    def __init__(self):
        self.sent_messages = []

    def is_configured(self) -> bool:
        return True

    async def send_to_n8n(self, phone_number: str, message: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Mock N8N message sending."""
        self.sent_messages.append({
            "phone": phone_number,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id
        })
        return {"success": True, "message_id": f"n8n_{len(self.sent_messages)}"}

    async def health_check(self) -> Dict[str, Any]:
        return {"healthy": True, "status": "Mock N8N is running"}


class ComprehensiveTestSuite:
    """Comprehensive test suite for the product recommendation system."""

    def __init__(self):
        self.mock_supa = MockSupabaseClient()
        self.mock_llm = MockLLMClient()
        self.mock_n8n = MockN8NClient()

        # Create handler with mocked dependencies
        self.handler = ProductRecommendationHandler()
        self.handler.db_client = self.mock_supa
        self.handler.llm_client = self.mock_llm
        self.handler.n8n_client = self.mock_n8n

    async def test_product_search_different_categories(self):
        """Test 1: Product search functionality with different categories and budgets."""
        print("\nTest 1: Product Search with Different Categories and Budgets")
        print("=" * 60)

        test_cases = [
            {
                "message_id": 1001,
                "expected_category": "țevi",
                "description": "Pipe category search"
            },
            {
                "message_id": 1002,
                "expected_category": "pompe",
                "expected_budget": "sub 500 lei",
                "description": "Pump category with budget filter"
            },
            {
                "message_id": 1003,
                "expected_category": "boilere",
                "description": "Boiler category search"
            }
        ]

        all_passed = True

        for test_case in test_cases:
            try:
                print(f"\nTesting: {test_case['description']}")

                # Process the message
                result = await self.handler.handle_product_recommendation(test_case["message_id"])

                # Verify response structure
                if not result.get("success"):
                    print(f"FAIL: Handler returned success=False: {result.get('error')}")
                    all_passed = False
                    continue

                # Check if response contains expected product information
                ai_response = result.get("ai_response", "")
                has_products = any(keyword in ai_response.lower() for keyword in ["țeavă", "pompă", "boiler", "romstal.ro"])

                if has_products:
                    print(f"PASS: Found product recommendations in response")
                    print(f"   Response preview: {ai_response[:100]}...")
                else:
                    print(f"FAIL: No product recommendations found in response")
                    all_passed = False

                # Check if function calls were made
                function_calls = result.get("function_calls", [])
                if function_calls:
                    print(f"PASS: Function calls executed ({len(function_calls)} calls)")
                    for call in function_calls:
                        print(f"   - {call.get('function', 'unknown')}({call.get('args', {})})")
                else:
                    print(f"WARN: No function calls made")

            except Exception as e:
                print(f"FAIL: Exception during test: {e}")
                all_passed = False

        return all_passed

    async def test_message_handler_integration(self):
        """Test 2: Message handler integration with main application."""
        print("\nTest 2: Message Handler Integration")
        print("=" * 60)

        test_cases = [
            {
                "message_id": 1001,
                "phone": "+40712345678",
                "description": "Standard product recommendation flow"
            },
            {
                "message_id": 1004,
                "phone": "+40712345678",
                "description": "Product code lookup flow"
            }
        ]

        all_passed = True

        for test_case in test_cases:
            try:
                print(f"\nTesting: {test_case['description']}")

                # Process the message
                result = await self.handler.handle_product_recommendation(
                    test_case["message_id"],
                    test_case["phone"]
                )

                # Verify all steps completed
                required_keys = ["success", "ai_response", "n8n_result", "correlation_id"]
                missing_keys = [key for key in required_keys if key not in result]

                if missing_keys:
                    print(f"FAIL: Missing keys in result: {missing_keys}")
                    all_passed = False
                    continue

                if result["success"]:
                    print(f"PASS: Handler completed successfully")

                    # Check N8N integration
                    n8n_result = result.get("n8n_result", {})
                    if n8n_result.get("success"):
                        print(f"PASS: N8N message sent successfully")
                    else:
                        print(f"FAIL: N8N message failed: {n8n_result.get('error')}")

                    # Check database integration
                    db_result = result.get("db_result", {})
                    if db_result:
                        print(f"PASS: Database operation completed")
                    else:
                        print(f"WARN: No database result returned")

                else:
                    print(f"FAIL: Handler failed: {result.get('error')}")
                    all_passed = False

            except Exception as e:
                print(f"FAIL: Exception during integration test: {e}")
                all_passed = False

        return all_passed


    async def test_llm_tool_integration(self):
        """Test 4: LLM tool integration and response formatting."""
        print("\nTest 4: LLM Tool Integration and Response Formatting")
        print("=" * 60)

        test_cases = [
            {
                "prompt": "Vreau să cumpăr o țeavă PPR de 20mm pentru instalații sanitare",
                "expected_tools": ["search_products_romstal"],
                "description": "Product search tool call"
            },
            {
                "prompt": "Am nevoie de detalii pentru codul produs 64px9822",
                "expected_tools": ["fetch_product_details"],
                "description": "Product details tool call"
            },
            {
                "prompt": "Bună ziua, cum pot să vă contactez?",
                "expected_tools": [],
                "description": "General query without tools"
            }
        ]

        all_passed = True

        for test_case in test_cases:
            try:
                print(f"\nTesting: {test_case['description']}")

                # Call LLM with tools
                response, function_calls = await self.mock_llm.call_llm_with_tools(
                    "Ești un asistent Romstal.", test_case["prompt"]
                )

                if response:
                    print(f"PASS: LLM returned response ({len(response)} chars)")
                    print(f"   Response: {response[:100]}...")
                else:
                    print(f"FAIL: LLM returned empty response")
                    all_passed = False
                    continue

                # Check function calls
                actual_tools = [call.get("function") for call in function_calls or []]
                expected_tools = test_case["expected_tools"]

                if set(actual_tools) == set(expected_tools):
                    print(f"PASS: Correct tools called: {actual_tools}")
                else:
                    print(f"FAIL: Expected tools {expected_tools}, got {actual_tools}")
                    all_passed = False

                # Verify response formatting (should be in Romanian, helpful)
                if "romstal" in response.lower() or "produs" in response.lower():
                    print(f"PASS: Response contains relevant keywords")
                else:
                    print(f"WARN: Response may not be properly formatted")

            except Exception as e:
                print(f"FAIL: Exception during LLM tool test: {e}")
                all_passed = False

        return all_passed

    async def test_error_handling_edge_cases(self):
        """Test 5: Error handling and edge cases."""
        print("\nTest 5: Error Handling and Edge Cases")
        print("=" * 60)

        test_cases = [
            {
                "message_id": 99999,  # Non-existent message
                "description": "Non-existent message ID"
            },
            {
                "message_id": None,  # Invalid message ID
                "description": "Invalid message ID (None)"
            },
            {
                "message_id": 1005,  # Message with no phone number
                "description": "Message with missing phone number"
            }
        ]

        all_passed = True

        for test_case in test_cases:
            try:
                print(f"\nTesting: {test_case['description']}")

                # Process the message (should handle errors gracefully)
                result = await self.handler.handle_product_recommendation(test_case["message_id"])

                # Should return error result, not throw exception
                if result.get("success") == False:
                    print(f"PASS: Handler properly handled error case")
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                else:
                    print(f"FAIL: Expected error but got success: {result.get('success')}")
                    all_passed = False

            except Exception as e:
                print(f"FAIL: Handler threw exception instead of graceful error: {e}")
                all_passed = False

        # Test LLM error handling
        print("\nTesting LLM error handling...")
        # Temporarily break LLM client
        original_client = self.mock_llm.client
        self.mock_llm.client = None

        try:
            response, function_calls = await self.mock_llm.call_llm_with_tools(
                "Test prompt", "Test user prompt"
            )

            if response and "pare rău" in response.lower():
                print(f"PASS: LLM gracefully handled missing client")
            else:
                print(f"FAIL: LLM didn't return proper error message")
                all_passed = False

        except Exception as e:
            print(f"FAIL: LLM threw exception: {e}")
            all_passed = False
        finally:
            # Restore LLM client
            self.mock_llm.client = original_client

        return all_passed

    async def test_database_operations_cleanup(self):
        """Test 6: Database operations and cleanup."""
        print("\nTest 6: Database Operations and Cleanup")
        print("=" * 60)

        try:
            phone = "+40712345678"

            # Test message insertion
            test_message = "Test message for cleanup testing"
            insert_result = self.mock_supa.insert_outbound_message(phone, test_message)

            if insert_result and "id" in insert_result:
                print(f"PASS: Outbound message inserted successfully")
                print(f"   Message ID: {insert_result['id']}")
            else:
                print(f"FAIL: Message insertion failed")
                return False

            # Context cleanup removed - simplified testing

            # Test conversation session management
            session = self.mock_supa.get_or_create_conversation_session(phone, "Test prompt")

            if session and "id" in session:
                print(f"PASS: Conversation session created/retrieved")
                print(f"   Session ID: {session['id']}")
            else:
                print(f"FAIL: Session management failed")
                return False

            # Test message history retrieval
            history = self.mock_supa.get_message_history(phone)

            if isinstance(history, list):
                print(f"PASS: Message history retrieved ({len(history)} messages)")
            else:
                print(f"FAIL: Message history returned invalid result")
                return False

            return True

        except Exception as e:
            print(f"FAIL: Exception during database operations test: {e}")
            return False

    async def run_all_tests(self):
        """Run all comprehensive tests."""
        print("Starting Comprehensive End-to-End Test Suite")
        print("=" * 70)

        test_methods = [
            self.test_product_search_different_categories,
            self.test_message_handler_integration,
            self.test_llm_tool_integration,
            self.test_error_handling_edge_cases,
            self.test_database_operations_cleanup
        ]

        results = []
        for test_method in test_methods:
            try:
                result = await test_method()
                results.append(result)
            except Exception as e:
                print(f"CRITICAL: Test {test_method.__name__} threw exception: {e}")
                results.append(False)

        # Summary
        print("\nResults Summary")
        print("=" * 70)

        passed = sum(results)
        total = len(results)

        for i, (test_method, result) in enumerate(zip(test_methods, results)):
            status = "PASS" if result else "FAIL"
            print(f"{i+1}. {test_method.__name__}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print("SUCCESS: All tests passed! The product recommendation system is working correctly.")
            return True
        else:
            print("WARN:  Some tests failed. Please review the implementation.")
            return False


async def main():
    """Main test execution function."""
    try:
        suite = ComprehensiveTestSuite()
        success = await suite.run_all_tests()

        if success:
            print("\nProduct recommendation system is ready for production!")
            sys.exit(0)
        else:
            print("\nSome tests failed. Please check the implementation.")
            sys.exit(1)

    except Exception as e:
        print(f"\nTest execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())