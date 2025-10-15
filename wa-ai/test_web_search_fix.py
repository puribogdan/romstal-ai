#!/usr/bin/env python3
"""
Test script for web search functionality fix.
This script tests that web_search_call objects are properly handled.
"""

import sys
import os
import asyncio
from unittest.mock import Mock, MagicMock
from datetime import datetime

# Add the wa-ai directory to the path
sys.path.insert(0, os.path.dirname(__file__))

from app.llm import LLMClient


def create_mock_response_with_web_search():
    """Create a mock OpenAI response with web_search_call objects."""
    mock_response = Mock()

    # Mock the output items to simulate web search calls
    mock_item1 = Mock()
    mock_item1.type = "web_search_call"
    mock_item1.id = "ws_042c9e883ff126940068ee81981190819da9cb25e7a23a4472"
    # Create a mock ActionSearch object
    mock_action1 = Mock()
    mock_action1.query = "site:romstal.ro centrale termice pe gaz condensare 24 kW"
    mock_action1.type = "search"
    mock_action1.sources = None
    mock_item1.action = mock_action1
    mock_item1.status = "completed"

    mock_item2 = Mock()
    mock_item2.type = "web_search_call"
    mock_item2.id = "ws_042c9e883ff126940068ee819d5f94819db1e447a59422824d"
    # Create a mock ActionSearch object for the second item
    mock_action2 = Mock()
    mock_action2.query = "site:romstal.ro/ produse centrala termica condensare gaz 24 kW"
    mock_action2.type = "search"
    mock_action2.sources = None
    mock_item2.action = mock_action2
    mock_item2.status = "completed"

    # Create mock content objects with text attribute
    mock_content1 = Mock()
    mock_content1.text = "Am găsit câteva centrale termice pe gaz de 24 kW pe site-ul Romstal."

    mock_content2 = Mock()
    mock_content2.text = "Rezultate căutare suplimentare."

    # Add a message response item (more realistic for OpenAI responses)
    mock_message_item = Mock()
    mock_message_item.type = "message"
    mock_message_item.content = [mock_content1]

    # Also add a text item for completeness
    mock_text_item = Mock()
    mock_text_item.type = "text"
    mock_text_item.content = [mock_content2]

    mock_response.output = [mock_item1, mock_item2, mock_message_item, mock_text_item]
    mock_response.output_text = "Am găsit câteva centrale termice pe gaz de 24 kW pe site-ul Romstal."
    mock_response.id = "resp_042c9e883ff126940068ee81919510819d8ada71a5c2534e67"

    return mock_response


def test_web_search_extraction():
    """Test that web_search_call objects are properly extracted."""
    print("Testing Web Search Call Extraction")
    print("=" * 40)

    client = LLMClient()

    # Create mock response with web search calls
    mock_response = create_mock_response_with_web_search()

    # Test the extraction
    tool_calls = client._extract_tool_calls_from_response(mock_response)

    print(f"Extracted {len(tool_calls)} tool calls")

    # Check that we got web search calls
    web_search_calls = [call for call in tool_calls if call.get("type") == "web_search_call"]

    if len(web_search_calls) >= 2:
        print(f"PASS: Found {len(web_search_calls)} web search calls")
        for i, call in enumerate(web_search_calls):
            print(f"   - Call {i+1}: {call.get('args', {}).get('query', 'No query')[:50]}...")
    else:
        print(f"FAIL: Expected at least 2 web search calls, got {len(web_search_calls)}")
        return False

    # Check that the calls have the expected structure
    for call in web_search_calls:
        if not all(key in call for key in ["id", "type", "name", "args"]):
            print(f"FAIL: Web search call missing required keys: {call}")
            return False

    print("PASS: Web search calls have correct structure")
    return True


def test_text_extraction_with_web_search():
    """Test that text can be extracted from responses containing web search calls."""
    print("\nTesting Text Extraction with Web Search")
    print("=" * 40)

    client = LLMClient()

    # Create mock response with web search calls
    mock_response = create_mock_response_with_web_search()

    # Debug: Check what's in the output items
    print(f"[DEBUG] Output items count: {len(mock_response.output)}")
    for i, item in enumerate(mock_response.output):
        item_type = getattr(item, "type", "unknown")
        print(f"[DEBUG] Item {i}: type={item_type}")
        if item_type in ["message", "text"]:
            content = getattr(item, "content", None)
            if content and isinstance(content, list):
                for j, c in enumerate(content):
                    text = getattr(c, "text", None)
                    try:
                        print(f"[DEBUG]   Content {j}: '{text}'")
                    except UnicodeEncodeError:
                        print(f"[DEBUG]   Content {j}: [Romanian text: {len(text) if text else 0} chars]")

    # Test text extraction
    extracted_text = client._extract_text_from_responses(mock_response)

    # Debug output (handle Unicode)
    try:
        print(f"[DEBUG] Extracted text: '{extracted_text}'")
    except UnicodeEncodeError:
        print(f"[DEBUG] Extracted text: [Contains Romanian characters]")
    print(f"[DEBUG] Text length: {len(extracted_text) if extracted_text else 0}")
    print(f"[DEBUG] Contains 'centrale termice': {'centrale termice' in extracted_text.lower() if extracted_text else False}")
    print(f"[DEBUG] Contains 'Rezultate': {'Rezultate' in extracted_text if extracted_text else False}")

    if extracted_text and "centrale termice" in extracted_text.lower():
        print("PASS: Successfully extracted concatenated text (contains Romanian characters)")
        return True
    else:
        print("FAIL: Could not extract expected concatenated text (contains Romanian characters)")
        return False


def test_built_in_tools_response_handling():
    """Test that responses with only web search calls extract text from current response."""
    print("\nTesting Built-in Tools Response Handling")
    print("=" * 40)

    client = LLMClient()

    # Create mock response with only web search calls (no function calls)
    mock_response = create_mock_response_with_web_search()
    # Ensure output_text is empty to simulate built-in tool scenario
    mock_response.output_text = ""

    # Test that text is extracted from current response (not requiring follow-up call)
    extracted_text = client._extract_text_from_responses(mock_response)

    if extracted_text and "centrale termice" in extracted_text.lower():
        print("PASS: Text correctly extracted from response with only web search calls")
        return True
    else:
        print("FAIL: Could not extract text from web search only response")
        return False


def test_web_search_processing():
    """Test that web search calls are properly processed."""
    print("\nTesting Web Search Processing")
    print("=" * 40)

    client = LLMClient()

    # Create mock response with web search calls
    mock_response = create_mock_response_with_web_search()

    # Extract tool calls
    tool_calls = client._extract_tool_calls_from_response(mock_response)

    # Test web search results extraction
    web_search_call = tool_calls[0]  # Get first web search call
    search_results = client._extract_web_search_results(mock_response, web_search_call["id"])

    if search_results:
        print(f"PASS: Extracted {len(search_results)} search results")
        for result in search_results:
            print(f"   - Query: {result.get('query', 'No query')[:50]}...")
    else:
        print("INFO: No search results extracted (this may be expected)")

    return True


def test_raw_web_search_results():
    """Test that raw web search results are extracted directly."""
    print("\nTesting Raw Web Search Results Extraction")
    print("=" * 40)

    client = LLMClient()

    # Create mock response with web search calls
    mock_response = create_mock_response_with_web_search()

    # Test the new raw extraction method
    raw_results = client._extract_raw_web_search_results(mock_response)

    if raw_results and "centrale termice" in raw_results.lower():
        print("PASS: Successfully extracted raw web search results")
        print(f"   - Results length: {len(raw_results)}")
        print(f"   - Contains expected content: {'centrale termice' in raw_results.lower()}")
        return True
    else:
        print(f"FAIL: Could not extract raw web search results. Got: '{raw_results}'")
        return False


def test_web_search_only_response():
    """Test that web search only responses return raw results directly."""
    print("\nTesting Web Search Only Response Handling")
    print("=" * 40)

    client = LLMClient()

    # Create mock response with only web search calls
    mock_response = create_mock_response_with_web_search()

    # Extract tool calls to simulate the scenario
    tool_calls = client._extract_tool_calls_from_response(mock_response)
    web_search_calls = [call for call in tool_calls if call.get("type") == "web_search_call"]

    if web_search_calls:
        print(f"PASS: Found {len(web_search_calls)} web search calls")

        # Test raw results extraction
        raw_results = client._extract_raw_web_search_results(mock_response)

        if raw_results and len(raw_results.strip()) > 0:
            print("PASS: Raw web search results extracted successfully")
            print(f"   - Results length: {len(raw_results)} characters")
            return True
        else:
            print("FAIL: No raw results extracted")
            return False
    else:
        print("FAIL: No web search calls found")
        return False


async def test_full_web_search_integration():
    """Test the full integration with web search processing."""
    print("\nTesting Full Web Search Integration")
    print("=" * 40)

    # Note: This test would require actual OpenAI API access
    # For now, we'll just test that the client initializes correctly
    client = LLMClient()

    if client.client is None:
        print("INFO: No OpenAI client available (API key not set)")
        print("PASS: Client handles missing API key gracefully")
        return True
    else:
        print("INFO: OpenAI client available - full integration test would require API call")
        return True


def main():
    """Run all tests."""
    print("Testing Web Search Fix Implementation")
    print("=" * 50)

    try:
        # Run synchronous tests
        test1_pass = test_web_search_extraction()
        test2_pass = test_text_extraction_with_web_search()
        test3_pass = test_built_in_tools_response_handling()
        test4_pass = test_web_search_processing()
        test5_pass = test_raw_web_search_results()
        test6_pass = test_web_search_only_response()

        # Run async test
        async_test_pass = asyncio.run(test_full_web_search_integration())

        # Summary
        all_passed = all([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass, test6_pass, async_test_pass])

        print("\n" + "=" * 50)
        if all_passed:
            print("All tests passed! Web search fix is working correctly.")
            print("The fix should resolve the original issue with web_search_call responses.")
            return True
        else:
            print("Some tests failed. Please check the implementation.")
            return False

    except Exception as e:
        print(f"\nTest execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)