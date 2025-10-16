import logging
import httpx
import asyncio
import json
from typing import Optional, Dict, Any
from .settings import settings
from .correlation import generate_correlation_id, CorrelationContext

logger = logging.getLogger(__name__)


class ProductService:
    def __init__(self):
        self.api_url = settings.romstal_product_url
        self.timeout = 10.0
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds

    async def fetch_product_details(self, code: str) -> str:
        """
        Fetch product details from Romstal API endpoint.

        Args:
            code: The product code to look up

        Returns:
            String containing formatted product information
        """
        if not code or not code.strip():
            return "Cod produs invalid sau lipsă."

        correlation_id = generate_correlation_id()

        with CorrelationContext(correlation_id):
            logger.info(f"[PRODUCT] [{correlation_id}] Fetching details for code: {code}")

            if not self.api_url:
                logger.warning(f"[PRODUCT] [{correlation_id}] ROMSTAL_PRODUCT_URL not configured")
                return "Serviciul de căutare produse nu este configurat momentan."

            # Construct the full API URL
            # Note: api_url already includes "?code=" so we just append the code value
            api_endpoint = f"{self.api_url}{code.strip()}"

            last_exception = None

            for attempt in range(self.max_retries):
                try:
                    logger.info(f"[PRODUCT] [{correlation_id}] Attempt {attempt + 1}/{self.max_retries} to fetch product {code}")

                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.get(api_endpoint)

                        logger.info(f"[PRODUCT] [{correlation_id}] Response status: {response.status_code}")

                        # Check for successful response
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                logger.info(f"[PRODUCT] [{correlation_id}] Successfully parsed JSON response")

                                # Return raw JSON data for AI to process (no filtering)
                                return self._format_raw_json_for_ai(data, code)
                            except json.JSONDecodeError as e:
                                logger.error(f"[PRODUCT] [{correlation_id}] Invalid JSON response: {e}")
                                if attempt == self.max_retries - 1:  # Last attempt
                                    return f"Am găsit produsul cu codul {code}, dar răspunsul serverului este invalid. Te rog să încerci din nou."

                        elif response.status_code == 404:
                            logger.warning(f"[PRODUCT] [{correlation_id}] Product not found: {code}")
                            return f"Nu am găsit niciun produs cu codul '{code}'. Te rog să verifici codul și să încerci din nou."

                        elif response.status_code >= 500:
                            logger.error(f"[PRODUCT] [{correlation_id}] Server error {response.status_code}: {response.text}")
                            if attempt == self.max_retries - 1:  # Last attempt
                                return f"Eroare temporară a serverului ({response.status_code}). Te rog să încerci din nou peste câteva momente."

                        else:
                            logger.warning(f"[PRODUCT] [{correlation_id}] HTTP error {response.status_code}: {response.text}")
                            if attempt == self.max_retries - 1:  # Last attempt
                                return f"Eroare la căutarea produsului ({response.status_code}). Te rog să încerci din nou."

                except httpx.TimeoutException as e:
                    last_exception = e
                    logger.warning(f"[PRODUCT] [{correlation_id}] Timeout on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

                except httpx.ConnectError as e:
                    last_exception = e
                    logger.error(f"[PRODUCT] [{correlation_id}] Connection error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

                except httpx.NetworkError as e:
                    last_exception = e
                    logger.error(f"[PRODUCT] [{correlation_id}] Network error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

                except Exception as e:
                    last_exception = e
                    logger.exception(f"[PRODUCT] [{correlation_id}] Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

            # All retries failed
            logger.error(f"[PRODUCT] [{correlation_id}] All {self.max_retries} attempts failed. Last error: {type(last_exception).__name__}: {last_exception}")
            return f"Eroare la căutarea produsului {code}. Te rog să încerci din nou mai târziu."

    def _format_raw_json_for_ai(self, data: Dict[str, Any], code: str) -> str:
        """
        Format raw JSON data for AI processing without filtering.

        Args:
            data: Raw JSON response data from the API
            code: Product code for context

        Returns:
            Formatted string containing complete JSON data for AI analysis
        """
        try:
            # Return complete JSON data as formatted string for AI to analyze
            json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)

            return f"Date complete produs {code} (JSON brut):\n\n```json\n{json_str}\n```\n\nAI: Analizează aceste date și oferă informații relevante despre produs într-un format natural și helpful."

        except Exception as e:
            logger.error(f"Error formatting raw JSON for AI: {e}")
            return f"Am găsit produsul cu codul {code}, dar nu am putut procesa datele JSON. Date brute: {data}"

    def _format_product_data(self, data: Dict[str, Any], code: str) -> str:
        """
        Format product data into a user-friendly string with comprehensive field extraction.

        Args:
            data: JSON response data from the API
            code: Product code for fallback

        Returns:
            Formatted product information string
        """
        try:
            # Debug logging to see the actual API response structure
            logger.info(f"[PRODUCT] Raw API response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            logger.info(f"[PRODUCT] Raw API response: {data}")

            # Handle different possible response structures
            if not data:
                return f"Am găsit produsul cu codul {code}, dar nu sunt disponibile detalii suplimentare."

            # Handle nested structures (check if data is wrapped in another object)
            product_data = data
            if 'product' in data:
                product_data = data['product']
            elif 'data' in data:
                product_data = data['data']
            elif 'result' in data:
                product_data = data['result']

            # If product_data is still a dict but might contain a list or nested structure
            if isinstance(product_data, list) and len(product_data) > 0:
                product_data = product_data[0]  # Take first item if it's a list

            # Comprehensive field extraction with multiple naming variations
            fields_to_check = {
                'name': ['name', 'nume', 'title', 'denumire', 'product_name', 'nume_produs'],
                'description': ['description', 'descriere', 'details', 'detalii', 'specifications', 'specs'],
                'price': ['price', 'pret', 'cost', 'pret_vanzare', 'selling_price'],
                'availability': ['availability', 'disponibilitate', 'stock', 'in_stock', 'status'],
                'category': ['category', 'categoria', 'category_name', 'nume_categorie'],
                'brand': ['brand', 'marca', 'manufacturer', 'producator'],
                'code': ['code', 'cod', 'product_code', 'cod_produs'],
                'unit': ['unit', 'unitate', 'measure_unit', 'unitate_masura'],
                'weight': ['weight', 'greutate', 'mass', 'masa'],
                'dimensions': ['dimensions', 'dimensiuni', 'size', 'marime']
            }

            extracted_info = {}
            for field_name, variations in fields_to_check.items():
                for variation in variations:
                    value = self._get_nested_value(product_data, variation)
                    if value is not None:
                        extracted_info[field_name] = value
                        logger.info(f"[PRODUCT] Found {field_name}: {value}")
                        break

            # Build comprehensive response
            parts = []

            # Product name/title (most important)
            if 'name' in extracted_info:
                name = str(extracted_info['name']).strip()
                if name:
                    parts.append(f"**{name}**")

            # Product code if different from searched code
            if 'code' in extracted_info and str(extracted_info['code']) != code:
                parts.append(f"Cod produs: {extracted_info['code']}")

            # Category and brand
            if 'category' in extracted_info:
                parts.append(f"Categorie: {extracted_info['category']}")
            if 'brand' in extracted_info:
                parts.append(f"Marcă: {extracted_info['brand']}")

            # Price (try to format properly)
            if 'price' in extracted_info:
                price_str = str(extracted_info['price'])
                try:
                    # Handle various price formats
                    price_clean = price_str.replace(' RON', '').replace(' lei', '').replace(',', '').replace(' ', '')
                    price_float = float(price_clean)
                    parts.append(f"💰 Preț: {price_float:.2f} RON")
                except (ValueError, AttributeError):
                    parts.append(f"💰 Preț: {price_str}")

            # Availability/Stock status
            if 'availability' in extracted_info:
                availability = str(extracted_info['availability']).lower().strip()
                if availability in ['true', '1', 'yes', 'da', 'disponibil', 'in stock', 'available']:
                    parts.append("✅ În stoc")
                elif availability in ['false', '0', 'no', 'nu', 'indisponibil', 'out of stock', 'unavailable']:
                    parts.append("❌ Indisponibil")
                else:
                    parts.append(f"📦 Stoc: {extracted_info['availability']}")

            # Unit of measure
            if 'unit' in extracted_info:
                parts.append(f"📏 Unitate: {extracted_info['unit']}")

            # Physical properties
            if 'weight' in extracted_info:
                parts.append(f"⚖️ Greutate: {extracted_info['weight']}")
            if 'dimensions' in extracted_info:
                parts.append(f"📐 Dimensiuni: {extracted_info['dimensions']}")

            # Description/Details (most detailed information)
            if 'description' in extracted_info:
                description = str(extracted_info['description']).strip()

                # Clean up the description
                import re
                # Remove HTML tags if present
                description = re.sub(r'<[^>]+>', '', description)
                # Remove extra whitespace
                description = re.sub(r'\s+', ' ', description).strip()

                if len(description) > 1000:
                    description = description[:1000] + "..."

                if description:
                    parts.append(f"📋 Descriere: {description}")

            # If we have no useful information extracted
            if not parts:
                # Try to return the entire response as a string if it's meaningful
                data_str = str(data).strip()
                if len(data_str) > 10 and data_str not in ['{}', '[]', 'null']:
                    if len(data_str) > 500:
                        data_str = data_str[:500] + "..."
                    return f"Am găsit produsul cu codul {code} în sistemul intern Romstal, dar fișa returnată este incompletă. Iată ce informații avem acum:\n\n{data_str}"
                else:
                    return f"Am găsit produsul cu codul {code}, dar informațiile disponibile sunt limitate."

            return "\n".join(parts)

        except Exception as e:
            logger.error(f"Error formatting product data: {e}")
            logger.error(f"Data type: {type(data)}, Data content: {data}")
            return f"Am găsit produsul cu codul {code}, dar nu am putut procesa informațiile. Te rog să încerci din nou."

    def _get_nested_value(self, data: Any, key_path: str) -> Any:
        """
        Extract value from nested dictionary/list structure using dot notation.

        Args:
            data: The data structure to search
            key_path: Key path like 'field' or 'nested.field' or 'array.0.field'

        Returns:
            The value if found, None otherwise
        """
        try:
            keys = key_path.split('.')
            current = data

            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return None

                if current is None:
                    return None

            return current
        except (KeyError, IndexError, TypeError, ValueError):
            return None


# Global product service instance
product_service = ProductService()