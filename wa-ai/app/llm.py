import json
import logging
import os
import asyncio
import time
import re
from typing import Optional, List, Dict, Any, Tuple
from openai import OpenAI
from bs4 import BeautifulSoup
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
            "description": "Ia detalii de produs Romstal când utilizatorul furnizează un cod de produs (ex: CP12345, 64px9822). Apelează DOAR dacă mesajul conține clar un cod.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Codul produsului cerut de utilizator."
                    }
                },
                "required": ["code"]
            }
        },
        {
            "type": "function",
            "name": "search_products_romstal",
            "description": "Caută produse pe romstal.ro după categorie, buget și cerințe. Folosește pentru recomandări de produse cu filtru pe domeniu romstal.ro.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Categoria produselor (ex: 'țevi', 'fitinguri', 'pompe', 'boilere', 'radiatoare')"
                    },
                    "budget": {
                        "type": "string",
                        "description": "Bugetul aproximativ (ex: 'sub 500 lei', '500-1000 lei', 'peste 1000 lei')"
                    },
                    "requirements": {
                        "type": "string",
                        "description": "Cerințe specifice sau caracteristici dorite (ex: 'material plastic', 'presiune ridicată', 'eficient energetic')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Numărul maxim de rezultate (default: 5, maxim: 10)",
                        "default": 5
                    }
                },
                "required": ["category"]
            }
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
                # Skip reasoning items and function calls, look for text content
                item_type = getattr(item, "type", None)
                if item_type in ["reasoning", "function_call"]:
                    continue

                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        t = getattr(c, "text", None)
                        if isinstance(t, str) and t.strip():
                            return t.strip()

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
        Returnează o listă de dicturi:
          {"id": <tool_call_id>, "name": <tool_name>, "args": <dict>}
        Suportă atât stilul 'function_call' (cu call_id) cât și 'tool_use'.
        """
        calls = []
        output = getattr(resp, "output", []) or []

        for item in output:
            typ = getattr(item, "type", None)
            contents = getattr(item, "content", None)

            # Stil 'function_call' (cel mai frecvent în Responses acum)
            if typ == "function_call" and getattr(item, "name", None):
                raw_args = getattr(item, "arguments", None)
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {}
                calls.append({
                    "id": getattr(item, "call_id", None),
                    "name": getattr(item, "name", None),
                    "args": args
                })

            # Stil 'tool_use' (fallback)
            if isinstance(contents, list):
                for c in contents:
                    if getattr(c, "type", None) == "tool_use" and getattr(c, "name", None):
                        calls.append({
                            "id": getattr(c, "id", None),   # tool_use_id
                            "name": getattr(c, "name", None),
                            "args": getattr(c, "input", {}) or {}
                        })
        return calls

    def extract_relevant_product_data(self, raw: dict) -> dict:
        """Extrage doar câmpurile esențiale din JSON-ul API Romstal."""
        result = {}

        # 1) info (core product)
        info = raw.get("info", {}) or {}
        result["info"] = {k: info.get(k) for k in [
            "productid", "masterid", "code", "barcode", "product", "subtitle",
            "master_product", "description", "brand", "brandid", "category",
            "categoryid", "categoryurl", "url", "price", "oldprice", "currencyid",
            "vat", "emag_price", "ispromo", "promo", "promo_from", "promo_to",
            "warranty", "stock", "stockno", "stockid", "isvariant",
            "variants_count", "modified_on"
        ] if k in info}

        # 2) rating
        rating = raw.get("rating", {}) or {}
        result["rating"] = {
            "rating": rating.get("rating"),
            "noreview": rating.get("noreview")
        }

        # 3) related.1.products (esențiale)
        related = raw.get("related", {}) or {}
        related_1 = related.get("1", {}) or {}
        products = related_1.get("products", []) or []
        essentials = []
        for p in products:
            essentials.append({k: p.get(k) for k in [
                "productid", "code", "product", "brand",
                "price", "stock", "url", "categoryid", "params"
            ] if k in p})
        result["related_products"] = essentials

        return result

    def remove_none_fields(self, obj):
        """Curăță None / "" / [] din structura rezultată (opțional dar recomandat)."""
        if isinstance(obj, dict):
            return {k: self.remove_none_fields(v) for k, v in obj.items() if v not in [None, "", []]}
        if isinstance(obj, list):
            return [self.remove_none_fields(v) for v in obj if v not in [None, "", []]]
        return obj

    async def tool_fetch_product_details(self, code: str) -> dict:
        """Handler pentru OpenAI tool - cheamă API-ul Romstal și filtrează răspunsul."""
        import httpx

        url = f"https://www.romstalpartener.ro/cs-includes/evolio/product.php?code={code}"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url)
                r.raise_for_status()
                data = r.json()  # API-ul tău returnează JSON

            # Temporarily commented out filtering to return all data
            # clean = self.extract_relevant_product_data(data)
            # clean = self.remove_none_fields(clean)
            return {"ok": True, "code": code, "data": data}

        except Exception as e:
            logger.error(f"[tool_fetch_product_details] {e}")
            return {"ok": False, "code": code, "error": str(e)}

    async def tool_search_products_romstal(self, category: Optional[str] = None, budget: Optional[str] = None, requirements: Optional[str] = None, limit: int = 5) -> dict:
        """Handler pentru OpenAI tool - oferă recomandări de produse Romstal cu filtru pe domeniu."""
        import json

        try:
            logger.info(f"[tool_search_products_romstal] Generating product recommendations for category: {category}, budget: {budget}, requirements: {requirements}")

            # Dacă nu avem categorie, returnează un mesaj helpful
            if not category:
                return {
                    "ok": True,
                    "message": "Pentru recomandări de produse, te rog să specifici categoria dorită (ex: țevii, pompe, boilere, radiatoare, etc.)",
                    "suggestion": "Încearcă cu: 'țevi pentru instalații sanitare' sau 'pompe sub 500 lei'",
                    "products": []
                }

            # Generează recomandări bazate pe categoria specificată
            products = self._generate_romstal_product_recommendations(category, budget, requirements, limit)

            # Aplică filtre suplimentare dacă sunt specificate
            if budget:
                products = self._filter_products_by_budget(products, budget)

            if requirements:
                products = self._filter_products_by_requirements(products, requirements)

            # Formatează rezultatul
            result = {
                "ok": True,
                "search_params": {
                    "category": category,
                    "budget": budget,
                    "requirements": requirements,
                    "limit": limit
                },
                "products": products[:limit],
                "total_found": len(products),
                "related_products": self._generate_related_products_section(products[:3]),
                "note": "Recomandări generate pe baza catalogului Romstal. Pentru detalii exacte, vizitează romstal.ro"
            }

            logger.info(f"[tool_search_products_romstal] Generated {len(products)} product recommendations for category: {category}")
            return result

        except Exception as e:
            logger.error(f"[tool_search_products_romstal] Error generating recommendations: {e}")
            return {
                "ok": False,
                "error": str(e),
                "search_params": {
                    "category": category,
                    "budget": budget,
                    "requirements": requirements,
                    "limit": limit
                }
            }

    def _generate_romstal_product_recommendations(self, category: str, budget: Optional[str] = None, requirements: Optional[str] = None, limit: int = 5) -> list:
        """Generează recomandări de produse Romstal pe baza categoriei."""
        # Bază de date de produse Romstal simulate (într-o implementare reală, acestea ar veni dintr-o bază de date sau API)
        romstal_products_db = {
            'tevi': [
                {'name': 'Țeavă PPR pentru apă caldă și rece', 'price': '45', 'url': 'https://www.romstal.ro/teava-ppr-20mm', 'description': 'Țeavă din polipropilenă random pentru instalații sanitare'},
                {'name': 'Țeavă PVC pentru canalizare', 'price': '120', 'url': 'https://www.romstal.ro/teava-pvc-110mm', 'description': 'Țeavă PVC pentru sisteme de canalizare'},
                {'name': 'Țeavă din cupru pentru instalații', 'price': '280', 'url': 'https://www.romstal.ro/teava-cupru-15mm', 'description': 'Țeavă din cupru pentru instalații de încălzire'},
                {'name': 'Țeavă multicu pentru apă', 'price': '85', 'url': 'https://www.romstal.ro/teava-multistrat-16mm', 'description': 'Țeavă multistrat pentru apă potabilă'},
                {'name': 'Țeavă PEX pentru încălzire în pardoseală', 'price': '65', 'url': 'https://www.romstal.ro/teava-pex-16mm', 'description': 'Țeavă PEX pentru sisteme de încălzire'},
            ],
            'pompe': [
                {'name': 'Pompă submersibilă pentru puțuri', 'price': '450', 'url': 'https://www.romstal.ro/pompa-submersibila-750w', 'description': 'Pompă submersibilă pentru puțuri adânci'},
                {'name': 'Pompă de suprafață pentru irigații', 'price': '320', 'url': 'https://www.romstal.ro/pompa-suprafata-550w', 'description': 'Pompă de suprafață pentru sisteme de irigații'},
                {'name': 'Pompă circulatie pentru încălzire', 'price': '180', 'url': 'https://www.romstal.ro/pompa-circulatie-25-60', 'description': 'Pompă de circulație pentru centrale termice'},
                {'name': 'Hidrofor complet cu pompă', 'price': '890', 'url': 'https://www.romstal.ro/hidrofor-50l-750w', 'description': 'Hidrofor complet cu rezervor și pompă'},
                {'name': 'Pompă de drenaj pentru apă murdară', 'price': '275', 'url': 'https://www.romstal.ro/pompa-drenaj-400w', 'description': 'Pompă submersibilă pentru apă murdară'},
            ],
            'boilere': [
                {'name': 'Boiler electric 80L vertical', 'price': '650', 'url': 'https://www.romstal.ro/boiler-electric-80l', 'description': 'Boiler electric cu capacitate 80 litri'},
                {'name': 'Boiler electric 100L orizontal', 'price': '720', 'url': 'https://www.romstal.ro/boiler-electric-100l', 'description': 'Boiler electric cu montare orizontală'},
                {'name': 'Boiler electric 50L cu afișaj digital', 'price': '480', 'url': 'https://www.romstal.ro/boiler-electric-50l-digital', 'description': 'Boiler electric compact cu control digital'},
                {'name': 'Boiler electric 120L cu anod de magneziu', 'price': '890', 'url': 'https://www.romstal.ro/boiler-electric-120l', 'description': 'Boiler electric cu protecție anticorozivă'},
                {'name': 'Boiler electric 30L pentru chiuvetă', 'price': '290', 'url': 'https://www.romstal.ro/boiler-electric-30l', 'description': 'Boiler electric mic pentru chiuvete'},
            ],
            'radiatoare': [
                {'name': 'Radiator aluminiu 600x800mm', 'price': '185', 'url': 'https://www.romstal.ro/radiator-aluminiu-10-elemente', 'description': 'Radiator din aluminiu cu 10 elementi'},
                {'name': 'Radiator oțel 600x1000mm', 'price': '220', 'url': 'https://www.romstal.ro/radiator-otel-22-elemente', 'description': 'Radiator din oțel pentru încălzire centrală'},
                {'name': 'Radiator baie cromat 500x800mm', 'price': '340', 'url': 'https://www.romstal.ro/radiator-baie-cromat', 'description': 'Radiator special pentru baie cu finisaj cromat'},
                {'name': 'Radiator fontă 800x600mm', 'price': '420', 'url': 'https://www.romstal.ro/radiator-fonta-8-elemente', 'description': 'Radiator tradițional din fontă'},
                {'name': 'Radiator electric cu termostat', 'price': '380', 'url': 'https://www.romstal.ro/radiator-electric-1500w', 'description': 'Radiator electric cu control termostat'},
            ],
            'fitinguri': [
                {'name': 'Fiting PPR cot 90 grade 20mm', 'price': '8', 'url': 'https://www.romstal.ro/fiting-ppr-cot-90-20mm', 'description': 'Cot PPR 90 grade pentru țevi 20mm'},
                {'name': 'Fiting PPR mufă 25mm', 'price': '12', 'url': 'https://www.romstal.ro/fiting-ppr-mufa-25mm', 'description': 'Mufă PPR pentru îmbinarea țevilor'},
                {'name': 'Fiting compresie pentru multicu 16mm', 'price': '15', 'url': 'https://www.romstal.ro/fiting-compresie-16mm', 'description': 'Fiting de compresie pentru țevi multistrat'},
                {'name': 'Fiting PPR tee 20mm', 'price': '18', 'url': 'https://www.romstal.ro/fiting-ppr-tee-20mm', 'description': 'Fiting tee PPR pentru ramificații'},
                {'name': 'Fiting PPR capăt 20mm', 'price': '6', 'url': 'https://www.romstal.ro/fiting-ppr-capat-20mm', 'description': 'Capăt PPR pentru închiderea țevilor'},
            ]
        }

        # Normalizează categoria pentru căutare
        category_lower = category.lower().strip()

        # Găsește produsele relevante pentru categoria dată
        matching_products = []

        for cat_key, products_list in romstal_products_db.items():
            if cat_key in category_lower or category_lower in cat_key:
                matching_products.extend(products_list)

        # Dacă nu găsim potriviri exacte, căutăm după cuvinte cheie
        if not matching_products:
            keywords = category_lower.split()
            for cat_key, products_list in romstal_products_db.items():
                for product in products_list:
                    product_text = f"{product['name']} {product['description']}".lower()
                    if any(keyword in product_text for keyword in keywords):
                        if product not in matching_products:
                            matching_products.append(product)

        # Dacă încă nu avem produse, returnăm o selecție generală
        if not matching_products:
            # Ia primele produse din fiecare categorie pentru diversitate
            for products_list in romstal_products_db.values():
                matching_products.extend(products_list[:2])  # 2 produse din fiecare categorie

        return matching_products

    def _extract_product_from_card(self, card, base_url: str) -> Optional[dict]:
        """Extrage informații despre produs dintr-un card HTML."""
        try:
            # Încearcă să găsească numele produsului
            name_element = card.find(['h1', 'h2', 'h3', 'h4', '.title', '.name', '.product-title'])
            name = name_element.get_text(strip=True) if name_element else ""

            # Încearcă să găsească URL-ul produsului
            link_element = card.find('a', href=True)
            url = ""
            if link_element:
                href = link_element.get('href', '')
                url = f"{base_url}{href}" if href.startswith('/') else href

            # Încearcă să găsească prețul
            price_element = card.find(['.price', '.cost', '.amount', '[data-price]'])
            price = ""
            if price_element:
                price_text = price_element.get_text(strip=True)
                # Extrage doar numerele și moneda
                price_match = re.search(r'(\d+[.,]\d+|\d+)', price_text)
                price = price_match.group(1) if price_match else price_text

            # Încearcă să găsească imaginea
            img_element = card.find('img')
            image = img_element.get('src', '') if img_element else ""

            # Încearcă să găsească descrierea
            desc_element = card.find(['.description', '.summary', 'p'])
            description = desc_element.get_text(strip=True) if desc_element else ""

            # Numai dacă avem nume și URL, considerăm că e un produs valid
            if name and url:
                return {
                    'name': name,
                    'url': url,
                    'price': price,
                    'image': f"{base_url}{image}" if image.startswith('/') else image,
                    'description': description[:200] + '...' if len(description) > 200 else description,
                    'source': 'product_card'
                }

        except Exception as e:
            logger.warning(f"[tool_search_products_romstal] Error in _extract_product_from_card: {e}")

        return None

    def _filter_products_by_budget(self, products: list, budget: str) -> list:
        """Filtrează produsele după buget."""
        if not budget or not products:
            return products

        filtered_products = []

        # Normalizează bugetul pentru comparație
        budget_lower = budget.lower()
        budget_range = {'min': 0, 'max': float('inf')}

        if 'sub' in budget_lower or 'mai puțin' in budget_lower:
            if '500' in budget_lower:
                budget_range['max'] = 500
            elif '1000' in budget_lower:
                budget_range['max'] = 1000
            elif '2000' in budget_lower:
                budget_range['max'] = 2000
        elif 'peste' in budget_lower or 'mai mult' in budget_lower:
            if '1000' in budget_lower:
                budget_range['min'] = 1000
            elif '2000' in budget_lower:
                budget_range['min'] = 2000
        elif '-' in budget_lower:
            # Parsează range-ul (ex: "500-1000 lei")
            range_match = re.search(r'(\d+)[^\d]*(\d+)', budget_lower)
            if range_match:
                budget_range['min'] = int(range_match.group(1))
                budget_range['max'] = int(range_match.group(2))

        # Filtrează produsele după preț
        for product in products:
            try:
                price_str = str(product.get('price', ''))
                price_match = re.search(r'(\d+[.,]\d+|\d+)', price_str)
                if price_match:
                    price = float(price_match.group(1).replace('.', '').replace(',', '.'))
                    if budget_range['min'] <= price <= budget_range['max']:
                        filtered_products.append(product)
            except (ValueError, AttributeError):
                # Dacă nu putem parsa prețul, includem produsul
                filtered_products.append(product)

        return filtered_products if filtered_products else products

    def _filter_products_by_requirements(self, products: list, requirements: str) -> list:
        """Filtrează produsele după cerințe specifice."""
        if not requirements or not products:
            return products

        filtered_products = []
        requirements_lower = requirements.lower()

        for product in products:
            try:
                # Verifică dacă cerințele se potrivesc cu numele, descrierea sau categoria produsului
                name = str(product.get('name', '')).lower()
                description = str(product.get('description', '')).lower()
                category = str(product.get('category', '')).lower()

                text_to_search = f"{name} {description} {category}"

                # Verifică cuvinte cheie comune
                keywords = requirements_lower.split()
                matches = sum(1 for keyword in keywords if keyword in text_to_search)

                # Dacă cel puțin jumătate din cuvintele cheie se potrivesc, includem produsul
                if matches >= len(keywords) / 2:
                    filtered_products.append(product)

            except (AttributeError, TypeError):
                # Dacă nu putem procesa produsul, îl includem
                filtered_products.append(product)

        return filtered_products if filtered_products else products

    def _generate_related_products_section(self, products: list) -> str:
        """Generează o secțiune cu produse similare."""
        if not products:
            return "Nu am găsit produse similare în această căutare."

        related_text = "Produse similare găsite:\n"
        for i, product in enumerate(products[:3], 1):
            name = product.get('name', 'Produs necunoscut')
            url = product.get('url', '')
            price = product.get('price', 'Preț indisponibil')

            related_text += f"{i}. {name}"
            if price:
                related_text += f" - {price} lei"
            if url:
                related_text += f"\n   Vezi: {url}"
            related_text += "\n"

        return related_text.strip()

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
                response = self.client.responses.create(
                    model=settings.openai_model,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "Ești asistent Romstal pe WhatsApp. "
                                "Dacă utilizatorul furnizează clar un cod de produs (ex: 64px9822), "
                                "apelează funcția `fetch_product_details`. "
                                "Dacă nu există cod, cere politicos codul. "
                                "Nu modifica URL-urile sau alte date. "
                                "Răspunde prietenos, în română.\n\n"
                                + system_prompt
                            )
                        },
                        {"role": "user", "content": user_prompt},
                    ],
                    tools=self.OPENAI_TOOLS,
                    tool_choice="auto",
                    max_output_tokens=2500,  # Increased from 700 to allow for reasoning + response
                    reasoning={"effort": "minimal"}
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
                        elif function_name == "search_products_romstal":
                            result = await self.tool_search_products_romstal(
                                category=args.get("category", ""),
                                budget=args.get("budget"),
                                requirements=args.get("requirements"),
                                limit=args.get("limit", 5)
                            )
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
                    if tool_outputs:
                        logger.info("[OpenAI] Making follow-up API call with function results")
                        logger.info(f"[DEBUG] Tool outputs count: {len(tool_outputs)}")
                        logger.info(f"[DEBUG] Tool outputs structure: {json.dumps([{'id': to['tool_call_id'], 'ok': to['output'].get('ok')} for to in tool_outputs], indent=2)}")

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

                        # Add function call outputs to input
                        for tool_output in tool_outputs:
                            follow_up_input.append({
                                "type": "function_call_output",
                                "call_id": tool_output["tool_call_id"],
                                "output": json.dumps(tool_output["output"], ensure_ascii=False)
                            })

                        logger.info(f"[DEBUG] Follow-up input length: {len(follow_up_input)} items")
                        logger.info(f"[DEBUG] Using previous_response_id: {response.id}")

                        follow_up_response = self.client.responses.create(
                            model=settings.openai_model,
                            input=follow_up_input,
                            tools=self.OPENAI_TOOLS,
                            previous_response_id=response.id,  # Thread the conversation
                            max_output_tokens=2500,  # Ensure enough tokens for response after function call
                            reasoning={"effort": "minimal"}
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