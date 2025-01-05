from langchain_community.tools import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI
import os
from typing import List, Dict, Optional, Set, Tuple
import re
import asyncio
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
from functools import lru_cache
import time
import aiohttp
import json
from dotenv import load_dotenv
import logging
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import tiktoken
import sqlite3
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue


# Constants
PROBABILITY_OF_RECIPE_IN_URL = 0.7
MAX_CHUNKS_TO_PROCESS = 5  # Limit chunks to process to control API usage


class RecipeAnalysisResult(BaseModel):
    """Structure for LLM output when analyzing recipe content."""
    content_summary: str = Field(
        description="Summary of the content analyzed so far")
    recipe_probability: float = Field(
        description="Probability that this is a recipe page (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation for the probability assessment")


class URLProcessingMonitor:
    def __init__(self, log_file: str = "url_processing_live.jsonl"):
        self.log_file = Path(log_file)
        self.queue = Queue()
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _monitor_loop(self):
        """Continuously monitor and log URL processing updates."""
        while self.is_running:
            try:
                entry = self.queue.get(timeout=1)
                self._write_entry(entry)
                self.queue.task_done()
            except:
                continue

    def _write_entry(self, entry: dict):
        """Write an entry to the log file."""
        entry['timestamp'] = datetime.now().isoformat()
        with self.log_file.open('a') as f:
            f.write(json.dumps(entry) + '\n')

        # Print real-time update
        status = "✅" if entry.get(
            'probability', 0) >= PROBABILITY_OF_RECIPE_IN_URL else "❌"
        print(f"\r{status} Processing: {entry['url'][:60]}{'...' if len(entry['url']) > 60 else ''} "
              f"(prob: {entry.get('probability', 0):.2f})", flush=True)

    def log_url_processing(self, **kwargs):
        """Add a URL processing entry to the monitor."""
        self.queue.put(kwargs)

    def stop(self):
        """Stop the monitoring thread."""
        self.is_running = False
        self.monitor_thread.join()


class RecipeSearch:
    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo"):
        """Initialize the recipe search tool with Tavily API key."""
        logging.debug("Initializing RecipeSearch with model: %s", model)
        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key

        self.search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )
        logging.debug("Initialized Tavily search tool with max_results=5")

        self.llm = ChatOpenAI(model=model, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

        self._seen_urls: Set[str] = set()
        self._last_request_time = 0
        self._min_request_interval = 1.0  # seconds between requests

        # Initialize text splitter for recipe analysis
        self.token_splitter = TokenTextSplitter(
            model_name=model,
            chunk_size=1000,
            chunk_overlap=100
        )

        # Setup prompt template for recipe analysis
        self.recipe_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing webpage content to determine if it's a recipe page.
                Consider these indicators:
                - Ingredients list (essential)
                - Cooking instructions
                - Nutritional information
                - Serving size information
                - Food-related images
                - Recipe-specific terminology
                
                Use ONLY the provided content chunk and context to update your assessment.
                """),
            ("human", """Previous summary: {previous_summary}
                Current probability: {current_probability}
                
                New content to analyze: {content_chunk}
                
                Analyze this content and update the probability that this is a recipe page.
                Consider both positive and negative indicators.
                If there's clear evidence this is NOT a recipe (e.g., it's a blog post or review),
                adjust probability accordingly."""),
        ])

        self.output_parser = PydanticOutputParser(
            pydantic_object=RecipeAnalysisResult)

        # Initialize SQLite in-memory database
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()

        # Create URL processing log table
        self.cursor.execute('''
            CREATE TABLE url_processing_log (
                url TEXT PRIMARY KEY,
                probability REAL,
                used_llm BOOLEAN,
                timestamp DATETIME,
                processing_time REAL,
                content_length INTEGER,
                error_message TEXT
            )
        ''')
        self.conn.commit()
        logging.debug("Initialized URL processing log in SQLite")

        # Add monitor
        self.monitor = URLProcessingMonitor()
        logging.debug("Initialized URL processing monitor")

        # Add URL analysis prompt
        self.url_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing URL paths to determine if they lead to recipe pages or recipe listings.
                Consider:
                - Words related to food, cooking, recipes
                - Numbers that might indicate servings or cooking time
                - Path structure typical of recipe websites
                - Whether the path suggests a single recipe or a collection
                
                Respond with:
                1. A probability between 0 and 1
                2. The type of page (recipe_page or recipe_listing)
                3. A brief explanation
                """),
            ("human", """Analyze this URL path: {url_path}
                
                Determine if this is likely to be:
                1. A single recipe page
                2. A recipe listing/collection page
                3. Neither (non-recipe content)
                
                Consider these patterns:
                
                Single Recipe URLs (recipe_page):
                - Contains specific dish names
                - Has ingredient names in path
                - Uses descriptive terms (quick, easy, homemade)
                - Often longer, more specific paths
                Examples:
                - recipes/chicken-parmesan
                - recipe/quick-vegetarian-curry
                - cooking/desserts/chocolate-cake
                
                Recipe Listing URLs (recipe_listing):
                - Contains collection terms (category, index, all)
                - Uses broader food categories
                - Has filtering/sorting parameters
                - Usually shorter, more general paths
                Examples:
                - recipes/vegetarian
                - category/dinner
                - collections/quick-meals
                
                Non-Recipe URLs:
                - blog/cooking-tips
                - articles/best-kitchen-tools
                - about/our-chefs
                
                Respond in this format:
                {
                    "probability": <float between 0 and 1>,
                    "path_type": "recipe_page" or "recipe_listing" or "non_recipe",
                    "reasoning": "<brief explanation of classification>"
                }
                """)
        ])

    def _is_recipe_listing_page(self, url: str) -> bool:
        """Check if URL is likely a recipe listing page rather than individual recipe."""
        listing_indicators = [
            '/category/', '/recipes/', '/gallery/',
            'meal-plan', 'diet-plan', 'recipe-collection',
            'index', 'browse', 'search'
        ]
        is_listing = any(indicator in url.lower()
                         for indicator in listing_indicators)
        logging.debug("URL %s is recipe listing page: %s", url, is_listing)
        return is_listing

    async def _rate_limited_request(self, url: str) -> str:
        """Make a rate-limited HTTP request."""
        now = time.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self._min_request_interval:
            delay = self._min_request_interval - time_since_last
            logging.debug(
                "Rate limiting: waiting %.2f seconds before request to %s", delay, url)
            await asyncio.sleep(delay)

        logging.debug("Making HTTP request to: %s", url)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                self._last_request_time = time.time()
                return await response.text()

    def _log_url_processing(self, url: str, probability: float, used_llm: bool,
                            processing_time: float, content_length: int = 0,
                            error_message: str | None = None):
        """Log URL processing details to SQLite and monitor."""
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO url_processing_log 
                (url, probability, used_llm, timestamp, processing_time, content_length, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                url,
                probability,
                used_llm,
                datetime.now().isoformat(),
                processing_time,
                content_length,
                error_message
            ))
            self.conn.commit()

            # Add to monitor
            self.monitor.log_url_processing(
                url=url,
                probability=probability,
                used_llm=used_llm,
                processing_time=processing_time,
                content_length=content_length,
                error_message=error_message
            )

        except Exception as e:
            logging.error("Failed to log URL processing: %s", e)

    async def _calculate_recipe_probability(self, url: str, content: str | None = None, use_llm: bool = False) -> float:
        """Calculate probability that a URL points to a recipe page."""
        start_time = time.time()
        probability = 0.0
        content_length = 0
        error_message = None

        try:
            # Check if we've already processed this URL
            self.cursor.execute(
                'SELECT probability FROM url_processing_log WHERE url = ?', (url,))
            result = self.cursor.fetchone()
            if result is not None:
                logging.debug(
                    "Using cached probability for %s: %.2f", url, result[0])
                return result[0]

            if not content:
                try:
                    content = await self._rate_limited_request(url)
                except Exception as e:
                    error_message = f"Failed to fetch content: {str(e)}"
                    logging.error("Failed to fetch content for %s: %s", url, e)
                    return 0.0

            content_length = len(content) if content else 0

            # Calculate probability
            probability = self._calculate_url_probability(url)
            if probability > 0:
                soup = BeautifulSoup(content, 'html.parser')

                if self._check_for_ingredients(soup):
                    probability *= self._analyze_content_indicators(soup)

                    if use_llm and probability > 0:
                        probability = await self._deep_content_analysis(content, probability)
                else:
                    probability = 0.0

        except Exception as e:
            error_message = str(e)
            logging.error(
                "Error calculating recipe probability for %s: %s", url, e)
            probability = 0.0

        finally:
            processing_time = time.time() - start_time
            self._log_url_processing(
                url=url,
                probability=probability,
                used_llm=use_llm,
                processing_time=processing_time,
                content_length=content_length,
                error_message=error_message
            )

        return probability

    def _calculate_url_probability(self, url: str) -> float:
        """Calculate initial probability based on URL patterns."""
        url_lower = url.lower()

        # Negative indicators (immediate rejections)
        negative_indicators = [
            'subscription', 'category', 'index', 'search',
            'blog', 'article', 'news', 'about', 'contact'
        ]
        if any(indicator in url_lower for indicator in negative_indicators):
            return 0.0

        # Positive indicators
        positive_indicators = {
            '/recipe/': 0.6,
            '/recipes/': 0.5,
            'cooking': 0.3,
            'food': 0.2,
            'meal': 0.2
        }

        probability = 0.1  # Base probability
        for indicator, weight in positive_indicators.items():
            if indicator in url_lower:
                probability = max(probability, weight)

        return probability

    def _check_for_ingredients(self, soup: BeautifulSoup) -> bool:
        """Check if page has an ingredients section."""
        ingredient_indicators = [
            'ingredients',
            'what you need',
            'shopping list',
            'you will need',
            'what youll need'  # Common variant without apostrophe
        ]

        measurement_patterns = [
            r'\b\d+(?:\.\d+)?\s*(?:cup|tbsp|tsp|g|oz|ml|pound|kg|lb|gram|ounce|tablespoon|teaspoon)s?\b',
            r'\b(?:half|quarter|third|one|two|three)\s+(?:cup|tbsp|tsp|pound|kg|lb)s?\b',
            r'\b\d+\s*(?:small|medium|large|whole)\b',
            r'\b(?:pinch|dash|handful|splash|drizzle|sprinkle)\s+of\b'
        ]

        # Check headers and list items
        for indicator in ingredient_indicators:
            # Look in headers
            if soup.find(['h1', 'h2', 'h3', 'h4'], string=re.compile(indicator, re.I)):
                return True

            # Look for ingredient section in divs/spans with specific classes
            if soup.find(['div', 'span'], class_=re.compile(r'ingredient', re.I)):
                return True

        # Check for measurement patterns in list items
        for pattern in measurement_patterns:
            if soup.find('li', string=re.compile(pattern, re.I)):
                return True

        return False

    def _analyze_content_indicators(self, soup: BeautifulSoup) -> float:
        """Analyze content for recipe indicators."""
        score = 1.0

        # Check for recipe schema (strong indicator)
        if self._check_recipe_schema(soup):
            score *= 1.5

        # Check for serving size
        if re.search(r'serves?\s+\d+|yield:?\s+\d+|portions?:?\s+\d+', str(soup), re.I):
            score *= 1.2

        # Check for nutritional information
        nutrition_patterns = [
            r'nutrition(?:al)?\s+(?:info|information|facts)',
            r'calories[^a-z]',
            r'protein|carbs|fat\b',
            r'per\s+serving'
        ]
        if any(re.search(pattern, str(soup), re.I) for pattern in nutrition_patterns):
            score *= 1.2

        # Check for recipe-related images
        food_images = soup.find_all('img', alt=re.compile(
            r'food|recipe|dish|meal|ingredient|step|preparation', re.I))
        if food_images:
            score *= 1.1

        # Check for cooking instructions
        instruction_patterns = [
            r'instructions|directions|method|steps?',
            r'prep(?:aration)?\s+time',
            r'cook(?:ing)?\s+time',
            r'total\s+time',
            r'ready\s+in'
        ]
        if any(re.search(pattern, str(soup), re.I) for pattern in instruction_patterns):
            score *= 1.2

        # Check for recipe metadata
        metadata_patterns = [
            r'difficulty:\s*\w+',
            r'cuisine:\s*\w+',
            r'category:\s*\w+',
            r'prep\s+time:\s*\d+',
            r'cook\s+time:\s*\d+'
        ]
        if any(re.search(pattern, str(soup), re.I) for pattern in metadata_patterns):
            score *= 1.1

        return min(score, 1.0)

    def _check_recipe_schema(self, soup: BeautifulSoup) -> bool:
        """Check for recipe schema markup."""
        schema_indicators = [
            'application/ld+json',
            'schema.org/Recipe',
            'itemtype="http://schema.org/Recipe"'
        ]

        # Check for JSON-LD schema
        scripts = soup.find_all('script', {'type': 'application/ld+json'})
        for script in scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    if data.get('@type') == 'Recipe':
                        return True
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and item.get('@type') == 'Recipe':
                            return True
            except:
                continue

        # Check for microdata
        if soup.find(itemtype=re.compile(r'schema.org/Recipe', re.I)):
            return True

        return False

    async def _deep_content_analysis(self, content: str, initial_probability: float) -> float:
        """Perform deep content analysis using LLM."""
        chunks = self.token_splitter.split_text(content)
        current_probability = initial_probability
        current_summary = "No content analyzed yet."

        for i, chunk in enumerate(chunks[:MAX_CHUNKS_TO_PROCESS]):
            try:
                result = await self.llm.apredict_messages([
                    *self.recipe_analysis_prompt.format_messages(
                        previous_summary=current_summary,
                        current_probability=current_probability,
                        content_chunk=chunk
                    )
                ])

                parsed_result = self.output_parser.parse(result.content)
                current_probability = parsed_result.recipe_probability
                current_summary = parsed_result.content_summary

                logging.debug("Chunk %d analysis: prob=%.2f, reasoning=%s",
                              i, parsed_result.recipe_probability, parsed_result.reasoning)

                # Early exit if probability drops too low
                if current_probability < 0.2:
                    break

            except Exception as e:
                logging.error("Error in LLM analysis: %s", e)
                break

        return current_probability

    def _extract_url_path(self, url: str) -> str:
        """Extract path component from URL, excluding domain and query parameters."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')

        # Remove common file extensions
        path = re.sub(r'\.(html|php|asp|jsp)$', '', path)

        # Remove numeric IDs from path
        path = re.sub(r'/\d+/?$', '', path)

        logging.debug("Extracted path from %s: %s", url, path)
        return path

    async def _analyze_url_path(self, url: str) -> Tuple[float, str, str]:
        """
        Use LLM to analyze if URL path suggests a recipe or recipe listing page.

        Returns:
            Tuple[float, str, str]: (probability, path_type, reasoning)
            where path_type is one of: 'recipe_page', 'recipe_listing', 'non_recipe'
        """
        try:
            url_path = self._extract_url_path(url)
            logging.debug("Analyzing URL path: %s", url_path)

            # Skip empty paths
            if not url_path:
                return 0.0, "non_recipe", "Empty URL path"

            # Quick check for obvious non-recipe paths
            non_recipe_indicators = [
                'login', 'register', 'about', 'contact',
                'privacy', 'terms', 'search', 'tag'
            ]
            if any(indicator in url_path.lower() for indicator in non_recipe_indicators):
                return 0.0, "non_recipe", "Path contains non-recipe indicators"

            # Update the prompt to distinguish between recipe types
            result = await self.llm.apredict_messages([
                *self.url_analysis_prompt.format_messages(
                    url_path=url_path,
                    context="""
                    Analyze if this URL path likely leads to:
                    1. A single recipe page (e.g., 'recipes/chicken-parmesan', 'recipe/vegetarian-curry')
                    2. A recipe listing/collection page (e.g., 'recipes/all', 'category/dinner', 'collections/vegetarian')
                    
                    Common patterns:
                    Single Recipe URLs:
                    - Contain specific dish names
                    - Include ingredient names
                    - Have detailed descriptors (quick, easy, homemade)
                    - Often longer, more specific paths
                    
                    Recipe Listing URLs:
                    - Contains words like: category, collection, all, index
                    - More general terms (dinner, vegetarian, quick)
                    - Usually shorter paths
                    - Often include sorting/filtering terms
                    
                    Respond in this format:
                    {
                        "probability": <float between 0 and 1>,
                        "path_type": "recipe_page" or "recipe_listing",
                        "reasoning": "<brief explanation>"
                    }
                    """
                )
            ])

            # Parse the response
            try:
                response = json.loads(result.content)
                probability = float(response.get('probability', 0))
                path_type = response.get('path_type', 'non_recipe')
                reasoning = response.get(
                    'reasoning', 'No explanation provided')

                # Adjust probability based on additional heuristics
                if path_type == "recipe_page":
                    if 'recipe' in url_path.lower():
                        probability = min(1.0, probability * 1.2)
                    if any(term in url_path.lower() for term in ['how-to', 'guide', 'tips']):
                        probability *= 0.7
                elif path_type == "recipe_listing":
                    if any(term in url_path.lower() for term in ['category', 'collection', 'index']):
                        probability = min(1.0, probability * 1.2)

                logging.debug(
                    "URL path analysis for %s: prob=%.2f, type=%s, reason=%s",
                    url_path, probability, path_type, reasoning
                )

                return probability, path_type, reasoning

            except Exception as e:
                logging.error("Failed to parse LLM response: %s", e)
                return 0.0, "non_recipe", f"Error parsing LLM response: {e}"

        except Exception as e:
            logging.error("Error analyzing URL path: %s", e)
            return 0.0, "non_recipe", f"Error: {e}"

    @lru_cache(maxsize=100)
    async def _extract_recipe_links(self, url: str) -> List[str]:
        """Extract individual recipe links from a recipe listing page with caching."""
        logging.debug("Extracting recipe links from: %s", url)
        try:
            content = await self._rate_limited_request(url)
            soup = BeautifulSoup(content, 'html.parser')

            links = soup.find_all('a', href=True)
            recipe_links = []

            for link in links:
                href = urljoin(url, link['href'])
                if href in self._seen_urls:
                    logging.debug("Skipping already seen URL: %s", href)
                    continue

                self._seen_urls.add(href)

                # First check URL path with LLM
                path_probability, path_type, reasoning = await self._analyze_url_path(href)

                if path_type == "recipe_page" and path_probability >= 0.7:
                    logging.debug(
                        "URL path suggests single recipe (%s): %.2f - %s",
                        href, path_probability, reasoning
                    )

                    # If URL path looks promising, do full content analysis
                    probability = await self._calculate_recipe_probability(href)
                    logging.debug(
                        "Full content analysis for %s: %.2f", href, probability)

                    if probability >= PROBABILITY_OF_RECIPE_IN_URL:
                        recipe_links.append(href)
                        logging.debug(
                            "Added recipe link: %s (path_prob=%.2f, content_prob=%.2f, type=%s)",
                            href, path_probability, probability, path_type
                        )
                elif path_type == "recipe_listing":
                    logging.debug(
                        "Found nested recipe listing (%s): %.2f - %s",
                        href, path_probability, reasoning
                    )
                    # Optionally process nested recipe listings
                    nested_links = await self._extract_recipe_links(href)
                    recipe_links.extend(nested_links)
                else:
                    logging.debug(
                        "Skipping URL based on path analysis (%s): %.2f - %s - %s",
                        href, path_probability, path_type, reasoning
                    )

            return list(set(recipe_links))

        except Exception as e:
            logging.error("Error extracting links from %s: %s", url, e)
            return []

    async def _validate_recipe(self, content: str) -> Dict:
        """Use LLM to validate and extract recipe information."""
        logging.debug(
            "Validating recipe content length: %d characters", len(content))
        schema = {
            "properties": {
                "is_recipe": {"type": "boolean"},
                "calories_per_serving": {"type": "integer"},
                "recipe_name": {"type": "string"},
                "cooking_time_minutes": {"type": "integer"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
                "servings": {"type": "integer"},
                "macros": {
                    "type": "object",
                    "properties": {
                        "protein": {"type": "number"},
                        "carbs": {"type": "number"},
                        "fat": {"type": "number"}
                    }
                },
                "instructions": {"type": "array", "items": {"type": "string"}},
                "dietary_info": {
                    "type": "object",
                    "properties": {
                        "vegetarian": {"type": "boolean"},
                        "vegan": {"type": "boolean"},
                        "gluten_free": {"type": "boolean"}
                    }
                }
            },
            "required": ["is_recipe", "calories_per_serving", "recipe_name", "ingredients"]
        }

        try:
            chain = create_extraction_chain(schema, self.llm)
            result = await chain.arun(content)
            recipe_info = result[0] if result else {}
            logging.debug("Recipe validation result: %s", {
                "is_recipe": recipe_info.get("is_recipe"),
                "name": recipe_info.get("recipe_name"),
                "calories": recipe_info.get("calories_per_serving"),
                "ingredients_count": len(recipe_info.get("ingredients", []))
            })
            return recipe_info
        except Exception as e:
            logging.error("Error validating recipe: %s", e)
            return {}

    async def _process_recipe_page(self, url: str, min_calories: int, max_calories: int) -> Optional[Dict]:
        """Process a single recipe page and validate it meets the criteria."""
        logging.debug("Processing recipe page: %s", url)
        try:
            loader = WebBaseLoader(url)
            docs = await loader.aload()  # type: ignore
            content = docs[0].page_content
            logging.debug("Loaded content from %s: %d characters",
                          url, len(content))

            # Validate and extract recipe info
            recipe_info = await self._validate_recipe(content)

            calories = recipe_info.get("calories_per_serving", 0)
            if (recipe_info.get("is_recipe") and
                    min_calories <= calories <= max_calories):
                logging.debug("Valid recipe found: %s (calories: %d)",
                              recipe_info.get("recipe_name"), calories)
                return {
                    "title": recipe_info.get("recipe_name", ""),
                    "url": url,
                    "content": content,
                    "calories": calories,
                    "cooking_time": recipe_info.get("cooking_time_minutes"),
                    "ingredients": recipe_info.get("ingredients", [])
                }
            else:
                logging.debug("Recipe rejected: is_recipe=%s, calories=%d",
                              recipe_info.get("is_recipe"), calories)
        except Exception as e:
            logging.error("Error processing %s: %s", url, e)
        return None

    async def search_recipes(self, min_calories: int = 200, max_calories: int = 750) -> List[Dict]:
        """Search for vegetarian recipes within specified calorie range."""
        logging.debug("Starting recipe search: calories %d-%d",
                      min_calories, max_calories)
        query = (
            f"vegetarian recipe websites with recipes between {
                min_calories} and {max_calories} calories"
        )

        logging.debug("Executing Tavily search with query: %s", query)
        initial_results = self.search_tool.invoke({"query": query})
        logging.debug("Got %d initial search results", len(initial_results))

        # Collect all recipe pages
        recipe_pages = []
        for result in initial_results:
            if isinstance(result, dict):
                url = result.get("url", "")
                content = result.get("content", "")

                if self._is_recipe_listing_page(url):
                    recipe_links = await self._extract_recipe_links(url)
                    recipe_pages.extend(recipe_links)
                    logging.debug("Added %d recipe links from listing page %s",
                                  len(recipe_links), url)
                elif self._is_likely_recipe_page(url, content):
                    recipe_pages.append(url)
                    logging.debug("Added likely recipe page: %s", url)
                else:
                    logging.debug("Skipping non-recipe page: %s", url)

        logging.debug("Found total of %d potential recipe pages",
                      len(recipe_pages))

        # Process recipes in batches
        batch_size = 3
        valid_recipes = []

        for i in range(0, len(recipe_pages[:10]), batch_size):
            batch = recipe_pages[i:i + batch_size]
            logging.debug("Processing batch %d-%d of recipes",
                          i, i + len(batch))

            batch_results = await self._process_url_batch(batch, min_calories, max_calories)
            valid_recipes.extend(batch_results)

            if len(valid_recipes) >= 5:
                logging.debug(
                    "Reached target of 5 valid recipes, stopping search")
                break

            await asyncio.sleep(0.5)  # Small delay between batches

        logging.debug("Search complete. Returning %d recipes",
                      len(valid_recipes[:5]))
        return valid_recipes[:5]

    def _recipes_are_similar(self, recipe1: Dict, recipe2: Dict, threshold: float = 0.7) -> bool:
        """Check if two recipes are similar based on ingredients and instructions."""
        if not recipe1 or not recipe2:
            return False

        # Compare ingredients
        ingredients1 = set(i.lower() for i in recipe1.get('ingredients', []))
        ingredients2 = set(i.lower() for i in recipe2.get('ingredients', []))

        if not ingredients1 or not ingredients2:
            return False

        intersection = ingredients1.intersection(ingredients2)
        union = ingredients1.union(ingredients2)

        similarity = len(intersection) / len(union)
        logging.debug("Recipe similarity: %.2f (threshold: %.2f) between %s and %s",
                      similarity, threshold, recipe1.get('title'), recipe2.get('title'))
        return similarity > threshold

    def _is_likely_recipe_page(self, url: str, content: str | None = None) -> bool:
        """
        Check if a URL and its content (if provided) are likely to be a recipe page.

        Args:
            url: The URL to check
            content: Optional page content to analyze

        Returns:
            bool: True if likely a recipe page, False otherwise
        """
        logging.debug("Checking if %s is a recipe page", url)

        # URL-based checks
        recipe_indicators_in_url = [
            '/recipe/', '-recipe', '/recipes/',
            'cooking', 'food', 'kitchen',
            'dish', 'meal', 'dinner', 'lunch', 'breakfast'
        ]

        url_lower = url.lower()
        url_suggests_recipe = any(indicator in url_lower
                                  for indicator in recipe_indicators_in_url)

        if not url_suggests_recipe:
            logging.debug("URL does not suggest recipe content: %s", url)
            return False

        # If content is provided, do additional checks
        if content:
            # Common recipe page indicators
            content_indicators = [
                'ingredients:', 'instructions:', 'directions:',
                'preparation:', 'cook time:', 'prep time:',
                'serving size:', 'servings:', 'yield:',
                'nutrition', 'calories'
            ]

            content_lower = content.lower()
            content_matches = sum(1 for indicator in content_indicators
                                  if indicator in content_lower)

            # Check for common recipe patterns
            has_ingredient_list = bool(re.search(r'\b\d+\s*(?:cup|tbsp|tsp|g|oz|ml|pound|kg)\b',
                                                 content_lower))
            has_numbered_steps = bool(re.search(r'\b(?:step\s*\d|[1-9]\.)\s+\w+',
                                                content_lower))

            score = content_matches + has_ingredient_list + has_numbered_steps
            is_likely_recipe = score >= 3  # Require at least 3 indicators

            logging.debug("Content analysis for %s: score=%d, has_ingredient_list=%s, has_numbered_steps=%s",
                          url, score, has_ingredient_list, has_numbered_steps)

            return is_likely_recipe

        # If no content provided, rely on URL check only
        logging.debug(
            "No content provided, using URL-based check only for %s", url)
        return url_suggests_recipe

    def get_url_processing_history(self, min_probability: float = 0.0) -> List[Dict]:
        """Retrieve URL processing history."""
        try:
            self.cursor.execute('''
                SELECT url, probability, used_llm, timestamp, processing_time, content_length, error_message
                FROM url_processing_log
                WHERE probability >= ?
                ORDER BY timestamp DESC
            ''', (min_probability,))

            columns = [description[0]
                       for description in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
        except Exception as e:
            logging.error("Failed to retrieve URL processing history: %s", e)
            return []

    def __del__(self):
        """Clean up resources."""
        try:
            self.monitor.stop()
            self.conn.close()
        except:
            pass

    def _get_cached_url_probability(self, url: str) -> Optional[float]:
        """Get cached probability for a URL if it exists."""
        try:
            self.cursor.execute(
                'SELECT probability FROM url_processing_log WHERE url = ?', (url,))
            result = self.cursor.fetchone()
            if result is not None:
                logging.debug(
                    "Found cached probability for %s: %.2f", url, result[0])
                return result[0]
            return None
        except Exception as e:
            logging.error("Error retrieving cached URL probability: %s", e)
            return None

    async def _process_url_batch(self, urls: List[str], min_calories: int, max_calories: int) -> List[Dict]:
        """Process a batch of URLs, using cached results when available."""
        results = []
        new_urls = []

        # Check cache first
        for url in urls:
            cached_prob = self._get_cached_url_probability(url)
            if cached_prob is not None and cached_prob >= PROBABILITY_OF_RECIPE_IN_URL:
                logging.debug(
                    "Using cached high-probability result for %s", url)
                # Fetch full recipe details for cached high-probability URLs
                recipe = await self._process_recipe_page(url, min_calories, max_calories)
                if recipe:
                    results.append(recipe)
            else:
                new_urls.append(url)

        if new_urls:
            # Process new URLs in parallel
            tasks = [
                self._process_recipe_page(url, min_calories, max_calories)
                for url in new_urls
            ]
            new_results = await asyncio.gather(*tasks)
            results.extend([r for r in new_results if r is not None])

        return results

    def view_url_log(self, min_probability: float = 0.0, show_errors: bool = True) -> None:
        """
        View the current URL processing log.

        Args:
            min_probability: Minimum probability to show (0-1)
            show_errors: Whether to show entries with errors
        """
        try:
            self.cursor.execute('''
                SELECT 
                    url,
                    probability,
                    used_llm,
                    timestamp,
                    processing_time,
                    content_length,
                    error_message
                FROM url_processing_log
                WHERE probability >= ?
                ORDER BY timestamp DESC
            ''', (min_probability,))

            results = self.cursor.fetchall()

            if not results:
                print("No URLs processed yet.")
                return

            print("\nURL Processing Log:")
            print("="*80)

            for row in results:
                url, prob, used_llm, timestamp, proc_time, content_len, error = row

                # Skip entries with errors if show_errors is False
                if not show_errors and error:
                    continue

                status = "✅" if prob >= PROBABILITY_OF_RECIPE_IN_URL else "❌"
                print(f"\n{status} {url[:70]}{'...' if len(url) > 70 else ''}")
                print(f"   Probability: {prob:.2f}")
                print(f"   LLM Used: {'Yes' if used_llm else 'No'}")
                print(f"   Processing Time: {proc_time:.2f}s")
                print(f"   Content Length: {content_len:,} chars")

                if error:
                    print(f"   ⚠️  Error: {error}")

                print("-"*80)

            # Print summary statistics
            total = len(results)
            successful = sum(
                1 for r in results if r[1] >= PROBABILITY_OF_RECIPE_IN_URL)
            errors = sum(1 for r in results if r[6])
            llm_usage = sum(1 for r in results if r[2])
            avg_time = sum(r[4] for r in results) / total if total else 0

            print("\nSummary:")
            print(f"Total URLs: {total}")
            print(f"Successful: {successful} ({successful/total*100:.1f}%)")
            print(f"Errors: {errors} ({errors/total*100:.1f}%)")
            print(f"LLM Usage: {llm_usage} ({llm_usage/total*100:.1f}%)")
            print(f"Average Processing Time: {avg_time:.2f}s")

        except Exception as e:
            print(f"Error viewing log: {e}")

    def analyze_failures(self) -> None:
        """Analyze URLs that failed to be identified as recipes."""
        try:
            self.cursor.execute('''
                SELECT url, probability, error_message
                FROM url_processing_log
                WHERE probability < ?
                ORDER BY probability DESC
            ''', (PROBABILITY_OF_RECIPE_IN_URL,))

            results = self.cursor.fetchall()

            if not results:
                print("No failed URLs found.")
                return

            print("\nFailed URL Analysis:")
            print("="*80)

            # Group by probability ranges
            ranges = {
                (0.5, 0.7): [],
                (0.3, 0.5): [],
                (0.1, 0.3): [],
                (0.0, 0.1): []
            }

            for url, prob, error in results:
                for (min_p, max_p) in ranges.keys():
                    if min_p <= prob < max_p:
                        ranges[(min_p, max_p)].append((url, prob, error))
                        break

            # Print analysis
            for (min_p, max_p), urls in ranges.items():
                if urls:
                    print(f"\nProbability {min_p:.1f}-{max_p:.1f}:")
                    print("-"*40)
                    for url, prob, error in urls:
                        print(f"• {url[:70]}{'...' if len(url) > 70 else ''}")
                        print(f"  Probability: {prob:.2f}")
                        if error:
                            print(f"  Error: {error}")

            print("\nCommon patterns in failed URLs:")
            patterns = {}
            for url, _, _ in results:
                for pattern in ['/blog/', '/article/', '/news/', '/category/',
                                'subscription', 'index', 'search']:
                    if pattern in url.lower():
                        patterns[pattern] = patterns.get(pattern, 0) + 1

            for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"• {pattern}: {count} occurrences")

        except Exception as e:
            print(f"Error analyzing failures: {e}")


async def test_recipe_search():
    """Test the RecipeSearch class with constraints from JSON files."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('recipe_search.log')
            ]
        )
        logging.info("Starting recipe search test")

        print("\n" + "="*60)
        print("RECIPE SEARCH TEST".center(60))
        print("="*60 + "\n")

        # Load all constraint files
        print("Loading constraint files...")
        with open('meal_planner_agent/recipe_constraints.json', 'r') as f:
            recipe_constraints = json.load(f)
        with open('meal_planner_agent/day_constraints.json', 'r') as f:
            day_constraints = json.load(f)
        with open('meal_planner_agent/meal_plan_constraints.json', 'r') as f:
            meal_plan_constraints = json.load(f)

        # Extract relevant constraints for recipe search
        calories = recipe_constraints.get('calories', {})
        min_calories = calories.get('min', 150)
        max_calories = calories.get('max', 750)

        dietary_restrictions = recipe_constraints.get(
            'dietary_restrictions', {})
        ingredients_to_avoid = recipe_constraints.get(
            'ingredients_to_avoid', [])

        print("\nSearch Parameters:")
        print("-"*30)
        print(f"Calorie Range: {min_calories}-{max_calories}")
        print("\nDietary Restrictions:")
        for restriction, value in dietary_restrictions.items():
            if not restriction.startswith('$'):  # Skip metadata fields
                print(f"- {restriction}: {value}")
        print("\nAvoiding Ingredients:")
        for ingredient in ingredients_to_avoid:
            print(f"- {ingredient}")

        # Initialize recipe searcher
        print("\nInitializing recipe search...")
        recipe_searcher = RecipeSearch()

        # Search for recipes
        print("\nSearching for recipes...")
        recipes = await recipe_searcher.search_recipes(
            min_calories=min_calories,
            max_calories=max_calories
        )

        # Print results
        print(f"\nFound {len(recipes)} valid recipes:")
        print("="*60)

        for i, recipe in enumerate(recipes, 1):
            print(f"\nRECIPE {i}:")
            print("-"*30)
            print(f"Title: {recipe['title']}")
            print(f"URL: {recipe['url']}")
            print(f"Calories per serving: {
                  recipe.get('calories', 'Not specified')}")
            print(f"Cooking time: {recipe.get(
                'cooking_time', 'Not specified')} minutes")

            if recipe.get('ingredients'):
                print("\nIngredients:")
                for ingredient in recipe['ingredients']:
                    print(f"  • {ingredient}")

            # Check for avoided ingredients
            conflicts = [ing for ing in ingredients_to_avoid
                         if ing.lower() in recipe.get('content', '').lower()]
            if conflicts:
                print("\n⚠️  WARNING:")
                print(f"Recipe contains avoided ingredients: {
                      ', '.join(conflicts)}")

            print("\n" + "="*60)

        # Test similar recipe detection if we have at least 2 recipes
        if len(recipes) >= 2:
            print("\nSimilarity Analysis:")
            print("-"*30)
            for i in range(len(recipes)-1):
                for j in range(i+1, len(recipes)):
                    similarity = recipe_searcher._recipes_are_similar(
                        recipes[i], recipes[j])
                    print(
                        f"Recipes {i+1} and {j+1} are: {'Similar' if similarity else 'Different'}")

        # After recipe search is complete, print URL processing statistics
        print("\nURL Processing Statistics:")
        print("-"*30)

        history = recipe_searcher.get_url_processing_history()

        # Calculate statistics
        total_urls = len(history)
        llm_used = sum(1 for entry in history if entry['used_llm'])
        high_prob = sum(
            1 for entry in history if entry['probability'] >= PROBABILITY_OF_RECIPE_IN_URL)
        avg_time = sum(entry['processing_time']
                       for entry in history) / total_urls if total_urls else 0

        print(f"Total URLs processed: {total_urls}")
        print(f"URLs using LLM analysis: {llm_used}")
        print(f"High probability recipes (≥{
              PROBABILITY_OF_RECIPE_IN_URL}): {high_prob}")
        print(f"Average processing time: {avg_time:.2f}s")

        # Print detailed log for high probability URLs
        print("\nHigh Probability Recipe URLs:")
        for entry in history:
            if entry['probability'] >= PROBABILITY_OF_RECIPE_IN_URL:
                print(f"\nURL: {entry['url']}")
                print(f"Probability: {entry['probability']:.2f}")
                print(f"Used LLM: {entry['used_llm']}")
                print(f"Processing Time: {entry['processing_time']:.2f}s")
                if entry['error_message']:
                    print(f"Error: {entry['error_message']}")

        print("\nMonitoring URL processing in real-time...")
        print("Check url_processing_live.jsonl for detailed logs")
        print("-"*60)

        print("\nViewing URL Processing Log:")
        # Show URLs with prob >= 0.3
        recipe_searcher.view_url_log(min_probability=0.3)

        print("\nAnalyzing Failed URLs:")
        recipe_searcher.analyze_failures()

        print("\nTest Complete!")
        print("="*60 + "\n")

    except Exception as e:
        print("\n❌ Error during testing:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        raise

if __name__ == "__main__":
    # Add imports needed for testing
    import json
    import asyncio
    from dotenv import load_dotenv
    import sys

    # Load environment variables
    load_dotenv(dotenv_path="./env")

    # Verify required environment variables
    required_vars = ['TAVILY_API_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("\n❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  • {var}")
        print("\nPlease set these variables in your .env file")
        sys.exit(1)

    # Run the test
    print("🔍 Starting recipe search test...")
    try:
        asyncio.run(test_recipe_search())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {type(e).__name__}: {str(e)}")
        raise
