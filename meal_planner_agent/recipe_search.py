from __future__ import annotations
from langchain_community.tools import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI
import os
from typing import Any, Generic, List, Dict, Literal, Optional, Set, Tuple, TypeVar, TypedDict
import re
import asyncio
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse, urlunparse
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
import webbrowser
from urllib.parse import urlunparse
from tqdm import tqdm
import sys
import select
import hashlib
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from umap import UMAP
import pacmap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import base64


# Constants
PROBABILITY_OF_RECIPE_IN_URL = 0.7
MAX_CHUNKS_TO_PROCESS = 5  # Limit chunks to process to control API usage


class Config:
    RECIPE_SEARCH_MODEL = "gpt-3.5-turbo"
    RECIPE_SEARCH_MAX_RESULTS = 5
    RECIPE_SEARCH_MAX_CHUNKS_TO_PROCESS = 5
    RECIPE_SEARCH_MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
    RECIPE_SEARCH_PROBABILITY_OF_RECIPE_IN_URL = 0.7
    RECIPE_SEARCH_MAX_CHUNKS_TO_PROCESS = 5
    RECIPE_SEARCH_TOKEN_SPLITTER_OVERLAP = 100
    RECIPE_SEARCH_TOKEN_SPLITTER_CHUNK_SIZE = 1000
    RECIPE_SEARCH_TEXT_SPLITTER_OVERLAP = 200
    RECIPE_SEARCH_TEXT_SPLITTER_CHUNK_SIZE = 2000
    NESTED_LINKS_MAX_RECURSION_LEVEL = 2
    MIN_PROBABILITY_OF_RECIPE_IN_URL_PATH = 0.49
    RECIPE_SEARCH_MAX_LLM_CALLS = 100  # Maximum number of LLM calls per session


TSLI = TypeVar('TSLI', bound=Literal['integer', 'string', 'boolean'])


class _SchemaTypeVal(TypedDict, Generic[TSLI]):
    type: TSLI


class _SchemaTypeList(TypedDict, Generic[TSLI]):
    type: Literal['array']
    items: _SchemaTypeVal[TSLI]


class _SchemaTypeObject(TypedDict, Generic[TSLI]):
    type: Literal['object']
    properties: Dict[str, _SchemaTypeVal[TSLI]]


TSVI = TypeVar('TSVI', bound=int | _SchemaTypeVal[Literal['integer']])
TSAI = TypeVar('TSAI', bound=list[int] | _SchemaTypeList[Literal['integer']])
TSDI = TypeVar('TSDI', bound=dict[str, int] |
               _SchemaTypeObject[Literal['integer']])
TSVS = TypeVar('TSVS', bound=str | _SchemaTypeVal[Literal['string']])
TSAS = TypeVar('TSAS', bound=list[str] | _SchemaTypeList[Literal['string']])
TSDS = TypeVar('TSDS', bound=dict[str, str] |
               _SchemaTypeObject[Literal['string']])
TSVB = TypeVar('TSVB', bound=bool | _SchemaTypeVal[Literal['boolean']])
TSAB = TypeVar('TSAB', bound=list[bool] | _SchemaTypeList[Literal['boolean']])
TSDB = TypeVar('TSDB', bound=dict[str, bool] |
               _SchemaTypeObject[Literal['boolean']])
# TSI = TSVI | TSDI | TSAI
# TSB = TSVB | TSDB | TSAB
# TSS = TSVS | TSDS | TSAS
TSI = TypeVar('TSI', bound=int | _SchemaTypeVal[Literal['integer']] |
              _SchemaTypeList[Literal['integer']] | _SchemaTypeObject[Literal['integer']])
TSB = TypeVar('TSB', bound=bool | _SchemaTypeVal[Literal['boolean']] |
              _SchemaTypeList[Literal['boolean']] | _SchemaTypeObject[Literal['boolean']])
TSS = TypeVar('TSS', bound=str | _SchemaTypeVal[Literal['string']] |
              _SchemaTypeList[Literal['string']] | _SchemaTypeObject[Literal['string']])


class ExtractRecipeDataSchemaMacros(TypedDict, Generic[TSI]):
    protein: TSI
    carbs: TSI
    fat: TSI


class ExtractRecipeDataSchemaMicros(TypedDict, Generic[TSI]):
    vitamin_a: TSI
    vitamin_c: TSI
    vitamin_d: TSI
    vitamin_e: TSI
    vitamin_k: TSI
    calcium: TSI
    iron: TSI
    magnesium: TSI
    phosphorus: TSI


class ExtractRecipeDataSchemaDietaryInfo(TypedDict, Generic[TSB]):
    vegetarian: TSB
    vegan: TSB
    gluten_free: TSB


class ExtractRecipeDataSchemaProperties(TypedDict, Generic[TSI, TSS, TSB]):
    """
    Properties of the recipe data schema.
    """
    recipe_name: TSS
    calories_per_serving: TSI
    is_recipe: TSB
    servings: TSI
    title: TSS
    calories: TSI
    macros: ExtractRecipeDataSchemaMacros[TSI] | _SchemaTypeObject[Literal["integer"]]
    micros: ExtractRecipeDataSchemaMicros[TSI] | _SchemaTypeObject[Literal["integer"]]
    images: list[TSS] | _SchemaTypeList[Literal["string"]]
    cooking_time_minutes: TSI
    ingredients: list[TSS] | _SchemaTypeList[Literal["string"]]
    instructions: list[TSS] | _SchemaTypeList[Literal["string"]]
    dietary_info: ExtractRecipeDataSchemaDietaryInfo[TSB] | _SchemaTypeObject[Literal["boolean"]]
    content: TSS
    url: TSS
    last_updated: TSS


class ExtractRecipeDataSchema(TypedDict, Generic[TSI, TSS, TSB]):
    properties: ExtractRecipeDataSchemaProperties[TSI, TSS, TSB]
    required: list[str]


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
    def __init__(self, api_key: str | None = None, model: str = "gpt-3.5-turbo", auto_open_browser: bool = False):
        """
        Initialize the recipe search tool.

        Args:
            api_key: Optional Tavily API key
            model: LLM model to use
            auto_open_browser: Whether to automatically open URLs in browser for inspection
        """
        logging.debug("Initializing RecipeSearch with model: %s", model)
        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key

        self.search_tool = TavilySearchResults(
            max_results=Config.RECIPE_SEARCH_MAX_RESULTS,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )
        logging.debug("Initialized Tavily search tool with max_results=%s",
                      Config.RECIPE_SEARCH_MAX_RESULTS)

        self.llm = ChatOpenAI(model=model, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.RECIPE_SEARCH_TEXT_SPLITTER_CHUNK_SIZE,
            chunk_overlap=Config.RECIPE_SEARCH_TEXT_SPLITTER_OVERLAP
        )

        self._seen_urls: Set[str] = set()
        self._last_request_time = 0
        # seconds between requests
        self._min_request_interval = Config.RECIPE_SEARCH_MIN_REQUEST_INTERVAL

        # Initialize text splitter for recipe analysis
        self.token_splitter = TokenTextSplitter(
            model_name=model,
            chunk_size=Config.RECIPE_SEARCH_TOKEN_SPLITTER_CHUNK_SIZE,
            chunk_overlap=Config.RECIPE_SEARCH_TOKEN_SPLITTER_OVERLAP
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

        self.schema_extract_recipe_data: ExtractRecipeDataSchema[
            _SchemaTypeVal[Literal["integer"]],
            _SchemaTypeVal[Literal["string"]],
            _SchemaTypeVal[Literal["boolean"]]
        ] = {
            "properties": {
                "is_recipe": {"type": "boolean"},
                "recipe_name": {"type": "string"},
                "title": {"type": "string"},
                "calories": {"type": "integer"},
                "calories_per_serving": {"type": "integer"},
                "cooking_time_minutes": {"type": "integer"},
                "servings": {"type": "integer"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
                "instructions": {"type": "array", "items": {"type": "string"}},
                "images": {"type": "array", "items": {"type": "string"}},
                "content": {"type": "string"},
                "macros": {
                    "type": "object",
                    "properties": {
                        "protein": {"type": "integer"},
                        "carbs": {"type": "integer"},
                        "fat": {"type": "integer"}
                    }
                },
                "micros": {
                    "type": "object",
                    "properties": {
                        "vitamin_a": {"type": "integer"},
                        "vitamin_c": {"type": "integer"},
                        "vitamin_d": {"type": "integer"},
                        "vitamin_e": {"type": "integer"},
                        "vitamin_k": {"type": "integer"},
                        "calcium": {"type": "integer"},
                        "iron": {"type": "integer"},
                        "magnesium": {"type": "integer"},
                        "phosphorus": {"type": "integer"}
                    }
                },
                "dietary_info": {
                    "type": "object",
                    "properties": {
                        "vegetarian": {"type": "boolean"},
                        "vegan": {"type": "boolean"},
                        "gluten_free": {"type": "boolean"}
                    }
                }
            },
            "required": ["is_recipe", "recipe_name", "ingredients"]
        }

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

                For example:
                Example 1:
                url_path: recipes/chicken-meatballs-quinoa-curried-cauliflower
                {{
                    "probability": 0.8,
                    "path_type": "recipe_page",
                    "reasoning": "The URL path suggests a single recipe page because it contains a specific dish name and is subpath of recipes/."
                }}
                Example 2:
                url_path: recipes/collection/700-calorie-meal-recipes#main-navigation-popup
                {{
                    "probability": 0.8,
                    "path_type": "recipe_listing",
                    "reasoning": "The URL path suggests a recipe listing page because it contains the word 'recipes' in the final segment of the url path and '700 calorie meal recipes'. The index string after the # is ignored."
                }}

                Pay close attention to the case of the words in the final segment of the url path.

                Respond in this format:
                {{
                    "probability": <float between 0 and 1>,
                    "path_type": "recipe_page" or "recipe_listing" or "non_recipe",
                    "reasoning": "<brief explanation of classification>"
                }}
                """)
        ])

        # Browser settings
        self.auto_open_browser = auto_open_browser
        self.browser = webbrowser.get()  # Get default browser
        logging.debug("Initialized browser: %s", self.browser.name)

        # Initialize persistent SQLite database
        db_path = Path('recipe_database.sqlite')
        self.persistent_conn = sqlite3.connect(str(db_path))
        self.persistent_cursor = self.persistent_conn.cursor()

        # Create recipe_pages table
        self.persistent_cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipe_pages (
                url TEXT PRIMARY KEY,
                probability REAL,
                path_type TEXT,
                last_checked DATETIME,
                content_hash TEXT
            )
        ''')

        # Drop and recreate valid_recipes table to match our schema
        self.persistent_cursor.execute('DROP TABLE IF EXISTS valid_recipes')
        self.persistent_cursor.execute('''
            CREATE TABLE IF NOT EXISTS valid_recipes (
                url TEXT PRIMARY KEY,
                recipe_name TEXT,
                title TEXT,
                calories INTEGER,
                calories_per_serving INTEGER,
                cooking_time_minutes INTEGER,
                servings INTEGER,
                ingredients TEXT,  -- JSON array
                instructions TEXT, -- JSON array
                content TEXT,
                images TEXT,      -- JSON array
                last_updated DATETIME,
                is_recipe BOOLEAN,
                macros TEXT,      -- JSON object
                micros TEXT,      -- JSON object
                dietary_info TEXT, -- JSON object
                is_modified BOOLEAN DEFAULT FALSE,
                original_recipe_url TEXT,
                user_approved BOOLEAN DEFAULT FALSE,
                modification_notes TEXT
            )
        ''')
        self.persistent_conn.commit()

        # Create URL path analysis table
        self.persistent_cursor.execute('''
            CREATE TABLE IF NOT EXISTS url_path_analysis (
                domain TEXT,
                path TEXT,
                probability REAL,
                path_type TEXT,
                last_checked DATETIME,
                processing_status TEXT,
                processed_at DATETIME,
                PRIMARY KEY (domain, path)
            )
        ''')

        self.persistent_conn.commit()

        # Add LLM call tracking
        self.llm_calls = 0
        # Maximum number of LLM calls per session
        self.max_llm_calls = Config.RECIPE_SEARCH_MAX_LLM_CALLS

        # Initialize FAISS for recipe similarity search
        self.embeddings = OpenAIEmbeddings()
        self.faiss_store_path = Path('recipe_embeddings.faiss')
        self.recipe_store = self._initialize_recipe_store()

        # Recipe adaptation prompt
        self.recipe_adaptation_prompt = PromptTemplate.from_template("""
            Given the following recipe constraints:
            {constraints}

            And this recipe:
            {recipe}

            Please analyze if:
            1. The recipe fully satisfies all constraints
            2. If not, can it be modified to satisfy the constraints?
            3. If it can be modified, please provide the modified recipe that satisfies all constraints.

            Respond in this JSON format:
            {
                "satisfies_constraints": boolean,
                "can_be_modified": boolean,
                "modified_recipe": {recipe_object} or null,
                "modifications_made": [list of changes] or null,
                "reasoning": "explanation"
            }
            """)

        # Initialize UMAP for visualization
        self.umap_reducer = UMAP(n_components=2, random_state=42)
        self.pacmap_reducer = pacmap.PaCMAP(
            n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
        )
        self.embedded_recipes = None
        self.recipe_names = None

    def _initialize_recipe_store(self) -> FAISS:
        """Initialize FAISS store with existing valid recipes."""
        try:
            # Try to load existing FAISS index
            if self.faiss_store_path.exists():
                logging.info("Loading existing FAISS index...")
                return FAISS.load_local(str(self.faiss_store_path), self.embeddings)

            # If no existing index, create new one from database
            logging.info("Creating new FAISS index from database...")
            self.persistent_cursor.execute('''
                SELECT * FROM valid_recipes
            ''')
            recipes = self.persistent_cursor.fetchall()

            if not recipes:
                # Create empty store if no recipes exist
                store = FAISS.from_texts(["placeholder"], self.embeddings)
                store.save_local(str(self.faiss_store_path))
                return store

            # Convert recipes to texts for embedding
            texts = [self._recipe_to_text(recipe) for recipe in recipes]
            metadatas = [json.loads(recipe['content']) for recipe in recipes]

            # Create and save the FAISS store
            store = FAISS.from_texts(
                texts, self.embeddings, metadatas=metadatas)
            store.save_local(str(self.faiss_store_path))
            return store

        except Exception as e:
            logging.error(f"Error initializing recipe store: {e}")
            store = FAISS.from_texts(["placeholder"], self.embeddings)
            store.save_local(str(self.faiss_store_path))
            return store

    def _recipe_to_text(self, recipe: Dict | ExtractRecipeDataSchemaProperties) -> str:
        """Convert recipe to searchable text format."""
        return f"""
        Recipe: {recipe.get('title', '')}
        Calories: {recipe.get('calories_per_serving', 0)}
        Protein: {recipe.get('macros', {}).get('protein', 0)}g
        Ingredients: {', '.join(recipe.get('ingredients', []))}
        Dietary Info: {json.dumps(recipe.get('dietary_info', {}))}
        Cooking Time: {recipe.get('cooking_time_minutes', 0)} minutes
        """

    async def _find_similar_recipes(self, constraints: Dict) -> List[Dict]:
        """Find recipes similar to constraints using FAISS."""
        query = self._constraints_to_query(constraints)
        similar_docs = self.recipe_store.similarity_search(query, k=5)
        return [doc.metadata for doc in similar_docs]

    def _constraints_to_query(self, constraints: Dict) -> str:
        """Convert constraints to search query text."""
        return f"""
        Recipe with:
        Calories between {constraints.get('min_calories', 0)} and {constraints.get('max_calories', 1000)}
        Cooking time under {constraints.get('max_cooking_time', 60)} minutes
        Dietary restrictions: {', '.join(constraints.get('dietary_restrictions', []))}
        Must include: {', '.join(constraints.get('must_include', []))}
        Must avoid: {', '.join(constraints.get('must_avoid', []))}
        """

    async def _validate_and_adapt_recipe(self, recipe: Dict, constraints: Dict) -> Optional[Dict]:
        """Validate recipe against constraints and adapt if possible."""
        prompt = self.recipe_adaptation_prompt.format(
            constraints=json.dumps(constraints, indent=2),
            recipe=json.dumps(recipe, indent=2)
        )

        response = await self.llm.apredict(prompt)
        result = json.loads(response)

        if result['satisfies_constraints']:
            return recipe
        elif result['can_be_modified'] and result['modified_recipe']:
            print("\nProposed recipe modifications:")
            print("\n".join(result['modifications_made']))

            if input("\nAccept modified recipe? (y/n): ").lower() == 'y':
                modified_recipe = result['modified_recipe']
                modified_recipe['is_modified'] = True
                modified_recipe['original_recipe_url'] = recipe['url']
                modified_recipe['user_approved'] = True
                return modified_recipe

        return None

    async def search_recipes(
        self,
        min_calories: int = 200,
        max_calories: int = 750,
        constraints: Optional[Dict] = None
    ) -> List[Dict]:
        """Enhanced recipe search with similarity matching and adaptation."""
        valid_recipes = []
        constraints = constraints or {
            'min_calories': min_calories,
            'max_calories': max_calories
        }

        # First try finding similar recipes from existing database
        similar_recipes = await self._find_similar_recipes(constraints)
        for recipe in similar_recipes:
            adapted_recipe = await self._validate_and_adapt_recipe(recipe, constraints)
            if adapted_recipe:
                valid_recipes.append(adapted_recipe)
                if len(valid_recipes) >= Config.RECIPE_SEARCH_MAX_RESULTS:
                    return valid_recipes

        # If we need more recipes, search for new ones
        if len(valid_recipes) < Config.RECIPE_SEARCH_MAX_RESULTS:
            # Use existing search logic but skip already processed URLs
            query = f"vegetarian recipe websites with recipes between {
                min_calories} and {max_calories} calories"
            initial_results = self.search_tool.invoke({"query": query})

            for result in initial_results:
                if isinstance(result, dict):
                    url = result.get("url", "")

                    # Skip if URL already processed
                    if self._is_url_processed(url):
                        continue

                    # Process new URLs
                    if self._is_recipe_listing_page(url):
                        recipe_links = await self._extract_recipe_links(url)
                        new_links = [
                            link for link, probability in recipe_links if not self._is_url_processed(link)]

                        for link in new_links:
                            recipe = await self.process_url(link)
                            if recipe:
                                # Add new recipe to FAISS store
                                self.recipe_store.add_texts(
                                    [self._recipe_to_text(recipe)],
                                    metadatas=[recipe]
                                )

                                # Try adapting the recipe
                                adapted_recipe = await self._validate_and_adapt_recipe(recipe, constraints)
                                if adapted_recipe:
                                    valid_recipes.append(adapted_recipe)
                                    if len(valid_recipes) >= Config.RECIPE_SEARCH_MAX_RESULTS:
                                        return valid_recipes

        return valid_recipes

    def _is_url_processed(self, url: str) -> bool:
        """Check if URL has already been processed."""
        parsed = urlparse(url)
        self.persistent_cursor.execute('''
            SELECT processing_status
            FROM url_path_analysis
            WHERE domain = ? AND path = ?
        ''', (parsed.netloc, parsed.path.strip('/')))

        result = self.persistent_cursor.fetchone()
        return bool(result and result[0] not in ['pending', None])

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
                (url, probability, used_llm, timestamp,
                 processing_time, content_length, error_message)
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
                    prob_loss = 1.0 / (1.0 - probability)
                    prob_loss += self._analyze_content_indicators(soup)
                    probability = 1.0 - (1.0 / prob_loss)

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
        init_score = 10.0
        score = 10.0

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

        return max(score, 0.0) - init_score

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
            # Check cache first
            cached_result = self._get_cached_path_analysis(url)
            if cached_result:
                return cached_result

            # Proceed with LLM analysis
            url_path = self._extract_url_path(url)
            logging.debug("Analyzing URL path: %s", url_path)

            if not url_path:
                return 0.0, "non_recipe", "Empty URL path"

            # Quick check for obvious non-recipe paths
            non_recipe_indicators = [
                'login', 'register', 'about', 'contact',
                'privacy', 'terms', 'search', 'tag'
            ]
            if any(indicator in url_path.lower() for indicator in non_recipe_indicators):
                self._save_path_analysis(url, 0.0, "non_recipe")
                return 0.0, "non_recipe", "Path contains non-recipe indicators"

            # Check LLM call limit
            if self.llm_calls >= self.max_llm_calls:
                logging.warning(
                    "Reached maximum LLM calls limit (%d)", self.max_llm_calls)
                return 0.0, "non_recipe", "LLM call limit reached"

            # Increment LLM call counter
            self.llm_calls += 1

            result = await self.llm.apredict_messages([
                *self.url_analysis_prompt.format_messages(
                    url_path=url_path
                )
            ])

            # Parse and save the response
            try:
                response = json.loads(result.content)
                probability = float(response.get('probability', 0))
                path_type = response.get('path_type', 'non_recipe')
                reasoning = response.get(
                    'reasoning', 'No explanation provided')

                # Save analysis to cache
                self._save_path_analysis(url, probability, path_type)

                return probability, path_type, reasoning

            except Exception as e:
                logging.error("Failed to parse LLM response: %s", e)
                return 0.0, "non_recipe", f"Error parsing LLM response: {e}"

        except Exception as e:
            logging.error("Error analyzing URL path: %s", e)
            return 0.0, "non_recipe", f"Error: {e}"

    @lru_cache(maxsize=100)
    async def _extract_recipe_links(self, url: str, recursion_level: int = 0) -> List[tuple[str, float]]:
        """
        Extract individual recipe links from a recipe listing page with caching.

        Args:
            url: URL to process
            recursion_level: Current depth of recursion

        Returns:
            List of tuples containing (url, probability)
        """
        if recursion_level >= Config.NESTED_LINKS_MAX_RECURSION_LEVEL:
            logging.debug(
                "Reached maximum recursion level (%d) for %s",
                Config.NESTED_LINKS_MAX_RECURSION_LEVEL, url
            )
            return []

        logging.debug("Extracting recipe links from: %s (level %d)",
                      url, recursion_level)
        try:
            content = await self._rate_limited_request(url)
            soup = BeautifulSoup(content, 'html.parser')

            # filter only the links that have the same domain.
            _links = soup.find_all('a', href=True)
            base_domain = urlparse(url).netloc
            base_url = self.ensure_url_scheme(base_domain)

            # Process links, ensuring proper URL format
            _hrefs = []
            for link in _links:
                href = link['href']
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == base_domain or parsed_url.netloc == base_url:
                    _hrefs.append(full_url)

            hrefs = list(set(_hrefs))  # Remove duplicates
            # List of (url, probability) tuples
            recipe_links_with_prob: list[tuple[str, float]] = []

            for href in hrefs:
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

                    # If auto_open_browser is enabled, open the URL for inspection
                    if self.auto_open_browser:
                        await self.open_in_browser(href)

                    # If URL path looks promising, do full content analysis
                    probability = await self._calculate_recipe_probability(href)
                    logging.debug(
                        "Full content analysis for %s: %.2f", href, probability)

                    if probability >= PROBABILITY_OF_RECIPE_IN_URL:
                        recipe_links_with_prob.append((href, path_probability))
                        logging.debug(
                            "Added recipe link: %s (path_prob=%.2f, content_prob=%.2f, type=%s)",
                            href, path_probability, probability, path_type
                        )
                    else:
                        logging.debug(
                            "Skipping URL based on path analysis probability <= %s (%s): %.2f - %s - %s",
                            PROBABILITY_OF_RECIPE_IN_URL, href, path_probability, path_type, reasoning
                        )
                elif path_type == "recipe_listing":
                    logging.debug(
                        "Found nested recipe listing (%s): %.2f - %s",
                        href, path_probability, reasoning
                    )
                    # Process nested recipe listings with incremented recursion level
                    nested_links = await self._extract_recipe_links(href, recursion_level + 1)
                    recipe_links_with_prob.extend(nested_links)
                else:
                    logging.debug(
                        "Skipping URL based on path analysis (%s): %.2f - %s - %s",
                        href, path_probability, path_type, reasoning
                    )

            # Sort by probability and return top results
            sorted_links = sorted(recipe_links_with_prob,
                                  key=lambda x: x[1], reverse=True)
            return sorted_links

        except Exception as e:
            logging.error("Error extracting links from %s: %s", url, e)
            return []

    async def _extract_recipe_data(self, content: str, url: str) -> Optional[ExtractRecipeDataSchemaProperties[int, str]]:
        """Extract recipe data from content using LLM."""
        try:
            schema = self.schema_extract_recipe_data

            chain = create_extraction_chain(schema, self.llm)

            class RecipeDataChainInvoked(TypedDict):
                text: list[ExtractRecipeDataSchemaProperties]
                input: str

            result: RecipeDataChainInvoked = await chain.ainvoke(content)

            if result and len(result['text']) > 0:
                recipe_data: ExtractRecipeDataSchemaProperties = result['text'][0]
                recipe_data['url'] = url
                recipe_data['last_updated'] = datetime.now().isoformat()
                return recipe_data
            return None

        except Exception as e:
            logging.error(f"Error extracting recipe data: {e}")
            return None

    # async def _validate_recipe(self, content: str) -> Dict:
    #     """Use LLM to validate and extract recipe information."""
    #     logging.debug(
    #         "Validating recipe content length: %d characters", len(content))
    #     try:
    #         schema = self.schema_extract_recipe_data
    #         chain = create_extraction_chain(schema, self.llm)
    #         result = await chain.arun(content)
    #         recipe_info = result[0] if result else {}
    #         logging.debug("Recipe validation result: %s", {
    #             "is_recipe": recipe_info.get("is_recipe"),
    #             "name": recipe_info.get("recipe_name"),
    #             "calories": recipe_info.get("calories_per_serving"),
    #             "ingredients_count": len(recipe_info.get("ingredients", []))
    #         })
    #         return recipe_info
    #     except Exception as e:
    #         logging.error("Error validating recipe: %s", e)
    #         return {}

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
            # recipe_info = await self._validate_recipe(content)
            recipe_info = await self._extract_recipe_data(content, url)
            if not recipe_info:
                logging.debug("No recipe info found for %s", url)
                return None

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

            # Save both recipe pages and valid recipes to database
            for recipe in batch_results:
                self._save_valid_recipe(recipe)
                self._save_recipe_page(
                    recipe['url'],
                    recipe.get('probability', 0),
                    'recipe_page'
                )

            valid_recipes.extend(batch_results)

            if len(valid_recipes) >= Config.RECIPE_SEARCH_MAX_RESULTS:
                logging.debug(
                    "Reached target of %d valid recipes, stopping search",
                    Config.RECIPE_SEARCH_MAX_RESULTS
                )
                break

            await asyncio.sleep(0.5)  # Small delay between batches

        logging.debug("Search complete. Found %d valid recipes",
                      len(valid_recipes))
        return valid_recipes[:Config.RECIPE_SEARCH_MAX_RESULTS]

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
        """Cleanup database connections."""
        if hasattr(self, 'persistent_conn'):
            self.persistent_conn.close()
        if hasattr(self, 'conn'):
            self.conn.close()

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

    async def open_in_browser(self, url: str, delay: float = 2.0) -> None:
        """
        Open a URL in the default browser with optional delay.

        Args:
            url: URL to open
            delay: Delay in seconds before opening (to avoid overwhelming the browser)
        """
        try:
            logging.debug("Opening URL in browser: %s", url)
            self.browser.open(url, new=2)  # new=2 means open in new tab
            if delay > 0:
                await asyncio.sleep(delay)
        except Exception as e:
            logging.error("Failed to open URL in browser: %s", e)

    def ensure_url_scheme(self, url: str, default_scheme: str = 'https') -> str:
        """Ensure URL has a scheme (http/https), adding default if missing."""
        parsed = urlparse(url)
        if not parsed.scheme:
            # Reconstruct URL with scheme
            parsed = parsed._replace(scheme=default_scheme)
            if not parsed.netloc and parsed.path:
                # If there's no netloc but there is a path, the domain is probably in the path
                parts = parsed.path.split('/', 1)
                parsed = parsed._replace(
                    netloc=parts[0], path=parts[1] if len(parts) > 1 else '')
        return urlunparse(parsed)

    def _save_recipe_page(self, url: str, probability: float, path_type: str):
        """Save or update recipe page information."""
        try:
            content = requests.get(url).text
            content_hash = hashlib.md5(content.encode()).hexdigest()

            self.persistent_cursor.execute('''
                INSERT OR REPLACE INTO recipe_pages
                (url, probability, path_type, last_checked, content_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', (url, probability, path_type, datetime.now().isoformat(), content_hash))
            self.persistent_conn.commit()
        except Exception as e:
            logging.error("Error saving recipe page %s: %s", url, e)
            exit(1)

    def _save_valid_recipe(self, recipe: ExtractRecipeDataSchemaProperties):
        """Save or update valid recipe information."""
        try:
            # Save to SQLite
            self.persistent_cursor.execute('''
                INSERT OR REPLACE INTO valid_recipes
                (url, recipe_name, title, calories, calories_per_serving,
                 cooking_time_minutes, servings, ingredients, instructions,
                 content, images, last_updated, is_recipe,
                 macros, micros, dietary_info,
                 is_modified, original_recipe_url, user_approved, modification_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                recipe['url'],
                recipe.get('recipe_name', ''),
                recipe.get('title', ''),
                recipe.get('calories', 0),
                recipe.get('calories_per_serving', 0),
                recipe.get('cooking_time_minutes', 0),
                recipe.get('servings', 0),
                json.dumps(recipe.get('ingredients', [])),
                json.dumps(recipe.get('instructions', [])),
                recipe.get('content', ''),
                json.dumps(recipe.get('images', [])),
                datetime.now().isoformat(),
                recipe.get('is_recipe', False),
                json.dumps(recipe.get('macros', {})),
                json.dumps(recipe.get('micros', {})),
                json.dumps(recipe.get('dietary_info', {})),
                recipe.get('is_modified', False),
                recipe.get('original_recipe_url', None),
                recipe.get('user_approved', False),
                recipe.get('modification_notes', None)
            ))
            self.persistent_conn.commit()

            # Update FAISS index
            recipe_text = self._recipe_to_text(recipe)
            self.recipe_store.add_texts([recipe_text], metadatas=[recipe])
            self.recipe_store.save_local(str(self.faiss_store_path))

            # Verify index integrity
            if not self.verify_faiss_index():
                logging.warning("FAISS index out of sync, rebuilding...")
                self.rebuild_faiss_index()

        except Exception as e:
            logging.error("Error saving recipe %s: %s", recipe['url'], e)
            exit(1)

    def _get_cached_path_analysis(self, url: str) -> Optional[Tuple[float, str, str]]:
        """Get cached path analysis if it exists for this URL or its parent paths."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path.strip('/')
            path_parts = path.split('/')

            # Check all parent paths from most specific to least specific
            for i in range(len(path_parts), 0, -1):
                parent_path = '/'.join(path_parts[:i])

                self.persistent_cursor.execute('''
                    SELECT probability, path_type, last_checked
                    FROM url_path_analysis
                    WHERE domain = ? AND path = ?
                ''', (domain, parent_path))

                result = self.persistent_cursor.fetchone()
                if result:
                    probability, path_type, last_checked = result
                    # If parent path is non_recipe with low probability, skip all child paths
                    if (path_type == 'non_recipe' and
                            probability < Config.MIN_PROBABILITY_OF_RECIPE_IN_URL_PATH):
                        logging.debug(
                            "Skipping URL due to low-probability parent path %s: %.2f",
                            parent_path, probability
                        )
                        return probability, path_type, "Parent path indicates non-recipe content"

                    # If this is the exact path (not a parent), return the cached result
                    if parent_path == path:
                        return probability, path_type, "Using cached analysis"

            return None
        except Exception as e:
            logging.error("Error checking cached path analysis: %s", e)
            return None

    def _save_path_analysis(self, url: str, probability: float, path_type: str, processing_status: str = 'pending'):
        """Save path analysis results to cache with processing status."""
        try:
            parsed = urlparse(url)
            self.persistent_cursor.execute('''
                INSERT OR REPLACE INTO url_path_analysis
                (domain, path, probability, path_type, last_checked, processing_status, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                parsed.netloc,
                parsed.path.strip('/'),
                probability,
                path_type,
                datetime.now().isoformat(),
                processing_status,
                None
            ))
            self.persistent_conn.commit()
        except Exception as e:
            logging.error("Error saving path analysis: %s", e)

    async def process_url(self, url: str) -> Optional[ExtractRecipeDataSchemaProperties]:
        """Process a URL and determine if it contains a valid recipe."""
        try:
            # First check if URL has already been processed
            parsed = urlparse(url)
            self.persistent_cursor.execute('''
                SELECT processing_status
                FROM url_path_analysis
                WHERE domain = ? AND path = ?
            ''', (parsed.netloc, parsed.path.strip('/')))

            result = self.persistent_cursor.fetchone()
            if result and result[0] not in ['pending', 'error']:
                logging.info(f"URL already processed with status: {result[0]}")
                return None

            # Process the URL
            content = await self._fetch_and_parse_url(url)
            if not content:
                self._update_processing_status(url, 'non_recipe')
                return None

            # Analyze content for recipe
            recipe_data = await self._extract_recipe_data(content, url)

            if recipe_data and recipe_data.get('is_recipe', False):
                self._save_valid_recipe(recipe_data)
                self._update_processing_status(url, 'valid_recipe')
                return recipe_data
            else:
                self._update_processing_status(url, 'invalid_recipe')
                return None

        except Exception as e:
            logging.error(f"Error processing URL {url}: {e}")
            self._update_processing_status(url, 'error')
            return None

    def _update_processing_status(self, url: str, status: str):
        """Update the processing status for a URL."""
        try:
            parsed = urlparse(url)
            self.persistent_cursor.execute('''
                UPDATE url_path_analysis
                SET processing_status = ?, processed_at = ?
                WHERE domain = ? AND path = ?
            ''', (
                status,
                datetime.now().isoformat(),
                parsed.netloc,
                parsed.path.strip('/')
            ))
            self.persistent_conn.commit()
        except Exception as e:
            logging.error(f"Error updating processing status: {e}")

    async def _fetch_and_parse_url(self, url: str) -> Optional[str]:
        """Fetch and parse URL content."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    # Remove script and style elements
                    for element in soup(['script', 'style']):
                        element.decompose()
                    return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logging.error(f"Error fetching URL {url}: {e}")
            return None

    def rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from scratch using the database."""
        try:
            logging.info("Rebuilding FAISS index from database...")

            # Get all recipes from database
            self.persistent_cursor.execute('''
                SELECT * FROM valid_recipes
            ''')
            recipes = self.persistent_cursor.fetchall()

            if not recipes:
                logging.warning("No recipes found in database to build index")
                store = FAISS.from_texts(["placeholder"], self.embeddings)
                store.save_local(str(self.faiss_store_path))
                self.recipe_store = store
                return

            # Convert recipes to texts and metadata
            texts = [self._recipe_to_text(recipe) for recipe in recipes]
            metadatas = [json.loads(recipe['content']) for recipe in recipes]

            # Create new store
            store = FAISS.from_texts(
                texts, self.embeddings, metadatas=metadatas)

            # Save and update
            store.save_local(str(self.faiss_store_path))
            self.recipe_store = store

            logging.info(f"Successfully rebuilt index with {
                         len(recipes)} recipes")

        except Exception as e:
            logging.error(f"Error rebuilding FAISS index: {e}")
            raise

    def verify_faiss_index(self) -> bool:
        """Verify FAISS index matches database content."""
        try:
            # Get count of recipes in database
            self.persistent_cursor.execute(
                'SELECT COUNT(*) FROM valid_recipes')
            db_count = self.persistent_cursor.fetchone()[0]

            # Get count of recipes in FAISS (subtract 1 for placeholder if empty)
            faiss_count = len(
                self.recipe_store.index_to_docstore_id) - (1 if db_count == 0 else 0)

            if db_count != faiss_count:
                logging.warning(f"Index mismatch: Database has {
                                db_count} recipes, FAISS has {faiss_count}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error verifying FAISS index: {e}")
            return False

    def _get_recipe_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get embeddings and recipe names from FAISS store."""
        if not self.recipe_store or not self.recipe_store.index:
            raise ValueError("No recipes in FAISS store")

        # Get all embeddings from FAISS
        embeddings = self.recipe_store.index.reconstruct_n(
            0, self.recipe_store.index.ntotal)

        # Get recipe names
        recipe_names = [
            self.recipe_store.docstore.search(idx).get('title', f'Recipe {i}')
            for i, idx in enumerate(self.recipe_store.index_to_docstore_id.values())
        ]

        return embeddings, recipe_names

    def _reduce_dimensions(self, query_embedding: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Reduce dimensions of embeddings using UMAP."""
        embeddings, recipe_names = self._get_recipe_embeddings()

        # Cache recipe names
        self.recipe_names = recipe_names

        # Fit UMAP on recipe embeddings
        if self.embedded_recipes is None:
            self.embedded_recipes = self.umap_reducer.fit_transform(embeddings)
            # self.embedded_recipes = self.pacmap_reducer.fit_transform(np.array(embeddings), init="pca",)

        # Transform query if provided
        query_2d = None
        if query_embedding is not None:
            query_2d = self.umap_reducer.transform(
                query_embedding.reshape(1, -1))

        return self.embedded_recipes, query_2d

    def visualize_store_interactive(self, query: Optional[str] = None) -> go.Figure:
        """Create interactive Plotly visualization of recipe embeddings."""
        # Get query embedding if provided
        query_embedding = None
        if query:
            query_embedding = self.embeddings.embed_query(query)

        # Reduce dimensions
        recipe_embeddings_2d, query_2d = self._reduce_dimensions(
            query_embedding)

        # Create scatter plot
        fig = go.Figure()

        # Add recipe points
        fig.add_trace(go.Scatter(
            x=recipe_embeddings_2d[:, 0],
            y=recipe_embeddings_2d[:, 1],
            mode='markers+text',
            marker=dict(color='blue', size=8),
            text=self.recipe_names,
            textposition="top center",
            name='Recipes'
        ))

        # Add query point if provided
        if query_2d is not None:
            fig.add_trace(go.Scatter(
                x=query_2d[:, 0],
                y=query_2d[:, 1],
                mode='markers',
                marker=dict(color='red', size=12, symbol='star'),
                name='Query'
            ))

        # Update layout
        fig.update_layout(
            title="Recipe Embedding Space",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            showlegend=True
        )

        return fig

    def save_visualization(self, query: Optional[str] = None, path: str = "recipe_viz.png") -> str:
        """Save static visualization of recipe embeddings and return base64 encoded image."""
        # Get query embedding if provided
        query_embedding = None
        if query:
            query_embedding = self.embeddings.embed_query(query)

        # Reduce dimensions
        recipe_embeddings_2d, query_2d = self._reduce_dimensions(
            query_embedding)

        # Create plot
        plt.figure(figsize=(12, 8))

        # Plot recipes
        plt.scatter(
            recipe_embeddings_2d[:, 0],
            recipe_embeddings_2d[:, 1],
            c='blue',
            alpha=0.5
        )

        # Add labels for some points
        for i, (x, y) in enumerate(recipe_embeddings_2d):
            if i % 3 == 0:  # Label every third point to avoid overcrowding
                plt.annotate(
                    self.recipe_names[i],
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points'
                )

        # Add query point if provided
        if query_2d is not None:
            plt.scatter(
                query_2d[:, 0],
                query_2d[:, 1],
                c='red',
                marker='*',
                s=200,
                label='Query'
            )

        plt.title("Recipe Embedding Space")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")

        # Save to file if path provided
        if path:
            plt.savefig(path)

        # Convert to base64 for web display
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()

        # Encode to base64
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()

        return f"data:image/png;base64,{img_str}"


def get_input_with_timeout(prompt: str, timeout: int = 3) -> str:
    """Get user input with a timeout and progress bar."""
    print(prompt, end='', flush=True)

    # Setup progress bar
    with tqdm(total=timeout, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}s') as pbar:
        start_time = time.time()
        input_ready = False
        response = ''

        while time.time() - start_time < timeout:
            # Check if there's input ready
            if select.select([sys.stdin], [], [], 0)[0]:
                response = sys.stdin.readline().strip()
                input_ready = True
                break

            # Update progress bar
            elapsed = time.time() - start_time
            pbar.n = min(timeout, int(elapsed))
            pbar.refresh()
            time.sleep(0.1)

    print()  # New line after progress bar
    return response if input_ready else 'n'  # Default to 'n' if no input


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

        # Ask user about browser inspection with timeout
        inspect_in_browser = get_input_with_timeout(
            "\nWould you like to inspect URLs in browser? (y/n): ") == 'y'

        if inspect_in_browser:
            print("URLs will open in your default browser for inspection.")
            try:
                delay = float(get_input_with_timeout(
                    "Enter delay between URLs (seconds, default 2.0): ") or "2.0")
            except ValueError:
                delay = 2.0
                print("Invalid input, using default delay of 2.0 seconds")
        else:
            delay = 0

        # Initialize recipe searcher with browser settings
        print("\nInitializing recipe search...")
        recipe_searcher = RecipeSearch(
            auto_open_browser=inspect_in_browser,
            model=Config.RECIPE_SEARCH_MODEL
        )

        # Update the open_in_browser method's delay if specified
        if inspect_in_browser:
            recipe_searcher.open_in_browser = lambda url: recipe_searcher.open_in_browser(
                url, delay=delay)

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
            print(
                f"Calories per serving: {recipe.get('calories', 'Not specified')}")
            print(
                f"Cooking time: {recipe.get('cooking_time', 'Not specified')} minutes")

            if recipe.get('ingredients'):
                print("\nIngredients:")
                for ingredient in recipe['ingredients']:
                    print(f"  • {ingredient}")

            # Check for avoided ingredients
            conflicts = [ing for ing in ingredients_to_avoid
                         if ing.lower() in recipe.get('content', '').lower()]
            if conflicts:
                print("\n⚠️  WARNING:")
                print(
                    f"Recipe contains avoided ingredients: {', '.join(conflicts)}")

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
        print(f"High probability recipes(≥{
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


async def test_recipe_validate_existing_urls():
    """Process existing URLs from the URL Analysis table and save valid recipes to the database"""
    try:
        # Initialize recipe searcher
        recipe_searcher = RecipeSearch(model=Config.RECIPE_SEARCH_MODEL)

        # Get all unprocessed URLs with high probability
        recipe_searcher.persistent_cursor.execute('''
            SELECT domain, path, probability
            FROM url_path_analysis
            WHERE (processing_status IS NULL OR processing_status = 'pending')
            AND probability >= ?
            ORDER BY probability DESC
        ''', (Config.MIN_PROBABILITY_OF_RECIPE_IN_URL_PATH,))

        urls_to_process = recipe_searcher.persistent_cursor.fetchall()

        if not urls_to_process:
            print("No pending URLs to process")
            return

        print(f"\nProcessing {len(urls_to_process)} URLs...")
        print("="*60)

        # Process URLs with progress bar
        valid_recipes = []
        with tqdm(total=len(urls_to_process), desc="Processing URLs") as pbar:
            for domain, path, probability in urls_to_process:
                url = f"https://{domain}/{path}"

                print(f"\nProcessing URL: {url}")
                print(f"Initial probability: {probability:.2f}")

                recipe_data = await recipe_searcher.process_url(url)

                if recipe_data and recipe_data.get('is_recipe', False):
                    valid_recipes.append(recipe_data)
                    print("✅ Valid recipe found!")
                    print(f"Title: {recipe_data['recipe_name']}")
                    print(
                        f"Calories: {recipe_data.get('calories_per_serving', 'Unknown')}")
                else:
                    print("❌ No valid recipe found")

                pbar.update(1)
                await asyncio.sleep(0.5)  # Small delay between requests

        # Print summary
        print("\n" + "="*60)
        print("Processing Complete!")
        print(f"Total URLs processed: {len(urls_to_process)}")
        print(f"Valid recipes found: {len(valid_recipes)}")

        # Print processing status distribution
        recipe_searcher.persistent_cursor.execute('''
            SELECT processing_status, COUNT(*) as count
            FROM url_path_analysis
            GROUP BY processing_status
        ''')

        print("\nProcessing Status Distribution:")
        for status, count in recipe_searcher.persistent_cursor.fetchall():
            print(f"{status}: {count}")

    except Exception as e:
        print(f"\n❌ Error during URL validation: {type(e).__name__}: {str(e)}")
        raise
    finally:
        recipe_searcher.monitor.stop()


async def reset_valid_recipes_to_pending():
    """One-off helper to reset all valid recipes back to pending status."""
    try:
        # Initialize database connection
        db_path = Path('recipe_database.sqlite')
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Get count before update
        cursor.execute('''
            SELECT COUNT(*) 
            FROM url_path_analysis 
            WHERE processing_status = 'valid_recipe'
        ''')
        count_before = cursor.fetchone()[0]

        # Update the processing status
        cursor.execute('''
            UPDATE url_path_analysis 
            SET processing_status = 'pending',
                processed_at = NULL
            WHERE processing_status = 'valid_recipe'
        ''')

        # Commit the changes
        conn.commit()

        print(f"Reset {count_before} valid recipes to pending status")

    except Exception as e:
        print(f"Error resetting valid recipes: {e}")
        raise
    finally:
        conn.close()

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

    # Run both tests
    try:
        # print("🔍 Starting recipe search tests...")
        # asyncio.run(test_recipe_search())
        print("🔄 Resetting valid recipes to pending...")
        asyncio.run(reset_valid_recipes_to_pending())
        print("\n🔍 Starting URL validation...")
        asyncio.run(test_recipe_validate_existing_urls())
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Tests failed: {type(e).__name__}: {str(e)}")
        raise
