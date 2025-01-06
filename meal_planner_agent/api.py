from fastapi import FastAPI, Query, HTTPException, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import sqlite3
from datetime import datetime
import json
from meal_planner_agent.recipe_search import RecipeSearch
from pydantic import BaseModel
from pathlib import Path
from icecream import ic
from fastapi.responses import JSONResponse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Recipe Search Results API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()


class RecipePage(BaseModel):
    url: str
    probability: float
    path_type: str
    last_checked: datetime
    content_hash: str


class Recipe(BaseModel):
    url: str
    title: str
    calories: int
    cooking_time: int
    ingredients: List[str]
    content: str
    last_updated: datetime


class URLPathAnalysis(BaseModel):
    domain: str
    path: str
    probability: float
    path_type: str
    last_checked: datetime


class RecipeVisualizationResponse(BaseModel):
    interactive_plot: dict  # Plotly figure JSON
    static_image: str      # Base64 encoded PNG


def get_db():
    """Get database connection."""
    db_path = Path('recipe_database.sqlite')
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
    return sqlite3.connect(str(db_path))


@app.get("/recipe-pages", response_model=List[RecipePage])
async def get_recipe_pages(
    min_prob: float = Query(0.0, ge=0.0, le=1.0),
    path_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get recipe pages with filtering and pagination."""
    try:
        conn = get_db()
        cursor = conn.cursor()

        query = """
            SELECT url, probability, path_type, last_checked, content_hash
            FROM recipe_pages
            WHERE probability >= ?
        """
        params: list[float | str] = [min_prob]

        if path_type:
            query += " AND path_type = ?"
            params.append(path_type)

        query += " ORDER BY probability DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        results = cursor.fetchall()

        return [
            RecipePage(
                url=row[0],
                probability=row[1],
                path_type=row[2],
                last_checked=datetime.fromisoformat(row[3]),
                content_hash=row[4]
            )
            for row in results
        ]
    finally:
        conn.close()


@app.get("/recipes", response_model=List[Recipe])
async def get_recipes(
    min_calories: Optional[int] = None,
    max_calories: Optional[int] = None,
    search: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get recipes with filtering and pagination."""
    try:
        conn = get_db()
        cursor = conn.cursor()

        query = "SELECT * FROM valid_recipes WHERE 1=1"
        params = []

        if min_calories is not None:
            query += " AND calories >= ?"
            params.append(min_calories)

        if max_calories is not None:
            query += " AND calories <= ?"
            params.append(max_calories)

        if search:
            query += """ AND (
                title LIKE ? OR 
                content LIKE ? OR 
                ingredients LIKE ?
            )"""
            search_term = f"%{search}%"
            params.extend([search_term] * 3)

        query += " ORDER BY last_updated DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        results = cursor.fetchall()

        return [
            Recipe(
                url=row[0],
                title=row[1],
                calories=row[2],
                cooking_time=row[3],
                ingredients=json.loads(row[4]),
                content=row[5],
                last_updated=datetime.fromisoformat(row[6])
            )
            for row in results
        ]
    finally:
        conn.close()


@app.get("/url-analysis", response_model=List[URLPathAnalysis])
async def get_url_analysis(
    domain: Optional[str] = None,
    path_type: Optional[str] = None,
    min_prob: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get URL path analysis results with filtering and pagination."""
    try:
        conn = get_db()
        cursor = conn.cursor()
        print(
            f"min_prob: {min_prob}, domain: {domain}, path_type: {path_type}, limit: {limit}, offset: {offset}")

        query = """
            SELECT domain, path, probability, path_type, last_checked
            FROM url_path_analysis
            WHERE probability >= ?
        """
        params: list[float | str] = [min_prob]

        if domain:
            query += " AND domain LIKE ?"
            params.append(f"%{domain}%")

        if path_type:
            query += " AND path_type = ?"
            params.append(path_type)

        query += " ORDER BY probability DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        results = cursor.fetchall()

        res = [
            URLPathAnalysis(
                domain=row[0],
                path=row[1],
                probability=row[2],
                path_type=row[3],
                last_checked=datetime.fromisoformat(row[4])
            )
            for row in results
        ]
        return ic(res)
    finally:
        conn.close()


@app.get("/stats")
async def get_stats():
    """Get overall statistics."""
    try:
        conn = get_db()
        cursor = conn.cursor()

        stats = {}

        # Recipe pages stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(probability) as avg_prob,
                SUM(CASE WHEN path_type = 'recipe_page' THEN 1 ELSE 0 END) as recipe_pages,
                SUM(CASE WHEN path_type = 'recipe_listing' THEN 1 ELSE 0 END) as recipe_listings
            FROM recipe_pages
        """)
        row = cursor.fetchone()
        stats['recipe_pages'] = {
            'total': row[0],
            'avg_probability': row[1],
            'recipe_pages_count': row[2],
            'recipe_listings_count': row[3]
        }

        # Valid recipes stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                AVG(calories) as avg_calories,
                AVG(cooking_time) as avg_cooking_time
            FROM valid_recipes
        """)
        row = cursor.fetchone()
        stats['valid_recipes'] = {
            'total': row[0],
            'avg_calories': row[1],
            'avg_cooking_time': row[2]
        }

        return stats
    finally:
        conn.close()


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error processing request: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info("\n=== FastAPI Request ===")
    logger.info(f"Method: {request.method}")
    logger.info(f"URL: {request.url}")
    logger.info(
        f"Client: {request.client.host if request.client else 'Unknown'}")
    logger.info("Headers:")
    for name, value in request.headers.items():
        logger.info(f"  {name}: {value}")
    logger.info("=== End Request ===\n")

    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise


@router.get("/recipes/visualize", response_model=RecipeVisualizationResponse)
async def visualize_recipes(query: Optional[str] = None):
    recipe_search = RecipeSearch()

    # Get both visualizations
    interactive_fig = recipe_search.visualize_store_interactive(query)
    static_image = recipe_search.save_visualization(query)

    return RecipeVisualizationResponse(
        interactive_plot=interactive_fig.to_dict(),
        static_image=static_image
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
