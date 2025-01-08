from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

from meal_planner_agent.meal_plan_constraints_pd import DietaryRestrictions


class ProductVitamins(BaseModel):
    """Vitamin content per 100g"""
    vitamin_a_mcg: float = Field(0.0, description="Vitamin A in micrograms")
    vitamin_c_mg: float = Field(0.0, description="Vitamin C in milligrams")
    vitamin_d_mcg: float = Field(0.0, description="Vitamin D in micrograms")
    vitamin_e_mg: float = Field(0.0, description="Vitamin E in milligrams")
    vitamin_k_mcg: float = Field(0.0, description="Vitamin K in micrograms")
    thiamin_mg: float = Field(0.0, description="Thiamin (B1) in milligrams")
    riboflavin_mg: float = Field(
        0.0, description="Riboflavin (B2) in milligrams")
    niacin_mg: float = Field(0.0, description="Niacin (B3) in milligrams")
    vitamin_b6_mg: float = Field(0.0, description="Vitamin B6 in milligrams")
    folate_mcg: float = Field(0.0, description="Folate (B9) in micrograms")
    vitamin_b12_mcg: float = Field(
        0.0, description="Vitamin B12 in micrograms")


class ProductMinerals(BaseModel):
    """Mineral content per 100g"""
    calcium_mg: float = Field(0.0, description="Calcium in milligrams")
    iron_mg: float = Field(0.0, description="Iron in milligrams")
    magnesium_mg: float = Field(0.0, description="Magnesium in milligrams")
    phosphorus_mg: float = Field(0.0, description="Phosphorus in milligrams")
    potassium_mg: float = Field(0.0, description="Potassium in milligrams")
    sodium_mg: float = Field(0.0, description="Sodium in milligrams")
    zinc_mg: float = Field(0.0, description="Zinc in milligrams")
    selenium_mcg: float = Field(0.0, description="Selenium in micrograms")


class NutritionalInfo(BaseModel):
    """Nutritional information per 100g"""
    calories_per_100g: float = Field(..., description="Calories per 100g")
    protein_g: float = Field(..., description="Protein content in grams")
    carbs_g: float = Field(..., description="Carbohydrate content in grams")
    fat_g: float = Field(..., description="Fat content in grams")
    fiber_g: float = Field(0.0, description="Fiber content in grams")
    sugar_g: float = Field(0.0, description="Sugar content in grams")
    salt_g: float = Field(0.0, description="Salt content in grams")
    vitamins: Optional[ProductVitamins] = Field(
        default_factory=lambda: None,
        description="Detailed vitamin content"
    )
    minerals: Optional[ProductMinerals] = Field(
        default_factory=lambda: None,
        description="Detailed mineral content"
    )


class SupermarketProduct(BaseModel):
    """Supermarket product details with nutritional information"""
    product_name: str = Field(..., description="Name of the product")
    price: float = Field(..., description="Current price in GBP")
    price_per_unit: str = Field(...,
                                description="Price per unit (e.g., '£2.50/kg')")
    ingredients: List[str] = Field(
        ..., description="List of ingredients in the product")
    allergens: List[str] = Field(
        default_factory=list, description="List of allergens")
    product_id: str = Field(..., description="Unique product identifier")
    description: str = Field(..., description="Product description")
    url: str = Field(..., description="Product page URL")
    image_url: str = Field(..., description="URL of product image")
    nutritional_info: NutritionalInfo = Field(
        ..., description="Detailed nutritional information")
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of last update"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Organic Whole Milk",
                "price": 1.99,
                "price_per_unit": "£1.75/L",
                "ingredients": ["Organic Whole Milk"],
                "allergens": ["Milk"],
                "product_id": "12345",
                "description": "Fresh organic whole milk from grass-fed cows",
                "url": "https://example.com/products/organic-milk",
                "image_url": "https://example.com/images/milk.jpg",
                "nutritional_info": {
                    "calories_per_100g": 65,
                    "protein_g": 3.3,
                    "carbs_g": 4.7,
                    "fat_g": 3.6,
                    "fiber_g": 0,
                    "sugar_g": 4.7,
                    "salt_g": 0.1
                }
            }
        }


class Meal(BaseModel):
    """A meal with recipe details and cost information."""
    recipe_name: str = Field(..., description="Name of the recipe")
    recipe_url: str = Field(..., description="URL of the recipe")
    servings: int = Field(..., description="Number of servings")
    total_cost_gbp: float = Field(..., description="Total cost in GBP")
    cost_per_serving_gbp: float = Field(...,
                                        description="Cost per serving in GBP")
    ingredient_costs_gbp: Dict[str, float] = Field(
        ..., description="Cost breakdown by ingredient")
    calories_per_serving: int = Field(..., description="Calories per serving")
    protein_g: float = Field(..., description="Protein content in grams")
    carbs_g: float = Field(..., description="Carbohydrate content in grams")
    fat_g: float = Field(..., description="Fat content in grams")
    nutritional_info: NutritionalInfo = Field(
        ..., description="Detailed nutritional information")
    cooking_time_minutes: int = Field(...,
                                      description="Cooking time in minutes")
    ingredients: List[str] = Field(..., description="List of ingredients")
    instructions: List[str] = Field(..., description="Cooking instructions")
    dietary_info: DietaryRestrictions

    class Config:
        json_schema_extra = {
            "example": {
                "recipe_name": "Spaghetti Carbonara",
                "recipe_url": "https://example.com/recipes/carbonara",
                "servings": 4,
                "total_cost_gbp": 12.50,
                "cost_per_serving_gbp": 3.125,
                "ingredient_costs_gbp": {
                    "spaghetti": 1.50,
                    "eggs": 1.00,
                    "pecorino cheese": 3.00,
                    "pancetta": 5.00,
                    "black pepper": 0.10
                },
                "calories_per_serving": 650,
                "protein_g": 25.0,
                "carbs_g": 70.0,
                "fat_g": 30.0,
                "cooking_time_minutes": 20,
                "ingredients": [
                    "400g spaghetti",
                    "4 large eggs",
                    "100g pecorino cheese",
                    "150g pancetta",
                    "2 tsp black pepper"
                ],
                "instructions": [
                    "Cook pasta in salted water",
                    "Fry pancetta until crispy",
                    "Mix eggs and cheese",
                    "Combine all ingredients",
                    "Season with black pepper"
                ],
                "dietary_info": ["contains dairy", "contains eggs", "contains pork"]
            }
        }
