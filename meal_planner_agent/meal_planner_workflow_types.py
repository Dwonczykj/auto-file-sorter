from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


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
        default_factory=ProductVitamins,
        description="Detailed vitamin content"
    )
    minerals: Optional[ProductMinerals] = Field(
        default_factory=ProductMinerals,
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
