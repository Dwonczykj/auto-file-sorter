from typing import List, Literal, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict
from datetime import date
from enum import Enum


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    VERY_ACTIVE = "very_active"
    EXTRA_ACTIVE = "extra_active"


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


breakdowns_enum = Literal['daily', 'per_meal', 'per_ingredient']
micronutrients_enum = Literal["none", "low", "normal"]
YesNoEnum = Literal["no", "yes"]
MealSizeEnum = Literal["none", "small", "medium", "large"]

T = TypeVar('T')


class Calories(BaseModel):
    max: int
    min: int


class Protein(BaseModel):
    max: int
    min: int


class LipidsPcntNonProteinCalories(BaseModel):
    max: int
    min: int


class MicronutrientsRDA(BaseModel):
    vitamin_b12: float
    vitamin_b6: float
    vitamin_d: float
    vitamin_c: float
    iron: float
    zinc: float
    magnesium: float
    selenium: float
    vitamin_a: float
    vitamin_e: float
    vitamin_k: float
    vitamin_b1: float
    vitamin_b2: float
    vitamin_b3: float
    vitamin_b5: float
    vitamin_b7: float
    vitamin_b9: float
    omega_3_dha: float
    omega_3_epa: float


class CookingTimeMinutes(BaseModel):
    max: int
    min: int


class BudgetPerWeekGBP(BaseModel):
    max: int
    min: int


class MaxMin(BaseModel, Generic[T]):
    max: T
    min: T


class ServingSizes(BaseModel):
    default: MaxMin[MealSizeEnum]
    lunch: MaxMin[MealSizeEnum]
    dinner: MaxMin[MealSizeEnum]


class Supplements(BaseModel):
    creatine: str


class MealFrequency(BaseModel):
    max: int
    min: int


class DietaryRestrictions(BaseModel):
    gluten: micronutrients_enum
    dairy: micronutrients_enum
    soy: micronutrients_enum
    nuts: micronutrients_enum
    shellfish: micronutrients_enum
    fish: micronutrients_enum
    eggs: micronutrients_enum
    meat: micronutrients_enum
    vegetarian: YesNoEnum
    pescetarian: YesNoEnum
    carnivore: YesNoEnum


class VarietyRepetition(BaseModel):
    total_recipes_per_week: int
    total_ingredients_per_week: int


class Breakdowns(BaseModel):
    macros_breakdown: breakdowns_enum
    micronutrients_breakdown: breakdowns_enum
    cost_gbp_breakdown: breakdowns_enum


class GenerationArguments(BaseModel):
    breakdowns: Breakdowns
    micronutrients_include: YesNoEnum


class MealPlanConstraints(BaseModel):
    model_config = ConfigDict(
        title="Meal Plan Constraints",
        json_schema_extra={
            "example": {
                "sex": "male",
                "DoB": "1990-01-01",
                "height": 180,
                "weight": 80
                # ... add more examples
            }
        }
    )

    calories: Calories = Field(description="Daily caloric intake limits")
    protein: Protein = Field(
        description="Daily protein intake limits in grams")
    lipids_pcnt_non_protein_calories: LipidsPcntNonProteinCalories = Field(
        description="Percentage of non-protein calories from lipids")
    micronutrients_rda: MicronutrientsRDA = Field(
        description="Recommended daily allowances for micronutrients")
    cooking_time_minutes: CookingTimeMinutes = Field(
        description="Acceptable cooking time range")
    budget_per_week_gbp: BudgetPerWeekGBP = Field(
        description="Weekly budget range in GBP")
    serving_sizes: ServingSizes = Field(
        description="Serving size preferences for different meals")
    left_overs: str = Field(description="Preferences for handling leftovers")
    supplements: Supplements
    meal_frequency: MealFrequency = Field(
        description="Number of meals per day")
    dietary_restrictions: DietaryRestrictions
    ingredients_to_avoid: List[str] = Field(
        default_factory=list, description="Ingredients to completely avoid")
    ingredients_to_avoid_in_recipes: List[str] = Field(
        default_factory=list, description="Ingredients to avoid in recipes but acceptable as traces")
    ingredients_to_include: List[str] = Field(
        default_factory=list, description="Ingredients that must be included")
    ingredients_I_love: List[str] = Field(
        default_factory=list, description="Preferred ingredients")
    cuisine_preferences: List[str] = Field(
        default_factory=list, description="Preferred cuisine types")
    variety_repitition: VarietyRepetition = Field(
        description="Constraints on recipe and ingredient variety")
    sex: Sex = Field(description="Biological sex for nutritional calculations")
    DoB: date = Field(description="Date of birth")
    height: int = Field(description="Height in centimeters", ge=0, le=300)
    weight: int = Field(description="Weight in kilograms", ge=0, le=500)
    daily_steps: int = Field(description="Average daily steps", ge=0)
    activity_level: ActivityLevel = Field(
        description="Physical activity level")
    goal: str = Field(description="Health or fitness goal")
    generation_arguments: GenerationArguments
