from typing import TypedDict, List, Literal, Union, Generic, TypeVar

breakdowns_enum = Literal['daily', 'per_meal', 'per_ingredient']
micronutrients_enum = Literal["none", "low", "normal"]
YesNoEnum = Literal["no", "yes"]
MealSizeEnum = Literal["none", "small", "medium", "large"]

T = TypeVar('T')


class Calories(TypedDict):
    max: int
    min: int


class Protein(TypedDict):
    max: int
    min: int


class LipidsPcntNonProteinCalories(TypedDict):
    max: int
    min: int


class MicronutrientsRDA(TypedDict):
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


class CookingTimeMinutes(TypedDict):
    max: int
    min: int


class BudgetPerWeekGBP(TypedDict):
    max: int
    min: int


class ServingSizes(TypedDict):
    default: 'MaxMin[MealSizeEnum]'
    lunch: 'MaxMin[MealSizeEnum]'
    dinner: 'MaxMin[MealSizeEnum]'


class MaxMin(TypedDict, Generic[T]):
    max: T
    min: T


class Supplements(TypedDict):
    creatine: str


class MealFrequency(TypedDict):
    max: int
    min: int


class DietaryRestrictions(TypedDict):
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


class VarietyRepetition(TypedDict):
    total_recipes_per_week: int
    total_ingredients_per_week: int


class GenerationArguments(TypedDict):
    breakdowns: 'Breakdowns'
    micronutrients_include: YesNoEnum


class Breakdowns(TypedDict):
    macros_breakdown: breakdowns_enum
    micronutrients_breakdown: breakdowns_enum
    cost_gbp_breakdown: breakdowns_enum


class MealPlanConstraints(TypedDict):
    calories: Calories
    protein: Protein
    lipids_pcnt_non_protein_calories: LipidsPcntNonProteinCalories
    micronutrients_rda: MicronutrientsRDA
    cooking_time_minutes: CookingTimeMinutes
    budget_per_week_gbp: BudgetPerWeekGBP
    serving_sizes: ServingSizes
    left_overs: str
    supplements: Supplements
    meal_frequency: MealFrequency
    dietary_restrictions: DietaryRestrictions
    ingredients_to_avoid: List[str]
    ingredients_to_avoid_in_recipes: List[str]
    ingredients_to_include: List[str]
    ingredients_I_love: List[str]
    cuisine_preferences: List[str]
    variety_repitition: VarietyRepetition
    sex: str
    DoB: str
    height: int
    weight: int
    daily_steps: int
    activity_level: str
    goal: str
    generation_arguments: GenerationArguments
