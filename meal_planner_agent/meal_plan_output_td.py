from typing import TypedDict, List, Dict


class MacroNutrients(TypedDict):
    protein: float
    carbs: float
    fats: float
    calories: float


class MicroNutrients(TypedDict):
    vitamin_a: float
    vitamin_c: float
    calcium: float
    iron: float
    # Add other micronutrients as needed


class Ingredient(TypedDict):
    name: str
    quantity: str
    macros: MacroNutrients


class Meal(TypedDict):
    name: str
    ingredients: List[Ingredient]
    instructions: str
    time: int  # in minutes
    cost: float  # in GBP
    servings: int
    macros: MacroNutrients
    micronutrients: MicroNutrients


class DayMeals(TypedDict):
    breakfast: Meal
    lunch: Meal
    lunch_desert: Meal
    snack: Meal
    dinner: Meal
    dinner_desert: Meal


class WeeklyMealPlan(TypedDict):
    day_1: DayMeals
    day_2: DayMeals
    day_3: DayMeals
    day_4: DayMeals
    day_5: DayMeals
    day_6: DayMeals
    day_7: DayMeals


# Example usage
example_meal_plan: WeeklyMealPlan = {
    "day_1": {
        "breakfast": {
            "name": "Oatmeal",
            "ingredients": [
                {"name": "Oats", "quantity": "50g", "macros": {
                    "protein": 5, "carbs": 30, "fats": 3, "calories": 150}},
                {"name": "Milk", "quantity": "200ml", "macros": {
                    "protein": 7, "carbs": 10, "fats": 5, "calories": 100}}
            ],
            "instructions": "Mix oats with milk and cook for 5 minutes.",
            "time": 10,
            "cost": 1.5,
            "servings": 1,
            "macros": {"protein": 12, "carbs": 40, "fats": 8, "calories": 250},
            "micronutrients": {"vitamin_a": 0.5, "vitamin_c": 0, "calcium": 300, "iron": 1.5}
        },
        "lunch": {
            # Define lunch meal
            ...
        },
        "lunch_desert": {
            # Define lunch desert meal
            ...
        },
        "snack": {
            # Define snack meal
            ...
        },
        "dinner": {
            # Define dinner meal
            ...
        },
        "dinner_desert": {
            # Define dinner desert meal
            ...
        }
    },
    # Define other days similarly
}
