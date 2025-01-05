from pydantic import BaseModel
from typing import List
from dataclasses import dataclass


@dataclass
class MacroNutrients:
    protein: float
    carbs: float
    fats: float
    calories: float


@dataclass
class MicroNutrients:
    vitamin_a: float
    vitamin_c: float
    calcium: float
    iron: float
    # Add other micronutrients as needed


@dataclass
class Ingredient:
    name: str
    quantity: str
    macros: MacroNutrients


@dataclass
class Meal:
    name: str
    ingredients: List[Ingredient]
    instructions: str
    time: int  # in minutes
    cost: float  # in GBP
    servings: int
    macros: MacroNutrients
    micronutrients: MicroNutrients


@dataclass
class DayMeals:
    breakfast: Meal
    lunch: Meal
    lunch_desert: Meal
    snack: Meal
    dinner: Meal
    dinner_desert: Meal


@dataclass
class WeeklyMealPlan:
    day_1: DayMeals
    day_2: DayMeals
    day_3: DayMeals
    day_4: DayMeals
    day_5: DayMeals
    day_6: DayMeals
    day_7: DayMeals


# Example usage
mock_example_meal_plan = WeeklyMealPlan(
    day_1=DayMeals(
        breakfast=Meal(
            name="Oatmeal",
            ingredients=[
                Ingredient(name="Oats", quantity="50g", macros=MacroNutrients(
                    protein=5, carbs=30, fats=3, calories=150)),
                Ingredient(name="Milk", quantity="200ml", macros=MacroNutrients(
                    protein=7, carbs=10, fats=5, calories=100))
            ],
            instructions="Mix oats with milk and cook for 5 minutes.",
            time=10,
            cost=1.5,
            servings=1,
            macros=MacroNutrients(protein=12, carbs=40, fats=8, calories=250),
            micronutrients=MicroNutrients(
                vitamin_a=0.5, vitamin_c=0, calcium=300, iron=1.5)
        ),
        lunch=Meal(
            name="Salad",
            ingredients=[
                Ingredient(name="Lettuce", quantity="100g", macros=MacroNutrients(
                    protein=1, carbs=10, fats=0, calories=50))
            ],
            instructions="Mix lettuce with dressing.",
            time=5,
            cost=0.5,
            servings=1,
            macros=MacroNutrients(protein=1, carbs=10, fats=0, calories=50),
            micronutrients=MicroNutrients(
                vitamin_a=0, vitamin_c=0, calcium=0, iron=0)
        ),
        lunch_desert=Meal(
            name="Salad",
            ingredients=[
                Ingredient(name="Lettuce", quantity="100g", macros=MacroNutrients(
                    protein=1, carbs=10, fats=0, calories=50))
            ],
            instructions="Mix lettuce with dressing.",
            time=5,
            cost=0.5,
            servings=1,
            macros=MacroNutrients(protein=1, carbs=10, fats=0, calories=50),
            micronutrients=MicroNutrients(
                vitamin_a=0, vitamin_c=0, calcium=0, iron=0)
        ),
        snack=Meal(
            name="Salad",
            ingredients=[
                Ingredient(name="Lettuce", quantity="100g", macros=MacroNutrients(
                    protein=1, carbs=10, fats=0, calories=50))
            ],
            instructions="Mix lettuce with dressing.",
            time=5,
            cost=0.5,
            servings=1,
            macros=MacroNutrients(protein=1, carbs=10, fats=0, calories=50),
            micronutrients=MicroNutrients(
                vitamin_a=0, vitamin_c=0, calcium=0, iron=0)
        ),
        dinner=Meal(
            name="Salad",
            ingredients=[
                Ingredient(name="Lettuce", quantity="100g", macros=MacroNutrients(
                    protein=1, carbs=10, fats=0, calories=50))
            ],
            instructions="Mix lettuce with dressing.",
            time=5,
            cost=0.5,
            servings=1,
            macros=MacroNutrients(protein=1, carbs=10, fats=0, calories=50),
            micronutrients=MicroNutrients(
                vitamin_a=0, vitamin_c=0, calcium=0, iron=0)
        ),
        dinner_desert=Meal(
            name="Salad",
            ingredients=[
                Ingredient(name="Lettuce", quantity="100g", macros=MacroNutrients(
                    protein=1, carbs=10, fats=0, calories=50))
            ],
            instructions="Mix lettuce with dressing.",
            time=5,
            cost=0.5,
            servings=1,
            macros=MacroNutrients(protein=1, carbs=10, fats=0, calories=50),
            micronutrients=MicroNutrients(
                vitamin_a=0, vitamin_c=0, calcium=0, iron=0)
        ),
    ),
    day_2=DayMeals(breakfast=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   snack=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0))),
    day_3=DayMeals(breakfast=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   snack=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0))),
    # Copy the same empty structure for days 4-7
    day_4=DayMeals(breakfast=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   snack=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0))),
    day_5=DayMeals(breakfast=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   snack=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0))),
    day_6=DayMeals(breakfast=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   snack=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0))),
    day_7=DayMeals(breakfast=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   lunch_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   snack=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(
                       protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)),
                   dinner_desert=Meal(name="", ingredients=[], instructions="", time=0, cost=0, servings=0, macros=MacroNutrients(protein=0, carbs=0, fats=0, calories=0), micronutrients=MicroNutrients(vitamin_a=0, vitamin_c=0, calcium=0, iron=0)))
)
