import json
from typing import Literal
import meal_planner_agent.meal_plan_constraints_pd as mpc


def _load_recipe_constraints():
    with open('meal_planner_agent/recipe_constraints.json', 'r') as f:
        recipe_constraints_raw = json.load(f)
        constraints = recipe_constraints_raw
        recipe_constraints = mpc.RecipeConstraints(**constraints)
        return recipe_constraints, constraints


def _load_day_constraints():
    recipe_constraints, recipe_constraints_raw = _load_recipe_constraints()
    with open('meal_planner_agent/day_constraints.json', 'r') as f:
        day_constraints_raw = json.load(f)
        constraints = {**recipe_constraints_raw, **day_constraints_raw}
        day_constraints = mpc.DayConstraints(**constraints)
        return day_constraints, constraints


def _load_meal_plan_constraints():
    day_constraints, day_constraints_raw = _load_day_constraints()
    with open('meal_planner_agent/recipe_constraints.json', 'r') as f:
        meal_constraints_raw = json.load(f)
        constraints = {**day_constraints_raw, **meal_constraints_raw}
        meal_plan_constraints = mpc.MealPlanConstraints(**constraints)
        return meal_plan_constraints, constraints


def load_recipe_constraints():
    return _load_recipe_constraints()[0]


def load_day_constraints():
    return _load_day_constraints()[0]


def load_meal_plan_constraints():
    return _load_meal_plan_constraints()[0]


def test_load_constraints(level: Literal['recipe', 'day', 'meal_plan']):

    if level == 'recipe':
        constraints = load_recipe_constraints()
        assert constraints.calories.max < 1000
    elif level == 'day':
        constraints = load_day_constraints()
        assert constraints.calories.min > 1000
    elif level == 'meal_plan':
        constraints = load_meal_plan_constraints()
        assert constraints.calories.min > 1000


if __name__ == "__main__":
    test_load_constraints('recipe')
    test_load_constraints('day')
    test_load_constraints('meal_plan')
