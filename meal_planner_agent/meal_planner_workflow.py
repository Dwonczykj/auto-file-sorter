# https://pub.towardsai.net/pydantic-ai-web-scraper-llama-3-3-python-powerful-ai-research-agent-6d634a9565fe
# Pydantic AI Web Scraper with Ollama and Streamlit AI Chatbot

import re
from meal_planner_agent.meal_planner_workflow_types import ProductMinerals, ProductVitamins, SupermarketProduct, Meal, NutritionalInfo
import random
import sqlite3
from urllib.parse import urlparse
import warnings
from meal_planner_agent.load_constraints import load_meal_plan_constraints, load_recipe_constraints, load_day_constraints
from meal_planner_agent.recipe_search_types import ExtractRecipeDataSchemaPropertiesWithSourcePydantic
from recipe_search import RecipeSearch, ValidateAndAdaptRecipeResult, ExtractRecipeDataSchemaPropertiesPydantic
import json
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings, UsageLimits

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import langchain.document_loaders as lcdl
import langchain.document_transformers as lcdt
import langchain.text_splitter as lcts
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.chains import create_extraction_chain
from auto_file_sorter.logging.logging_config import configure_logging
from meal_planner_agent.meal_plan_constraints_pd import DietaryRestrictions, MealPlanConstraints, MealSizeEnum
import meal_planner_agent.meal_plan_constraints_pd as mpc
from meal_planner_agent.meal_plan_output_pd import DayMeals, WeeklyMealPlan, mock_example_meal_plan
import streamlit as st
from openai import AsyncOpenAI
from typing_extensions import TypedDict
# from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
import logging
import os
import asyncio
import datetime
from typing import Any, Dict, Literal, Union, Optional
from dataclasses import dataclass
from langgraph.graph import START
from meal_planner_agent.prompt_templates import generate_meal_plan
from tqdm import tqdm

import nest_asyncio
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
configure_logging()

load_dotenv(dotenv_path='.env')


def get_pydantic_llm(use_ollama: bool):
    if use_ollama:
        #  Ollama now has built-in compatibility with the OpenAI Chat Completions API, making it possible to use more tooling and applications with Ollama locally.
        client = AsyncOpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required but not used,
            # api_key=os.getenv("OPENAI_API_KEY"),
        )

        # smaller model at 3.8GB
        model = OpenAIModel('llama2', openai_client=client)
        # model = OpenAIModel('llama3.3:latest', openai_client=client) # llama3.3:latest is 42GB too big
    else:
        model = OpenAIModel('gpt-4o', api_key=os.getenv("OPENAI_API_KEY"))
    return model


def get_langchain_llm(use_ollama: bool):
    if use_ollama:
        #  Ollama now has built-in compatibility with the OpenAI Chat Completions API, making it possible to use more tooling and applications with Ollama locally.
        client = AsyncOpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',  # required but not used,
            # api_key=os.getenv("OPENAI_API_KEY"),
        )

        # smaller model at 3.8GB
        model = ChatOpenAI(model='llama2', async_client=client)
        # model = OpenAIModel('llama3.3:latest', openai_client=client) # llama3.3:latest is 42GB too big
    else:
        model = ChatOpenAI(model='gpt-4o')
    return model


def get_tavily_client():
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    if not TAVILY_API_KEY:
        raise ValueError("Please set TAVILY_API_KEY environment variable.")

    tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
    return tavily_client


@dataclass
class SearchDataclassForDI:
    max_results: int
    todays_date: str
    constraints: str


@dataclass
class ResearchDependencies:
    todays_date: str


class ResearchResult(BaseModel):
    research_title: str = Field(
        description='Markdown heading describing the article topic, prefixed with #')
    research_main: str = Field(
        description='A main section that provides a detailed news article')
    research_bullets: str = Field(
        description='A set of bullet points summarizing key points')


class OverallMealPlannerState(BaseModel):
    meal_plan: WeeklyMealPlan = Field(
        description='A detailed meal plan for the week')
    constraints: MealPlanConstraints = Field(
        description='The constraints for the meal plan in JSON format')
    calculations: list[str] = Field(
        description='A list of calculations performed to generate the meal plan')


llm_gpt_4o = get_langchain_llm(use_ollama=False)
# llm_with_tools = llm_gpt_4o.bind_tools([...])

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with generating a meal plan " +
                        "for the week based on user provided constraints. " +
                        "When creating the meal plan, you may use the provided search tool to find recipes and ingredients and check nutritional information. " +
                        "Then combine the results to form the input for calculations to validate the meal plan."
                        )
prompt = PromptTemplate(
    input_variables=generate_meal_plan[0],
    template=generate_meal_plan[1],
)


# Create the agent
pydantic_llm = get_pydantic_llm(use_ollama=False)
search_agent = Agent(
    pydantic_llm,
    # passsed in the do_search function which calls agent.run() and passes the deps.
    deps_type=SearchDataclassForDI,
    result_type=WeeklyMealPlan,
    system_prompt=f"{sys_msg.content}",
    # system_prompt='You are a helpful research assistant, you are an expert in research. '
    #               'When given a query, you will identify strong keywords to do 3-5 searches using the provided search tool. '
    #               'Then combine results into a detailed response.'
)

# Next we define the tools below.

# @search_agent.system_prompt
# async def add_current_date(ctx: RunContext[SearchDataclass]) -> str:
#     todays_date = ctx.deps.todays_date
#     system_prompt = (
#         f"You're a helpful research assistant and an expert in research. "
#         f"When given a question, write strong keywords to do 3-5 searches in total "
#         f"(each with a query_number) and then combine the results. "
#         f"If you need today's date it is {todays_date}. "
#         f"Focus on providing accurate and current information."
#     )
#     system_prompt = (
#         f"{sys_msg.content} "
#         f"If you need today's date it is {todays_date}. "
#         f"Focus on providing accurate and current information."
#     )
#     return system_prompt

tavily_client = get_tavily_client()


@search_agent.tool
async def get_search(search_data: RunContext[SearchDataclassForDI], query: str, query_number: int) -> dict[str, Any]:
    """Perform a search using the Tavily client."""
    max_results = search_data.deps.max_results
    results = await tavily_client.get_search_context(query=query, max_results=max_results)
    logging.info(f"Search results [{type(results)}]: {results}")
    # Convert results to dict if not already
    if not isinstance(results, dict):
        return {"results": str(results)}
    return results

# a main function

# could convert this into a node in our graph using the inptu to search as the input to the graph
# would just need to decide how we store results if even in the graph for use by the assistant node.


# async def do_search(constraints: str, max_results: int):
#     # Prepare dependencies
#     current_date = datetime.date.today()
#     date_string = current_date.strftime("%Y-%m-%d")
#     deps = SearchDataclassForDI(
#         todays_date=date_string, constraints=constraints, max_results=max_results)
#     query = prompt.format(CONSTRAINTS_JSON=constraints)
#     # Inject dependencies into the agent
#     logging.info(f"Query: {query}")
#     logging.info(f"Deps: {deps}")

#     result = await search_agent.run(
#         query,
#         deps=deps,
#         usage_limits=UsageLimits(request_limit=10, total_tokens_limit=10000),
#         model_settings=ModelSettings(max_tokens=10000))
#     logging.info(f"Result: {result}")
#     return result

# Langgraph Section
# Node
# def assistant(state: OverallMealPlannerState):
#     return {"messages": [llm_gpt_4o.invoke([sys_msg] + state["messages"])]}


# meal_planner_workflow = StateGraph(OverallMealPlannerState)
# meal_planner_workflow.add_node("assistant", assistant)

# meal_planner_workflow.add_edge(START, "assistant")
# End of Langgraph Section

# # streamlit app
# def main3():
#     st.title("Pydantic AI Web Scraper")
#     query = st.text_input("Enter your query")
#     if st.button("Search"):
#         result = asyncio.run(do_search(query, max_results=5))
#         st.write(result)


def main2():
    st.set_page_config(page_title="AI News Researcher", layout="centered")

    st.title("Large Language Model News Researcher")
    st.write(
        "Stay updated on the latest trends and developments in Large Language Model.")

    # User input section
    st.sidebar.title("Search Parameters")
    query = st.sidebar.text_input(
        "Enter your query:", value="latest Large Language Model news")
    max_results = st.sidebar.slider(
        "Number of search results:", min_value=3, max_value=10, value=5)

    st.write("Use the sidebar to adjust search parameters.")

    if st.button("Get Latest Large Language Model News"):
        with st.spinner("Researching, please wait..."):
            result_data = asyncio.run(do_search(query, max_results))

        if not result_data.data:
            st.error("Failed to get research results")
            return

        st.markdown(result_data.data.research_title)
        # A bit of styling for the main article
        st.markdown(
            f"<div style='line-height:1.6;'>{result_data.data.research_main}</div>", unsafe_allow_html=True)

        st.markdown("### Key Takeaways")
        st.markdown(result_data.data.research_bullets)


def main_manual_no_streamlit():
    meal_plan_constraints = load_meal_plan_constraints()
    recipe_constraints = load_recipe_constraints()
    day_constraints = load_day_constraints()

    workflow = MealPlannerWorkflow()
    weekly_meal_plan = asyncio.run(
        workflow.generate_meal_plan(meal_plan_constraints, recipe_constraints, day_constraints))

    # result = asyncio.run(do_search(constraints, max_results=2))
    # logging.info(result)
    # if result and result.data:
    #     weekly_meal_plan = result.data
    # else:
    #     # Mock result for demonstration purposes
    #     weekly_meal_plan = mock_example_meal_plan

    #     st.error("Failed to get meal plan")
    return weekly_meal_plan


def main_manual():
    st.set_page_config(page_title="AI Meal Planner", layout="centered")

    st.title("AI Meal Planner Manual")

    with open('meal_planner_agent/meal_plan_constraints.json', 'r') as f:
        constraints = json.load(f)
    result = asyncio.run(do_search(constraints, max_results=2))
    logging.info(result)
    if result and result.data:
        weekly_meal_plan = result.data
    else:
        # Mock result for demonstration purposes
        weekly_meal_plan = mock_example_meal_plan

        st.error("Failed to get meal plan")
        return

    # Display the meal plan using tabs and expanders
    days = list(weekly_meal_plan.__dict__.keys())
    tabs = st.tabs(days)

    for i, day in enumerate(days):
        with tabs[i]:
            st.header(f"Meal Plan for {day.replace('_', ' ').title()}")
            meals: DayMeals = weekly_meal_plan.__dict__[day]
            for meal_name, meal_details in meals.__dict__.items():
                with st.expander(meal_name.replace('_', ' ').title()):
                    st.subheader(meal_details['name'])
                    st.write(f"Instructions: {
                        meal_details['instructions']}")
                    st.write(f"Time: {meal_details['time']} minutes")
                    st.write(f"Cost: £{meal_details['cost']}")
                    st.write(f"Servings: {meal_details['servings']}")
                    st.write("Macros:")
                    st.json(meal_details['macros'])
                    st.write("Micronutrients:")
                    st.json(meal_details['micronutrients'])
                    st.write("Ingredients:")
                    for ingredient in meal_details['ingredients']:
                        st.write(
                            f"- {ingredient['name']}: {ingredient['quantity']}")
                        st.json(ingredient['macros'])


def main():
    st.set_page_config(page_title="AI Meal Planner", layout="centered")

    st.title("AI Meal Planner")

    # User input section for MealPlanConstraints
    st.sidebar.title("Meal Plan Constraints")

    # Calories
    calories_min = st.sidebar.number_input(
        "Min Calories", min_value=0, value=1200)
    calories_max = st.sidebar.number_input(
        "Max Calories", min_value=0, value=1890)

    # Protein
    protein_min = st.sidebar.number_input(
        "Min Protein", min_value=0, value=170)
    protein_max = st.sidebar.number_input(
        "Max Protein", min_value=0, value=210)

    # Lipids percentage of non-protein calories
    lipids_min = st.sidebar.number_input("Min Lipids %", min_value=0, value=10)
    lipids_max = st.sidebar.number_input("Max Lipids %", min_value=0, value=30)

    # Cooking time
    cooking_time_min = st.sidebar.number_input(
        "Min Cooking Time (minutes)", min_value=0, value=1)
    cooking_time_max = st.sidebar.number_input(
        "Max Cooking Time (minutes)", min_value=0, value=20)

    # Budget
    budget_min = st.sidebar.number_input(
        "Min Budget per Week (GBP)", min_value=0, value=30)
    budget_max = st.sidebar.number_input(
        "Max Budget per Week (GBP)", min_value=0, value=100)

    # Serving sizes
    serving_size_default: MealSizeEnum = st.sidebar.selectbox(
        "Default Serving Size", ["none", "small", "medium", "large"])
    lunch_serving_size_min: MealSizeEnum = st.sidebar.selectbox(
        "Min Lunch Serving Size", ["none", "small", "medium", "large"])
    lunch_serving_size_max: MealSizeEnum = st.sidebar.selectbox(
        "Max Lunch Serving Size", ["none", "small", "medium", "large"])
    dinner_serving_size_min: MealSizeEnum = st.sidebar.selectbox(
        "Min Dinner Serving Size", ["none", "small", "medium", "large"])
    dinner_serving_size_max: MealSizeEnum = st.sidebar.selectbox(
        "Max Dinner Serving Size", ["none", "small", "medium", "large"])

    # Leftovers
    left_overs = st.sidebar.selectbox("Leftovers", ["yes", "no"])

    # Supplements
    creatine = st.sidebar.text_input("Creatine Supplement", value="5g")

    # Meal frequency
    meal_frequency_min = st.sidebar.number_input(
        "Min Meal Frequency", min_value=0, value=2)
    meal_frequency_max = st.sidebar.number_input(
        "Max Meal Frequency", min_value=0, value=3)

    # Dietary restrictions
    dietary_restrictions: DietaryRestrictions = {
        "gluten": st.sidebar.selectbox("Gluten", ["none", "low", "normal", "no", "yes"]),
        "dairy": st.sidebar.selectbox("Dairy", ["none", "low", "normal", "no", "yes"]),
        "soy": st.sidebar.selectbox("Soy", ["none", "low", "normal", "no", "yes"]),
        "nuts": st.sidebar.selectbox("Nuts", ["none", "low", "normal", "no", "yes"]),
        "shellfish": st.sidebar.selectbox("Shellfish", ["none", "low", "normal", "no", "yes"]),
        "fish": st.sidebar.selectbox("Fish", ["none", "low", "normal", "no", "yes"]),
        "eggs": st.sidebar.selectbox("Eggs", ["none", "low", "normal", "no", "yes"]),
        "meat": st.sidebar.selectbox("Meat", ["none", "low", "normal", "no", "yes"]),
        "vegetarian": st.sidebar.selectbox("Vegetarian", ["none", "low", "normal", "no", "yes"]),
        "vegan": st.sidebar.selectbox("Vegan", ["none", "low", "normal", "no", "yes"]),
        "pescetarian": st.sidebar.selectbox("Pescetarian", ["none", "low", "normal", "no", "yes"]),
        "carnivore": st.sidebar.selectbox("Carnivore", ["none", "low", "normal", "no", "yes"]),
    }

    # Ingredients to avoid
    ingredients_to_avoid = st.sidebar.text_area(
        "Ingredients to Avoid", value="salmon, walnuts, rice, pizza").split(',')

    # Ingredients to include
    ingredients_to_include = st.sidebar.text_area(
        "Ingredients to Include", value="goats cheese, quorn, pumpkin seeds").split(',')

    # Ingredients I love
    ingredients_I_love = st.sidebar.text_area(
        "Ingredients I Love", value="tenderstem broccoli, asparagus, mushrooms, peas").split(',')

    # Cuisine preferences
    cuisine_preferences = st.sidebar.multiselect("Cuisine Preferences", [
                                                 "healthy", "Mediterranean", "Mexican", "Greek", "Asian", "Thai", "Lebanese"])

    # Variety repetition
    total_recipes_per_week = st.sidebar.number_input(
        "Total Recipes per Week", min_value=0, value=5)
    total_ingredients_per_week = st.sidebar.number_input(
        "Total Ingredients per Week", min_value=0, value=30)

    # Personal details
    sex = st.sidebar.selectbox("Sex", ["male", "female", "other"])

    # Set a reasonable default date (e.g., 30 years ago)
    default_date = datetime.date.today() - datetime.timedelta(days=365*30)
    dob = st.sidebar.date_input("Date of Birth", value=default_date)

    # Validate date input
    if dob and dob >= datetime.date.today():
        st.sidebar.error("Date of birth cannot be in the future!")
        dob = default_date

    height = st.sidebar.number_input("Height (cm)", min_value=0, value=181)
    weight = st.sidebar.number_input("Weight (kg)", min_value=0, value=89)
    daily_steps = st.sidebar.number_input(
        "Daily Steps", min_value=0, value=12000)
    activity_level = st.sidebar.selectbox(
        "Activity Level", ["sedentary", "light", "moderate", "active", "very active"])
    goal = st.sidebar.selectbox(
        "Goal", ["lose_weight", "maintain_weight", "gain_weight"])

    # Generation arguments
    breakdowns = {
        "macros_breakdown": st.sidebar.selectbox("Macros Breakdown", ["daily", "per_meal", "per_ingredient"]),
        "micronutrients_breakdown": st.sidebar.selectbox("Micronutrients Breakdown", ["daily", "per_meal", "per_ingredient"]),
        "cost_gbp_breakdown": st.sidebar.selectbox("Cost GBP Breakdown", ["daily", "per_meal", "per_ingredient"]),
    }
    micronutrients_include = st.sidebar.selectbox(
        "Include Micronutrients", ["yes", "no"])

    # Button to run the search agent
    if st.button("Generate Meal Plan"):

        # Create the MealPlanConstraints instance
        meal_plan_constraints = mpc.MealPlanConstraints.default()

        # Convert constraints to JSON for further processing
        constraints_json = meal_plan_constraints.model_dump_json(indent=2)
        st.json(constraints_json)  # Display the constraints for debugging
        # result = asyncio.run(do_search(constraints_json, max_results=2))
        # st.success("Meal plan generated successfully!")
        # if result and result.data:
        #     weekly_meal_plan = result.data
        # else:
        #     # Mock result for demonstration purposes
        #     weekly_meal_plan = mock_example_meal_plan

        #     st.error("Failed to get meal plan")

        # # Display the meal plan using tabs and expanders
        # days = list(weekly_meal_plan.__dict__.keys())
        # tabs = st.tabs(days)

        workflow = MealPlannerWorkflow()
        weekly_meal_plan = asyncio.run(
            workflow.generate_meal_plan(
                meal_plan_constraints=meal_plan_constraints,
                recipe_constraints=mpc.RecipeConstraints.default(),
                day_constraints=mpc.DayConstraints.default(),
            )
        )

        # Rest of your display code remains the same
        if weekly_meal_plan:
            # Display the meal plan using tabs and expanders
            days = list(weekly_meal_plan.__dict__.keys())
            tabs = st.tabs(days)

            for i, day in enumerate(days):
                with tabs[i]:
                    st.header(f"Meal Plan for {day.replace('_', ' ').title()}")
                    meals: DayMeals = weekly_meal_plan.__dict__[day]
                    for meal_name, meal_details in meals.__dict__.items():
                        with st.expander(meal_name.replace('_', ' ').title()):
                            st.subheader(meal_details['name'])
                            st.write(f"Instructions: {
                                     meal_details['instructions']}")
                            st.write(f"Time: {meal_details['time']} minutes")
                            st.write(f"Cost: £{meal_details['cost']}")
                            st.write(f"Servings: {meal_details['servings']}")
                            st.write("Macros:")
                            st.json(meal_details['macros'])
                            st.write("Micronutrients:")
                            st.json(meal_details['micronutrients'])
                            st.write("Ingredients:")
                            for ingredient in meal_details['ingredients']:
                                st.write(
                                    f"- {ingredient['name']}: {ingredient['quantity']}")
                                st.json(ingredient['macros'])


class MealPlannerWorkflow:
    def __init__(self):
        self.recipe_searcher = RecipeSearch()

    async def find_suitable_recipes(self, constraints: mpc.ConstraintsType):
        """Find recipes that match the given constraints."""

        recipes = await self.recipe_searcher.search_recipes(
            min_calories=constraints.calories.min,
            max_calories=constraints.calories.max,
            constraints=constraints
        )

        # Filter recipes based on other constraints
        filtered_recipes: list[ValidateAndAdaptRecipeResult] = []
        for recipe in recipes:
            # Skip recipes with avoided ingredients
            if any(ingredient in [ing.lower() for ing in recipe.ingredients]
                   for ingredient in constraints.ingredients_to_avoid):
                continue

            # Check cooking time if available in recipe content
            if recipe.cooking_time_minutes:
                if constraints.cooking_time_minutes:
                    max_time = constraints.cooking_time_minutes.max
                    if max_time and recipe.cooking_time_minutes > max_time:
                        continue

            if recipe.dietary_info.contains_meat and constraints.dietary_restrictions.meat == 'no':
                continue
            if recipe.dietary_info.contains_dairy and constraints.dietary_restrictions.dairy == 'no':
                continue
            if recipe.dietary_info.contains_soy and constraints.dietary_restrictions.soy == 'no':
                continue
            if recipe.dietary_info.contains_nuts and constraints.dietary_restrictions.nuts == 'no':
                continue
            if recipe.dietary_info.contains_shellfish and constraints.dietary_restrictions.shellfish == 'no':
                continue
            if recipe.dietary_info.contains_fish and constraints.dietary_restrictions.fish == 'no':
                continue
            if recipe.dietary_info.contains_eggs and constraints.dietary_restrictions.eggs == 'no':
                continue

            filtered_recipes.append(recipe)

        return filtered_recipes

    async def generate_meal_plan(
            self,
            meal_plan_constraints: mpc.MealPlanConstraints,
            recipe_constraints: mpc.RecipeConstraints,
            day_constraints: mpc.DayConstraints
    ) -> WeeklyMealPlan:
        """Generate a complete meal plan using recipe search and constraints."""
        # Get suitable recipes
        recipes = await self.find_suitable_recipes(recipe_constraints)
        day_plans: list[DayMeals] = []

        day_plan = await self.build_day_plan(day_constraints, recipes, day_plans)
        day_plans.append(day_plan)
        if not self.check_day_plan_against_constraints(day_plan, day_constraints):
            raise ValueError(
                "Day plan does not satisfy day constraints, needs to be run again")

        week_plan = self.build_week_plan(
            meal_plan_constraints, day_constraints, recipes)
        if not self.check_week_plan_against_constraints(week_plan, meal_plan_constraints):
            raise ValueError(
                "Week plan does not satisfy week constraints, needs to be run again")

        self.save_week_plan(week_plan)
        self.export_week_plan(week_plan)
        return week_plan

    async def build_day_plan(self, day_constraints: mpc.DayConstraints, recipes: list[ValidateAndAdaptRecipeResult], existing_day_plans: list[DayMeals]) -> DayMeals:
        """Build a day's meal plan that satisfies the given constraints."""
        # Initialize meals for the day
        day_plan = DayMeals()
        # create shallow copies of the lists so we can modify the list not the underlying items
        recipes = [*recipes]
        existing_day_plans = [*existing_day_plans]
        # Calculate target calories per meal based on daily total
        daily_calories_max = day_constraints.calories.max
        remaining_calories = daily_calories_max
        remaining_calories_min = remaining_calories - (
            day_constraints.calories.max - day_constraints.calories.min)
        order_of_exec = [
            'dinner',
            'lunch',
            'dinner_desert',
            'lunch_desert',
            'breakfast',
            'snacks'
        ]
        if day_constraints.meal_frequency.max == 2:
            order_of_exec.remove('breakfast')
            order_of_exec.remove('snacks')
        elif day_constraints.meal_frequency.max == 1:
            order_of_exec.remove('breakfast')
            order_of_exec.remove('lunch')
            order_of_exec.remove('lunch_desert')
            order_of_exec.remove('snacks')
        elif day_constraints.meal_frequency.max == 0:
            raise ValueError("Day constraints require at least one meal")

        serving_size_proportion = {
            'breakfast': (1, 1),
            'lunch': (1, 1),
            'dinner': (1, 1),
            'snacks': (1, 1),
            'lunch_desert': (1, 1),
            'dinner_desert': (1, 1),
        }
        for k in serving_size_proportion.keys():
            if k not in order_of_exec:
                serving_size_proportion[k] = (0, 0)
                continue
            if hasattr(day_constraints.serving_sizes, k):
                multi: mpc.MaxMin[MealSizeEnum] = getattr(
                    day_constraints.serving_sizes, k)
                if multi.max == 'small':
                    serving_size_proportion[k] = (
                        serving_size_proportion[k][0], 1)
                elif multi.max == 'medium':
                    serving_size_proportion[k] = (
                        serving_size_proportion[k][0], 2)
                elif multi.max == 'large':
                    serving_size_proportion[k] = (
                        serving_size_proportion[k][0], 3)
                if multi.min == 'small':
                    serving_size_proportion[k] = (
                        1, serving_size_proportion[k][1])
                elif multi.min == 'medium':
                    serving_size_proportion[k] = (
                        2, serving_size_proportion[k][1])
                elif multi.min == 'large':
                    serving_size_proportion[k] = (
                        3, serving_size_proportion[k][1])

        proportions_units = {
            k: serving_size_proportion[k][1] for k in order_of_exec}
        proportions_sum = sum(proportions_units.values())
        proportions = {k: p/proportions_sum for k,
                       p in proportions_units.items()}

        # We need a certain amount of shuffling randomisation to ensure recipes are not sorted and we get different meal plans each time to keep them fresh.
        random.shuffle(recipes)

        recipes_used_counter = {r.url: 0 for r in recipes}

        async def assign_recipes_to_meals(proportions: dict[str, float], call_depth: int):
            for k in order_of_exec:
                remaining_calories = daily_calories_max - sum(
                    r.calories_per_serving
                    for meal in [
                        day_plan.breakfast,
                        day_plan.lunch,
                        day_plan.dinner,
                        day_plan.snacks,
                        day_plan.lunch_desert,
                        day_plan.dinner_desert
                    ]
                    for r in meal
                )
                remaining_calories_min = remaining_calories - (
                    day_constraints.calories.max - day_constraints.calories.min)
                if 'dessert' in k:
                    warnings.warn(
                        f"Dessert meals for [{k}] have not yet been hard implemented dessert recipes to my liking like berry bowls")
                    suitable_recipes = [
                        r for r in recipes if r.suitable_for_meal == 'dessert']
                elif 'snacks' in k:
                    suitable_recipes = [
                        r for r in recipes if r.suitable_for_meal == 'snacks']
                elif 'breakfast' == k:
                    suitable_recipes = [
                        r for r in recipes if r.suitable_for_meal == 'breakfast']
                else:
                    suitable_recipes = [r for r in recipes if r.suitable_for_meal not in [
                        'dessert', 'snacks', 'breakfast']]
                calories_range = (
                    proportions[k]*day_constraints.calories.min, proportions[k]*day_constraints.calories.max)
                recipes_by_calories = [r for r in suitable_recipes if r.calories_per_serving >=
                                       calories_range[0] and r.calories_per_serving <= calories_range[1]]
                if remaining_calories < 0 or not recipes_by_calories:
                    ind = order_of_exec.index(k)
                    for i, _k in enumerate(order_of_exec):
                        if i < ind:
                            proportions_units[_k] = proportions_units[_k] - 1
                            proportions_sum = sum(proportions_units.values())
                            proportions = {k: p/proportions_sum for k,
                                           p in proportions_units.items()}
                    await assign_recipes_to_meals(
                        proportions, call_depth=call_depth+1)
                    raise ValueError(
                        f"No suitable recipes found for {k} with calories range {calories_range} from {len(suitable_recipes)} recipes available for {k}.\n May have gone too heavy on earlier meals for this day, algorithm should try to fix itself and rerun the days by reducing previous meals calories multipliers.")
                elif remaining_calories_min < 0:
                    warnings.warn(
                        f"Remaining calories for {k} are less than 0, may have gone too light on earlier meals for this day, algorithm should try to fix itself and rerun the days by increasing previous meals calories multipliers.")
                # choose and use a recipe for today, do not use the same recipe twice in one day unless when looking at meal_plan_constraints we have a variety set to # recipes <= 7
                chosen_recipe = random.choice(recipes_by_calories)
                meals: list[Meal] = getattr(day_plan, k)
                meal = await self.convert_recipe_to_meal(chosen_recipe)
                meals.append(meal)
                recipes_used_counter[chosen_recipe.url] += 1
                if k in ['breakfast', 'lunch', 'dinner']:
                    recipes.remove(chosen_recipe)

        await assign_recipes_to_meals(proportions, call_depth=0)

        return day_plan

    def build_week_plan(self,
                        week_constraints: mpc.MealPlanConstraints,
                        day_constraints: mpc.DayConstraints,
                        recipes: list[ValidateAndAdaptRecipeResult]) -> WeeklyMealPlan:
        """Build a week's meal plan from daily plans."""
        # depending on variety constraints, we should calculate the number of different days to build and then assign them out between the 7 days. i.e. if variety of 3 differnet day plans, then we have dayA, dayB, dayC, dayA, dayB, dayC, dayA
        recipes_spans_days = week_constraints.variety_repitition.total_recipes_per_week / \
            week_constraints.meal_frequency.max
        num_days_to_vary: int = int(7 // recipes_spans_days)
        for i in range(num_days_to_vary):
            day_plan = self.build_day_plan(
                day_constraints, recipes, existing_day_plans=day_plans)
            day_plans.append(day_plan)
        week_plan = WeeklyMealPlan()
        for i, day in enumerate(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
            setattr(week_plan, day, day_plans[i % num_days_to_vary])

        return week_plan

    def check_day_plan_against_constraints(self, day_plan: DayMeals, day_constraints: mpc.ConstraintsType) -> bool:
        """Verify that a day plan meets all constraints."""
        # Calculate daily totals
        daily_calories = sum(
            recipe.calories_per_serving
            for meal in [day_plan.breakfast, day_plan.lunch, day_plan.dinner, day_plan.snacks]
            for recipe in meal
        )

        daily_protein = sum(
            recipe.macros.protein
            for meal in [day_plan.breakfast, day_plan.lunch, day_plan.dinner, day_plan.snacks]
            for recipe in meal
        )

        # Check calorie constraints
        if not (day_constraints.calories.min <= daily_calories <= day_constraints.calories.max):
            return False

        # Check protein constraints
        if not (day_constraints.protein.min <= daily_protein <= day_constraints.protein.max):
            return False

        # Check meal frequency
        total_meals = sum(
            1 for meal in [day_plan.breakfast, day_plan.lunch, day_plan.dinner, day_plan.snacks]
            if meal
        )
        if not (day_constraints.meal_frequency.min <= total_meals <= day_constraints.meal_frequency.max):
            return False

        return True

    def check_week_plan_against_constraints(self, week_plan: WeeklyMealPlan, week_constraints: mpc.ConstraintsType) -> bool:
        """Verify that a week plan meets all constraints."""
        # Check variety constraints
        used_recipes = set()
        used_ingredients = set()

        for day_attr in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            day_plan = getattr(week_plan, day_attr)
            for meal in [day_plan.breakfast, day_plan.lunch, day_plan.dinner, day_plan.snacks]:
                for recipe in meal:
                    used_recipes.add(recipe.url)
                    used_ingredients.update(recipe.ingredients)

        # Check recipe variety
        if len(used_recipes) < week_constraints.variety_repitition.total_recipes_per_week:
            return False

        # Check ingredient variety
        if len(used_ingredients) < week_constraints.variety_repitition.total_ingredients_per_week:
            return False

        # Check weekly budget
        weekly_cost = sum(
            recipe.cost_gbp
            for day_attr in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            for meal in [getattr(week_plan, day_attr).breakfast,
                         getattr(week_plan, day_attr).lunch,
                         getattr(week_plan, day_attr).dinner,
                         getattr(week_plan, day_attr).snacks]
            for recipe in meal
        )

        if not (week_constraints.budget_per_week_gbp.min <= weekly_cost <= week_constraints.budget_per_week_gbp.max):
            return False

        return True

    def save_week_plan(self, week_plan: WeeklyMealPlan):
        """Save the week plan to the database."""
        # Convert week plan to JSON
        week_plan_json = week_plan.model_dump_json()

        # Get database connection
        conn = sqlite3.connect('recipe_database.sqlite')
        cursor = conn.cursor()

        try:
            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS week_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Insert the week plan
            cursor.execute(
                'INSERT INTO week_plans (plan_json) VALUES (?)',
                (week_plan_json,)
            )

            conn.commit()
            logging.info(f"Saved week plan to database with ID {
                         cursor.lastrowid}")

        finally:
            conn.close()

    def export_week_plan(self, week_plan: WeeklyMealPlan):
        """Export the week plan in a user-friendly format."""
        # Create a detailed text report
        report = ["Weekly Meal Plan\n" + "="*50 + "\n"]

        for day_attr in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            day_plan = getattr(week_plan, day_attr)
            report.append(f"\n{day_attr.title()}\n{'-'*20}")

            for meal_type in ['breakfast', 'lunch', 'dinner', 'snacks']:
                meals = getattr(day_plan, meal_type)
                if meals:
                    report.append(f"\n{meal_type.title()}:")
                    for recipe in meals:
                        report.append(f"- {recipe.title}")
                        report.append(
                            f"  Calories: {recipe.calories_per_serving}")
                        report.append(f"  Protein: {recipe.macros.protein}g")
                        report.append(f"  Cooking time: {
                                      recipe.cooking_time_minutes} minutes")
                        report.append(f"  Cost: £{recipe.cost_gbp:.2f}")

        # Save to file
        with open('meal_plan_export.txt', 'w') as f:
            f.write('\n'.join(report))

        # Log the report
        logging.info('\n'.join(report))

        return report

    async def add_my_own_recipe_from_url(self, recipe_url: str, constraints: Optional[mpc.ConstraintsType] = None) -> ValidateAndAdaptRecipeResult | None:
        """Add a recipe from a URL to the database."""
        # try selenium for authenticated url routes?
        if recipe_url.startswith('https://myfitnesspal.com'):
            loader = lcdl.SeleniumURLLoader(
                urls=[recipe_url], browser='chrome', headless=False)
            recipes = loader.load()
            raise Exception(
                f"SeleniumURLLoader not quite implemented for {recipe_url}")
        else:
            return await self.recipe_searcher.process_and_adapt_recipe_from_url(url=recipe_url, constraints=constraints)

    async def add_my_own_recipe_from_file(
        self,
        recipe_file: str,
        constraints: Optional[mpc.ConstraintsType] = None
    ) -> ValidateAndAdaptRecipeResult:
        """Add a recipe from a file to the database."""
        # Get the file extension and appropriate loader
        file_type = recipe_file.split('.')[-1].lower()
        loader = self._get_file_loader(file_type, recipe_file)

        # Load and process the document
        docs = loader.load()

        # Create extraction chain for recipe data
        # Use more capable model for extraction
        llm = get_langchain_llm(use_ollama=False)
        extraction_chain = create_extraction_chain(
            schema=self.recipe_searcher.schema_extract_recipe_data,
            llm=llm
        )

        # Extract recipe data from document
        combined_text = "\n".join(doc.page_content for doc in docs)
        extraction_result = await extraction_chain.ainvoke({"input": combined_text})

        if not extraction_result or not extraction_result.get('text'):
            raise ValueError("Could not extract recipe data from file")

        # Convert to Pydantic model
        recipe_data = ExtractRecipeDataSchemaPropertiesPydantic(
            **extraction_result['text'][0])

        # Validate and adapt recipe if needed
        if constraints:
            recipe = await self.recipe_searcher._validate_and_adapt_recipe(
                recipe_data, constraints)
            if not recipe:
                raise ValueError(
                    "Recipe does not meet constraints and cannot be adapted")
        else:
            recipe = ValidateAndAdaptRecipeResult(**recipe_data.model_dump())

        # Save to database
        self.recipe_searcher._save_valid_recipe(recipe)

        # Update FAISS index
        recipe_text = self.recipe_searcher._recipe_to_text(recipe)
        self.recipe_searcher.recipe_store.add_texts(
            [recipe_text],
            metadatas=[recipe.model_dump()]
        )

        return recipe

    def _get_file_loader(self, file_type: str, file_path: str) -> Any:
        """Get appropriate document loader for file type."""
        loaders = {
            'pdf': lcdl.PyPDFLoader,
            'docx': lcdl.Docx2txtLoader,
            'txt': lcdl.TextLoader,
            'md': lcdl.MarkdownLoader,
            'html': lcdl.BSHTMLLoader,
            'xlsx': lcdl.UnstructuredExcelLoader,
            'csv': lcdl.CSVLoader,
            'json': lambda p: lcdl.JSONLoader(p, jq_schema='.'),
            'xml': lcdl.UnstructuredXMLLoader
        }

        if file_type not in loaders:
            raise ValueError(f"Unsupported file type: {file_type}")

        return loaders[file_type](file_path)

    async def add_my_own_recipe_pre_formed(self, recipe: ExtractRecipeDataSchemaPropertiesPydantic) -> ValidateAndAdaptRecipeResult:
        """Add a recipe from a memory to the database."""
        recipe = self.find_recipe_from_memory(recipe)
        return recipe

    async def add_my_own_recipe_from_natural_language(
        self,
        recipe_string: str,
        image: Optional[bytes] = None
    ) -> Optional[ValidateAndAdaptRecipeResult]:
        """Add a recipe from natural language description with interactive follow-up questions."""

        llm = get_langchain_llm(use_ollama=False)
        max_attempts = 3
        attempts = 0

        # Initial extraction prompt
        structure_prompt = PromptTemplate.from_template("""
            You are a helpful assistant that extracts recipe information.
            Please analyze this recipe description and extract all relevant information.
            If information is missing, indicate what's missing so we can ask the user.
            
            Recipe description:
            {recipe_text}
            
            Required information:
            - Title
            - Ingredients with quantities
            - Step by step instructions
            - Cooking time
            - Servings
            - Nutritional information (calories, protein, etc.)
            - Cuisine type
            - Dietary restrictions/tags
        """)

        while attempts < max_attempts:
            # Extract recipe data
            extraction_chain = create_extraction_chain(
                schema=self.recipe_searcher.schema_extract_recipe_data,
                llm=llm
            )

            # Get structured recipe text first
            structured_result = await llm.ainvoke(
                structure_prompt.format(recipe_text=recipe_string)
            )

            # Try to extract recipe data
            extraction_result = await extraction_chain.ainvoke({
                "input": structured_result.content
            })

            if not extraction_result or not extraction_result.get('text'):
                print("Could not extract recipe data, gathering more information...")
            else:
                # Check for missing required fields
                recipe_data = extraction_result['text'][0]
                missing_fields = self._check_missing_recipe_fields(recipe_data)

                if not missing_fields:
                    # All required fields are present
                    recipe = ExtractRecipeDataSchemaPropertiesPydantic(
                        **recipe_data)
                    recipe.url = f"user_recipe_{
                        datetime.datetime.now().isoformat()}"
                    recipe = ExtractRecipeDataSchemaPropertiesWithSourcePydantic(
                        **recipe.model_dump(),
                        source="user_input",
                        image=image
                    )
                    # Save to database and index
                    self.recipe_searcher._save_valid_recipe(recipe)
                    recipe_text = self.recipe_searcher._recipe_to_text(recipe)
                    self.recipe_searcher.recipe_store.add_texts(
                        [recipe_text],
                        metadatas=[recipe.model_dump()]
                    )
                    return await self.recipe_searcher._validate_and_adapt_recipe(
                        recipe=recipe,
                        constraints=None
                    )

                # Gather missing information through interactive prompts
                print(
                    "\nSome information is missing from your recipe. Please provide the following details:")

                for field, question in missing_fields.items():
                    user_input = input(f"{question}: ").strip()
                    if field in ['ingredients', 'instructions']:
                        # Split lists on newlines or commas
                        recipe_data[field] = [
                            item.strip()
                            for item in user_input.replace('\n', ',').split(',')
                            if item.strip()
                        ]
                    elif field in ['cooking_time_minutes', 'calories_per_serving']:
                        try:
                            recipe_data[field] = int(user_input)
                        except ValueError:
                            print(f"Invalid input for {
                                  field}. Please enter a number.")
                            continue
                    else:
                        recipe_data[field] = user_input

                # Update recipe string with new information
                recipe_string = self._format_recipe_with_new_info(recipe_data)

            attempts += 1
            if attempts == max_attempts:
                missing = ', '.join(missing_fields.keys())
                raise ValueError(
                    f"Could not extract complete recipe after {
                        max_attempts} attempts. "
                    f"Missing information: {missing}"
                )

        raise ValueError("Failed to extract recipe data")

    def _check_missing_recipe_fields(self, recipe_data: Dict) -> Dict[str, str]:
        """Check which required fields are missing from the recipe data."""
        required_fields = {
            'title': 'What is the name of the recipe?',
            'ingredients': 'Please list the ingredients with quantities (one per line)',
            'instructions': 'Please provide the cooking instructions (one step per line)',
            'cooking_time_minutes': 'How many minutes does it take to cook?',
            'servings': 'How many servings does this recipe make?',
            'calories_per_serving': 'How many calories per serving (approximate)?',
            'cuisine_type': 'What type of cuisine is this recipe?',
            'dietary_tags': 'Any dietary tags (e.g., vegetarian, gluten-free)?'
        }

        missing_fields = {}
        for field, question in required_fields.items():
            if field not in recipe_data or not recipe_data[field]:
                missing_fields[field] = question

        return missing_fields

    def _format_recipe_with_new_info(self, recipe_data: Dict) -> str:
        """Format recipe data back into a string for re-extraction."""
        return f"""
        Recipe: {recipe_data.get('title', '')}

        Cuisine: {recipe_data.get('cuisine_type', '')}
        Cooking Time: {recipe_data.get('cooking_time_minutes', '')} minutes
        Servings: {recipe_data.get('servings', '')}
        Calories per serving: {recipe_data.get('calories_per_serving', '')}

        Ingredients:
        {chr(10).join(recipe_data.get('ingredients', []))}

        Instructions:
        {chr(10).join(recipe_data.get('instructions', []))}

        Dietary Tags: {', '.join(recipe_data.get('dietary_tags', []))}
        """

    def _calculate_recipe_suitability(self, recipe: ValidateAndAdaptRecipeResult, constraints: mpc.ConstraintsType) -> float:
        """Calculate how well a recipe matches the given constraints."""
        score = 0.0

        # Calorie match (0-1 score)
        target_calories = (constraints.calories.min +
                           constraints.calories.max) / 2
        calorie_diff = abs(recipe.calories_per_serving - target_calories)
        score += 1 - (calorie_diff / target_calories)

        # Protein match (0-1 score)
        target_protein = (constraints.protein.min +
                          constraints.protein.max) / 2
        protein_diff = abs(recipe.macros.protein - target_protein)
        score += 1 - (protein_diff / target_protein)

        # Cooking time match (0-1 score)
        if recipe.cooking_time_minutes <= constraints.cooking_time_minutes.max:
            score += 1

        # Dietary restrictions match (0-1 score)
        if all(restriction not in recipe.dietary_info
               for restriction in constraints.dietary_restrictions.to_label().split(', ')):
            score += 1

        return score / 4  # Normalize to 0-1

    def _vary_day_plan(self, base_plan: DayMeals, constraints: mpc.ConstraintsType) -> DayMeals:
        """Create a varied version of a day plan while maintaining nutritional balance."""
        # Get new recipes that match the nutritional profile of each meal
        varied_plan = DayMeals(
            breakfast=[],
            lunch=[],
            dinner=[],
            snacks=[]
        )

        # Get all available recipes
        all_recipes = asyncio.run(self.find_suitable_recipes(constraints))

        # For each meal type, find alternative recipes with similar nutritional profiles
        for meal_type in ['breakfast', 'lunch', 'dinner', 'snacks']:
            base_meals = getattr(base_plan, meal_type)
            varied_meals = getattr(varied_plan, meal_type)

            for base_recipe in base_meals:
                # Find alternative recipes with similar nutritional profile
                alternative = self._find_alternative_recipe(
                    base_recipe=base_recipe,
                    available_recipes=all_recipes,
                    used_recipes=self._get_used_recipes(varied_plan),
                    constraints=constraints
                )

                if alternative:
                    varied_meals.append(alternative)
                else:
                    # If no suitable alternative found, use the original recipe
                    varied_meals.append(base_recipe)

        return varied_plan

    def _find_alternative_recipe(
        self,
        base_recipe: ValidateAndAdaptRecipeResult,
        available_recipes: list[ValidateAndAdaptRecipeResult],
        used_recipes: set[str],
        constraints: mpc.ConstraintsType
    ) -> Optional[ValidateAndAdaptRecipeResult]:
        """Find an alternative recipe with similar nutritional profile but different ingredients."""

        # Calculate acceptable ranges for key nutrients
        calorie_margin = 0.2  # 20% margin
        protein_margin = 0.2
        target_calories = base_recipe.calories_per_serving
        target_protein = base_recipe.macros.protein

        # Filter suitable alternatives
        candidates = []
        for recipe in available_recipes:
            # Skip if recipe already used
            if recipe.url in used_recipes:
                continue

            # Skip if same recipe
            if recipe.url == base_recipe.url:
                continue

            # Check if nutritionally similar
            calorie_diff = abs(recipe.calories_per_serving -
                               target_calories) / target_calories
            protein_diff = abs(recipe.macros.protein -
                               target_protein) / target_protein

            if (calorie_diff <= calorie_margin and
                protein_diff <= protein_margin and
                    recipe.cooking_time_minutes <= constraints.cooking_time_minutes.max):

                # Calculate ingredient overlap
                base_ingredients = set(i.lower()
                                       for i in base_recipe.ingredients)
                new_ingredients = set(i.lower() for i in recipe.ingredients)
                overlap = len(base_ingredients & new_ingredients) / \
                    len(base_ingredients)

                # Add to candidates with similarity score
                similarity_score = self._calculate_recipe_similarity(
                    base_recipe, recipe, overlap)
                candidates.append((recipe, similarity_score))

        # Sort by similarity score (lower is better - we want different but nutritionally similar)
        candidates.sort(key=lambda x: x[1])

        # Return the best candidate if any found
        return candidates[0][0] if candidates else None

    def _calculate_recipe_similarity(
        self,
        recipe1: ValidateAndAdaptRecipeResult,
        recipe2: ValidateAndAdaptRecipeResult,
        ingredient_overlap: float
    ) -> float:
        """Calculate similarity score between two recipes (lower means more varied but nutritionally similar)."""
        score = 0.0

        # Ingredient variety (0-1, lower is better)
        score += ingredient_overlap

        # Cuisine variety (0 or 1)
        if recipe1.cuisine_type == recipe2.cuisine_type:
            score += 1

        # Cooking method variety (0 or 1)
        if recipe1.cooking_method == recipe2.cooking_method:
            score += 1

        # Nutritional similarity (0-1, higher is better)
        calorie_diff = abs(recipe1.calories_per_serving -
                           recipe2.calories_per_serving) / recipe1.calories_per_serving
        protein_diff = abs(recipe1.macros.protein -
                           recipe2.macros.protein) / recipe1.macros.protein
        nutritional_similarity = 1 - ((calorie_diff + protein_diff) / 2)

        # We want different recipes that are nutritionally similar
        return score - nutritional_similarity

    def _get_used_recipes(self, plan: DayMeals) -> set[str]:
        """Get set of recipe URLs already used in a day plan."""
        used_recipes = set()
        for meal_type in ['breakfast', 'lunch', 'dinner', 'snacks']:
            meals = getattr(plan, meal_type)
            for recipe in meals:
                used_recipes.add(recipe.url)
        return used_recipes

    async def extract_recipe_from_image(self, image: bytes) -> ValidateAndAdaptRecipeResult:
        """Extract recipe data from an image."""
        # TODO: use a vision model to extract the recipe data from the image in the structured format from the recipe_searcher.schema_extract_recipe_data schema using the langchain.chains.create_extraction_chain function and a langchain graph with a node hosting the vision model to extract all ingredients and amounts, a node hosting the recipe_searcher.schema_extract_recipe_data schema to validate the extracted data, and a node hosting the recipe_searcher.schema_adapt_recipe_data schema to adapt the extracted data to the schema, a node to save the recipe data to the database, and a node to return the recipe data, a node to add the recipe data to the FAISS index, a node to ask the Human in the loop if they are happy with the recipe data, and a node to save the recipe data to the database if they are happy with it.
        # TODO: the above code should be in this function that returns the ValidateAndAdaptRecipeResult
        ...

    async def scrape_product(self, product_name: str, supermarket_name: str) -> SupermarketProduct:
        """
        Scrape product details from a supermarket website and store in database.
        """
        # Initialize search tool and extraction chain
        llm = get_langchain_llm(use_ollama=False)

        # Define product extraction schema
        # TODO: Type Annotate this schema to ensure it matches the SupermarketProduct Pydantic model
        product_schema = {
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "price_per_unit": {"type": "string"},
                "ingredients": {"type": "array", "items": {"type": "string"}},
                "allergens": {"type": "array", "items": {"type": "string"}},
                "product_id": {"type": "string"},
                "description": {"type": "string"},
                "url": {"type": "string"},
                "image_url": {"type": "string"},
                "nutritional_info": {
                    "type": "object",
                    "properties": {
                        "calories_per_100g": {"type": "number"},
                        "protein_g": {"type": "number"},
                        "carbs_g": {"type": "number"},
                        "fat_g": {"type": "number"},
                        "fiber_g": {"type": "number"},
                        "sugar_g": {"type": "number"},
                        "salt_g": {"type": "number"},
                        "vitamins": {
                            "type": "object",
                            "properties": {
                                "vitamin_a_mcg": {"type": "number"},
                                "vitamin_c_mg": {"type": "number"},
                                "vitamin_d_mcg": {"type": "number"},
                                "vitamin_e_mg": {"type": "number"},
                                "vitamin_k_mcg": {"type": "number"},
                                "thiamin_mg": {"type": "number"},
                                "riboflavin_mg": {"type": "number"},
                                "niacin_mg": {"type": "number"},
                                "vitamin_b6_mg": {"type": "number"},
                                "folate_mcg": {"type": "number"},
                                "vitamin_b12_mcg": {"type": "number"}
                            }
                        },
                        "minerals": {
                            "type": "object",
                            "properties": {
                                "calcium_mg": {"type": "number"},
                                "iron_mg": {"type": "number"},
                                "magnesium_mg": {"type": "number"},
                                "phosphorus_mg": {"type": "number"},
                                "potassium_mg": {"type": "number"},
                                "sodium_mg": {"type": "number"},
                                "zinc_mg": {"type": "number"},
                                "selenium_mcg": {"type": "number"}
                            }
                        }
                    }
                }
            }
        }

        # Create extraction chain
        extraction_chain = create_extraction_chain(
            schema=product_schema,
            # schema=self.recipe_searcher.schema_extract_recipe_data,
            llm=llm
        )

        # Search for product listing
        query = f"{product_name} {supermarket_name} grocery product page"
        search_results = await self.recipe_searcher.search_tool.ainvoke({"query": query})

        # Process search results
        for result in search_results:
            url = result.get("url", "")
            if not url or not self._is_valid_supermarket_url(url, supermarket_name):
                continue

            try:
                # Load and process the product page
                loader = lcdl.AsyncChromiumLoader([url])
                docs = await loader.aload()

                # Extract product data
                html_content = "\n".join(doc.page_content for doc in docs)
                extraction_result = await extraction_chain.ainvoke({"input": html_content})

                if not extraction_result or not extraction_result.get('text'):
                    continue

                product_data = extraction_result['text'][0]
                product = SupermarketProduct(**product_data)
                product.url = url

                # Confirm with user
                print("\nFound product details:")
                print(f"Name: {product.product_name}")
                print(f"Price: £{product.price}")
                print(f"Ingredients: {', '.join(product.ingredients)}")
                print(f"URL: {product.url}")

                if input("\nIs this the correct product? (y/n): ").lower() == 'y':
                    # Save to database
                    await self._save_product_to_db(product)

                    # Add to FAISS index
                    product_text = self._product_to_text(product)
                    self.recipe_searcher.recipe_store.add_texts(
                        [product_text],
                        metadatas=[product.model_dump()]
                    )

                    return product

            except Exception as e:
                logging.error(f"Error processing product URL {url}: {e}")
                continue

        raise ValueError(f"Could not find valid product listing for {
                         product_name} at {supermarket_name}")

    def _is_valid_supermarket_url(self, url: str, supermarket_name: str) -> bool:
        """Check if URL is from the specified supermarket."""
        supermarket_domains = {
            "tesco": ["tesco.com"],
            "sainsburys": ["sainsburys.co.uk"],
            "asda": ["asda.com"],
            "morrisons": ["morrisons.com"],
            "waitrose": ["waitrose.com"],
            "aldi": ["aldi.co.uk"],
            "lidl": ["lidl.co.uk"]
        }

        domain = urlparse(url).netloc
        return any(domain.endswith(d) for d in supermarket_domains.get(supermarket_name.lower(), []))

    async def _save_product_to_db(self, product: SupermarketProduct):
        """Save product information to database."""
        # Create products table if it doesn't exist
        self.recipe_searcher.persistent_cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                url TEXT PRIMARY KEY,
                product_name TEXT,
                price REAL,
                price_per_unit TEXT,
                ingredients TEXT,
                allergens TEXT,
                product_id TEXT,
                description TEXT,
                image_url TEXT,
                nutritional_info TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert or update product data
        self.recipe_searcher.persistent_cursor.execute('''
            INSERT OR REPLACE INTO products (
                url, product_name, price, price_per_unit, ingredients,
                allergens, product_id, description, image_url, nutritional_info,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            product.url,
            product.product_name,
            product.price,
            product.price_per_unit,
            json.dumps(product.ingredients),
            json.dumps(product.allergens),
            product.product_id,
            product.description,
            product.image_url,
            product.nutritional_info.model_dump_json(),
            datetime.datetime.now().isoformat()
        ))

        self.recipe_searcher.persistent_conn.commit()

    def _product_to_text(self, product: SupermarketProduct) -> str:
        """Convert product data to searchable text format."""
        return f"""
        Product: {product.product_name}
        Price: £{product.price}
        Description: {product.description}
        Ingredients: {', '.join(product.ingredients)}
        Allergens: {', '.join(product.allergens)}
        Nutrition per 100g:
        - Calories: {product.nutritional_info.calories_per_100g}
        - Protein: {product.nutritional_info.protein_g}g
        - Carbs: {product.nutritional_info.carbs_g}g
        - Fat: {product.nutritional_info.fat_g}g
        """

    async def convert_recipe_to_meal(
        self,
        recipe: ValidateAndAdaptRecipeResult,
        servings: Optional[int] = None
    ) -> Meal:
        """Convert a recipe to a meal with cost calculations."""

        # Initialize cost tracking
        total_cost = 0.0
        ingredient_costs: Dict[str, float] = {}

        # Use provided servings or recipe default
        servings = servings or recipe.servings

        # Extract ingredients with quantities
        ingredients_with_quantities = self._parse_ingredients(
            recipe.ingredients)

        # Search and calculate costs for each ingredient
        for ingredient, quantity in ingredients_with_quantities.items():
            try:
                # Search for ingredient in product database first
                product = await self._find_product_in_db(ingredient)

                if not product:
                    # If not in database, search and scrape from supermarket
                    product = await self.scrape_product(
                        product_name=ingredient,
                        supermarket_name="tesco"  # Default to Tesco, could be configurable
                    )

                # Calculate cost based on quantity needed
                ingredient_cost = self._calculate_ingredient_cost(
                    product=product,
                    required_quantity=quantity
                )

                ingredient_costs[ingredient] = ingredient_cost
                total_cost += ingredient_cost

            except Exception as e:
                logging.warning(f"Could not get cost for {ingredient}: {e}")
                ingredient_costs[ingredient] = 0.0

        # Create the meal object
        meal = Meal(
            recipe_name=recipe.title,
            recipe_url=recipe.url,
            servings=servings,
            total_cost_gbp=total_cost,
            cost_per_serving_gbp=total_cost / servings if servings else 0,
            ingredient_costs_gbp=ingredient_costs,
            calories_per_serving=recipe.calories_per_serving,
            protein_g=recipe.macros.protein,
            carbs_g=recipe.macros.carbs,
            fat_g=recipe.macros.fat,
            cooking_time_minutes=recipe.cooking_time_minutes,
            ingredients=recipe.ingredients,
            instructions=recipe.instructions,
            dietary_info=DietaryRestrictions(
                gluten='none' if recipe.dietary_info.gluten_free else 'normal',
                dairy='none' if recipe.dietary_info.contains_dairy else 'normal',
                soy='none' if recipe.dietary_info.contains_soy else 'normal',
                nuts='none' if recipe.dietary_info.contains_nuts else 'normal',
                shellfish='none' if recipe.dietary_info.contains_shellfish else 'normal',
                fish='none' if recipe.dietary_info.contains_fish else 'normal',
                eggs='none' if recipe.dietary_info.contains_eggs else 'normal',
                meat='none' if recipe.dietary_info.contains_meat else 'normal',
                vegetarian='yes' if recipe.dietary_info.vegetarian else 'no',
                vegan='yes' if recipe.dietary_info.vegan else 'no',
                pescetarian='yes' if recipe.dietary_info.contains_fish and not recipe.dietary_info.contains_meat else 'no',
                carnivore='yes' if recipe.dietary_info.contains_meat else 'no',
                suitable_for_diet=recipe.dietary_info.suitable_for_diet
            ),
            nutritional_info=NutritionalInfo(
                calories_per_100g=recipe.calories_per_serving,
                protein_g=recipe.macros.protein,
                carbs_g=recipe.macros.carbs,
                fat_g=recipe.macros.fat,
                fiber_g=recipe.macros.fiber,
                sugar_g=recipe.macros.sugar,
                salt_g=recipe.macros.salt,
                vitamins=ProductVitamins(
                    vitamin_a_mcg=recipe.micros.vitamin_a,
                    vitamin_c_mg=recipe.micros.vitamin_c,
                    vitamin_d_mcg=recipe.micros.vitamin_d,
                    vitamin_e_mg=recipe.micros.vitamin_e,
                    vitamin_k_mcg=recipe.micros.vitamin_k,
                    thiamin_mg=recipe.micros.thiamin,
                    riboflavin_mg=recipe.micros.riboflavin,
                    niacin_mg=recipe.micros.niacin,
                    vitamin_b6_mg=recipe.micros.vitamin_b6,
                    folate_mcg=recipe.micros.folate,
                    vitamin_b12_mcg=recipe.micros.vitamin_b12
                ),
                minerals=ProductMinerals(
                    calcium_mg=recipe.micros.calcium,
                    iron_mg=recipe.micros.iron,
                    magnesium_mg=recipe.micros.magnesium,
                    phosphorus_mg=recipe.micros.phosphorus,
                    potassium_mg=recipe.micros.potassium,
                    sodium_mg=recipe.micros.sodium,
                    zinc_mg=recipe.micros.zinc,
                    selenium_mcg=recipe.micros.selenium
                )
            )
        )

        return meal

    async def delete_recipe(self, recipe_url: str):
        """Delete a recipe from the database and the FAISS index."""
        self.recipe_searcher.persistent_cursor.execute(
            "DELETE FROM recipes WHERE url = ?", (recipe_url,)
        )
        self.recipe_searcher.persistent_conn.commit()

        # Delete from FAISS index
        warnings.warn(f"Deleting recipe {recipe_url} from FAISS index")
        results = self.recipe_searcher.recipe_store.similarity_search(
            recipe_url,
            k=1
        )
        self.recipe_searcher.recipe_store.delete(results[0].metadata)

    def _parse_ingredients(self, ingredients: list[str]) -> Dict[str, float]:
        """Parse ingredients list to extract quantities and units."""
        # Initialize LLM for ingredient parsing
        llm = get_langchain_llm(use_ollama=False)

        # Create extraction chain for ingredient parsing
        ingredient_schema = {
            "properties": {
                "ingredient": {"type": "string"},
                "quantity": {"type": "number"},
                "unit": {"type": "string"}
            }
        }

        extraction_chain = create_extraction_chain(
            schema=ingredient_schema,
            llm=llm
        )

        parsed_ingredients = {}
        for ingredient in ingredients:
            try:
                # Extract quantity information
                result = extraction_chain.invoke({"input": ingredient})
                if result and result.get('text'):
                    data = result['text'][0]
                    # Convert to standard units (e.g., grams, ml)
                    quantity = self._standardize_quantity(
                        data['quantity'],
                        data['unit']
                    )
                    parsed_ingredients[data['ingredient']] = quantity
                else:
                    # If parsing fails, store ingredient without quantity
                    parsed_ingredients[ingredient] = 0.0
            except Exception as e:
                logging.warning(f"Error parsing ingredient {ingredient}: {e}")
                parsed_ingredients[ingredient] = 0.0

        return parsed_ingredients

    def _standardize_quantity(self, quantity: float, unit: str) -> float:
        """Convert quantities to standard units (g/ml)."""
        # Conversion factors
        conversions = {
            # Weight
            'kg': 1000,
            'g': 1,
            'mg': 0.001,
            'oz': 28.35,
            'lb': 453.592,
            # Volume
            'l': 1000,
            'ml': 1,
            'cup': 240,
            'tbsp': 15,
            'tsp': 5,
            'fl oz': 29.574
        }

        unit = unit.lower().strip()
        if unit in conversions:
            return quantity * conversions[unit]
        return quantity

    async def _find_product_in_db(self, ingredient: str) -> Optional[SupermarketProduct]:
        """Search for ingredient in product database."""
        try:
            # Search FAISS index first
            results = self.recipe_searcher.recipe_store.similarity_search(
                ingredient,
                k=1
            )

            if results:
                product_data = results[0].metadata
                return SupermarketProduct(**product_data)

        except Exception as e:
            logging.warning(f"Error searching product database: {e}")

        return None

    def _calculate_ingredient_cost(
        self,
        product: SupermarketProduct,
        required_quantity: float
    ) -> float:
        """Calculate cost of ingredient based on required quantity."""
        # Parse price per unit
        try:
            unit_price = self._parse_price_per_unit(product.price_per_unit)

            # Convert required quantity to same unit as price
            converted_quantity = required_quantity / 1000  # Convert g to kg or ml to l

            return unit_price * converted_quantity
        except:
            # Fallback to simple division by package size if available
            # Assume 100g/ml portions
            return (required_quantity / 100) * product.price

    def _parse_price_per_unit(self, price_per_unit: str) -> float:
        """Parse price per unit string (e.g., '£2.50/kg')."""
        try:
            # Extract numeric value
            price = float(re.search(r'£?(\d+\.?\d*)', price_per_unit).group(1))

            # Convert to per gram/ml if needed
            if '/kg' in price_per_unit or '/l' in price_per_unit:
                return price
            elif '/100g' in price_per_unit or '/100ml' in price_per_unit:
                return price * 10

            return price
        except:
            raise ValueError(f"Could not parse price per unit: {
                             price_per_unit}")


async def test_add_my_own_recipe_from_natural_language():
    recipe_text = """
    Make spaghetti with tomato sauce.
    Cook pasta, add sauce with garlic and basil.
    Takes about 20 minutes.
    """

    try:
        workflow = MealPlannerWorkflow()
        recipe = await workflow.add_my_own_recipe_from_natural_language(recipe_text)
        print(recipe)
        workflow.delete_recipe(recipe.url)
    except ValueError as e:
        print(f"Error: {e}")


def run_tests():
    asyncio.run(test_add_my_own_recipe_from_natural_language())


if __name__ == "__main__":
    # streamlit run src/mains/pydantic_ai_streamlit_web_scraper.py
    # main_manual()
    main_manual_no_streamlit()
    # main()
