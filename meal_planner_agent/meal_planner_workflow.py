# https://pub.towardsai.net/pydantic-ai-web-scraper-llama-3-3-python-powerful-ai-research-agent-6d634a9565fe
# Pydantic AI Web Scraper with Ollama and Streamlit AI Chatbot

from recipe_search import RecipeSearch
import json
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings, UsageLimits

from langchain_openai import ChatOpenAI
from auto_file_sorter.logging.logging_config import configure_logging
from meal_planner_agent.meal_plan_constraints_pd import DietaryRestrictions, MealPlanConstraints, MealSizeEnum
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
from typing import Any, Literal, Union
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


async def do_search(constraints: str, max_results: int):
    # Prepare dependencies
    current_date = datetime.date.today()
    date_string = current_date.strftime("%Y-%m-%d")
    deps = SearchDataclassForDI(
        todays_date=date_string, constraints=constraints, max_results=max_results)
    query = prompt.format(CONSTRAINTS_JSON=constraints)
    # Inject dependencies into the agent
    logging.info(f"Query: {query}")
    logging.info(f"Deps: {deps}")

    # TODO: Add human in the loop breakpoints so thtat it has max iterations and doesnt get too expensive.
    result = await search_agent.run(
        query,
        deps=deps,
        usage_limits=UsageLimits(request_limit=10, total_tokens_limit=10000),
        model_settings=ModelSettings(max_tokens=10000))
    logging.info(f"Result: {result}")
    return result

# Langgraph Section
# Node
# def assistant(state: OverallMealPlannerState):
#     return {"messages": [llm_gpt_4o.invoke([sys_msg] + state["messages"])]}


# meal_planner_workflow = StateGraph(OverallMealPlannerState)
# meal_planner_workflow.add_node("assistant", assistant)

# meal_planner_workflow.add_edge(START, "assistant")
# End of Langgraph Section

# streamlit app
def main3():
    st.title("Pydantic AI Web Scraper")
    query = st.text_input("Enter your query")
    if st.button("Search"):
        result = asyncio.run(do_search(query, max_results=5))
        st.write(result)


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
    with open('meal_planner_agent/recipe_constraints.json', 'r') as f:
        recipe_constraints = json.load(f)
    with open('meal_planner_agent/day_constraints.json', 'r') as f:
        day_constraints = json.load(f)
    with open('meal_planner_agent/meal_plan_constraints.json', 'r') as f:
        meal_plan_constraints = json.load(f)
    _constraints_raw = {
        **recipe_constraints,
        **day_constraints,
        **meal_plan_constraints
    }
    logging.info(f"Constraints: {_constraints_raw}")
    constraints = MealPlanConstraints(**_constraints_raw)
    workflow = MealPlannerWorkflow()
    weekly_meal_plan = asyncio.run(
        workflow.generate_meal_plan(constraints))

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
        constraints = MealPlanConstraints(
            calories={"min": calories_min, "max": calories_max},
            protein={"min": protein_min, "max": protein_max},
            lipids_pcnt_non_protein_calories={
                "min": lipids_min, "max": lipids_max},
            micronutrients_rda={
                "vitamin_b12": 2,
                "vitamin_b6": 2,
                "vitamin_d": 1.5,
                "vitamin_c": 1.5,
                "iron": 1.5,
                "zinc": 1.5,
                "magnesium": 1.5,
                "selenium": 1.5,
                "vitamin_a": 1.5,
                "vitamin_e": 1.5,
                "vitamin_k": 1.5,
                "vitamin_b1": 1.5,
                "vitamin_b2": 1.5,
                "vitamin_b3": 1.5,
                "vitamin_b5": 1.5,
                "vitamin_b7": 1.5,
                "vitamin_b9": 1.5,
                "omega_3_dha": 1.5,
                "omega_3_epa": 1.5
            },  # Add micronutrient values as needed
            cooking_time_minutes={
                "min": cooking_time_min, "max": cooking_time_max},
            budget_per_week_gbp={"min": budget_min, "max": budget_max},
            serving_sizes={
                "default": {"min": serving_size_default, "max": serving_size_default},
                "lunch": {"min": lunch_serving_size_min, "max": lunch_serving_size_max},
                "dinner": {"min": dinner_serving_size_min, "max": dinner_serving_size_max}
            },
            left_overs=left_overs,
            supplements={"creatine": creatine},
            meal_frequency={"min": meal_frequency_min,
                            "max": meal_frequency_max},
            dietary_restrictions=dietary_restrictions,
            ingredients_to_avoid=ingredients_to_avoid,
            ingredients_to_avoid_in_recipes=[],  # Add as needed
            ingredients_to_include=ingredients_to_include,
            ingredients_I_love=ingredients_I_love,
            cuisine_preferences=cuisine_preferences,
            variety_repitition={"total_recipes_per_week": total_recipes_per_week,
                                "total_ingredients_per_week": total_ingredients_per_week},
            sex=sex,
            DoB=dob,  # Now we're sure dob is a valid date
            height=height,
            weight=weight,
            daily_steps=daily_steps,
            activity_level=activity_level,
            goal=goal,
            generation_arguments={
                "breakdowns": breakdowns,
                "micronutrients_include": micronutrients_include
            }
        )

        # Convert constraints to JSON for further processing
        constraints_json = json.dumps(constraints)
        st.json(constraints_json)  # Display the constraints for debugging
        # result = asyncio.run(do_search(constraints_json, max_results=2))
        # st.success("Meal plan generated successfully!")
        # # TODO parse the results and display them to the user formatted pretty so that the user can use a tab controller to open each day, and then a tree view to drill down into each meal and investigate all the details, macros, micronutrients, cost, etc for the meal and ingredients breakdowns.
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
            workflow.generate_meal_plan(constraints))

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

    async def find_suitable_recipes(self, constraints: MealPlanConstraints):
        """Find recipes that match the given constraints."""
        min_calories = constraints.calories.min
        max_calories = constraints.calories.max

        # Consider dietary restrictions
        query_modifiers = []
        if constraints.dietary_restrictions:
            if constraints.dietary_restrictions.vegetarian in ["yes", "normal"]:
                query_modifiers.append("vegetarian")
            if constraints.dietary_restrictions.vegan in ["yes", "normal"]:
                query_modifiers.append("vegan")
            # Add other dietary restrictions as needed

        recipes = await self.recipe_searcher.search_recipes(
            min_calories=min_calories,
            max_calories=max_calories
        )

        # Filter recipes based on other constraints
        filtered_recipes = []
        for recipe in recipes:
            # Skip recipes with avoided ingredients
            if any(ingredient in recipe.get("content", "").lower()
                   for ingredient in constraints.ingredients_to_avoid):
                continue

            # Check cooking time if available in recipe content
            if "time" in recipe.get("content", "").lower():
                # This is a simple check - you might want to use regex or better parsing
                if constraints.cooking_time_minutes:
                    max_time = constraints.cooking_time_minutes.max
                    if max_time and "minutes" in recipe.get("content", ""):
                        # Basic time extraction - could be improved
                        try:
                            time_st: str | None = (recipe.get("content", "").split("minutes")[0].split()[-1]
                                                   if "minutes" in recipe.get("content", "") else None)
                            if time_str and time_str.isdigit() and int(time_str) > max_time:
                                continue
                        except:
                            pass

            filtered_recipes.append(recipe)

        return filtered_recipes

    async def generate_meal_plan(self, constraints: MealPlanConstraints) -> WeeklyMealPlan:
        """Generate a complete meal plan using recipe search and constraints."""
        # Get suitable recipes
        recipes = await self.find_suitable_recipes(constraints)

        recipe_data = [
            {
                "name": recipe.get("title", ""),
                "url": recipe.get("url", ""),
                "description": recipe.get("content", "")
            }
            for recipe in recipes
        ]

        # Convert constraints to dict and ensure date serialization
        constraints_dict = constraints.model_dump(
            mode='json',  # This automatically converts dates to ISO format
            exclude_none=True  # Optionally exclude None values
        )

        # Add the recipes to your constraints
        constraints_dict["available_recipes"] = recipe_data

        # Call your existing meal plan generation logic
        result = await do_search(json.dumps(constraints_dict), max_results=2)

        return result.data if result and result.data else mock_example_meal_plan


if __name__ == "__main__":
    # streamlit run src/mains/pydantic_ai_streamlit_web_scraper.py
    # main_manual()
    main_manual_no_streamlit()
    # main()
