# https://chatgpt.com/share/67765baa-d3c8-800a-8552-bc7eb6b725e6
generate_meal_plan = (["CONSTRAINTS_JSON"], """
You are given the following JSON which represents all the constraints and requirements for creating a 7-day meal plan:

{CONSTRAINTS_JSON}

Using only these constraints:

1. **Overall Dietary and Nutritional Constraints**  
   - Daily calories must be between the `"calories.min"` and `"calories.max"`.  
   - Daily protein must be between the `"protein.min"` and `"protein.max"`.  
   - Lipids must be between the `"lipids_pcnt_non_protein_calories.min"` and `"lipids_pcnt_non_protein_calories.max"` percent of total non-protein calories.  
   - Comply with all `"dietary_restrictions"`, ensuring ingredients marked as `"none"` or `"no"` do not appear, and honoring any `"low"` or `"normal"` restriction levels.  
   - Avoid any items listed in `"ingredients_to_avoid"` and `"ingredients_to_avoid_in_recipes"`.  
   - Include `"ingredients_to_include"` where possible.  
   - Prioritize frequent use of `"ingredients_I_love"`.  
   - Adhere to `"meal_frequency.min"` through `"meal_frequency.max"` meals per day.  
   - Use `"serving_sizes"` as specified, with default and meal-specific min/max sizes.  
   - Observe the RDA multipliers in `"micronutrients_rda"` (or `"micronutrients_include": "yes"`) to meet or exceed 1.0× the specified amounts where the multiplier is the recommended daily allowance and should be converted into it's equivalent in grams or milligrams per day and then divided by the `"meal_frequency"` to get the total amount for this meal.
   - Observe `"cooking_time_minutes.min"` and `"cooking_time_minutes.max"`.  
   - Stay within `"budget_per_week_gbp.min"` and `"budget_per_week_gbp.max"`.  

2. **Leftovers and Servings**  
   - Since `"left_overs"` is `"yes"`, indicate how many servings to prepare for each meal so leftovers can be used on subsequent meals or days as appropriate.  

3. **Cuisine and Variety**  
   - Follow any `"cuisine_preferences"`.  
   - Respect `"variety_repitition.total_recipes_per_week"` and `"variety_repitition.total_ingredients_per_week"`.  

4. **Supplements**  
   - The plan must reflect usage of any `"supplements"` if relevant.  

5. **User Profile**  
   - The plan should be appropriate for a `sex` of `"male"`, the stated `DoB`, `height`, `weight`, `daily_steps`, `activity_level`, and `"goal"` of `"lose_weight"`.  

6. **Generation Arguments / Output Breakdown**  
   - Per `"generation_arguments.breakdowns"`, provide a **macronutrient breakdown per ingredient**, a **micronutrient breakdown per meal**, and a **cost in GBP per meal**.  
   - Make sure to include all relevant micronutrient details as `"micronutrients_include"` is `"yes"`.  

7. **Final Output Requirements**  
   - **Output a 7-day meal plan** in valid **JSON format**.  
   - Each day must have separate entries for each meal (e.g., “breakfast”, “lunch”, “dinner”, or however you structure it to respect `"meal_frequency"`).  
   - For each meal, list:
     - The meal name.
     - The ingredients used (respecting all avoid/include rules).
     - The serving size (from the allowed `"serving_sizes"`).
     - The number of servings you are making (if leftovers apply).
     - A micronutrient breakdown per meal.
     - A macronutrient breakdown **per ingredient**.
     - A cost GBP breakdown **per meal**.
   - Ensure the total daily sums and weekly sums adhere to the min/max constraints from the provided JSON.

Now, using **only** the constraints above and the provided JSON, generate the 7-day meal plan in JSON format. **Do not** include any additional information or deviate from these constraints.
""")
