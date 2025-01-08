from typing import Literal, TypeVar, TypedDict, Generic
from pydantic import BaseModel, Field

TSLI = TypeVar('TSLI', bound=Literal['integer', 'string', 'boolean'])
TANY = TypeVar('TANY', covariant=True)


class _SchemaTypeVal(TypedDict):
    type: Literal['integer', 'string', 'boolean']


class _SchemaTypeValEnum(TypedDict):
    type: Literal['string']
    enum: list[str]


class _SchemaTypeList(TypedDict):
    type: Literal['array']
    items: _SchemaTypeVal


class _SchemaTypeObject(TypedDict, Generic[TANY]):
    type: Literal['object']
    properties: TANY


class _CustomTypeDummyObjectProps(TypedDict):
    protein: _SchemaTypeVal | int
    carbs: _SchemaTypeVal | int
    fat: _SchemaTypeVal | int


DUMMY_VAL: _SchemaTypeVal | bool = {"type": "boolean"}
DUMMY_ARRAY: _SchemaTypeList | list[str] = {
    "type": "array", "items": {"type": "string"}}
DUMMY_OBJECT_PROPS: _CustomTypeDummyObjectProps = {"protein": {
    "type": "integer"}, "carbs": {"type": "integer"}, "fat": {"type": "integer"}}
DUMMY_OBJECT: _SchemaTypeObject | _CustomTypeDummyObjectProps = {
    "type": "object", "properties": DUMMY_OBJECT_PROPS}

DUMMY = {
    "is_recipe": {"type": "boolean"},
    "recipe_name": {"type": "string"},
    # "title": {"type": "string"},
    "calories": {"type": "integer"},
    # "calories_per_serving": {"type": "integer"},
    # "cooking_time_minutes": {"type": "integer"},
    # "servings": {"type": "integer"},
    "ingredients": {"type": "array", "items": {"type": "string"}},
    # "instructions": {"type": "array", "items": {"type": "string"}},
    # "images": {"type": "array", "items": {"type": "string"}},
    # "content": {"type": "string"},
    "macros": {
        "type": "object",
        "properties": {
            "protein": {"type": "integer"},
            "carbs": {"type": "boolean"},
            "fat": {"type": "string"}
        }
    },
    # "micros": {
    #     "type": "object",
    #     "properties": {
    #         "vitamin_a": {"type": "integer"},
    #         "vitamin_c": {"type": "integer"},
    #         "vitamin_d": {"type": "integer"},
    #         "vitamin_e": {"type": "integer"},
    #         "vitamin_k": {"type": "integer"},
    #         "calcium": {"type": "integer"},
    #         "iron": {"type": "integer"},
    #         "magnesium": {"type": "integer"},
    #         "phosphorus": {"type": "integer"}
    #     }
    # },
    # "dietary_info": {
    #     "type": "object",
    #     "properties": {
    #         "vegetarian": {"type": "boolean"},
    #         "vegan": {"type": "boolean"},
    #         "gluten_free": {"type": "boolean"}
    #     }
    # }
}


# TSVI = TypeVar('TSVI', bound=int | _SchemaTypeVal[Literal['integer']])
# TSAI = TypeVar('TSAI', bound=list[int] | _SchemaTypeList[Literal['integer']])
# TSDI = TypeVar('TSDI', bound=dict[str, int] |
#                _SchemaTypeObject[Literal['integer']])
# TSVS = TypeVar('TSVS', bound=str | _SchemaTypeVal[Literal['string']])
# TSAS = TypeVar('TSAS', bound=list[str] | _SchemaTypeList[Literal['string']])
# TSDS = TypeVar('TSDS', bound=dict[str, str] |
#                _SchemaTypeObject[Literal['string']])
# TSVB = TypeVar('TSVB', bound=bool | _SchemaTypeVal[Literal['boolean']])
# TSAB = TypeVar('TSAB', bound=list[bool] | _SchemaTypeList[Literal['boolean']])
# TSDB = TypeVar('TSDB', bound=dict[str, bool] |
#                _SchemaTypeObject[Literal['boolean']])
# TSI = TSVI | TSDI | TSAI
# TSB = TSVB | TSDB | TSAB
# TSS = TSVS | TSDS | TSAS

# TSI = TypeVar('TSI', bound=int | _SchemaTypeVal[Literal['integer']] |
#               _SchemaTypeList[Literal['integer']] | _SchemaTypeObject[Literal['integer']])
# TSB = TypeVar('TSB', bound=bool | _SchemaTypeVal[Literal['boolean']] |
#               _SchemaTypeList[Literal['boolean']] | _SchemaTypeObject[Literal['boolean']])
# TSS = TypeVar('TSS', bound=str | _SchemaTypeVal[Literal['string']] |
#               _SchemaTypeList[Literal['string']] | _SchemaTypeObject[Literal['string']])


class ExtractRecipeDataSchemaMacros(TypedDict):
    protein: int | _SchemaTypeVal
    carbs: int | _SchemaTypeVal
    fat: int | _SchemaTypeVal
    fiber: int | _SchemaTypeVal
    sugar: int | _SchemaTypeVal
    salt: int | _SchemaTypeVal


class ExtractRecipeDataSchemaMicros(TypedDict):
    vitamin_a: int | _SchemaTypeVal
    vitamin_c: int | _SchemaTypeVal
    vitamin_d: int | _SchemaTypeVal
    vitamin_e: int | _SchemaTypeVal
    vitamin_k: int | _SchemaTypeVal
    calcium: int | _SchemaTypeVal
    iron: int | _SchemaTypeVal
    magnesium: int | _SchemaTypeVal
    phosphorus: int | _SchemaTypeVal
    potassium: int | _SchemaTypeVal
    sodium: int | _SchemaTypeVal
    zinc: int | _SchemaTypeVal
    selenium: int | _SchemaTypeVal
    thiamin: int | _SchemaTypeVal
    riboflavin: int | _SchemaTypeVal
    niacin: int | _SchemaTypeVal
    vitamin_b6: int | _SchemaTypeVal
    folate: int | _SchemaTypeVal
    vitamin_b12: int | _SchemaTypeVal


class ExtractRecipeDataSchemaDietaryInfo(TypedDict):
    vegetarian: bool | _SchemaTypeVal
    vegan: bool | _SchemaTypeVal
    gluten_free: bool | _SchemaTypeVal
    contains_meat: bool | _SchemaTypeVal
    contains_dairy: bool | _SchemaTypeVal
    contains_nuts: bool | _SchemaTypeVal
    contains_soy: bool | _SchemaTypeVal
    contains_gluten: bool | _SchemaTypeVal
    contains_eggs: bool | _SchemaTypeVal
    contains_shellfish: bool | _SchemaTypeVal
    contains_peanuts: bool | _SchemaTypeVal
    contains_tree_nuts: bool | _SchemaTypeVal
    contains_wheat: bool | _SchemaTypeVal
    contains_fish: bool | _SchemaTypeVal
    suitable_for_diet: list[str]


class ExtractRecipeDataSchemaMacrosPydantic(BaseModel):
    protein: int
    carbs: int
    fat: int
    fiber: int
    sugar: int
    salt: int


class ExtractRecipeDataSchemaMicrosPydantic(BaseModel):
    vitamin_a: int
    vitamin_c: int
    vitamin_d: int
    vitamin_e: int
    vitamin_k: int
    calcium: int
    iron: int
    magnesium: int
    phosphorus: int
    potassium: int
    sodium: int
    zinc: int
    selenium: int
    thiamin: int
    riboflavin: int
    niacin: int
    vitamin_b6: int
    folate: int
    vitamin_b12: int


class ExtractRecipeDataSchemaDietaryInfoPydantic(BaseModel):
    vegetarian: bool
    vegan: bool
    gluten_free: bool
    contains_meat: bool
    contains_dairy: bool
    contains_nuts: bool
    contains_soy: bool
    contains_gluten: bool
    contains_eggs: bool
    contains_shellfish: bool
    contains_peanuts: bool
    contains_tree_nuts: bool
    contains_wheat: bool
    contains_fish: bool
    suitable_for_diet: list[str]


class ExtractRecipeDataSchemaPropertiesPydantic(BaseModel):
    """
    Pydantic Class Implementation of the TypedDict with TSS -> str, TSB -> bool, TSI -> int and ignoring _SchemaTypeObject, _SchemaTypeList, _SchemaTypeVal
    """
    recipe_name: str
    calories_per_serving: int
    is_recipe: bool
    servings: int
    title: str
    suitable_for_meal: str
    calories: int
    macros: ExtractRecipeDataSchemaMacrosPydantic
    micros: ExtractRecipeDataSchemaMicrosPydantic
    images: list[str]
    cooking_time_minutes: int
    ingredients: list[str]
    instructions: list[str]
    dietary_info: ExtractRecipeDataSchemaDietaryInfoPydantic
    cuisine_type: str
    cooking_method: str
    content: str
    url: str
    last_updated: str
    image: bytes | None


class ExtractRecipeDataSchemaPropertiesWithSourcePydantic(ExtractRecipeDataSchemaPropertiesPydantic):
    source: Literal['user_input', 'web_scrape']


class ValidateAndAdaptRecipeResult(ExtractRecipeDataSchemaPropertiesWithSourcePydantic):
    original_recipe_url: str
    is_modified: bool
    user_approved: bool


TSE = str | _SchemaTypeVal  # [Literal["string"]]
TSS = str | _SchemaTypeVal  # [Literal["string"]]
TSI = int | _SchemaTypeVal  # [Literal["integer"]]
TSB = bool | _SchemaTypeVal  # [Literal["boolean"]]


class ExtractRecipeDataSchemaProperties(TypedDict):
    """
    Properties of the recipe data schema.
    """
    recipe_name: TSS
    calories_per_serving: TSI
    is_recipe: TSB
    servings: TSI
    suitable_for_meal: Literal['breakfast', 'lunch',
                               'dinner', 'dessert', 'snacks'] | _SchemaTypeValEnum
    title: TSS
    calories: TSI
    macros: ExtractRecipeDataSchemaMacros | _SchemaTypeObject[ExtractRecipeDataSchemaMacros]
    micros: ExtractRecipeDataSchemaMicros | _SchemaTypeObject[ExtractRecipeDataSchemaMicros]
    images: list[str] | _SchemaTypeList
    cooking_time_minutes: TSI
    ingredients: list[str] | _SchemaTypeList
    instructions: list[str] | _SchemaTypeList
    dietary_info: ExtractRecipeDataSchemaDietaryInfo | _SchemaTypeObject[
        ExtractRecipeDataSchemaDietaryInfo]
    cuisine_type: TSS
    cooking_method: TSS
    content: TSS


class ExtractRecipeDataSchemaPropertiesWithMetadata(ExtractRecipeDataSchemaProperties):
    url: str
    last_updated: str


class ExtractRecipeDataSchema(TypedDict):
    properties: ExtractRecipeDataSchemaProperties
    required: list[str]


class RecipeAnalysisResult(BaseModel):
    """Structure for LLM output when analyzing recipe content."""
    content_summary: str = Field(
        description="Summary of the content analyzed so far")
    recipe_probability: float = Field(
        description="Probability that this is a recipe page (0-1)",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Explanation for the probability assessment")
