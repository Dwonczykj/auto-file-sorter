from typing import TypeGuard
from helpers.type_aliases import model_type_alias


def is_valid_model_type(model: str) -> TypeGuard[model_type_alias]:
    """
    Check if the given string is a valid model_type_alias.

    This function serves as a runtime type checker and a type guard for static type checkers.

    Args:
        model (str): The string to check.

    Returns:
        bool: True if the string is a valid model_type_alias, False otherwise.
    """
    valid_models = {"gpt-3.5-turbo", "gpt-4",
                    "gpt-4-turbo", "gpt-4o", "gpt-4o-latest", "whisper"}
    return model in valid_models
