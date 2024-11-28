import logging
import os
from typing import Literal, Optional, Type, TypeGuard, TypeVar, Union
from enum import EnumType, Enum
from datetime import timedelta
from datetime import datetime
import dateutil.parser
import pytz
import random
import string

EnumType = TypeVar('EnumType', bound=Enum)
T = TypeVar('T', bound=Optional[Enum])


def generate_random_uid(length=12):
    """Generate a random UID string of specified length."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def parse_datetime_robust(datetime_str: str) -> Optional[datetime]:
    """
    Robustly parse a datetime string in various formats.

    Args:
        datetime_str: String representation of a datetime

    Returns:
        datetime object if parsing successful, None if parsing fails

    Examples:
        >>> parse_datetime_robust("2024-11-18T13:16:21.364459")
        datetime(2024, 11, 18, 13, 16, 21, 364459)
        >>> parse_datetime_robust("2024-11-18")
        datetime(2024, 11, 18, 0, 0)
        >>> parse_datetime_robust("invalid")
        None
    """
    if not datetime_str:
        return None

    try:
        # Try parsing with dateutil first (handles most formats)
        dt = dateutil.parser.parse(datetime_str)

        # If no timezone info, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)

        return dt

    except (ValueError, TypeError):
        # Fallback parsing attempts
        formats_to_try = [
            "%Y-%m-%dT%H:%M:%S.%f",  # 2024-11-18T13:16:21.364459
            "%Y-%m-%dT%H:%M:%S",     # 2024-11-18T13:16:21
            "%Y-%m-%d %H:%M:%S",     # 2024-11-18 13:16:21
            "%Y-%m-%d",              # 2024-11-18
            "%d/%m/%Y %H:%M:%S",     # 18/11/2024 13:16:21
            "%d/%m/%Y",              # 18/11/2024
            "%d-%m-%Y %H:%M:%S",     # 18-11-2024 13:16:21
            "%d-%m-%Y",              # 18-11-2024
            "%Y/%m/%d",              # 2024/11/18
            "%b %d %Y %H:%M:%S",     # Nov 18 2024 13:16:21
            "%b %d %Y",              # Nov 18 2024
            "%d %b %Y %H:%M:%S",     # 18 Nov 2024 13:16:21
            "%d %b %Y",              # 18 Nov 2024
            "%Y-%m-%d %H:%M",        # 2024-11-18 13:16
            "%d/%m/%Y %H:%M",        # 18/11/2024 13:16
        ]

        for fmt in formats_to_try:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                return dt.replace(tzinfo=pytz.UTC)
            except ValueError:
                continue

        # If all parsing attempts fail
        return None


def format_time_to_mins_from_secs(secs: float):
    return f"{(secs//60)}mins and {(secs % 60)}secs"


def datetime_to_day_and_time(dt):
    # Get the day of the week
    day_of_week = dt.strftime("%A")

    # Get the hour (in 24-hour format)
    hour = dt.hour

    # Determine the time of day
    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"

    # Combine the day and time of day
    return f"{day_of_week} {time_of_day}"


def str_name_to_enum(enum_class: Type[EnumType], value: str) -> Optional[EnumType]:
    '''
    Converts a string matching uppercase enum *name* entry to the enum
    '''
    try:
        # Convert the string to uppercase if needed or match the exact case
        return enum_class[value.upper()] if value.upper() in enum_class.__members__ else None
    except KeyError:
        return None


def str_to_enum_nullable(enum_class: Type[EnumType], value: Union[str, None], default: Union[EnumType, None] = None) -> Optional[EnumType]:
    if value is None:
        return default
    for member in enum_class:
        if member.value == value:
            return member
    return default


def str_to_enum(enum_class: Type[EnumType], value: Union[str, None], default: EnumType) -> EnumType:
    if value is None:
        return default
    for member in enum_class:
        if member.value == value:
            return member
    return default


def compare_template_dicts(template1: dict, template2: dict):
    """
    Compares two template dictionaries and returns their differences.

    Args:
        template1: First template dictionary
        template2: Second template dictionary

    Returns:
        DeepDiff object containing the differences
    """
    from deepdiff import DeepDiff

    # Ignore order in lists since order might not matter for template components
    diff = DeepDiff(template1, template2, ignore_order=True)
    return diff


def pence_to_pounds_str(pence: float):
    return f"£{pence / 100:.2f}"


def time_delta_to_hours_and_mins_str(delta: timedelta) -> str:
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours} hours and {minutes} minutes" if hours > 0 else f"{minutes} minutes"


class ConversionRequestType(str, Enum):
    SUMMARISE_TO_BULLETS = "summarise to bullets"
    CONVERT_TO_ACTION_POINTS = "convert to action points"
    FULL_TRANSCRIPT = "full transcript"
    SUMMARISE_VOICE_NOTE = "summarise voice note"
    SHORT_SUMMARY = "short summary"
    DETAILED_SUMMARY = "detailed summary"
    PROOF_READ = "proof read"
    BESPOKE_PROCESSING = "bespoke processing"
    MIND_MAP = "mind map"
    TABLE = "table"


def conversionRequestTypeFromStr(request_type: str) -> ConversionRequestType:
    if request_type.replace("_", " ") == "full transcript":
        return ConversionRequestType.FULL_TRANSCRIPT
    elif request_type.replace("_", " ") == "summarise voice note":
        return ConversionRequestType.SUMMARISE_VOICE_NOTE
    elif request_type.replace("_", " ") == "summarise to bullets":
        return ConversionRequestType.SUMMARISE_TO_BULLETS
    elif request_type.replace("_", " ") == "convert to action points":
        return ConversionRequestType.CONVERT_TO_ACTION_POINTS
    elif request_type.replace("_", " ") == "short summary":
        return ConversionRequestType.SHORT_SUMMARY
    elif request_type.replace("_", " ") == "detailed summary":
        return ConversionRequestType.DETAILED_SUMMARY
    elif request_type.replace("_", " ") == "proof read":
        return ConversionRequestType.PROOF_READ
    elif request_type.replace("_", " ") == "save voicenote":
        return ConversionRequestType.FULL_TRANSCRIPT
    elif request_type.replace("_", " ") == "bespoke processing":
        return ConversionRequestType.BESPOKE_PROCESSING
    elif request_type.replace("_", " ") == "mind map":
        return ConversionRequestType.MIND_MAP
    elif request_type.replace("_", " ") == "table":
        return ConversionRequestType.TABLE
    else:
        raise Exception(
            f"Unknown request type passed to conversionRequestTypeFromStr(request_type={request_type})")


def convert_voice_note_file_name_to_txt_file_name(audio_file_name: str, request_type: ConversionRequestType):
    if request_type == ConversionRequestType.FULL_TRANSCRIPT:
        return audio_file_name.replace(
            '.ogg', '_transcription.txt')
    elif request_type == ConversionRequestType.SUMMARISE_VOICE_NOTE:
        return audio_file_name.replace(
            '.ogg', '_summary.txt')
    elif request_type == ConversionRequestType.SUMMARISE_TO_BULLETS:
        return audio_file_name.replace(
            '.ogg', '_bullet_summary.txt')
    elif request_type == ConversionRequestType.CONVERT_TO_ACTION_POINTS:
        return audio_file_name.replace(
            '.ogg', '_action_points.txt')
    elif request_type == ConversionRequestType.SHORT_SUMMARY:
        return audio_file_name.replace(
            '.ogg', '_short_summary.txt')
    elif request_type == ConversionRequestType.DETAILED_SUMMARY:
        return audio_file_name.replace(
            '.ogg', '_detailed_summary.txt')
    elif request_type == ConversionRequestType.PROOF_READ:
        return audio_file_name.replace(
            '.ogg', '_proof_read.txt')
    elif request_type == ConversionRequestType.MIND_MAP:
        return audio_file_name.replace(
            '.ogg', '_mind_map.txt')
    elif request_type == ConversionRequestType.TABLE:
        return audio_file_name.replace(
            '.ogg', '_table.txt')
    elif request_type == ConversionRequestType.BESPOKE_PROCESSING:
        return audio_file_name.replace(
            '.ogg', '_bespoke_processing.txt')
    else:
        raise Exception(
            f"Unknown request type passed to convert_voice_note_file_name_to_txt_file_name(request_type={request_type})")


# def is_valid_conversion_request_type(model: str) -> TypeGuard[conversion_request_type_alias]:
#     """
#     Check if the given string is a valid conversion_request_type_alias.

#     This function serves as a runtime type checker and a type guard for static type checkers.

#     Args:
#         model (str): The string to check.

#     Returns:
#         bool: True if the string is a valid model_type_alias, False otherwise.
#     """
#     valid_models = {"full transcript", "summarise voice note",
#                     "summarise to bullets", "convert to action points"}
#     return model in valid_models


def delete_local_file(file_path: str):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logging.debug(f"Deleted local file: {file_path}")
            return True
        except Exception as e:
            logging.debug(f"Error deleting local file: {e}")
            return False
    return True
