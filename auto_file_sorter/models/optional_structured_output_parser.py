from __future__ import annotations
from enum import Enum
import json
import logging
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_and_check_json_markdown, _json_markdown_re, _json_strip_chars
from typing import Dict, Any, List, Optional

_json_response_key = "json_response_key"


class StructuredOutput:
    def __init__(self, raw_response: str, parsed_json: Dict[str, Any] | None, parser_succeeded: bool):
        self.raw_response = raw_response
        self.parsed_json = parsed_json
        self.llm_resp_str_to_dict_parser_succeeded = parser_succeeded
        self.json_response_key = parsed_json.get(
            _json_response_key) if parsed_json and _json_response_key in parsed_json else None
        self.total_tokens_used = -1

    def set_total_tokens_used(self, total_tokens_used: int):
        self.total_tokens_used = total_tokens_used
        return self


class OptionalStructuredOutputParser(StructuredOutputParser):
    """Parse the output of an LLM call to a structured output with optional fields."""

    required_keys: List[str]
    """The keys that are required in the response."""

    @classmethod
    def from_response_schemas(
        cls,
        response_schemas: List[ResponseSchema],
        required_keys: List[str] | None = None
    ) -> "OptionalStructuredOutputParser":
        """Create a parser from a list of response schemas."""
        parser = cls(
            response_schemas=response_schemas,
            required_keys=required_keys or []
        )
        return parser

    def parse(self, text: str) -> Dict[str, Any]:
        """Parse the output of an LLM call."""
        try:
            # Get all possible keys
            expected_keys = [rs.name for rs in self.response_schemas]
            # But only check for required ones
            keys_to_check = self.required_keys if self.required_keys else expected_keys
            return parse_and_check_json_markdown(text, keys_to_check)
        except OutputParserException as e:
            if self.required_keys:
                # Only raise if missing required keys
                raise
            # For optional keys, return whatever we got
            from langchain_core.utils.json import parse_json_markdown
            return parse_json_markdown(text)


class StructuredOutputConditionalOn(Enum):
    """The type of conditional response schema."""
    OneRequestType = "one_request_type"
    OneResultType = "one_result_type"


class ResponseSchemaWithPattern(ResponseSchema):
    """A response schema with a pattern."""
    pattern: Optional[str] = ""
    """The pattern to match the response against."""
    examples: Optional[list] = []
    """The examples to match the response against."""

    @classmethod
    def schema_to_format_instructions(cls, schemae: dict[str, dict[str, ResponseSchemaWithPattern]], json_response_key="json_response_key") -> str:
        """
        Convert a schema dictionary with patterns and examples to format instructions string.

        Args:
            schema: Dictionary mapping response types to their field schemas with patterns and examples

        Returns:
            Format instructions as a string
        """
        formats = []

        for response_type, fields in schemae.items():
            format_obj: dict[str, Any] = {
                json_response_key: response_type
            }
            for field_name, field_schema in fields.items():
                # Use the first example if available, otherwise use a default
                example = field_schema.examples[0] if field_schema.examples else (
                    {} if field_schema.type == "object" else "..."
                )
                format_obj[field_name] = example

            formats.append(format_obj)
        # Convert format objects to JSON strings with proper indentation
        json_examples = [
            json.dumps(format_obj, indent=4)
            for format_obj in formats
        ]

        # Join the formats with "OR" between them
        format_str = """Respond to: "{input}"

""" + """Format as:
```json
{}
```""".format("\nOR\n".join(json_examples)).replace("{", "{{").replace("}", "}}")

    #     format_str = """Format as:
    # ```json
    # {
    #     "json_response_key": "template_response",
    #     "template_sid": "HX...",
    #     "content_variables": {...}
    # }
    # OR
    # {
    #     "json_response_key": "custom_response",
    #     "custom_response": "message"
    # }
    # ```"""

        return format_str


class ConditionalResponseSchema(ResponseSchema):
    """A response schema that is conditional on the value of another field."""
    conditional_on: str
    """The field to condition on."""
    conditional_type: StructuredOutputConditionalOn
    """The type of conditional response schema."""


class ConditionalResponseSchemaList():
    """A list of conditional response schemas."""

    def __init__(self,
                 output_schemae: Dict[str, Dict[str, ResponseSchema]],
                 #  condition_on_schema: Optional[list[ResponseSchema]] = []
                 ):
        self.sub_parsers = {
            output_schema_name: StructuredOutputParser.from_response_schemas(
                list(output_schema.values()))
            for output_schema_name, output_schema in output_schemae.items()
        }
        self.required_sub_schema = StructuredOutputParser.from_response_schemas([ResponseSchema(
            name=_json_response_key, type="str", description=f"The name of the key for this markdown code snippet")])
        condition_on_schema = [
            [
                ResponseSchema(
                    name=_json_response_key, type="str", description=f"The name of the key for this markdown code snippet which is '{k}'. This is used to to apply the correct schema to parse the response with."),
                *output_schemae[k].values(),
                # ResponseSchema(
                #     name="should_use_template", type="bool", description="Whether to use a predefined template"),
            ]
            for k in output_schemae.keys()
        ]
        # if condition_on_schema:
        #     required_sub_schema = condition_on_schema
        # else:
        #     required_sub_schema: list[ResponseSchema] = [ResponseSchema(name="request_type", description=f"The type of request being made which is always one of the following: {
        #         ', '.join(sub_parsers.keys())}", type="string")]
        # required_sub_parser = StructuredOutputParser.from_response_schemas(required_sub_schema)
        format_instructions_manual = f"The output should be ONE of the following markdown code snippets from the following list, including the leading and trailing trailing \"```json\" and \"```\":\n[\n"
        # for k, schema in output_schemae.items():
        #     add_schemae = list(schema.values()) + required_sub_schema
        #     # format_instructions_manual += f'\t"{k}": '
        #     format_instructions_manual += StructuredOutputParser.from_response_schemas(
        #         add_schemae).get_format_instructions(only_json=True)
        for schemae in condition_on_schema:
            format_instructions_manual += StructuredOutputParser.from_response_schemas(
                schemae).get_format_instructions(only_json=True)
        format_instructions_manual += "\n]\n"
        format_instructions_manual += f"Each snippet needs to include the following json keys [{
            ', '.join([f"[\"{rs.name}\" which is one of the following: {', '.join([f"\"{k}\"" for k in self.sub_parsers.keys()])}]" for rs in self.required_sub_schema.response_schemas])}] to allow me to differentiate between the markdown snippets in your response."
        self.format_instructions = format_instructions_manual

    def parse_response_type(self, response: str) -> StructuredOutput:
        try:
            required = self.required_sub_schema.parse(response)
            if required and required.get(_json_response_key):
                request_was = required.get(_json_response_key)
                if request_was and request_was in self.sub_parsers.keys():
                    sub_parser = self.sub_parsers[request_was]
                    return StructuredOutput(raw_response=response, parsed_json=sub_parser.parse(response), parser_succeeded=True)
            return StructuredOutput(raw_response=response, parsed_json=None, parser_succeeded=False)
        except Exception as e:
            logging.warning(f"structured_output: error parsing response: {
                            e}\nresponse: {response} with parsers: [{
                                '],\n['.join(
                                    [rs.name for rs in self.required_sub_schema.response_schemas])
            }]\nThe format instructions are: \n\"{self.format_instructions}\"")
            return StructuredOutput(raw_response=response, parsed_json=None, parser_succeeded=False)
