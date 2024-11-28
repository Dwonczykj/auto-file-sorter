from typing import Literal, Union


model_type_alias = Union[Literal["gpt-3.5-turbo"], Literal["gpt-4"],
                         Literal["gpt-4-turbo"], Literal["gpt-4o"], Literal["gpt-4o-latest"], Literal["whisper"]]
