from dataclasses import dataclass
from datetime import datetime
from typing import BinaryIO, Literal, Optional
import httpx
from openai.types.image import Image
from openai.types.chat import ChatCompletionMessageParam
import pytz


@dataclass
class ImageGenerationResponse:
    url: str
    cost: float
    image: Image

    def __str__(self):
        return f"ImageGenerationResponse(url={self.url}, cost={self.cost}, image={self.image})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.url == other.url and self.cost == other.cost

    async def get_img_bytes(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
            return response.content

    @property
    def b64_json(self) -> str | None:
        '''The base64-encoded JSON of the generated image, if response_format is b64_json.'''
        return self.image.b64_json


@dataclass
class TranscribeAudioResponse:
    text: str
    cost: float


@dataclass
class ChatCompletionMessageInput:
    role: Literal["user", "assistant", "function", "tool", "system"]
    content: str
    timestamp: datetime = datetime.now(pytz.utc).replace(tzinfo=None)
    audio: Optional[BinaryIO | str] = None
    name: Optional[str] = None
    function_call_name: str = ''
    tool_call_id: str = ''

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "audio": self.audio if self.audio else "",
            "name": self.name if self.name else "",
            "function_call_name": self.function_call_name if self.function_call_name else "",
            "tool_call_id": self.tool_call_id if self.tool_call_id else ""
        }

    def to_open_ai_schema(self) -> ChatCompletionMessageParam:
        if self.role == "user":
            return {"role": "user", "content": self.content}
        elif self.role == "assistant":
            return {"role": "assistant", "content": self.content}
        elif self.role == "function":
            return {"role": "function", "content": self.content, "name": self.function_call_name}
        elif self.role == "tool":
            return {"role": "tool", "content": self.content, "tool_call_id": self.tool_call_id}
        else:
            return {"role": "system", "content": self.content}
