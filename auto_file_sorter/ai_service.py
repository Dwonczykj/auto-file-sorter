from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import os
import io
import time
from typing import Callable, Literal, Optional, List, TypeVar, Union, Dict, Any
import typing
from langchain_openai import ChatOpenAI, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import BaseMessage, FunctionMessage, ToolMessage
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
import logging
import openai
import requests
from openai.types.chat import ChatCompletion as OpenAIChatCompletion
from openai.types import Completion as OpenAICompletion
from openai.types.audio import Transcription as OpenAITranscription
from openai.types.images_response import ImagesResponse as OpenAIImageResponse
from typing import BinaryIO
from openai import OpenAI as DirectOpenAI  # For direct API calls when needed
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
import audio_helpers
from auto_file_sorter.config import Config
from auto_file_sorter.models.optional_structured_output_parser import StructuredOutput
from models.temporary_file_from_bytes import TemporaryFileFromBytes, TemporaryFileFromStr
from ai_service_types import ImageGenerationResponse, TranscribeAudioResponse, ChatCompletionMessageInput

# TODO: I need to rewrite the ai_service and types based on what I now know in the @hugging_face_agent_tutorials dir.
# TODO: the AI service should be a simpler graphed agentic workflow that calls the right tools based on the request, we can then have different services based on if I need multimodal outputs or I can leverage the notebooks on how to generate multimodal ouptut with langgraph.
# TODO: All we need right now is prompts with RAG from sql, email html and user text.

T = TypeVar('T', requests.Response, OpenAIChatCompletion,
            OpenAITranscription, OpenAIImageResponse, OpenAICompletion)


def log_request_details(response: Callable[[], T]) -> T:
    from logging.logging_config import logging, add_file_logger
    logger = add_file_logger(name='openai_api_usage', log_level=logging.INFO)
    if isinstance(response, Callable):
        max_retries = 5
        backoff_factor = 2
        attempt = 0
        passed = False
        for attempt in range(max_retries):
            try:
                response_called = response()
                passed = True
                break

            except openai.RateLimitError as e:
                response_called = e.response
                if isinstance(response_called, requests.Response):
                    headers = response_called.headers
                    limit = headers.get('x-ratelimit-limit-requests')
                    remaining = headers.get('x-ratelimit-remaining-requests')

                    logger.info(f"Request made at {time.time()}")
                    logger.info(f"Rate Limit: {limit}")
                    logger.info(f"Remaining: {remaining}")
                    raise e
                if attempt == max_retries - 1:
                    raise e
                wait_time = backoff_factor ** attempt
                logger.info(f"Rate limit hit. Retrying in {
                            wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                raise e
        if not passed:
            raise Exception("log_request_details: max retries reached")
        # Add type assertion to ensure response_called matches type T
        response_called = typing.cast(T, response_called)

        assert response_called is not None, "log_request_details: response_called is None"
        try:
            if isinstance(response_called, requests.Response):
                headers = response_called.headers
                limit = headers.get('x-ratelimit-limit-requests')
                remaining = headers.get('x-ratelimit-remaining-requests')

                logger.info(f"Request made at {time.time()}")
                logger.info(f"Rate Limit: {limit}")
                logger.info(f"Remaining: {remaining}")
                return response_called
            assert isinstance(response_called, OpenAIChatCompletion | OpenAITranscription | OpenAIImageResponse |
                              OpenAICompletion), "log_request_details: response_called is of type Callable but the response_called is not an OpenAI response_called type"
            if isinstance(response_called, OpenAIChatCompletion):
                usage = response_called.usage
                if not usage:
                    logger.error("log_request_details: usage is None")
                    logger.info(response_called)
                    return response_called
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                # service_tier = response_called.service_tier
                headers = getattr(response_called, 'headers') if hasattr(
                    response_called, 'headers') else None
                if headers:
                    limit = headers.get('x-ratelimit-limit-requests')
                    remaining = headers.get('x-ratelimit-remaining-requests')
                    reset_time = headers.get('x-ratelimit-reset-requests')

                    logger.info(f"Rate Limit: {limit}")
                    logger.info(f"Remaining: {remaining}")
                    logger.info(f"Resets in: {reset_time} seconds")
                else:
                    logger.info(f"No headers found in response_called")
                limit = 'N/A'
                remaining = 'N/A'
                logger.info(f"Request made to openai [{response_called.model}] on tier [{
                    response_called.service_tier}] at {time.time()}")
                logger.info(f"Rate Limit: {limit}")
                logger.info(f"Remaining: {remaining}")
                logger.info(f"Prompt Tokens: {prompt_tokens}")
                logger.info(f"Completion Tokens: {completion_tokens}")
                logger.info(f"Total Tokens: {total_tokens}")
            elif isinstance(response_called, OpenAIImageResponse):
                logger.info(
                    f"Request made to openai image generation at {time.time()}")
            elif isinstance(response_called, OpenAITranscription):
                logger.info(
                    f"Request made to openai transcription at {time.time()}")
                logger.info(f"Model: {response_called.model_fields}")
            elif isinstance(response_called, OpenAICompletion):
                logger.info(
                    f"Request made to openai completion at {time.time()}")
                logger.info(f"Model: {response_called.model_fields}")
            else:
                logger.error(f"log_request_details: response_called is of type {
                    type(response_called)}")
        finally:
            return response_called
    else:
        logger.error(f"log_request_details: response is of type {
                     type(response)}")
        return response
# After making a request, call this function:
# log_request_details(response)


@dataclass
class _ChatCompletionResponse():
    response: str
    total_tokens_used: int


class AIService:
    _instance = None

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self._model_name = model_name
        self._chat_model = None
        self._completion_model = None
        self._direct_client = None

    @property
    def direct_client(self):
        if not self._direct_client:
            self._direct_client = DirectOpenAI(api_key=Config.OPENAI_API_KEY)
        return self._direct_client

    @property
    def chat_model(self):
        if not self._chat_model:
            if "gpt" in self._model_name.lower():
                self._chat_model = ChatOpenAI(
                    name=self._model_name,
                    temperature=0.7,
                )
            else:
                self._chat_model = ChatAnthropic(
                    model_name=self._model_name,
                    # anthropic_api_key=Config.ANTHROPIC_API_KEY,
                    timeout=10,
                    stop=["\n\n"],
                )
        return self._chat_model

    @property
    def completion_model(self):
        if not self._completion_model:
            if "gpt" in self._model_name.lower():
                self._completion_model = OpenAI(
                    name=self._model_name,
                    temperature=0.7,
                )
            else:
                self._completion_model = self.chat_model
        return self._completion_model

    async def _messages_to_langchain_format(self, messages: List[ChatCompletionMessageInput], system_message: Optional[str] = None) -> List[list[BaseMessage]]:
        langchain_messages: List[BaseMessage] = []

        if system_message:
            langchain_messages.append(SystemMessage(content=system_message))

        for message in messages:
            if message.role == "user":
                langchain_messages.append(
                    HumanMessage(content=message.content))
            elif message.role == "assistant":
                if message.audio:
                    logging.warning(f"Audio messages are not supported in AI chat completions. Falling back to transcription using ai_service.transcribe_audio_direct(). {
                                    message.content}", stacklevel=0)
                    transcription = "\n\n"
                    transcription += await self.transcribe_audio_direct(audio_file=message.audio)
                    # if isinstance(message.audio, str) and audio_helpers.audio_file_has_supported_extension(message.audio):
                    #     transcription += await self.transcribe_audio(audio_file=message.audio, audio_file_type=message.audio.split(".")[-1])
                    content_and_audio = f"{
                        message.content}\nThe following audio transcript was provided as context to this message:{transcription}"
                    append = AIMessage(content=content_and_audio)
                    # append = AIMessage(content=message.content, audio=message.audio)
                else:
                    append = AIMessage(content=message.content)
                langchain_messages.append(append)
            elif message.role == "system":
                langchain_messages.append(
                    SystemMessage(content=message.content))
            elif message.role == "function":
                langchain_messages.append(
                    FunctionMessage(content=message.content, name=message.function_call_name))
            elif message.role == "tool":
                langchain_messages.append(
                    ToolMessage(content=message.content, tool_call_id=message.tool_call_id))
            else:
                raise ValueError(f"Unsupported message role: {
                                 message.role}")
        return [langchain_messages]

    async def chat_completion(
        self,
        messages: List[ChatCompletionMessageInput],
        system_message: Optional[str] = None
    ) -> _ChatCompletionResponse:
        """Convert messages to LangChain format and get completion"""
        if not self.chat_model:
            raise ValueError("Chat model not initialized")

        langchain_messages = await self._messages_to_langchain_format(
            messages, system_message)

        with get_openai_callback() as cb:
            response = await self.chat_model.agenerate(langchain_messages)
            logging.debug(f"Total Tokens: {
                          cb.total_tokens} [{self._model_name}]")
            logging.debug(f"Prompt Tokens: {
                          cb.prompt_tokens} [{self._model_name}]")
            logging.debug(f"Completion Tokens: {
                          cb.completion_tokens} [{self._model_name}]")
            total_tokens_used = cb.total_tokens

        return _ChatCompletionResponse(response.generations[0][0].text, total_tokens_used=total_tokens_used)

    async def completion(self, prompt: str, max_tokens: int = 50) -> str:
        """Get completion from the model"""
        with get_openai_callback() as cb:
            prompts = [prompt]
            # messages: list[list[BaseMessage]] = [
            #     [HumanMessage(content=prompt)]]
            response = await self.completion_model.agenerate(
                prompts=prompts, max_tokens=max_tokens or 50)  # type: ignore
            logging.debug(f"Total Tokens: {cb.total_tokens}")

        return response.generations[0][0].text

    async def structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, ResponseSchema],
        message_history: List[ChatCompletionMessageInput] = [],
        system_message: str = '',
    ) -> StructuredOutput:
        """Get structured output based on schema
        prompt: `str`
        messages: `list[ChatCompletionMessageInput]`
        output_schema: `dict[str, langchain.output_parsers.ResponseSchema]` objects
        system_message: `str`
        Returns: `dict[str, Any]`
        """
        # Convert dict to list of ResponseSchema
        schema_list = list(output_schema.values())
        parser = StructuredOutputParser.from_response_schemas(schema_list)
        # parser = parser.configurable_alternatives(...)
        # parser = parser.with_types(...)
        format_instructions = parser.get_format_instructions(only_json=False)
        user_message_content = "{input}\n\n{format_instructions}"
        langchain_messages = await self._messages_to_langchain_format(
            [
                *message_history,
                ChatCompletionMessageInput(
                    role="user", content=user_message_content),
            ],
            system_message=f"You are a helpful assistant that outputs structured data.\n{
                system_message}. The current time is {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}."
        )
        named_message_arguments_for_prompt_template = {
            "input": f"Apply the following format instructions to the following input: \"{prompt}\"",
            "format_instructions": format_instructions
        }
        if Config.DEBUG:
            for prompt_named_key in named_message_arguments_for_prompt_template:
                assert prompt_named_key in user_message_content, f"Prompt named key {
                    prompt_named_key} not found in user_message_content"
        # prompt_template = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful assistant that outputs structured data."),
        #     # Use named placeholders
        #     ("user", "{input}\n\n{format_instructions}")
        # ])
        prompt_template = ChatPromptTemplate.from_messages(
            [("user", str(m.content)) if isinstance(m, HumanMessage) else ("system", str(m.content)) if isinstance(m, SystemMessage) else ("assistant", str(m.content)) for m in langchain_messages[0]])
        # prompt_template = ChatPromptTemplate.from_messages(
        #     langchain_messages[0])
        with get_openai_callback() as cb:
            chain = LLMChain(llm=self.chat_model, prompt=prompt_template)
            # Pass the input variables as a dictionary
            # response = await chain.arun(input=prompt, format_instructions=format_instructions)
            response = await chain.arun(**named_message_arguments_for_prompt_template, callbacks=None, tags=None, metadata=None)
            logging.debug(f"Total Tokens: {cb.total_tokens}")
        # TODO: Save the conversation to user converstation history in the calling method?
        try:
            return StructuredOutput(raw_response=response, parsed_json=parser.parse(response), parser_succeeded=True)
        except Exception as e:
            logging.warning(f"structured_output: error parsing response: {
                            e}\nresponse: {response}")
            return StructuredOutput(raw_response=response, parsed_json=None, parser_succeeded=False)

    async def from_prompt(
        self,
        prompt: ChatPromptTemplate,
        # user_message: str,
        **kwargs,
    ) -> str:
        total_tokens_used = 0
        with get_openai_callback() as cb:
            chain = LLMChain(llm=self.chat_model, prompt=prompt)
            # Pass the input variables as a dictionary
            # response = await chain.arun(input=prompt, format_instructions=format_instructions)
            try:
                response = await chain.arun(**kwargs, callbacks=None, tags=None, metadata=None)
            except Exception as e:
                logging.error(f"structured_output_conditional: error parsing response: {
                    e}")
                raise e
            logging.debug(f"Total Tokens: {cb.total_tokens}")
            total_tokens_used = cb.total_tokens
        return response

    async def generate_image(
        self,
        prompt: str,
        size: Literal['1024x1024', '1792x1024',
                      '1024x1792'] | Literal['256x256', '512x512', '1024x1024'] | None = "1024x1024",
        quality: Literal['standard', 'hd'] = "standard",
        n: int = 1,
        model: Optional[str] = None
    ) -> list[ImageGenerationResponse]:
        """Generate images using DALL-E through direct OpenAI API"""
        try:
            model = model or "dall-e-3"

            # Validate size based on model
            if model == "dall-e-3":
                if size not in ["1024x1024", "1792x1024", "1024x1792"]:
                    size = "1024x1024"  # Default for DALL-E 3
            elif model == "dall-e-2":
                if size not in ["256x256", "512x512", "1024x1024"]:
                    size = "1024x1024"  # Default for DALL-E 2

            with get_openai_callback() as cb:
                def response_lambda(): return self.direct_client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=n,
                )
                response = log_request_details(response_lambda)
                logging.info(f"Image generation cost: ${cb.total_cost}")

            # Return list of image URLs
            return [ImageGenerationResponse(url=img.url, cost=cb.total_cost, image=img) for img in response.data if img.url]
        except Exception as e:
            logging.error(f"Error generating image: {e}")
            raise

    async def transcribe_audio(
        self,
        audio_file: Union[str, BinaryIO, bytes],
        audio_file_type: audio_helpers.AudioFileAcceptedFormat,
        model: Optional[str] = None,
        response_format: Literal["text", "srt", "vtt"] = "text",
    ) -> str:
        """See whatsapp_transcriber_venv/lib/python3.12/site-packages/langchain_openai/chat_models/base.py:132"""
        if not audio_helpers.AudioFileAcceptedFormatTypeGuard(audio_file_type):
            raise ValueError(f"Unsupported audio file type: {
                             audio_file_type}")
        if isinstance(audio_file, str) and audio_file.startswith("https://storage.googleapis.com/"):
            # gcs_service = Services.gcs_service()
            # with TemporaryFileFromGoogleStorage(gcs_service, audio_file) as temp_file:
            #     audio_file = temp_file
            with requests.get(audio_file) as response:
                audio_file_bytes = response.content
            with TemporaryFileFromBytes(audio_file_bytes, audio_file_type) as temp_file:
                audio_file = temp_file
        elif isinstance(audio_file, str) and len(audio_file) < 150 and len(audio_file.strip()) > 0 and os.path.exists(audio_file):
            with open(audio_file, 'rb') as f:
                audio_file_bytes = f.read()
            with TemporaryFileFromBytes(audio_file_bytes, audio_file_type) as temp_file:
                audio_file = temp_file
        # elif isinstance(audio_file, str):
        #     with TemporaryFileFromStr(audio_file, audio_file_type) as temp_file:
        #         audio_file = temp_file
        elif isinstance(audio_file, bytes):
            with TemporaryFileFromBytes(audio_file, audio_file_type) as temp_file:
                audio_file = temp_file
        elif isinstance(audio_file, io.BytesIO):
            with TemporaryFileFromBytes(audio_file.getvalue(), audio_file_type) as temp_file:
                audio_file = temp_file
        elif isinstance(audio_file, io.BufferedReader):
            with TemporaryFileFromBytes(audio_file.read(), audio_file_type) as temp_file:
                audio_file = temp_file
        elif isinstance(audio_file, BinaryIO):
            with TemporaryFileFromBytes(audio_file.read(), audio_file_type) as temp_file:
                audio_file = temp_file
        else:
            raise ValueError(f"Unsupported audio file type: {
                             type(audio_file)}")
        response = await self.chat_completion(
            messages=[
                ChatCompletionMessageInput(
                    role="assistant", audio=audio_file, content="Please transcribe the attached audio file.")
            ],
        )
        logging.info(response)
        return response.response

    async def transcribe_audio_direct(
        self,
        audio_file: Union[str, BinaryIO],
        model: Optional[str] = None,
        response_format: Literal["text", "srt", "vtt"] = "text",
    ) -> str:
        """Transcribe audio using Whisper through direct OpenAI API"""
        try:
            with get_openai_callback() as cb:
                # Handle both file path strings and file objects
                if isinstance(audio_file, str):
                    # If it's a file path, open and read the file
                    with open(audio_file, 'rb') as f:
                        response = self.direct_client.audio.transcriptions.create(
                            file=f,
                            model=model or "whisper-1",
                            response_format=response_format
                        )
                else:
                    # If it's already a file object, use it directly
                    response = self.direct_client.audio.transcriptions.create(
                        file=audio_file,
                        model=model or "whisper-1",
                        response_format=response_format
                    )
                logging.info(f"Transcription cost: ${cb.total_cost}")

            return str(response)
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            raise

    # Add methods for other AI operations (transcription, image generation, etc.)
    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls, model_name="gpt-4o") -> 'AIService':
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance


class AIServiceFactory:
    @staticmethod
    def create_ai_service(model_name: str = "gpt-4o-mini") -> AIService:
        return AIService(model_name)


class TestImageGeneration:
    """https://python.langchain.com/docs/integrations/tools/"""

    def test_generate_image(self, ai_service: AIService):
        """Test image generation"""
        from langchain.chains import LLMChain
        from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI
        llm = OpenAI(temperature=0.9)
        prompt = PromptTemplate(
            input_variables=["image_desc"],
            template="Generate a detailed prompt to generate an image based on the following description: {image_desc}",
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        image_url = DallEAPIWrapper().run(chain.run("halloween night at a haunted museum"))
        logging.info(f"DALL-E Image URL: {image_url}")
        return image_url

    def test_generate_image_with_agent(self):
        from langchain.agents import initialize_agent, load_tools, AgentType

        tools = load_tools(["dalle-image-generator"])
        llm = OpenAI(temperature=0.9)
        agent = initialize_agent(
            tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        output = agent.run(
            "Create an image of a halloween night at a haunted museum")
        logging.info(f"DALL-E Image URL: {output}")
        return output

    def test_generate_speech_eleven_labs(self):
        """https://python.langchain.com/docs/integrations/tools/eleven_labs_tts/"""
        from langchain_community.tools import ElevenLabsText2SpeechTool
        import os
        assert os.environ.get("ELEVEN_API_KEY") and os.environ.get(
            "ELEVEN_API_KEY") != "needs-api-key"

        text_to_speak = "Hello world! I am the real slim shady"

        tts = ElevenLabsText2SpeechTool()
        logging.info(f"Eleven Labs TTS Tool: {tts.name}")
        speech_file = tts.run(text_to_speak)
        tts.play(speech_file)
        # or stream audio directly:
        # tts.stream_speech(text_to_speak)
        return

    def test_generate_speech_eleven_labs_agent(self):
        from langchain.agents import AgentType, initialize_agent, load_tools
        from langchain_openai import OpenAI
        from langchain_community.tools import ElevenLabsText2SpeechTool
        llm = OpenAI(temperature=0)
        tts = ElevenLabsText2SpeechTool()
        tools = load_tools(["eleven_labs_text2speech"])
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
        audio_file = agent.run("Tell me a joke and read it out for me.")
        tts.play(audio_file)

    def test_transcribe_audio_google(self, audio_file: Union[str, BinaryIO]):
        """https://python.langchain.com/docs/integrations/document_loaders/google_speech_to_text/#related"""
        from langchain.document_loaders import GoogleSpeechToTextLoader
        if isinstance(audio_file, str):
            with TemporaryFileFromStr(audio_file, ".mp3") as temp_file:
                loader = GoogleSpeechToTextLoader(
                    project_id="", file_path=temp_file)
        elif isinstance(audio_file, BinaryIO):
            with TemporaryFileFromBytes(audio_file.read(), ".mp3") as temp_file:
                loader = GoogleSpeechToTextLoader(
                    project_id="", file_path=temp_file)
        else:
            raise ValueError(f"Unsupported audio file type: {
                             type(audio_file)}")
        docs = loader.load()
        logging.info(docs)
        pass

    def test_url_selenium_llm(self):
        """https://python.langchain.com/docs/integrations/document_loaders/google_speech_to_text/#related"""
        from langchain.document_loaders.url_selenium import SeleniumURLLoader
        loader = SeleniumURLLoader(urls=["https://www.google.com"])
        docs = loader.load()
        logging.info(docs)
        pass

    def test_notion_llm(self):
        """https://python.langchain.com/docs/integrations/document_loaders/google_speech_to_text/#related"""
        from langchain.document_loaders.notion import NotionLoader
        loader = NotionLoader(
            "whatsapp_chat.txt")
        docs = loader.load()
        logging.info(docs)
        pass

    def test_whatsapp_llm(self):
        """https://python.langchain.com/docs/integrations/document_loaders/google_speech_to_text/#related"""
        from langchain.document_loaders.whatsapp_chat import WhatsappChatLoader
        loader = WhatsappChatLoader(
            "whatsapp_chat.txt")
        docs = loader.load()
        logging.info(docs)
        pass

    def test_employ_these(self):
        """
        https://python.langchain.com/docs/integrations/tools/memorize/
        https://python.langchain.com/docs/integrations/tools/serpapi/
        https://python.langchain.com/docs/integrations/tools/twilio/
        """
        ...
