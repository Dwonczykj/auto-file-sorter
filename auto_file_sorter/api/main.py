import os
from dotenv import load_dotenv
import ngrok
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, confloat
from typing import List, Dict, Any, Optional, Annotated
import logging
import json
import asyncio

from auto_file_sorter.gmail_rule_daemon import GmailAutomation, GmailRuleEngine
from auto_file_sorter.db.gmail_db import GmailDatabase
from auto_file_sorter.ai_service import AIService
from auto_file_sorter.models.archive_decision import ArchiveDecisionOutput

from langchain.graphs import Neo4jGraph
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

app = FastAPI()
app.mount(
    "/static", StaticFiles(directory="auto_file_sorter/api/static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionOutput(BaseModel):
    """Model for structured action output"""
    type: str
    value: str
    confidence: Annotated[float, confloat(ge=0.0, le=1.0)]
    parameters: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Model for chat messages"""
    content: str
    role: str = "user"


class ChatResponse(BaseModel):
    """Model for chat responses"""
    message: str
    actions: Optional[List[ActionOutput]] = None


# Define action schemas
ACTION_SCHEMAS = [
    ResponseSchema(
        name="type",
        description="The type of action to perform (e.g., label, archive, delete)",
        type="string"
    ),
    ResponseSchema(
        name="value",
        description="The main value for the action (e.g., label name, email address)",
        type="string"
    ),
    ResponseSchema(
        name="confidence",
        description="Confidence score between 0 and 1",
        type="number"
    ),
    ResponseSchema(
        name="parameters",
        description="Optional additional parameters as a dictionary",
        type="object"
    )
]


class ChatSession:
    """Manages chat session state and components"""

    def __init__(self):
        self.db = GmailDatabase()
        self.ai_service = AIService.get_instance(model_name="gpt-4")
        self.gmail = GmailAutomation(
            credentials_path='path/to/credentials.json',
            token_path='path/to/token.json',
            ai_service=self.ai_service,
            db=self.db
        )
        self.rule_engine = GmailRuleEngine(self.gmail, 'email_rules.json')

        # Initialize RAG components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="thenlper/gte-small")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize knowledge base
        self._init_knowledge_base()

        # Initialize the parser
        self.action_parser = StructuredOutputParser.from_response_schemas(
            ACTION_SCHEMAS)

    def _init_knowledge_base(self):
        """Initialize RAG knowledge base"""
        # Add documentation about available actions
        docs = [
            "label: Apply labels to emails. Parameters: label_name",
            "archive: Move emails to archive. No parameters needed.",
            "delete: Move emails to trash. No parameters needed.",
            "markRead: Mark emails as read. No parameters needed.",
            "star: Star important emails. No parameters needed.",
            "forward: Forward emails. Parameters: to_email",
            "block_sender: Block specific sender. Parameters: email",
            "block_domain: Block entire domain. Parameters: domain",
            "block_body_pattern: Block emails matching pattern. Parameters: pattern",
            "auto_archive: Auto-archive based on rules. Parameters: criteria",
            "summarize: Generate email summaries. No parameters needed.",
            "auto_reply: Send automatic replies. Parameters: context",
            "save_attachment: Save email attachments. Parameters: save_path"
        ]

        self.vectorstore = FAISS.from_texts(
            docs,
            self.embeddings
        )

    async def process_message(self, message: str) -> ChatResponse:
        """Process incoming chat message and determine actions"""
        try:
            # Get relevant context from knowledge base
            context_docs = self.vectorstore.similarity_search(message, k=3)
            context = "\n".join(doc.page_content for doc in context_docs)

            # Get format instructions
            format_instructions = self.action_parser.get_format_instructions()

            # Create prompt with format instructions
            prompt = self._create_intent_prompt(
                message, context, format_instructions)
            completion = await self.ai_service.chat_completion(
                messages=[{
                    "role": "system",
                    "content": prompt
                }]
            )

            # Parse AI response into structured actions
            try:
                parsed_output = self.action_parser.parse(completion.response)
                actions = [ActionOutput(**action) for action in parsed_output]
            except Exception as e:
                logger.error(f"Failed to parse AI response: {
                             completion.response}\nError: {e}")
                actions = []

            # Create natural language rule if needed
            if actions:
                rule_id = self.db.create_nl_rule(
                    message, [a.dict() for a in actions])
                if not rule_id:
                    raise Exception("Failed to create natural language rule")

            # Generate response message
            response_msg = await self._generate_response(message, actions)

            return ChatResponse(
                message=response_msg,
                actions=actions
            )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _create_intent_prompt(self, message: str, context: str, format_instructions: str) -> str:
        """Create prompt for intent classification"""
        return f"""
        You are an AI assistant that helps users manage their Gmail account.
        Analyze the user message and determine what Gmail automation actions should be taken.

        Available actions and their usage:
        {context}

        {format_instructions}

        User message: {message}
        """

    async def _generate_response(self, message: str, actions: List[ActionOutput]) -> str:
        """Generate natural language response"""
        action_desc = "\n".join(
            [f"- {a.type}: {a.value} (confidence: {a.confidence:.2f})"
             for a in actions]
        )

        prompt = f"""
        Generate a natural, helpful response to the user explaining what actions will be taken.
        For each action, explain why it was chosen and what it will do.

        User message: {message}

        Actions to be taken:
        {action_desc}
        """

        completion = await self.ai_service.chat_completion(
            messages=[{
                "role": "system",
                "content": prompt
            }]
        )
        return completion.response


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # Initialize chat session
    session = ChatSession()

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = ChatMessage(**data)

            # Process message
            response = await session.process_message(message.content)

            # Send response
            await websocket.send_json(response.dict())

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    session = ChatSession()
    return await session.process_message(message.content)


def run_app():
    load_dotenv()
    port = int(os.environ.get('PORT', 8000))
    ngrok_authtoken = os.environ.get('NGROK_AUTHTOKEN', None)
    ngrok.set_auth_token(ngrok_authtoken)  # type: ignore
    ngrok_fixed_domain = os.environ.get(
        'NGROK_FIXED_DOMAIN', "foo.ngrok-free.app")
    ngrok_tunnel = ngrok.connect(port, domain=ngrok_fixed_domain)
    public_url = ngrok_tunnel.url()
    print('Public URL:', public_url)
    logging.info(f"ngrok tunnel \"{
        public_url}\" -> \"http://127.0.0.1:{port}\"")
