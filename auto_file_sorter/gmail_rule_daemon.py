from dataclasses import dataclass, field
from datetime import datetime, timedelta
import base64
import os
import re
import logging
from pathlib import Path
from typing import List, Optional, Set, BinaryIO, Any, Dict, Callable, Tuple
from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
import fpdf
import time
import json
import httpx
import csv
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from auto_file_sorter.gmail_service_types import GmailServiceProtocol

from ai_service import AIService, ChatCompletionMessageInput
from auto_file_sorter.logging.logging_config import configure_logging
from auto_file_sorter.auth_base import GoogleServiceAuth
from auto_file_sorter.db.gmail_db import GmailDatabase
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from auto_file_sorter.models.unsubscribe_link import UnsubscribeLinkOutput
from email_validator import validate_email, EmailNotValidError
from auto_file_sorter.models.archive_decision import ArchiveDecisionOutput

configure_logging()


@dataclass
class UnreadTracker:
    sender: str
    subject: str
    timestamp: datetime
    days_unread: int


@dataclass
class EmailRule:
    name: str
    # e.g., {'from': 'example@gmail.com', 'subject': '.*invoice.*'}
    conditions: Dict[str, str]
    # e.g., [{'type': 'label', 'value': 'Invoices'}, {'type': 'save_attachment'}]
    actions: List[Dict[str, Any]]


@dataclass
class GmailFilterSize:
    greaterThan: bool
    sizeInMB: Optional[float]


@dataclass
class GmailFilterAction:
    delete: bool = False
    archive: bool = False
    markAsRead: bool = False
    star: bool = False
    label: str = ""
    forwardTo: str = ""


@dataclass
class NLRuleOutput:
    id: int
    name: str


@dataclass
class GmailFilter:
    from_: str = ""  # Using from_ to avoid Python keyword
    to: str = ""
    subject: str = ""
    hasWords: str = ""
    doesNotHaveWords: str = ""
    size: GmailFilterSize = field(
        default_factory=lambda: GmailFilterSize(False, None))
    hasAttachment: bool = False
    includeChats: bool = False
    action: GmailFilterAction = field(default_factory=GmailFilterAction)


# Define response schemas
NL_RULE_SCHEMAS = [
    ResponseSchema(
        name="id",
        description="List of rule IDs that match the email content",
        type="list"
    ),
    ResponseSchema(
        name="name",
        description="Name of the matched rule",
        type="string"
    )
]


class GmailAutomation(GoogleServiceAuth):
    """Class to handle Gmail automation tasks with AI integration"""

    service: GmailServiceProtocol

    @property
    def SCOPES(self) -> List[str]:
        return [
            'https://www.googleapis.com/auth/gmail.modify',
            'https://www.googleapis.com/auth/gmail.settings.basic',
            'https://www.googleapis.com/auth/gmail.settings.sharing'
        ]

    def __init__(self, credentials_path: str, token_path: str, ai_service: AIService, db: GmailDatabase):
        """Initialize Gmail automation with OAuth2 credentials and AI service"""
        super().__init__(credentials_path, token_path)
        self.ai_service = ai_service
        self.unread_tracking: Set[UnreadTracker] = set()
        self.db = db
        self.authenticate()
        # Initialize the parser
        self.nl_rule_parser = StructuredOutputParser.from_response_schemas(
            NL_RULE_SCHEMAS)

    def _build_service(self, credentials: Credentials) -> GmailServiceProtocol:
        """Build the Gmail API service"""
        service = build('gmail', 'v1', credentials=credentials)
        assert isinstance(service, GmailServiceProtocol)
        return service

    async def summarize_email(self, message_id: str) -> str:
        """Summarize email content using AI service"""
        try:
            # Get email content
            message = self.service.users().messages().get(
                # TODO: Fix this linter error
                userId='me', id=message_id, format='full').execute()

            # Extract email body
            body = ""
            if 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']).decode('utf-8')
            elif 'parts' in message['payload']:
                parts = message['payload']['parts']
                body = base64.urlsafe_b64decode(
                    parts[0]['body']['data']).decode('utf-8') if 'data' in parts[0]['body'] else ""
            else:
                body = ""
                return ""

            # Clean HTML if present
            soup = BeautifulSoup(body, 'html.parser')
            clean_text = soup.get_text()

            # Get summary using AI service
            completion = await self.ai_service.chat_completion(
                messages=[
                    ChatCompletionMessageInput(
                        role="system",
                        content="Please provide a concise summary of the following email:"
                    ),
                    ChatCompletionMessageInput(
                        role="user",
                        content=clean_text
                    )
                ]
            )

            return completion.response

        except HttpError as error:
            logging.error(f'An error occurred: {error}')
            return ""

    async def auto_reply(self, message_id: str, context: str = "", send_immediately: bool = False) -> None:
        """Generate and send an automatic reply using AI"""
        try:
            # Get the original message details
            msg = self.service.users().messages().get(
                userId='me', id=message_id).execute()
            thread_id = msg['threadId']

            # Create reply message
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='full').execute()
            headers = message['payload']['headers']
            subject = next(h['value']
                           for h in headers if h['name'] == 'Subject')
            from_email = next(h['value']
                              for h in headers if h['name'] == 'From')

            subject = f"Message ID: {message_id} [NO SUBJECT]"
            if 'subject' in message['payload']:
                subject = message['payload']['subject']
            else:
                logging.warning(
                    f"No subject found in message payload.")

            # Get email body for context
            if 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']).decode('utf-8')
            elif 'parts' in message['payload']:
                parts = message['payload']['parts']
                body = base64.urlsafe_b64decode(
                    parts[0]['body']['data']).decode('utf-8') if 'data' in parts[0]['body'] else ""
            else:
                body = ""
                logging.warning(
                    f"No body found in message payload with subject: {subject}. Returning early.")
                return

            # Clean HTML
            soup = BeautifulSoup(body, 'html.parser')
            clean_text = soup.get_text()

            # Generate reply using AI
            completion = await self.ai_service.chat_completion(
                messages=[
                    ChatCompletionMessageInput(
                        role="system",
                        content=f"Generate a professional email reply. Additional context: {
                            context}"
                    ),
                    ChatCompletionMessageInput(
                        role="user",
                        content=f"Original email:\n{clean_text}"
                    )
                ]
            )

            reply_text = completion.response

            reply_message = f"""From: me
To: {from_email}
Subject: Re: {subject}

{reply_text}"""

            # Encode and send the reply
            encoded_message = base64.urlsafe_b64encode(
                reply_message.encode('utf-8')).decode('utf-8')

            if send_immediately:
                self.service.users().messages().send(
                    userId='me',
                    body={
                        'raw': encoded_message,
                        'threadId': thread_id
                    }
                ).execute()
                logging.info(
                    f"Sent reply to message {message_id} with subject {subject}")
            else:
                self.service.users().drafts().create(
                    userId='me',
                    body={
                        'message': {
                            'raw': encoded_message,
                            'threadId': thread_id
                        }
                    }
                ).execute()
                logging.info(
                    f"Drafted reply to message {message_id} with subject {subject}")

        except HttpError as error:
            logging.error(f'An error occurred: {error}')

    async def apply_label(self, message_ids: List[str], label_name: str) -> None:
        """Apply a label to specified messages"""
        try:
            # Create label if it doesn't exist
            labels = self.service.users().labels().list(userId='me').execute()
            label_id = None

            for label in labels['labels']:
                if label['name'] == label_name:
                    label_id = label['id']
                    break

            if not label_id:
                label_body = {
                    'name': label_name,
                    'labelListVisibility': 'labelShow',
                    'messageListVisibility': 'show'
                }
                created_label = self.service.users().labels().create(
                    userId='me', body=label_body).execute()
                label_id = created_label['id']

            body = {'addLabelIds': [label_id], 'removeLabelIds': []}
            self.service.users().messages().batchModify(
                userId='me', body=body, ids=message_ids).execute()
            logging.info(
                f"Applied label {label_name} to messages {message_ids}")

        except HttpError as error:
            logging.error(f'An error occurred: {error}')

    async def save_attachments(
        self,
        sender_pattern: str,
        subject_pattern: str,
        save_path: Path
    ) -> None:
        """Save attachments from emails matching patterns"""
        try:
            query = f"from:({sender_pattern}) subject:({
                subject_pattern}) has:attachment"
            messages = self.service.users().messages().list(
                userId='me', q=query).execute()

            if 'messages' not in messages:
                return

            for message in messages['messages']:
                msg = self.service.users().messages().get(
                    userId='me', id=message['id']).execute()

                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if 'filename' in part and part['filename']:
                            attachment_id = part['body']['attachmentId']
                            attachment = self.service.users().messages().attachments().get(
                                userId='me', messageId=message['id'], id=attachment_id
                            ).execute()

                            file_data = base64.urlsafe_b64decode(
                                attachment['data'].encode('UTF-8'))

                            filepath = save_path / part['filename']
                            with open(filepath, 'wb') as f:
                                f.write(file_data)
                            logging.info(
                                f"Saved attachment {part['filename']} from message {message['id']}")

        except HttpError as error:
            logging.error(f'An error occurred: {error}')

    async def print_to_pdf(
        self,
        subject_pattern: str,
        include_thread: bool = False,
        output_path: Path = Path("email_pdfs")
    ) -> None:
        """logging.error matching emails to PDF"""
        try:
            output_path.mkdir(exist_ok=True)

            messages = self.service.users().messages().list(
                userId='me', q=f"subject:({subject_pattern})").execute()

            if 'messages' not in messages:
                return

            for message in messages['messages']:
                msg = self.service.users().messages().get(
                    userId='me', id=message['id'], format='full').execute()

                if include_thread:
                    thread = self.service.users().threads().get(
                        userId='me', id=msg['threadId']).execute()
                    messages_in_thread = thread['messages']
                else:
                    messages_in_thread = [msg]

                pdf = fpdf.FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)

                for thread_message in messages_in_thread:
                    headers = thread_message['payload']['headers']
                    subject = next(h['value']
                                   for h in headers if h['name'] == 'Subject')
                    sender = next(h['value']
                                  for h in headers if h['name'] == 'From')
                    date = next(h['value']
                                for h in headers if h['name'] == 'Date')

                    pdf.cell(0, 10, f"From: {sender}", ln=True)
                    pdf.cell(0, 10, f"Date: {date}", ln=True)
                    pdf.cell(0, 10, f"Subject: {subject}", ln=True)
                    pdf.cell(0, 10, "-" * 50, ln=True)

                    if 'data' in thread_message['payload']['body']:
                        body = base64.urlsafe_b64decode(
                            thread_message['payload']['body']['data']).decode('utf-8')
                    else:
                        parts = thread_message['payload']['parts']
                        body = base64.urlsafe_b64decode(
                            parts[0]['body']['data']).decode('utf-8')

                    soup = BeautifulSoup(body, 'html.parser')
                    clean_text = soup.get_text()

                    pdf.multi_cell(0, 10, clean_text)
                    pdf.cell(0, 10, "-" * 50, ln=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_path = output_path / f"email_{timestamp}.pdf"
                pdf.output(str(pdf_path))
                logging.info(
                    f"Saved email to PDF: {pdf_path} from subject: {subject_pattern} on email {message['id']}")

        except HttpError as error:
            logging.error(f'An error occurred: {error}')

    async def track_unread_emails(self) -> None:
        """Track emails that remain unread"""
        try:
            messages = self.service.users().messages().list(
                userId='me', q='is:unread').execute()

            if 'messages' not in messages:
                return

            for message in messages['messages']:
                msg = self.service.users().messages().get(
                    userId='me', id=message['id']).execute()

                headers = msg['payload']['headers']
                subject = next(h['value']
                               for h in headers if h['name'] == 'Subject')
                sender = next(h['value']
                              for h in headers if h['name'] == 'From')
                internal_date = int(msg['internalDate']) / 1000
                timestamp = datetime.fromtimestamp(internal_date)

                days_unread = (datetime.now() - timestamp).days

                tracker = UnreadTracker(
                    sender=sender,
                    subject=subject,
                    timestamp=timestamp,
                    days_unread=days_unread
                )

                self.unread_tracking.add(tracker)
                logging.info(
                    f"Tracked unread email: {tracker} added to in-memory tracker Set()")

        except HttpError as error:
            logging.error(f'An error occurred: {error}')

    async def list_folders(self) -> List[Dict[str, str]]:
        """List all folders/labels in the mailbox"""
        try:
            results = self.service.users().labels().list(userId='me').execute()
            return results.get('labels', [])
        except HttpError as error:
            logging.error(f'Error listing folders: {error}')
            return []

    async def create_folder(self, folder_name: str) -> Optional[str]:
        """Create a new folder/label and return its ID"""
        try:
            label_body = {
                'name': folder_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            created_label = self.service.users().labels().create(
                userId='me', body=label_body).execute()
            return created_label['id']
        except HttpError as error:
            logging.error(f'Error creating folder {folder_name}: {error}')
            return None

    async def block_sender(self, sender_email: str) -> Tuple[bool, str]:
        """
        Block a specific email address
        Returns (success, message)
        """
        try:
            # Validate email format
            validate_email(sender_email)

            # Add to database
            rule_id = self.db.create_blocked_sender(sender_email, "email")
            if rule_id:
                logging.info(f"Blocked sender: {sender_email}")
                return True, f"Successfully blocked {sender_email}"
            return False, "Failed to add to database"

        except EmailNotValidError as e:
            error_msg = f"Invalid email address: {str(e)}"
            logging.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error blocking sender: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    async def block_domain(self, domain_name: str) -> Tuple[bool, str]:
        """
        Block an entire domain
        Returns (success, message)
        """
        try:
            # Validate domain format
            domain_pattern = r'^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$'
            if not re.match(domain_pattern, domain_name):
                error_msg = f"Invalid domain format: {domain_name}"
                logging.error(error_msg)
                return False, error_msg

            # Create pattern to match all emails from this domain
            pattern = f".*@{re.escape(domain_name)}$"

            # Add to database
            rule_id = self.db.create_blocked_sender(pattern, "pattern")
            if rule_id:
                logging.info(f"Blocked domain: {domain_name}")
                return True, f"Successfully blocked domain {domain_name}"
            return False, "Failed to add to database"

        except Exception as e:
            error_msg = f"Error blocking domain: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    async def block_body_pattern(self, body_pattern: str) -> Tuple[bool, str]:
        """
        Block emails containing specific pattern in body
        Returns (success, message)
        """
        try:
            # Validate regex pattern
            try:
                re.compile(body_pattern)
            except re.error as e:
                error_msg = f"Invalid regex pattern: {str(e)}"
                logging.error(error_msg)
                return False, error_msg

            # Add to database
            rule_id = self.db.create_blocked_sender(
                body_pattern, "body_pattern")
            if rule_id:
                logging.info(f"Blocked body pattern: {body_pattern}")
                return True, f"Successfully blocked pattern: {body_pattern}"
            return False, "Failed to add to database"

        except Exception as e:
            error_msg = f"Error blocking body pattern: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    async def find_unsubscribe_link(self, message_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Find unsubscribe link in email headers or body using AI"""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='full').execute()

            # Extract email data
            headers = {h['name']: h['value']
                       for h in message['payload']['headers']}
            subject = headers.get('Subject', '')
            from_email = headers.get('From', '')

            # Get email body
            body_html = ""
            body_text = ""
            if 'data' in message['payload']['body']:
                body_html = base64.urlsafe_b64decode(
                    message['payload']['body']['data']).decode('utf-8')
            elif 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part.get('mimeType') == 'text/html' and 'data' in part['body']:
                        body_html = base64.urlsafe_b64decode(
                            part['body']['data']).decode('utf-8')
                    elif part.get('mimeType') == 'text/plain' and 'data' in part['body']:
                        body_text = base64.urlsafe_b64decode(
                            part['body']['data']).decode('utf-8')

            # Create email data structure for AI
            email_data = {
                "headers": {
                    "subject": subject,
                    "from": from_email,
                    "list_unsubscribe": headers.get('List-Unsubscribe', '')
                },
                "body_html": body_html,
                "body_text": body_text
            }

            # Create system prompt
            system_prompt = """You are an unsubscribe link detector. Analyze the email and find any unsubscribe links.
            Rules:
            1. First check the List-Unsubscribe header - this is the most reliable source
            2. If no header, look for links in the HTML that contain words like 'unsubscribe', 'opt-out', etc.
            3. For HTML links, provide the CSS selector path to locate the link
            4. Never hallucinate or create links - only return real links found in the email
            5. Assign a confidence score (0.0-1.0) based on how certain you are it's an unsubscribe link
            6. Explain your reasoning

            Return your findings in the following JSON format:
            {
                "link": "the unsubscribe URL or null if none found",
                "location": "header" or "CSS selector path",
                "confidence": float between 0.0 and 1.0,
                "reason": "explanation of your decision"
            }"""

            # Get AI analysis
            completion = await self.ai_service.chat_completion(
                messages=[
                    ChatCompletionMessageInput(
                        role="system",
                        content=system_prompt
                    ),
                    ChatCompletionMessageInput(
                        role="user",
                        content=json.dumps(email_data, indent=2)
                    )
                ]
            )

            # Parse AI response
            try:
                result = UnsubscribeLinkOutput.parse_raw(completion.response)

                if result.link and result.confidence >= 0.7:  # Only return high-confidence results
                    logging.info(f"Found unsubscribe link with confidence {
                                 result.confidence}: {result.link}")
                    logging.info(f"Reason: {result.reason}")
                    return result.link, result.location
                else:
                    if result.reason:
                        logging.info(f"No reliable unsubscribe link found: {
                                     result.reason}")
                    return None, None

            except Exception as e:
                logging.error(f"Error parsing AI response: {e}")
                logging.error(f"AI response was: {completion.response}")
                return None, None

        except Exception as e:
            logging.error(f'Error finding unsubscribe link: {e}')
            return None, None

    async def process_unsubscribes(self, folder_name: str, max_emails: int = 100) -> None:
        """Process emails in a folder to find and act on unsubscribe links"""
        try:
            # Create to_unsubscribe folder if it doesn't exist
            to_unsubscribe_id = await self.create_folder('to_unsubscribe')

            # Get folder ID
            folders = await self.list_folders()
            folder_id = next(
                (f['id'] for f in folders if f['name'] == folder_name), None)
            if not folder_id:
                logging.error(f'Folder {folder_name} not found')
                return

            # Get messages in folder
            messages = self.service.users().messages().list(
                userId='me', labelIds=[folder_id], maxResults=max_emails).execute()

            if 'messages' not in messages:
                return

            # Prepare CSV file
            unsubscribe_log = Path('unsubscribed.log')
            fieldnames = ['email_address', 'domain', 'unsubscribe_link']

            if not unsubscribe_log.exists():
                with open(unsubscribe_log, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

            async with httpx.AsyncClient() as client:
                for message in messages['messages']:
                    msg = self.service.users().messages().get(
                        userId='me', id=message['id'], format='full').execute()

                    # Get sender email
                    headers = {h['name']: h['value']
                               for h in msg['payload']['headers']}
                    from_email = headers.get('From', '')
                    email_match = re.search(r'<(.+@.+)>', from_email)
                    if email_match:
                        sender_email = email_match.group(1)
                    else:
                        sender_email = from_email

                    domain = sender_email.split(
                        '@')[-1] if '@' in sender_email else ''

                    # Find unsubscribe link
                    unsubscribe_url, source = await self.find_unsubscribe_link(message['id'])

                    if unsubscribe_url:
                        # Log the information
                        with open(unsubscribe_log, 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writerow({
                                'email_address': sender_email,
                                'domain': domain,
                                'unsubscribe_link': unsubscribe_url
                            })

                        # Try to unsubscribe
                        try:
                            response = await client.get(unsubscribe_url, follow_redirects=True, timeout=10.0)
                            if 'unsubscribed' in response.text.lower() or 'success' in response.text.lower():
                                logging.info(f'Successfully unsubscribed from {
                                             sender_email}')
                                # Create rule to auto-delete future emails
                                rule = {
                                    'name': f'Auto-delete {domain}',
                                    'conditions': {'from': f'.*@{re.escape(domain)}'},
                                    'actions': [{'type': 'delete'}]
                                }
                                # Add rule to rules file
                                self._add_rule_to_file(rule)
                            else:
                                # Move to to_unsubscribe folder for manual review
                                await self.apply_label([message['id']], 'to_unsubscribe')
                                logging.info(f'Moved email from {
                                             sender_email} to to_unsubscribe folder')
                        except Exception as e:
                            logging.error(f'Error unsubscribing from {
                                          sender_email}: {e}')
                            await self.apply_label([message['id']], 'to_unsubscribe')

        except Exception as e:
            logging.error(f'Error processing unsubscribes: {e}')

    def _add_rule_to_file(self, rule: Dict[str, Any]) -> None:
        """Add a new rule to the rules file"""
        rules_file = Path('email_rules.json')
        if rules_file.exists():
            with open(rules_file, 'r') as f:
                rules = json.load(f)
        else:
            rules = []

        rules.append(rule)
        with open(rules_file, 'w') as f:
            json.dump(rules, f, indent=4)
        logging.info(f'Added new rule for {rule["conditions"]["from"]}')

    def _sanitize_label(self, label: str) -> str:
        r"""Sanitize label to match [A-Za-z\s_\-] pattern"""
        # First, replace any invalid characters with spaces
        sanitized = re.sub(r'[^A-Za-z\s_\-]', ' ', label)
        # Replace multiple spaces with single space
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # Trim spaces from start and end
        return sanitized.strip()

    async def create_synced_label(self, label: str) -> Tuple[bool, Optional[str]]:
        """
        Create a label both on Gmail server and local database.
        Returns (success, error_message)
        """
        try:
            # Sanitize the label
            sanitized_label = self._sanitize_label(label)
            if not sanitized_label:
                return False, "Label is empty after sanitization"

            # Check if label already exists in Gmail
            existing_labels = self.service.users().labels().list(userId='me').execute()
            for existing in existing_labels.get('labels', []):
                if existing['name'].lower() == sanitized_label.lower():
                    # Label exists, store in local db if not already there
                    try:
                        self.db.create_label_with_uri(
                            sanitized_label,
                            existing.get('id')  # Gmail API uses id as URI
                        )
                    except Exception:
                        # Ignore if label already exists in local db
                        pass
                    return True, None

            # Create new label on Gmail
            label_body = {
                'name': sanitized_label,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }

            try:
                created_label = self.service.users().labels().create(
                    userId='me',
                    body=label_body
                ).execute()

                # Store in local database
                label_id = self.db.create_label_with_uri(
                    sanitized_label,
                    created_label.get('id')  # Gmail API uses id as URI
                )

                if label_id is None:
                    return False, "Failed to store label in local database"

                logging.info(
                    f"Created label '{sanitized_label}' on Gmail and local database")
                return True, None

            except HttpError as e:
                error_message = f"Failed to create label on Gmail: {str(e)}"
                logging.error(error_message)
                return False, error_message

        except Exception as e:
            error_message = f"Unexpected error creating label: {str(e)}"
            logging.error(error_message)
            return False, error_message

    async def add_natural_language_rule(self, rule: str, actions: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Add a natural language rule with associated actions.
        Returns (success, message)
        """
        if len(rule) > 50:
            return False, "Rule must be 50 characters or less"

        try:
            # Validate actions against Gmail SDK actions
            valid_actions = {"label", "archive",
                             "delete", "markRead", "star", "forward"}
            for action in actions:
                if action.get('type') not in valid_actions:
                    return False, f"Invalid action type: {action.get('type')}"

            # Add rule to database
            rule_id = self.db.create_nl_rule(rule, actions)
            if rule_id is None:
                return False, "Failed to create rule in database"

            return True, f"Successfully created rule with ID: {rule_id}"

        except Exception as e:
            error_msg = f"Error creating natural language rule: {str(e)}"
            logging.error(error_msg)
            return False, error_msg

    async def process_nl_rules(self, email_content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Process natural language rules against email content.
        Returns list of matching rules and their actions.
        """
        try:
            # Get all rules from database
            rules = self.db.get_all_nl_rules()
            if not rules:
                return None

            # Get format instructions
            format_instructions = self.nl_rule_parser.get_format_instructions()

            # Create system prompt
            rules_context = "\n".join(
                [f"Rule {r['id']}: {r['rule']}" for r in rules])
            system_prompt = f"""
            You are an email rule matching system. Given the following rules:

            {rules_context}

            Analyze the email content and return the IDs of any matching rules.
            {format_instructions}
            Only return rule IDs if you are highly confident they match.
            """

            # Create prompt template for LangChain
            prompt = PromptTemplate(
                input_variables=["email_content"],
                template=system_prompt + "\nEmail content: {email_content}"
            )

            # Get AI response
            completion = await self.ai_service.chat_completion(
                messages=[
                    ChatCompletionMessageInput(
                        role="system",
                        content=system_prompt
                    ),
                    ChatCompletionMessageInput(
                        role="user",
                        content=email_content
                    )
                ]
            )

            # Parse response to get matching rule IDs
            try:
                output = self.nl_rule_parser.parse(completion.response)
                matching_rules = []

                # Get actions for matching rules
                for rule_id in output['id']:
                    rule = self.db.get_nl_rule(rule_id)
                    if rule:
                        matching_rules.append(rule)

                return matching_rules

            except Exception as e:
                logging.error(f"Error parsing AI response: {e}")
                return None

        except Exception as e:
            logging.error(f"Error processing natural language rules: {e}")
            return None

    async def apply_nl_rule_actions(self, message_id: str, actions: List[Dict[str, Any]]) -> None:
        """Apply the actions from a natural language rule
        ### Example usage: (Apply multiple actions to a message)
        ```python
        actions = [
            {"type": "star"},
            {"type": "markRead"},
            {"type": "forward", "to": "someone@example.com"}
        ]
        await gmail.apply_nl_rule_actions(message_id, actions)
        ```
        """
        # TODO: Call this based on db stored filters that match emails and then apply the rules defined on the filters from the database.
        for action in actions:
            action_type = action.get('type')
            try:
                if action_type == 'label':
                    await self.apply_label([message_id], action['value'])
                elif action_type == 'archive':
                    await self._archive_message(message_id)
                elif action_type == 'delete':
                    await self._delete_message(message_id)
                elif action_type == 'markRead':
                    await self._mark_as_read(message_id)
                elif action_type == 'star':
                    await self._star_message(message_id)
                elif action_type == 'forward':
                    await self._forward_message(message_id, action.get('to'))
                else:
                    logging.warning(f"Unknown action type: {action_type}")
            except Exception as e:
                logging.error(f"Error applying action {action_type}: {e}")

    async def _archive_message(self, message_id: str) -> None:
        """Remove INBOX label to archive message"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={
                    'removeLabelIds': ['INBOX']
                }
            ).execute()
            logging.info(f"Archived message: {message_id}")
        except HttpError as e:
            logging.error(f"Error archiving message {message_id}: {e}")
            raise

    async def _delete_message(self, message_id: str) -> None:
        """Move message to trash"""
        try:
            self.service.users().messages().trash(
                userId='me',
                id=message_id
            ).execute()
            logging.info(f"Deleted message: {message_id}")
        except HttpError as e:
            logging.error(f"Error deleting message {message_id}: {e}")
            raise

    async def _mark_as_read(self, message_id: str) -> None:
        """Remove UNREAD label from message"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={
                    'removeLabelIds': ['UNREAD']
                }
            ).execute()
            logging.info(f"Marked message as read: {message_id}")
        except HttpError as e:
            logging.error(f"Error marking message {message_id} as read: {e}")
            raise

    async def _star_message(self, message_id: str) -> None:
        """Add STARRED label to message"""
        try:
            self.service.users().messages().modify(
                userId='me',
                id=message_id,
                body={
                    'addLabelIds': ['STARRED']
                }
            ).execute()
            logging.info(f"Starred message: {message_id}")
        except HttpError as e:
            logging.error(f"Error starring message {message_id}: {e}")
            raise

    async def _forward_message(self, message_id: str, to_email: str) -> None:
        """Forward message to specified email address"""
        if not to_email:
            logging.error("No 'to' email address provided for forward action")
            return

        try:
            # Get original message
            message = self.service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()

            # Extract headers
            headers = {h['name']: h['value']
                       for h in message['payload']['headers']}
            subject = headers.get('Subject', '')
            from_email = headers.get('From', '')

            # Get message content
            if 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']).decode('utf-8')
            elif 'parts' in message['payload']:
                parts = message['payload']['parts']
                body = base64.urlsafe_b64decode(
                    parts[0]['body']['data']).decode('utf-8') if 'data' in parts[0]['body'] else ""
            else:
                body = ""

            # Create forward message
            forward_message = f"""
From: me
To: {to_email}
Subject: Fwd: {subject}

---------- Forwarded message ---------
From: {from_email}
Subject: {subject}

{body}
"""
            # Encode and send
            encoded_message = base64.urlsafe_b64encode(
                forward_message.encode('utf-8')).decode('utf-8')

            self.service.users().messages().send(
                userId='me',
                body={
                    'raw': encoded_message
                }
            ).execute()
            logging.info(f"Forwarded message {message_id} to {to_email}")

        except HttpError as e:
            logging.error(f"Error forwarding message {message_id}: {e}")
            raise

    async def auto_archive_emails(self, max_emails: int = 100) -> None:
        """
        Automatically archive non-important emails and generate a report
        """
        try:
            # Get unprocessed emails
            messages = self.service.users().messages().list(
                userId='me',
                q='in:inbox -label:auto_archived',
                maxResults=max_emails
            ).execute()

            if 'messages' not in messages:
                return

            archived_emails = []
            kept_emails = []

            for message in messages['messages']:
                msg = self.service.users().messages().get(
                    userId='me', id=message['id'], format='full'
                ).execute()

                # Extract email data
                headers = {h['name']: h['value']
                           for h in message['payload']['headers']}
                subject = headers.get('Subject', '')
                from_email = headers.get('From', '')
                has_attachments = any(
                    'filename' in part for part in msg['payload'].get('parts', [])
                    if 'filename' in part
                )

                # Get email body
                body = self._get_message_body(msg) or ""

                # Create email context for AI
                email_data = {
                    "from": from_email,
                    "subject": subject,
                    "body": body[:1000],  # First 1000 chars for context
                    "has_attachments": has_attachments,
                    "date": datetime.fromtimestamp(
                        int(msg['internalDate']) / 1000
                    ).isoformat()
                }

                # Get AI decision
                decision = await self._get_archive_decision(email_data)

                if decision.can_archive and decision.confidence >= 0.8:
                    # Archive the email
                    await self._archive_message(message['id'])
                    # Add auto_archived label
                    await self.apply_label([message['id']], 'auto_archived')

                    archived_emails.append({
                        'from': from_email,
                        'subject': subject,
                        'reason': decision.reason,
                        'importance': decision.importance_score
                    })
                else:
                    kept_emails.append({
                        'from': from_email,
                        'subject': subject,
                        'reason': decision.reason,
                        'importance': decision.importance_score,
                        'summary': decision.summary
                    })

            # Generate and send report
            if archived_emails or kept_emails:
                await self._send_archive_report(archived_emails, kept_emails)

        except Exception as e:
            logging.error(f"Error in auto_archive_emails: {e}")

    async def _get_archive_decision(self, email_data: Dict[str, Any]) -> ArchiveDecisionOutput:
        """Get AI decision on whether to archive an email"""
        system_prompt = """You are an email importance analyzer. Determine if an email can be safely archived based on these rules:

        Can be archived if:
        1. Promotional or marketing content
        2. Automated notifications that don't require action
        3. Social media updates
        4. Newsletters without critical content
        5. Duplicate messages

        Must be kept if:
        1. Contains action items or requests
        2. Personal or direct communication
        3. Important business correspondence
        4. Financial or legal information
        5. Time-sensitive content

        Return your analysis in JSON format with these fields:
        {
            "can_archive": boolean,
            "confidence": float (0-1),
            "reason": "explanation of decision",
            "importance_score": float (0-1),
            "summary": "brief summary if important, null if not"
        }"""

        try:
            completion = await self.ai_service.chat_completion(
                messages=[
                    ChatCompletionMessageInput(
                        role="system",
                        content=system_prompt
                    ),
                    ChatCompletionMessageInput(
                        role="user",
                        content=json.dumps(email_data, indent=2)
                    )
                ]
            )

            return ArchiveDecisionOutput.parse_raw(completion.response)

        except Exception as e:
            logging.error(f"Error getting archive decision: {e}")
            # Default to not archiving on error
            return ArchiveDecisionOutput(
                can_archive=False,
                confidence=0.0,
                reason="Error analyzing email",
                importance_score=0.5
            )

    async def _send_archive_report(self, archived: List[Dict], kept: List[Dict]) -> None:
        """Send a report of archived and kept emails"""
        try:
            # Create report content
            report = f"""Email Archive Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}

Archived Emails ({len(archived)}):
{'=' * 50}
"""
            for email in archived:
                report += f"\nFrom: {email['from']}\nSubject: {
                    email['subject']}\nReason: {email['reason']}\n"

            report += f"""\n\nKept Emails ({len(kept)}):
{'=' * 50}
"""
            for email in kept:
                report += f"\nFrom: {email['from']
                                     }\nSubject: {email['subject']}\n"
                if email['summary']:
                    report += f"Summary: {email['summary']}\n"
                report += f"Reason: {email['reason']}\n"

            # Create message
            message = f"""From: me
To: me
Subject: Email Archive Report - {datetime.now().strftime('%Y-%m-%d')}

{report}
"""
            # Encode and send
            encoded_message = base64.urlsafe_b64encode(
                message.encode('utf-8')
            ).decode('utf-8')

            self.service.users().messages().send(
                userId='me',
                body={'raw': encoded_message}
            ).execute()

            logging.info("Sent archive report")

        except Exception as e:
            logging.error(f"Error sending archive report: {e}")


class GmailRuleEngine:
    def __init__(self, gmail_automation: GmailAutomation, rules_file: str):
        self.gmail = gmail_automation
        self.rules_file = rules_file
        self.rules: List[EmailRule] = []
        self.last_check_time = datetime.now().isoformat()
        self.last_archive_time = datetime.now()
        # Run auto-archive every 4 hours
        self.archive_interval = timedelta(hours=4)
        self.load_rules()

    def load_rules(self) -> None:
        """Load rules from JSON file"""
        try:
            with open(self.rules_file, 'r') as f:
                rules_data = json.load(f)
                self.rules = [EmailRule(**rule) for rule in rules_data]
                logging.info(
                    f"Loaded {len(self.rules)} rules from {self.rules_file}")
        except FileNotFoundError:
            logging.warning(f"Rules file not found: {self.rules_file}")
            self.rules = []

    async def run_scheduled_tasks(self) -> None:
        """Run scheduled tasks like auto-archiving"""
        try:
            current_time = datetime.now()

            # Check if it's time to run auto-archive
            if current_time - self.last_archive_time >= self.archive_interval:
                logging.info("Running scheduled auto-archive")
                await self.gmail.auto_archive_emails(max_emails=100)
                self.last_archive_time = current_time

        except Exception as e:
            logging.error(f"Error in scheduled tasks: {e}")

    async def process_message(self, message: Dict[str, Any]) -> None:
        """Process a single message against all rules"""
        try:
            # First check if sender is blocked
            if await self.check_blocked_sender(message):
                await self.gmail.apply_label([message['id']], 'Blocked')
                logging.info(f"Blocked message {
                             message['id']} from blocked sender")
                return

            headers = {h['name']: h['value']
                       for h in message['payload']['headers']}

            # Check for auto-archive conditions first
            if await self._should_auto_archive(message):
                await self.gmail.auto_archive_emails(max_emails=1)
                return

            # Continue with regular rule processing
            for rule in self.rules:
                matches = True
                for field, pattern in rule.conditions.items():
                    if field in headers:
                        if not re.search(pattern, headers[field], re.IGNORECASE):
                            matches = False
                            break
                    else:
                        matches = False
                        break

                if matches:
                    await self.apply_actions(message['id'], rule.actions)
                    logging.info(f"Applied rule '{
                                 rule.name}' to message {message['id']}")

        except Exception as e:
            logging.error(f"Error processing message: {e}")

    async def _should_auto_archive(self, message: Dict[str, Any]) -> bool:
        """Check if message meets auto-archive criteria"""
        try:
            headers = {h['name']: h['value']
                       for h in message['payload']['headers']}
            from_email = headers.get('From', '')
            subject = headers.get('Subject', '')
            body = self._get_message_body(message) or ""

            # Create email context for AI
            email_data = {
                "from": from_email,
                "subject": subject,
                "body": body[:1000],
                "has_attachments": self._has_attachments(message)
            }

            # Get AI decision
            decision = await self.gmail._get_archive_decision(email_data)
            return decision.can_archive and decision.confidence >= 0.8

        except Exception as e:
            logging.error(f"Error checking auto-archive criteria: {e}")
            return False

    def _has_attachments(self, message: Dict[str, Any]) -> bool:
        """Check if message has attachments"""
        try:
            return any(
                'filename' in part
                for part in message['payload'].get('parts', [])
                if 'filename' in part
            )
        except Exception:
            return False

    async def check_new_emails(self) -> None:
        """Check for new emails and process them"""
        try:
            # Run scheduled tasks first
            await self.run_scheduled_tasks()

            # Process new messages
            query = f'after:{self.last_check_time}'
            messages = self.gmail.service.users().messages().list(
                userId='me', q=query).execute()

            if 'messages' in messages:
                for message in messages['messages']:
                    full_message = self.gmail.service.users().messages().get(
                        userId='me', id=message['id'], format='full').execute()
                    await self.process_message(full_message)

            self.last_check_time = datetime.now().isoformat()

        except Exception as e:
            logging.error(f"Error checking new emails: {str(e)}")

    def load_blocked_senders(self) -> Set[str]:
        """Load blocked senders from database"""
        blocked_senders = set()
        try:
            db_patterns = self.gmail.db.get_all_blocked_senders()
            for pattern in db_patterns:
                blocked_senders.add(pattern['pattern'])
            return blocked_senders
        except Exception as e:
            logging.error(f"Error loading blocked senders from database: {e}")
            return set()

    async def check_blocked_sender(self, message: Dict[str, Any]) -> bool:
        """Check if sender is blocked using patterns from database"""
        headers = {h['name']: h['value']
                   for h in message['payload']['headers']}
        from_email = headers.get('From', '')
        body = self._get_message_body(message)

        blocked_patterns = self.gmail.db.get_all_blocked_senders()

        for pattern in blocked_patterns:
            pattern_text = pattern['pattern']
            pattern_type = pattern['type']

            if pattern_type == 'email' and pattern_text.lower() in from_email.lower():
                return True
            elif pattern_type == 'pattern' and re.search(pattern_text, from_email, re.IGNORECASE):
                return True
            elif pattern_type == 'body_pattern' and body and re.search(pattern_text, body, re.IGNORECASE):
                return True

        return False

    def _get_message_body(self, message: Dict[str, Any]) -> Optional[str]:
        """Extract message body"""
        try:
            if 'data' in message['payload']['body']:
                return base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')
            elif 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part.get('mimeType') == 'text/plain' and 'data' in part['body']:
                        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        except Exception as e:
            logging.error(f'Error getting message body: {e}')
        return None

    async def create_rule_from_prompt(self, prompt: str) -> None:
        """Create a new Gmail rule from a user prompt using AI"""
        try:
            system_prompt = """You are a system for creating Gmail filter rules based on user input. 
            Generate a Gmail filter rule in JSON format with the following structure:
            {
                "from": "",
                "to": "",
                "subject": "",
                "hasWords": "",
                "doesNotHaveWords": "",
                "size": {
                    "greaterThan": false,
                    "sizeInMB": null
                },
                "hasAttachment": false,
                "includeChats": false,
                "action": {
                    "delete": false,
                    "archive": false,
                    "markAsRead": false,
                    "star": false,
                    "label": "",
                    "forwardTo": ""
                }
            }
            Only respond with the JSON, no other text."""

            # Get rule JSON from AI
            completion = await self.gmail.ai_service.chat_completion(
                messages=[
                    ChatCompletionMessageInput(
                        role="system",
                        content=system_prompt
                    ),
                    ChatCompletionMessageInput(
                        role="user",
                        content=f"Create a Gmail rule for the following request: {
                            prompt}"
                    )
                ]
            )

            # Parse the JSON response
            try:
                rule_json = json.loads(completion.response)

                # Convert to our internal rule format
                filter_rule = GmailFilter(
                    from_=rule_json.get('from', ''),
                    to=rule_json.get('to', ''),
                    subject=rule_json.get('subject', ''),
                    hasWords=rule_json.get('hasWords', ''),
                    doesNotHaveWords=rule_json.get('doesNotHaveWords', ''),
                    size=GmailFilterSize(
                        rule_json.get('size', {}).get('greaterThan', False),
                        rule_json.get('size', {}).get('sizeInMB', None)
                    ),
                    hasAttachment=rule_json.get('hasAttachment', False),
                    includeChats=rule_json.get('includeChats', False),
                    action=GmailFilterAction(**rule_json.get('action', {}))
                )

                # Convert to email_rules.json format
                conditions = {}
                if filter_rule.from_:
                    conditions['from'] = filter_rule.from_
                if filter_rule.to:
                    conditions['to'] = filter_rule.to
                if filter_rule.subject:
                    conditions['subject'] = filter_rule.subject

                actions = []
                action = filter_rule.action
                if action.delete:
                    actions.append({'type': 'delete'})
                if action.archive:
                    actions.append({'type': 'archive'})
                if action.markAsRead:
                    actions.append({'type': 'mark_read'})
                if action.star:
                    actions.append({'type': 'star'})
                if action.label:
                    actions.append({'type': 'label', 'value': action.label})
                if action.forwardTo:
                    actions.append(
                        {'type': 'forward', 'value': action.forwardTo})

                new_rule = {
                    'name': f"AI Generated Rule - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'conditions': conditions,
                    'actions': actions
                }

                # Add to rules file
                self._add_rule_to_file(new_rule)
                logging.info(f"Created new rule from prompt: {prompt}")

            except json.JSONDecodeError as e:
                logging.error(f"Error parsing AI response as JSON: {e}")
                logging.error(f"AI response was: {completion.response}")

        except Exception as e:
            logging.error(f"Error creating rule from prompt: {e}")

    def _add_rule_to_file(self, rule: Dict[str, Any]) -> None:
        """Add a new rule to the rules file"""
        try:
            rules_file = Path(self.rules_file)
            if rules_file.exists():
                with open(rules_file, 'r') as f:
                    rules = json.load(f)
            else:
                rules = []

            rules.append(rule)
            with open(rules_file, 'w') as f:
                json.dump(rules, f, indent=4)

            # Also update the in-memory rules
            self.rules.append(EmailRule(**rule))
            logging.info(f'Added new rule: {rule["name"]}')
        except Exception as e:
            logging.error(f'Error adding rule to file: {e}')


# Example usage:

# Test Users are published here: https://console.cloud.google.com/apis/credentials/consent?authuser=1&invt=AbiK0Q&project=gmail-daemon-442511


async def main():
    # Initialize database
    db = GmailDatabase()

    ai_service = AIService.get_instance(model_name="gpt-4")
    gmail = GmailAutomation(
        credentials_path='path/to/credentials.json',
        token_path='path/to/token.json',
        ai_service=ai_service,
        db=db
    )

    rule_engine = GmailRuleEngine(gmail, 'email_rules.json')

    logging.info("Starting Gmail Rule Daemon...")
    try:
        while True:
            await rule_engine.check_new_emails()
            await asyncio.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logging.info("Shutting down Gmail Rule Daemon...")
    finally:
        db.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
