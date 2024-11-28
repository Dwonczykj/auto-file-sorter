from dataclasses import dataclass, field
from datetime import datetime
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
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import fpdf
import time
import json
import httpx
import csv
from urllib.parse import urlparse

from ai_service import AIService, ChatCompletionMessageInput
from auto_file_sorter.logging.logging_config import configure_logging

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


class GmailAutomation:
    """Class to handle Gmail automation tasks with AI integration"""

    SCOPES = [
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.settings.basic',
        'https://www.googleapis.com/auth/gmail.settings.sharing'
    ]

    def __init__(self, credentials_path: str, token_path: str, ai_service: AIService):
        """Initialize Gmail automation with OAuth2 credentials and AI service"""
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = self._authenticate()
        self.ai_service = ai_service
        self.unread_tracking: Set[UnreadTracker] = set()

    def _authenticate(self) -> Any:
        """Handle Gmail API authentication"""
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(
                self.token_path, self.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        return build('gmail', 'v1', credentials=creds)

    async def summarize_email(self, message_id: str) -> str:
        """Summarize email content using AI service"""
        try:
            # Get email content
            message = self.service.users().messages().get(
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

    async def find_unsubscribe_link(self, message_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Find unsubscribe link in email headers or body"""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id, format='full').execute()

            # Check List-Unsubscribe header first
            headers = {h['name']: h['value']
                       for h in message['payload']['headers']}
            if 'List-Unsubscribe' in headers:
                unsubscribe = headers['List-Unsubscribe']
                # Extract URL from <> brackets if present
                url_match = re.search(r'<(https?://[^>]+)>', unsubscribe)
                if url_match:
                    return url_match.group(1), 'header'

            # Check email body
            body = ""
            if 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(
                    message['payload']['body']['data']).decode('utf-8')
            elif 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part.get('mimeType') == 'text/html' and 'data' in part['body']:
                        body = base64.urlsafe_b64decode(
                            part['body']['data']).decode('utf-8')
                        break

            if body:
                soup = BeautifulSoup(body, 'html.parser')
                unsubscribe_links = soup.find_all(
                    'a', href=True, text=re.compile(r'unsubscribe', re.I))
                if unsubscribe_links:
                    return unsubscribe_links[0]['href'], 'body'

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


class GmailRuleEngine:
    def __init__(self, gmail_automation: GmailAutomation, rules_file: str):
        self.gmail = gmail_automation
        self.rules_file = rules_file
        self.rules: List[EmailRule] = []
        self.last_check_time = datetime.now().isoformat()
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

    async def process_message(self, message: Dict[str, Any]) -> None:
        """Process a single message against all rules"""
        # First check if sender is blocked
        if await self.check_blocked_sender(message):
            # Move to trash or apply blocked label
            await self.gmail.apply_label([message['id']], 'Blocked')
            logging.info(f"Blocked message {
                         message['id']} from blocked sender")
            return

        # Continue with existing rule processing...
        headers = {h['name']: h['value']
                   for h in message['payload']['headers']}

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

    async def apply_actions(self, message_id: str, actions: List[Dict[str, Any]]) -> None:
        """Apply the specified actions to a message"""
        for action in actions:
            action_type = action['type']
            try:
                if action_type == 'label':
                    await self.gmail.apply_label([message_id], action['value'])
                elif action_type == 'summarize':
                    summary = await self.gmail.summarize_email(message_id)
                    logging.info(f"Email summary: {summary}")
                elif action_type == 'auto_reply':
                    await self.gmail.auto_reply(message_id, action.get('context', ''))
                elif action_type == 'save_attachment':
                    save_path = Path(action.get('path', 'attachments'))
                    await self.gmail.save_attachments('', '', save_path)
            except Exception as e:
                logging.error(f"Error applying action {action_type}: {str(e)}")

    async def check_new_emails(self) -> None:
        """Check for new emails and process them"""
        try:
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
        """Load blocked senders from file"""
        blocked_file = Path(
            '/Users/joey/Library/Mobile Documents/iCloud~is~workflow~my~workflows/Documents/blocked_senders.txt')
        blocked_senders = set()

        if blocked_file.exists():
            with open(blocked_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        blocked_senders.add(line)

        return blocked_senders

    async def check_blocked_sender(self, message: Dict[str, Any]) -> bool:
        """Check if sender is blocked"""
        headers = {h['name']: h['value']
                   for h in message['payload']['headers']}
        from_email = headers.get('From', '')
        body = self._get_message_body(message)

        blocked_senders = self.load_blocked_senders()

        for blocked in blocked_senders:
            # Check if it's an email address
            if '@' in blocked and blocked.lower() in from_email.lower():
                return True
            # Check if it's a pattern for From field
            elif re.search(blocked, from_email, re.IGNORECASE):
                return True
            # Check if it's a pattern for body
            elif body and re.search(blocked, body, re.IGNORECASE):
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
    ai_service = AIService.get_instance(model_name="gpt-4")
    gmail = GmailAutomation(
        credentials_path='/Users/joey/Github_Keep/python_scripts/auto-file-sorter/Google Cloud Credentials.json',
        token_path='/Users/joey/Github_Keep/python_scripts/auto-file-sorter/gmail_token.json',
        ai_service=ai_service
    )

    rule_engine = GmailRuleEngine(gmail, 'email_rules.json')

    # Example of creating a rule from prompt
    await rule_engine.create_rule_from_prompt(
        "Create a rule to filter all promotional emails containing 'special offer' and mark them as read"
    )

    logging.info("Starting Gmail Rule Daemon...")
    while True:
        await rule_engine.check_new_emails()
        await asyncio.sleep(60)  # Check every minute


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
