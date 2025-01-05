from abc import ABC, abstractmethod
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import Resource
from pathlib import Path
import os
from typing import Any, List


class GoogleServiceAuth(ABC):
    """Abstract base class for Google service authentication"""

    def __init__(self, credentials_path: str, token_path: str):
        """Initialize with OAuth2 credentials and token paths"""
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.credentials = None
        self.script_dir = Path(__file__).parent
        self.service: Resource = Resource(
            None, None, None, None, None, None, None, None, "")

    @property
    @abstractmethod
    def SCOPES(self) -> List[str]:
        """Abstract property that must return the required OAuth2 scopes"""
        pass

    @abstractmethod
    def _build_service(self, credentials: Credentials) -> Resource:
        """Abstract method to build the specific Google service"""
        pass

    def authenticate(self) -> None:
        """Handle Google API authentication"""
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

            # Save the credentials for the next run
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        self.credentials = creds
        self.service = self._build_service(creds)
