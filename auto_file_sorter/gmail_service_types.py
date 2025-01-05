from typing import Any, Protocol, runtime_checkable, Optional, List, Dict


@runtime_checkable
class GmailServiceProtocol(Protocol):
    """Protocol defining the Gmail service interface"""

    def users(self) -> 'GmailUsersProtocol': ...


@runtime_checkable
class GmailUsersProtocol(Protocol):
    """Protocol for Gmail users() methods"""

    def messages(self) -> 'GmailMessagesProtocol': ...
    def labels(self) -> 'GmailLabelsProtocol': ...
    def drafts(self) -> 'GmailDraftsProtocol': ...
    def threads(self) -> 'GmailThreadsProtocol': ...


@runtime_checkable
class GmailMessagesProtocol(Protocol):
    """Protocol for Gmail messages() methods"""

    def list(self, userId: str, q: str = "",
             labelIds: Optional[List[str]] = None,
             maxResults: Optional[int] = None) -> Dict[str, Any]: ...

    def get(self, userId: str, id: str,
            format: Optional[str] = None) -> Dict[str, Any]: ...

    def send(self, userId: str, body: Dict[str, Any]) -> Dict[str, Any]: ...

    def batchModify(self, userId: str,
                    body: Dict[str, Any],
                    ids: List[str]) -> Dict[str, Any]: ...

    def attachments(self) -> 'GmailAttachmentsProtocol': ...


@runtime_checkable
class GmailLabelsProtocol(Protocol):
    """Protocol for Gmail labels() methods"""

    def list(self, userId: str) -> Dict[str, Any]: ...
    def create(self, userId: str, body: Dict[str, Any]) -> Dict[str, Any]: ...


@runtime_checkable
class GmailDraftsProtocol(Protocol):
    """Protocol for Gmail drafts() methods"""

    def create(self, userId: str, body: Dict[str, Any]) -> Dict[str, Any]: ...


@runtime_checkable
class GmailThreadsProtocol(Protocol):
    """Protocol for Gmail threads() methods"""

    def get(self, userId: str, id: str) -> Dict[str, Any]: ...


@runtime_checkable
class GmailAttachmentsProtocol(Protocol):
    """Protocol for Gmail attachments() methods"""

    def get(self, userId: str, messageId: str, id: str) -> Dict[str, Any]: ...
