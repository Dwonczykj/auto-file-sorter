from pydantic.dataclasses import ConfigDict


@ConfigDict
class User:
    id: str
    flags: list[str]
    blocked_domains: list[str]
    blocked_senders: list[str]
