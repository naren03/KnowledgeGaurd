from typing import Any, TypedDict


class ChatState(TypedDict, total=False):
    question: str
    user_id: str
    context: list[dict[str, Any]]
    answer: str
    audit_data: dict[str, Any]
