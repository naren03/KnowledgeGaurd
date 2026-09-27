from typing import TypedDict


class ChatState(TypedDict):
    question: str
    context: list[dict[str, object]]
    answer: str
