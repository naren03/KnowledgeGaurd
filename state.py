from typing import TypedDict

class ChatState(TypedDict):
    question: str
    context: list[str]
    answer: str