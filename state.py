from typing import TypedDict

class State(TypedDict):
    question: str
    context: list[str]
    answer: str