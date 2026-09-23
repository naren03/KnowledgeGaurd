from typing import TypedDict

from langchain_core.documents import Document


class IngestionState(TypedDict, total=False):
    file_path: str
    file_type: str
    raw_documents: list[Document]
    chunks: list[Document]
    error: str | None
