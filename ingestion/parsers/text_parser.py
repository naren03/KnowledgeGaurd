from pathlib import Path

from langchain_core.documents import Document


def parse_text(file_path: str) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8-sig")

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata={
                "source": file_path,
                "file_type": "txt",
            },
        )
    ]