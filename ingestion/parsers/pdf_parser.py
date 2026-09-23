from langchain_core.documents import Document
from pypdf import PdfReader


def parse_pdf(file_path: str) -> list[Document]:
    reader = PdfReader(file_path)
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "file_type": "pdf",
                        "page_number": page_number,
                    },
                )
            )

    return documents
