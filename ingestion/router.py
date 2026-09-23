from pathlib import Path

SUPPORTED_TYPES = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx",
    ".pptx": "pptx",
    ".txt": "txt",
    ".md": "md",
}


def detect_file_type(file_path: str) -> str:
    extension = Path(file_path).suffix.lower()

    if extension not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported file type: {extension}")

    return SUPPORTED_TYPES[extension]
