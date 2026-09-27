from typing import Literal

from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from parsers.docx_parser import parse_docx
from parsers.excel_parser import parse_excel
from parsers.markdown_parser import parse_markdown
from parsers.pdf_parser import parse_pdf
from parsers.pptx_parser import parse_pptx
from parsers.text_parser import parse_text
from router import detect_file_type

from state import IngestionState


def detect_type_node(state: IngestionState):
    file_type = detect_file_type(state["file_path"])

    return {
        "file_type": file_type,
        "error": None,
    }


def route_by_type(
    state: IngestionState,
) -> Literal[
    "parse_pdf",
    "parse_docx",
    "parse_excel",
    "parse_pptx",
    "parse_markdown",
    "parse_text",
]:
    file_type = state["file_type"]

    routes = {
        "pdf": "parse_pdf",
        "docx": "parse_docx",
        "xlsx": "parse_excel",
        "pptx": "parse_pptx",
        "md": "parse_markdown",
        "txt": "parse_text",
    }

    if file_type not in routes:
        raise ValueError(f"No parser configured for {file_type}")

    return routes[file_type]


def parse_pdf_node(state: IngestionState):
    docs = parse_pdf(state["file_path"])
    return {"raw_documents": docs}


def parse_docx_node(state: IngestionState):
    docs = parse_docx(state["file_path"])
    return {"raw_documents": docs}


def parse_excel_node(state: IngestionState):
    docs = parse_excel(state["file_path"])
    return {"raw_documents": docs}


def parse_pptx_node(state: IngestionState):
    docs = parse_pptx(state["file_path"])
    return {"raw_documents": docs}


def parse_markdown_node(state: IngestionState):
    docs = parse_markdown(state["file_path"])
    return {"raw_documents": docs}


def parse_text_node(state: IngestionState):
    docs = parse_text(state["file_path"])
    return {"raw_documents": docs}


builder = StateGraph(IngestionState)

builder.add_node("detect_type", detect_type_node)

builder.add_node("parse_pdf", parse_pdf_node)
builder.add_node("parse_docx", parse_docx_node)
builder.add_node("parse_excel", parse_excel_node)
builder.add_node("parse_pptx", parse_pptx_node)
builder.add_node("parse_markdown", parse_markdown_node)
builder.add_node("parse_text", parse_text_node)

builder.add_edge(START, "detect_type")

builder.add_conditional_edges(
    "detect_type",
    route_by_type,
)

builder.add_edge("parse_pdf", END)
builder.add_edge("parse_docx", END)
builder.add_edge("parse_excel", END)
builder.add_edge("parse_pptx", END)
builder.add_edge("parse_markdown", END)
builder.add_edge("parse_text", END)

ingestion_graph = builder.compile()
# ingestion_graph.get_graph().print_ascii()
