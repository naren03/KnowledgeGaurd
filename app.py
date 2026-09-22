import json
from pathlib import Path
from typing import TypedDict
from config import  PDF_PATH,VECTOR_STORE_PATH,MODEL_CACHE_PATH,MODEL,EMBEDDING_MODEL
from state import State
from vector_store import create_vector_store

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from sentence_transformers import SentenceTransformer


console = Console()

load_dotenv()

if Path(MODEL_CACHE_PATH).exists():
    console.print("[green]Loading existing embedding model...[/green]")
    embedding_model = SentenceTransformer(MODEL_CACHE_PATH)
else:
    console.print("[yellow]Downloading embedding model...[/yellow]")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embedding_model.save(MODEL_CACHE_PATH)


index_path = Path(VECTOR_STORE_PATH) / "index.faiss"
chunks_path = Path(VECTOR_STORE_PATH) / "chunks.json"

if index_path.exists() and chunks_path.exists():
    console.print("[green]Loading existing vector store...[/green]")
    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
else:
    console.print("[yellow]Creating vector store...[/yellow]")
    index, chunks = create_vector_store()


llm = ChatGroq(model=MODEL, temperature=0)


def retrieve(state: State):
    question_vector = embedding_model.encode([state["question"]], normalize_embeddings=True)
    question_vector = np.array(question_vector, dtype="float32")

    _, chunk_indexes = index.search(question_vector, k=4)
    selected_chunks = [chunks[i] for i in chunk_indexes[0]]

    return {"context": selected_chunks}


def generate(state: State):
    context = "\n\n".join(state["context"])

    prompt = f"""
            Answer the question using only the context below.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {state["question"]}
    """

    response = llm.invoke(prompt)
    return {"answer": response.content}


graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)


graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)


graph = graph_builder.compile()
# graph.get_graph().print_ascii()


def ask_pdf(question: str) -> str:
    """Answer a question using the local PDF RAG workflow."""
    result = graph.invoke({"question": question})
    return result["answer"]


pdf_rag_tool = StructuredTool.from_function(
    func=ask_pdf,
    name="pdf_rag_tool",
    description="Use this tool to answer questions from the local HR policy PDF.",
)
tools = [pdf_rag_tool]
chat_llm = llm.bind_tools(tools)
tools_by_name = {tool.name: tool for tool in tools}

messages = [
    SystemMessage(
        content=(
            "You are a helpful chat assistant. "
            "Use pdf_rag_tool when the user asks about the HR policy PDF, leave, "
            "vacation, health, wellbeing, sick leave, holidays, or anything that "
            "may be answered from the local document. "
            "For normal conversation, answer directly without using a tool."
        )
    )
]


console.print(
    Panel.fit(
        "Chat normally, or ask about the HR policy PDF. Type [bold]exit[/bold], [bold]quit[/bold], or [bold]q[/bold] to stop.",
        title="Chat App",
        border_style="cyan",
    )
)

while True:
    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

    if user_input.lower() in {"exit", "quit", "q"}:
        console.print("[cyan]Goodbye![/cyan]")
        break

    if not user_input:
        console.print("[red]Please enter a message.[/red]")
        continue

    with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
        messages.append(HumanMessage(content=user_input))
        ai_message = chat_llm.invoke(messages)
        messages.append(ai_message)

        if ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                tool = tools_by_name[tool_call["name"]]
                tool_result = tool.invoke(tool_call["args"])
                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                )

            final_message = llm.invoke(messages)
            messages.append(final_message)
            answer = final_message.content
        else:
            answer = ai_message.content

    console.print(
        Panel(
            Markdown(answer),
            title="Assistant",
            border_style="green",
        )
    )
