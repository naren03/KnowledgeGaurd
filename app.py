import json
import uuid
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    MODEL,
    MODEL_CACHE_PATH,
    VECTOR_STORE_PATH,
)
from state import ChatState
from vector_store import create_vector_store

# ============================================================
# 1. INITIALIZATION
# ============================================================

console = Console()

load_dotenv()

thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": thread_id}, "run_name": "KnowledgeGaurd"}
# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

if Path(MODEL_CACHE_PATH).exists():
    console.print("[green]Loading existing embedding model...[/green]")

    embedding_model = SentenceTransformer(MODEL_CACHE_PATH)

else:
    console.print("[yellow]Downloading embedding model...[/yellow]")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    embedding_model.save(MODEL_CACHE_PATH)


# ============================================================
# 3. LOAD OR CREATE VECTOR STORE
# ============================================================

index_path = Path(VECTOR_STORE_PATH) / "index.faiss"
chunks_path = Path(VECTOR_STORE_PATH) / "chunks.json"


# if index_path.exists() and chunks_path.exists():
console.print("[green]Loading existing vector store...[/green]")

index = faiss.read_index(str(index_path))

chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

# else:
#     console.print("[yellow]Creating vector store...[/yellow]")

#     index, chunks = create_vector_store(embedding_model)


# ============================================================
# 4. INITIALIZE LLM
# ============================================================

llm = ChatGroq(model=MODEL, temperature=0)


# ============================================================
# 5. RETRIEVE NODE
# ============================================================


def retrieve(state: ChatState):

    question = state["question"]

    # Convert question into embedding
    question_vector = embedding_model.encode([question], normalize_embeddings=True)

    question_vector = np.asarray(question_vector, dtype="float32")

    # Prevent requesting more chunks than available
    k = min(4, index.ntotal)

    if k == 0:
        return {"context": []}

    # Search FAISS
    _, chunk_indexes = index.search(question_vector, k=k)

    # Collect retrieved chunks
    selected_chunks = [chunks[i] for i in chunk_indexes[0] if i != -1]

    return {"context": selected_chunks}


# ============================================================
# 6. GENERATE NODE
# ============================================================


def generate(state: ChatState):

    question = state["question"]

    context = state.get("context", [])

    # No retrieved context
    if not context:
        return {
            "answer": ("I don't know based on the available Lumetra knowledge base.")
        }

    # Convert retrieved chunks into text
    context_text = "\n\n".join(str(chunk) for chunk in context)

    prompt = f"""
    You are KnowledgeGaurd, Lumetra's company knowledge assistant.

    Your task is to answer the user's question using ONLY
    the retrieved context from Lumetra's internal knowledge base.

    Rules:
    1. Use only the provided context.
    2. Do not use external knowledge.
    3. Do not invent facts or information.
    4. If the answer is not available in the context,
    respond exactly:
    "I don't know based on the available Lumetra knowledge base."
    5. Give a clear and helpful answer.
    6. If the context is insufficient, do not guess.
    7. After the answer, provide a "Sources" section.
    8. For each source, use ONLY the document metadata provided
    in the retrieved context.
    9. Include the document name/title and page number for each
    retrieved chunk that was actually used to answer the question.
    10. Do NOT guess, infer, or create document names or page numbers.
    11. If multiple chunks from the same document and page are used,
        list that source only once.
    12. Keep the Sources section at the end of the response.

    Retrieved Context:
    ------------------
    {context_text}
    ------------------

    User Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}


# ============================================================
# 7. BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(ChatState)

# Add nodes
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

# Add edges
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

# Compile graph
graph = graph_builder.compile()


# ============================================================
# 8. OPTIONAL GRAPH VISUALIZATION
# ============================================================

# Uncomment to see the graph in your terminal:
#
# graph.get_graph().print_ascii()


# ============================================================
# 9. ASK KNOWLEDGEGAURD
# ============================================================


def ask_knowledgeguard(question: str) -> str:
    """
    Ask a question using the fixed RAG workflow.

    Every question follows:
    START -> RETRIEVE -> GENERATE -> END
    """

    result = graph.invoke({"question": question}, config=config)

    return result["answer"]


# ============================================================
# 10. TERMINAL UI
# ============================================================

console.print(
    Panel.fit(
        "[bold cyan]Welcome to KnowledgeGaurd[/bold cyan]\n\n"
        "[bold white]Lumetra's Knowledge Assistant[/bold white]\n\n"
        "I'm an AI-powered chatbot designed to help you "
        "explore and understand Lumetra's knowledge base.\n\n"
        "Ask me anything related to Lumetra, including "
        "company information, policies, processes, "
        "documents, and more.\n\n"
        "[italic]How can I help you today?[/italic]\n\n"
        "[dim]Type [bold]exit[/bold], [bold]quit[/bold], "
        "or [bold]q[/bold] to stop.[/dim]",
        border_style="cyan",
        padding=(1, 2),
    )
)


# ============================================================
# 11. CHAT LOOP
# ============================================================

while True:
    user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

    if user_input.lower() in {"exit", "quit", "q"}:
        console.print("[cyan]Goodbye![/cyan]")

        break

    if not user_input:
        console.print("[red]Please enter a message.[/red]")

        continue

    with console.status(
        "[bold green]Searching Lumetra knowledge base...[/bold green]", spinner="dots"
    ):
        answer = ask_knowledgeguard(user_input)

    console.print(Panel(Markdown(answer), title="KnowledgeGaurd", border_style="green"))
