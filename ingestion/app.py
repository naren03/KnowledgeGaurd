import json
from pathlib import Path

from chunker import chunk_documents
from graph import ingestion_graph
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.traceback import install

# ==========================================
# RICH CONFIGURATION
# ==========================================

install()

console = Console()


# ==========================================
# PATH CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "docs"

FAISS_DIR = PROJECT_ROOT / "vector_store"

CHUNKS_JSON_PATH = FAISS_DIR / "chunks.json"
DOCUMENT_PERMISSIONS_PATH = PROJECT_ROOT / "document_permissions.json"
USERS_PATH = PROJECT_ROOT / "users.json"


# ==========================================
# EMBEDDING MODEL
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# ==========================================
# SUPPORTED EXTENSIONS
# ==========================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".md",
    ".txt",
}


# ==========================================
# FIND DOCUMENTS
# ==========================================


def get_document_files():

    return sorted(
        file_path
        for file_path in DOCS_DIR.rglob("*")
        if (file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS)
    )


def load_document_permissions():
    permission_data = json.loads(
        DOCUMENT_PERMISSIONS_PATH.read_text(encoding="utf-8")
    )
    users_data = json.loads(USERS_PATH.read_text(encoding="utf-8"))
    valid_groups = {
        group
        for user in users_data["users"]
        for group in user["groups"]
    }
    document_permissions = permission_data["documents"]

    if not isinstance(document_permissions, dict):
        raise ValueError("The 'documents' permission entry must be an object.")

    for document_path, groups in document_permissions.items():
        if not isinstance(groups, list) or any(
            group not in valid_groups for group in groups
        ):
            raise ValueError(
                f"Invalid group list for {document_path!r}; "
                f"allowed groups are {sorted(valid_groups)}."
            )

    return document_permissions


# ==========================================
# INGESTION
# ==========================================


def main():

    console.print(
        Panel.fit(
            "[bold cyan]Knowledge Guard[/bold cyan]\n"
            "[white]Multi-format document ingestion[/white]",
            border_style="cyan",
        )
    )

    console.print(f"\n[bold]Documents:[/bold] {DOCS_DIR}")

    if not DOCS_DIR.is_dir():
        console.print("[bold red]Docs directory not found![/bold red]")

        return

    document_files = get_document_files()

    if not document_files:
        console.print("[yellow]No supported documents found.[/yellow]")

        return

    document_permissions = load_document_permissions()

    console.print(f"[green]Found {len(document_files)} documents.[/green]\n")

    # One FAISS instance for the entire run.
    vector_store = None

    # Collect all chunks for optional JSON export.
    all_chunks = []

    total_files = 0
    total_chunks = 0
    failed_files = 0

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Processing documents...",
            total=len(document_files),
        )

        for file_path in document_files:
            progress.update(
                task,
                description=f"[cyan]Processing {file_path.name}",
            )

            try:
                # ------------------------------
                # STEP 1: INVOKE INGESTION GRAPH
                # ------------------------------

                console.print(
                    f"\n[bold blue]▶[/bold blue] [bold]{file_path.name}[/bold]"
                )

                result = ingestion_graph.invoke({"file_path": str(file_path)})

                raw_documents = result.get("raw_documents", [])

                if not raw_documents:
                    console.print("[yellow]No content extracted.[/yellow]")

                    results.append((file_path.name, 0, "No content"))

                    progress.advance(task)

                    continue

                console.print(
                    f"  [green]✓[/green] Extracted: {len(raw_documents)} documents"
                )

                # ------------------------------
                # STEP 2: ADD METADATA
                # ------------------------------

                relative_source = file_path.relative_to(PROJECT_ROOT).as_posix()
                allowed_groups = document_permissions.get(relative_source, [])

                if not allowed_groups:
                    console.print(
                        f"  [yellow]No groups configured for {relative_source}; "
                        "chunks will be inaccessible.[/yellow]"
                    )

                for document in raw_documents:
                    document.metadata.update(
                        {
                            "document_id": file_path.name,
                            "source_path": relative_source,
                            "file_name": file_path.name,
                            "file_type": file_path.suffix.lower(),
                            "allowed_groups": allowed_groups,
                        }
                    )

                # ------------------------------
                # STEP 3: CHUNK DOCUMENTS
                # ------------------------------

                chunks = chunk_documents(raw_documents)

                console.print(f"  [green]✓[/green] Generated chunks: {len(chunks)}")

                if not chunks:
                    console.print("[yellow]No chunks generated.[/yellow]")

                    results.append((file_path.name, 0, "No chunks"))

                    progress.advance(task)

                    continue

                # Collect chunks for JSON export.
                all_chunks.extend(chunks)

                # ------------------------------
                # STEP 4: ADD TO SAME FAISS
                # ------------------------------

                if vector_store is None:
                    vector_store = FAISS.from_documents(chunks, embeddings)

                else:
                    vector_store.add_documents(documents=chunks)

                total_files += 1

                total_chunks += len(chunks)

                results.append((file_path.name, len(chunks), "Success"))

                console.print("  [green]✓ Stored in FAISS[/green]")

            except Exception as error:
                failed_files += 1

                results.append((file_path.name, 0, "Failed"))

                console.print(f"  [bold red]✗ Failed:[/bold red] {error}")

            finally:
                progress.advance(task)

    # ==========================================
    # SAVE FAISS
    # ==========================================

    if vector_store is None:
        console.print("\n[bold red]No documents were indexed.[/bold red]")

        return

    console.print("\n[bold cyan]Saving FAISS index...[/bold cyan]")

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(FAISS_DIR))

    console.print("[green]✓ FAISS index saved successfully.[/green]")

    # ==========================================
    # SAVE CHUNKS TO JSON
    # ==========================================

    console.print("\n[bold cyan]Saving chunks to JSON...[/bold cyan]")

    chunk_records = [
        {
            "chunk_id": index + 1,
            "content": document.page_content,
            "metadata": document.metadata,
        }
        for index, document in enumerate(all_chunks)
    ]

    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(
            chunk_records,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    console.print(
        f"[green]✓ Saved {len(chunk_records)} chunks to:[/green] {CHUNKS_JSON_PATH}"
    )

    # ==========================================
    # SUMMARY TABLE
    # ==========================================

    table = Table(
        title="Ingestion Summary",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("File")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")

    for file_name, chunks, status in results:
        status_text = (
            "[green]Success[/green]"
            if status == "Success"
            else (
                "[yellow]" + status + "[/yellow]"
                if status != "Failed"
                else "[red]Failed[/red]"
            )
        )

        table.add_row(
            file_name,
            str(chunks),
            status_text,
        )

    console.print()
    console.print(table)

    console.print(
        Panel(
            f"[bold green]Ingestion completed[/bold green]\n\n"
            f"Files indexed: {total_files}\n"
            f"Total chunks: {total_chunks}\n"
            f"Failed files: {failed_files}\n"
            f"FAISS location: {FAISS_DIR}\n"
            f"Chunks JSON: {CHUNKS_JSON_PATH}",
            title="Final Result",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
