import pickle
from pathlib import Path

import faiss
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ==========================================
# RICH
# ==========================================

console = Console()


# ==========================================
# PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FAISS_DIR = PROJECT_ROOT / "vector_store"

INDEX_PATH = FAISS_DIR / "index.faiss"

PICKLE_PATH = FAISS_DIR / "index.pkl"

PREVIEW_LENGTH = 250


# ==========================================
# INSPECT
# ==========================================


def main():

    console.print(
        Panel.fit(
            "[bold cyan]FAISS Chunk Inspector[/bold cyan]\n"
            "[white]Direct folder inspection[/white]",
            border_style="cyan",
        )
    )

    if not INDEX_PATH.exists():
        console.print(f"[red]Missing:[/red] {INDEX_PATH}")

        return

    if not PICKLE_PATH.exists():
        console.print(f"[red]Missing:[/red] {PICKLE_PATH}")

        return

    # --------------------------------------
    # LOAD FAISS INDEX
    # --------------------------------------

    console.print("[cyan]Reading FAISS index...[/cyan]")

    index = faiss.read_index(str(INDEX_PATH))

    total_chunks = index.ntotal

    console.print(f"[green]✓ Total vectors:[/green] {total_chunks}")

    # --------------------------------------
    # LOAD DOCSTORE
    # --------------------------------------

    console.print("[cyan]Reading stored documents...[/cyan]")

    # IMPORTANT:
    # Only open this pickle if the index was
    # created by you or another trusted source.

    with open(PICKLE_PATH, "rb") as file:
        docstore, index_to_docstore_id = pickle.load(file)

    console.print("[green]✓ Docstore loaded.[/green]")

    # --------------------------------------
    # ITERATE ALL CHUNKS
    # --------------------------------------

    for index_position in range(total_chunks):
        docstore_id = index_to_docstore_id[index_position]

        document = docstore.search(docstore_id)

        if document is None:
            console.print(f"[yellow]Chunk {index_position + 1}: Not found[/yellow]")

            continue

        # ----------------------------------
        # CONTENT PREVIEW
        # ----------------------------------

        content = document.page_content.strip()

        preview = content[:PREVIEW_LENGTH]

        if len(content) > PREVIEW_LENGTH:
            preview += "..."

        # ----------------------------------
        # CHUNK HEADER
        # ----------------------------------

        console.print()

        console.rule(
            f"[bold cyan]Chunk {index_position + 1} / {total_chunks}[/bold cyan]"
        )

        # ----------------------------------
        # METADATA
        # ----------------------------------

        table = Table(
            title="Metadata",
            box=box.SIMPLE,
            header_style="bold magenta",
        )

        table.add_column(
            "Field",
            style="cyan",
        )

        table.add_column(
            "Value",
            style="white",
        )

        table.add_row(
            "Docstore ID",
            str(docstore_id),
        )

        for key, value in document.metadata.items():
            table.add_row(
                str(key),
                str(value),
            )

        console.print(table)

        # ----------------------------------
        # CONTENT
        # ----------------------------------

        console.print(
            Panel(
                preview,
                title="[bold green]Content Preview[/bold green]",
                border_style="green",
            )
        )

    # --------------------------------------
    # COMPLETE
    # --------------------------------------

    console.print()

    console.rule("[bold green]Inspection Complete[/bold green]")

    console.print(f"Total chunks inspected: {total_chunks}")


if __name__ == "__main__":
    main()
