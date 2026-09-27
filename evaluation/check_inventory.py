import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".pptx", ".md", ".txt"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_inventory(project_root=PROJECT_ROOT):
    docs_dir = project_root / "docs"
    permissions_path = project_root / "document_permissions.json"
    users_path = project_root / "users.json"
    chunks_path = project_root / "vector_store" / "chunks.json"
    index_path = project_root / "vector_store" / "index.faiss"
    errors = []

    for path in (docs_dir, permissions_path, users_path, chunks_path, index_path):
        if not path.exists():
            errors.append(f"required inventory asset not found: {path}")
    if errors:
        return errors

    permissions = read_json(permissions_path).get("documents")
    if not isinstance(permissions, dict):
        return ["document_permissions.json must contain a 'documents' object"]

    users = read_json(users_path).get("users", [])
    valid_groups = {
        group
        for user in users
        for group in user.get("groups", [])
        if isinstance(group, str)
    }
    document_paths = {
        path.relative_to(project_root).as_posix()
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }

    for source_path in sorted(document_paths - permissions.keys()):
        errors.append(f"document has no permission entry: {source_path}")
    for source_path in sorted(permissions.keys() - document_paths):
        errors.append(f"permission entry has no document: {source_path}")

    for source_path, groups in permissions.items():
        if not isinstance(groups, list):
            errors.append(f"permission groups must be a list: {source_path}")
        elif any(group not in valid_groups for group in groups):
            errors.append(f"permission entry contains an unknown group: {source_path}")

    chunks = read_json(chunks_path)
    indexed_paths = set()
    for chunk_number, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        if not isinstance(metadata, dict):
            errors.append(f"chunk {chunk_number} has invalid metadata")
            continue

        source_path = metadata.get("source_path")
        if not isinstance(source_path, str):
            errors.append(f"chunk {chunk_number} has no source_path")
            continue

        indexed_paths.add(source_path)
        if source_path not in document_paths:
            errors.append(f"chunk source does not exist in docs: {source_path}")
            continue

        allowed_groups = metadata.get("allowed_groups")
        expected_groups = permissions.get(source_path)
        if not isinstance(allowed_groups, list):
            errors.append(f"chunk {chunk_number} has invalid allowed_groups: {source_path}")
        elif expected_groups is not None and set(allowed_groups) != set(expected_groups):
            errors.append(f"chunk permission mismatch: {source_path}")
        if isinstance(allowed_groups, list) and any(
            group not in valid_groups for group in allowed_groups
        ):
            errors.append(f"chunk contains an unknown group: {source_path}")

    for source_path in sorted(document_paths - indexed_paths):
        errors.append(f"document has no indexed chunks: {source_path}")

    try:
        import faiss

        index = faiss.read_index(str(index_path))
        if index.ntotal != len(chunks):
            errors.append(
                f"FAISS/chunk count mismatch: index has {index.ntotal}, chunks has {len(chunks)}"
            )
    except ImportError:
        errors.append("faiss-cpu is required to verify index/chunk alignment")
    except RuntimeError as error:
        errors.append(f"could not read FAISS index: {error}")

    return errors


def main():
    errors = check_inventory()
    if errors:
        for error in errors:
            print(f"FAIL {error}")
        print(f"\nInventory check failed with {len(errors)} issue(s).")
        return 1

    print("PASS document, permission, chunk, and FAISS inventories are aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())