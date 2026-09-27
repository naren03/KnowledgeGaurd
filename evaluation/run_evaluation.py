import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import EMBEDDING_MODEL, MODEL_CACHE_PATH, VECTOR_STORE_PATH
from retrieval import retrieve_chunks


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_case(case, users_by_id, embedding_model, index, chunks):
    user = users_by_id.get(case["user_id"])
    if user is None:
        return [f"unknown user_id: {case['user_id']}"]

    selected = retrieve_chunks(
        question=case["question"],
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        user_groups=user["groups"],
        limit=4,
    )

    chunks_by_source = {}
    for chunk in selected:
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        source_path = metadata.get("source_path")
        if isinstance(source_path, str):
            chunks_by_source.setdefault(source_path, []).append(
                str(chunk.get("content", ""))
            )

    failures = []
    for expected in case.get("expected", []):
        source_path = expected["source_path"]
        source_content = "\n".join(chunks_by_source.get(source_path, []))
        if not source_content:
            failures.append(f"missing expected source: {source_path}")
            continue
        content_lower = source_content.casefold()
        for phrase in expected.get("contains", []):
            if phrase.casefold() not in content_lower:
                failures.append(f"missing expected fact in {source_path}: {phrase}")

    for source_path in case.get("forbidden_sources", []):
        if source_path in chunks_by_source:
            failures.append(f"forbidden source retrieved: {source_path}")

    return failures


def main():
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        print(f"Missing runtime dependency: {error}", file=sys.stderr)
        return 2

    users_path = PROJECT_ROOT / "users.json"
    chunks_path = PROJECT_ROOT / VECTOR_STORE_PATH / "chunks.json"
    index_path = PROJECT_ROOT / VECTOR_STORE_PATH / "index.faiss"
    cases_path = Path(__file__).resolve().parent / "cases.json"
    model_path = PROJECT_ROOT / MODEL_CACHE_PATH

    for required_path in (users_path, chunks_path, index_path, cases_path, model_path):
        if not required_path.exists():
            print(f"Required evaluation asset not found: {required_path}", file=sys.stderr)
            return 2

    users_by_id = {
        user["user_id"]: user for user in read_json(users_path)["users"]
    }
    chunks = read_json(chunks_path)
    cases = read_json(cases_path)
    index = faiss.read_index(str(index_path))
    embedding_model = SentenceTransformer(str(model_path))

    failures = 0
    for case in cases:
        case_failures = evaluate_case(case, users_by_id, embedding_model, index, chunks)
        if case_failures:
            failures += 1
            print(f"FAIL {case['id']}")
            for failure in case_failures:
                print(f"  - {failure}")
        else:
            print(f"PASS {case['id']}")

    passed = len(cases) - failures
    print(f"\n{passed}/{len(cases)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())