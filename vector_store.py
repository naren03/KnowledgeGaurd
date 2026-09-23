import json
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import PDF_PATH, VECTOR_STORE_PATH


def create_vector_store(embedding_model):
    reader = PdfReader(PDF_PATH)
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(pdf_text)

    # embedding_model = SentenceTransformer(MODEL_CACHE_PATH)
    vectors = embedding_model.encode(chunks, normalize_embeddings=True)
    vectors = np.array(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    vector_store_dir = Path(VECTOR_STORE_PATH)
    vector_store_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(vector_store_dir / "index.faiss"))
    (vector_store_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    return index, chunks
