# KnowledgeGaurd

KnowledgeGaurd retrieves relevant company-document chunks, filters them by a configured user's groups, and uses Groq to answer questions from the permitted context. The evaluation tools check retrieval and access control separately from generated-answer quality.

## Architecture

KnowledgeGaurd has two flows: ingestion builds a searchable, permission-tagged corpus; question answering retrieves from that corpus and only passes authorized chunks to the language model.

### Document Ingestion

```mermaid
flowchart LR
	A[Files in docs/] --> B[Detect file type]
	B --> C{Route to parser}
	C -->|PDF, DOCX, XLSX, PPTX, Markdown, text| D[Extract text and source metadata]
	D --> E[Look up path in document_permissions.json]
	E --> F[Attach allowed_groups and source_path]
	F --> G[Split into overlapping chunks]
	G --> H[Embed chunks]
	H --> I[(FAISS index)]
	G --> J[(chunks.json)]
	K[users.json defines valid groups] --> E
```

The ingestion graph routes each supported file to its format-specific parser. Permissions are attached using the document's repository-relative path before chunking; the splitter carries metadata into each chunk. The index stores vectors for similarity search, while `chunks.json` stores corresponding text and metadata. The chunk order must stay aligned with the FAISS row order.

### Question Answering

```mermaid
flowchart LR
	Q[Question] --> E[Embed and normalize question]
	E --> S[Search all FAISS vectors]
	S --> R[Ranked chunk IDs]
	U[Configured user ID] --> G[Load user groups]
	C[chunks.json metadata] --> F{allowed_groups intersects user groups?}
	R --> F
	G --> F
	F -->|No| X[Discard chunk]
	F -->|Yes| T[Rank authorized chunks with Hybrid FAISS + BM25 RRF]
	T --> L[Keep top 8 authorized chunks]
	L --> P[Build grounded prompt]
	P --> M[Groq chat model]
	M --> A[Answer with source guidance]
```

The app searches the shared index first, filters the ranked results by `allowed_groups`, and reranks authorized candidates using Hybrid FAISS + BM25 Reciprocal Rank Fusion with source diversity before building the Groq prompt. Unauthorized chunks are not passed to answer generation. Searching all vectors before filtering lets an authorized match be found even when restricted chunks rank ahead of it; retrieval returns up to eight authorized chunks.

The current app does not authenticate users. `USER_ID` in `app.py` selects a configured demo identity (currently `u002`, HR), and `users.json` supplies that identity's groups. This is a demonstration access-control boundary, not a replacement for application authentication, tenant isolation, or production security controls.

## How This Differs From A Typical RAG App

The core RAG pattern is the same: parse documents, chunk and embed them, retrieve relevant passages, then ask an LLM to answer from those passages. The differences are in access filtering and evaluation, not a different kind of language model or search algorithm.

| Area        | Typical basic RAG                             | KnowledgeGaurd                                                                                                                                  |
| ----------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Documents   | Often one format or a simple loader           | A LangGraph ingestion router dispatches PDF, DOCX, XLSX, PPTX, Markdown, and text to format-specific parsers.                                   |
| Permissions | Often one shared corpus for every user        | Chunks carry`allowed_groups`; retrieval keeps only chunks authorized for the configured user's groups before generation.                      |
| Retrieval   | Often asks the vector store for a fixed top-k | Searches the indexed vectors, filters by group, then reranks authorized candidates with Hybrid FAISS + BM25 RRF (up to 8 chunks).               |
| Identity    | May use an authenticated application identity | Currently selects a demo user ID from`users.json`; real authentication is not implemented.                                                    |
| Evaluation  | May rely on manual spot checks                | Includes fake-backed permission tests, a fact/source golden set, and an inventory check for docs, permissions, chunks, and FAISS row alignment. |

This does not make retrieval or answers automatically correct: vector similarity can miss relevant evidence, and Groq can still produce an incorrect answer. The golden evaluation measures retrieved facts and source metadata; it does not grade generated answer prose.

## Requirements

- Python and packages listed in `requirements.txt`.
- A Groq API key to run the interactive chat application.
- A populated `docs/` directory and matching entries in `document_permissions.json` to build the vector store.
- A local embedding model and vector store to run retrieval evaluation. The checked-in model and vector store can be used as-is.

Run commands from the repository root in PowerShell.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env` only if it does not already exist, then add your Groq API key:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

```env
GROQ_API_KEY=your_groq_api_key_here
```

The test and retrieval evaluation commands do not need a Groq key. The interactive app does.

## Build Or Refresh The Index

Run ingestion after adding or changing source documents or their permissions:

```powershell
python ingestion/app.py
```

Ingestion reads supported files from `docs/`, applies `document_permissions.json`, and writes the FAISS index and chunk records under `vector_store/`. It may download the embedding model and takes longer than the tests. Run the inventory check afterward to catch missing permission entries or stale indexed documents.

## Ask A Question

Start the interactive app:

```powershell
python app.py
```

Example question:

```text
How many vacation days do employees get?
```

The app loads the local embedding model and existing FAISS index. If the embedding model is not cached, it attempts to download it. It requires `GROQ_API_KEY` to generate answers. The demo currently uses user `u002` (HR), configured by `USER_ID` in `app.py`; use a different configured ID there to run the app as another user.

## Evaluation Commands

Run all deterministic, fake-backed unit tests (including retrieval permissions and citation/refusal evaluators):

```powershell
python -m unittest discover -s evaluation -p "test_*.py" -v
```

These tests check retrieval access-control logic, citation accuracy detection, hallucination checks, and refusal response handling. They run in milliseconds without needing Groq or model weights.

Run the golden retrieval cases against the actual local model and vector store:

```powershell
python evaluation/run_evaluation.py
```

This runs the questions in `evaluation/cases.json` for their configured users. It checks that expected fact phrases and source paths appear in retrieved chunks and that forbidden sources do not.

Run the comprehensive End-to-End RAG & Permission Evaluation:

```powershell
python evaluation/evaluate_rag.py
```

Options:
- `--deny-only`: Evaluate only permission-restricted questions to verify 0.0% ACL leakage and graceful refusals.
- `--limit 5`: Run the first 5 test cases.
- `--case <case_id>`: Run a specific test case by ID.
- `--no-llm-judge`: Skip LLM-as-a-judge calls to evaluate retrieval and citations rapidly.

This evaluates the full RAG pipeline with Groq:
1. **ACL Leakage Rate**: Ensures restricted documents never reach the LLM context.
2. **Refusal Correctness**: Verifies the standard refusal is returned on unauthorized queries.
3. **Faithfulness & Groundedness**: Uses an LLM judge to verify every claim against retrieved context.
4. **Citation Accuracy**: Verifies that the `Sources:` section references valid documents and pages.

Check the source, permission, chunk, and FAISS inventories:

```powershell
python evaluation/check_inventory.py
```

This checks that supported files in `docs/` have permission entries and indexed chunks, that chunk permissions match the configuration, and that the FAISS row count matches the chunk count.

---

## Observability & Enterprise Audit Logging

KnowledgeGuard includes enterprise access logging and tracing:

### 1. Observability Dashboard
View query volume, access denial rates, latency metrics, and top accessed vs. blocked documents:

```powershell
python observability/dashboard.py
```

### 2. Audit Logging
Every query writes a structured audit record to `logs/audit.jsonl`, detailing:
- User ID and active groups
- Candidates scanned vs. authorized vs. denied chunks
- Denied source paths and their required clearance groups
- Access decision (`GRANTED`, `PARTIAL_ACCESS`, `DENIED`, or `NO_MATCH`)
- Execution latency in milliseconds

### 3. Distributed Tracing (LangSmith / Arize Phoenix)
To trace LangGraph node transitions and Groq token usage with LangSmith, add your key to `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=KnowledgeGaurd
```

Tracing initializes automatically on startup without modifying code.
