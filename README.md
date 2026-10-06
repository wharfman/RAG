# Knowledge Base Assistant

Knowledge Base Assistant is a local full-stack retrieval-augmented generation (RAG)
application. It indexes Markdown and text files, stores their embeddings in PostgreSQL
with pgvector, retrieves relevant passages for a question, and asks an OpenAI model to
produce a grounded answer with source citations.

The repository includes:

- A responsive Next.js chat interface
- A FastAPI backend for document status, ingestion, health checks, and chat
- A Python command-line interface for validation, indexing, querying, and database reset
- PostgreSQL 17 with pgvector and full-text search for hybrid retrieval
- Automated tests and static-analysis commands

## How it works

```text
Files in data/
      |
      v
Discover -> normalize -> chunk -> embed -> PostgreSQL/pgvector + tsvector
                                                       |
User question -> pgvector cosine rank -----------------|
              -> tsvector BM25 rank -> RRF ------------|
                                                       v
                              retrieved excerpts + recent chat history
                                                   |
                                                   v
                                      OpenAI Responses API
                                                   |
                                                   v
                                    answer + cited sources
```

During ingestion, each file and generated chunk receives a SHA-256 hash. A later sync
skips unchanged files, replaces changed files, and removes database records for files
that no longer exist in `data/`. Source files are only read; they are never rewritten.

## Features

- Recursively discovers `.md`, `.markdown`, and `.txt` files
- Preserves Markdown heading paths as chunk metadata
- Uses bounded, line-aware fallback chunking for plain text
- Repairs common text-extraction encoding artifacts in memory with `ftfy`
- Generates embeddings in batches and stores them in pgvector
- Uses an HNSW cosine-distance index for semantic retrieval
- Uses a stored English `tsvector`, GIN index, and BM25 (`k1=1.2`, `b=0.75`)
  for lexical retrieval
- Fuses the independent rankings with reciprocal rank fusion (RRF, `k=60`), not
  weighted score blending
- Returns the retrieved file name, section heading, source identifier, and RRF score
- Includes up to eight recent chat messages when generating an answer
- Saves the current browser conversation in local storage
- Shows file indexing and database status in the frontend
- Prevents overlapping ingestion jobs within one API process

## Technology and tools

| Area | Software | Role |
| --- | --- | --- |
| Frontend | Next.js 16.3, React 19.2, TypeScript 7 | Web interface and browser client |
| Styling | CSS | Responsive application layout and components |
| Backend | Python 3.12, FastAPI, Uvicorn, Pydantic | HTTP API, validation, and application server |
| AI | OpenAI Python SDK and Responses API | Text embeddings and grounded answer generation |
| Database | PostgreSQL 17, pgvector, `tsvector`, SQLAlchemy, Psycopg | Metadata and hybrid search |
| Text processing | `ftfy`, Python standard library | Encoding repair, hashing, discovery, and chunking |
| Configuration | `python-dotenv` | Loads server settings from the root `.env` file |
| Containers | Docker Desktop and Docker Compose | Runs the local PostgreSQL/pgvector service |
| Python quality | pytest, Ruff | Automated tests, linting, and import/style checks |
| Frontend quality | TypeScript compiler, Next.js build | Type checking and production-build validation |
| Package management | pip, npm | Installs Python and frontend dependencies |

Exact Python dependency ranges are in `requirements.txt`. Exact frontend versions and
the reproducible dependency graph are in `frontend/package.json` and
`frontend/package-lock.json`.

## Project structure

```text
RAG/
|-- api.py                         # FastAPI application and HTTP endpoints
|-- main.py                        # Python CLI entry point
|-- docker-compose.yml             # PostgreSQL 17 + pgvector service
|-- requirements.txt               # Runtime and development Python dependencies
|-- pyproject.toml                 # pytest and Ruff configuration
|-- .env.example                   # Backend environment-variable template
|-- .gitignore                     # Local data, secrets, caches, and build exclusions
|-- data/                          # Local knowledge files (ignored by Git)
|-- frontend/
|   |-- app/
|   |   |-- layout.tsx             # Root layout and page metadata
|   |   |-- page.tsx               # Chat UI, API calls, and browser state
|   |   `-- globals.css            # Global styles and responsive layout
|   |-- .env.local.example         # Public backend URL template
|   |-- next.config.ts             # Next.js configuration
|   |-- tsconfig.json              # TypeScript configuration
|   |-- package.json               # Frontend scripts and direct dependencies
|   `-- package-lock.json          # Locked npm dependency tree
|-- src/
|   |-- config.py                  # Environment-backed application settings
|   |-- ingestion.py               # Directory sync and offline validation
|   |-- rag_pipeline.py            # RAG component orchestration
|   |-- interface/                 # Abstract component contracts and shared models
|   |   |-- base_datastore.py
|   |   |-- base_evaluator.py
|   |   |-- base_indexer.py
|   |   |-- base_response_generator.py
|   |   `-- base_retriever.py
|   |-- impl/                      # Current concrete implementations
|   |   |-- datastore.py           # pgvector + BM25 retrieval and RRF fusion
|   |   |-- evaluator.py           # Optional model-based answer evaluator
|   |   |-- indexer.py             # File discovery, normalization, and chunking
|   |   |-- response_generator.py  # Grounded OpenAI response generation
|   |   `-- retriever.py           # Retrieval adapter
|   `-- util/
|       |-- extract_xml.py         # Evaluator response parsing helper
|       `-- invoke_ai.py           # Shared OpenAI invocation helper
`-- tests/
    |-- test_chunking.py            # Discovery, chunking, headings, and encoding tests
    |-- test_config.py              # Default model and embedding configuration tests
    `-- test_hybrid_search.py       # Reciprocal rank fusion behavior tests
```

The abstract classes under `src/interface/` separate the pipeline from its concrete
implementations. This makes it possible to substitute another datastore, indexer,
retriever, response generator, or evaluator without redesigning the orchestrator.

## Prerequisites

Install the following before setting up the project:

- Python 3.12
- Node.js 20.9 or newer, with npm
- Docker Desktop with Docker Compose
- An OpenAI API key with access to the configured generation and embedding models
- On Windows, WSL 2 for Docker Desktop's Linux container engine

The commands below use PowerShell. The Python, npm, and Docker commands are the same on
macOS or Linux, but virtual-environment activation uses `source .venv/bin/activate`.

## First-time setup

### 1. Start PostgreSQL with pgvector

From the repository root:

```powershell
docker compose up -d postgres
docker compose ps
```

The Compose service starts the `pgvector/pgvector:pg17` image on port `5432` and stores
database files in the named Docker volume `rag_postgres_data`. Wait until the container
reports `healthy` before indexing files.

### 2. Create the Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Configure the backend

Copy the template and edit the new root `.env` file:

```powershell
Copy-Item .env.example .env
notepad .env
```

At minimum, replace the placeholder with a real server-side API key:

```dotenv
OPENAI_API_KEY=sk-your-real-key-here
```

Never place this key in `frontend/.env.local` or in a variable beginning with
`NEXT_PUBLIC_`. Next.js exposes public variables to the browser. The root `.env` file is
ignored by Git.

### 4. Add knowledge files

Place UTF-8 Markdown or plain-text files under `data/`. Nested directories are scanned
recursively.

```text
data/
|-- handbook.md
|-- policies.txt
`-- product/
    `-- guide.markdown
```

Only `.md`, `.markdown`, and `.txt` are currently supported. Files with other extensions
are ignored. The repository's current local sample corpus contains AWS documentation,
but the ingestion pipeline accepts any content in the supported formats.

### 5. Validate and index the files

Validation is offline: it does not call OpenAI and does not write to PostgreSQL.

```powershell
python main.py validate
```

If validation succeeds, create embeddings and synchronize the database:

```powershell
python main.py ingest
```

Ingestion calls the OpenAI embeddings API and therefore requires the API key, network
access, and a running database.

### 6. Install the frontend

```powershell
Set-Location frontend
npm ci
Copy-Item .env.local.example .env.local
Set-Location ..
```

The frontend defaults to `http://localhost:8000`, so `.env.local` is optional for the
standard local setup. It is useful when the backend runs at another URL.

## Run the application

Keep the database running and open two terminals from the repository root.

Terminal 1 — backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --reload --port 8000
```

Terminal 2 — frontend:

```powershell
Set-Location frontend
npm run dev
```

Open:

- Frontend: [http://localhost:3000](http://localhost:3000)
- API health: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

The frontend's **Sync documents** button invokes the same synchronization workflow as
`python main.py ingest`.

## Everyday startup and shutdown

After the first-time setup, the normal startup sequence is:

```powershell
# Start the database.
docker compose up -d postgres

# Start the backend in a separate terminal.
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --reload --port 8000

# Start the frontend in another terminal.
Set-Location frontend
npm run dev
```

Run `python main.py ingest` again after adding, editing, or deleting files under `data/`.
Unchanged files are skipped automatically.

Stop the foreground servers with `Ctrl+C`. Stop the database container without deleting
its data:

```powershell
docker compose stop postgres
```

To start the existing container again, use `docker compose start postgres`, or simply
run `docker compose up -d postgres`.

## Production-style local run

Build and serve an optimized frontend:

```powershell
Set-Location frontend
npm run build
npm run start
```

Run the backend without automatic reload:

```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

This repository does not currently include production deployment, TLS, authentication,
rate limiting, or a container image for the frontend/backend. Add those controls before
exposing the application to an untrusted network.

## Configuration reference

### Backend (`.env`)

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for AI calls | Server-side OpenAI credential |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag` | SQLAlchemy database URL |
| `OPENAI_GENERATION_MODEL` | `gpt-5.6-terra` | Model used to generate answers |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for document and query embeddings |
| `OPENAI_EMBEDDING_DIMENSIONS` | `1536` | Embedding width; the current schema requires exactly 1536 |
| `RAG_TOP_K` | `6` | Intended retrieval-count setting; the current retriever implementation uses 6 directly |
| `DATA_DIR` | Repository `data/` directory | Absolute or relative knowledge-file directory |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins allowed by FastAPI |

The API can report health and list local files without an OpenAI key. Ingestion, search,
and answer generation require it.

### Frontend (`frontend/.env.local`)

| Variable | Default | Description |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Browser-visible FastAPI base URL |

Restart the Next.js development server after changing frontend environment variables.

## HTTP API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Reports database availability, API-key presence, and document/chunk counts |
| `GET` | `/api/documents` | Lists supported local files and whether each current hash is indexed |
| `POST` | `/api/ingest` | Synchronizes `data/` with PostgreSQL and returns an ingestion report |
| `POST` | `/api/chat` | Retrieves context and returns an answer with sources |

Example chat request:

```json
{
  "question": "Summarize the main recommendations.",
  "history": [
    { "role": "user", "content": "What is this guide about?" },
    { "role": "assistant", "content": "It describes..." }
  ]
}
```

Input limits enforced by the API:

- Question: 1–4,000 characters
- History: at most 20 messages
- History message: 1–20,000 characters
- History roles: `user` or `assistant`

The response contains `answer` plus a `sources` array. Each source includes `source`,
`file_name`, `heading`, and a rounded RRF `score`.

## Command-line interface

Run commands from the repository root with the Python environment active.

| Command | Description | OpenAI required | Database required |
| --- | --- | --- | --- |
| `python main.py validate` | Discovers and chunks every supported file, then prints statistics | No | No |
| `python main.py ingest` | Synchronizes files, embeddings, and database records | Yes | Yes |
| `python main.py query "your question"` | Queries the indexed corpus and prints an answer | Yes | Yes |
| `python main.py reset` | Drops and recreates this application's `documents` and `chunks` tables | No | Yes |

`reset` is destructive for the application index. Source files and the PostgreSQL Docker
volume remain intact, but all documents must be ingested again.

## Document processing details

- Files are decoded as UTF-8 with optional byte-order-mark support.
- Line endings are normalized and trailing whitespace is removed in memory.
- Markdown headings from levels 1–6 form breadcrumb strings such as
  `Guide > Installation > Windows`.
- Chunks are at most 3,600 characters with up to 320 characters of overlap.
- Oversized paragraphs are split on a newline, then whitespace, then the hard limit.
- The whole-file SHA-256 determines whether re-indexing is necessary.
- Each chunk also receives a content SHA-256 and a stable per-file chunk index.
- Embeddings are requested in batches of 64.
- Documents are identified by file name in the current database schema. Avoid duplicate
  file names across nested `data/` directories because one can replace the other.

The database contains two application tables:

- `documents`: file name, file hash, byte size, chunk count, and indexing timestamp
- `chunks`: source, heading, content, content hash, 1,536-dimensional vector, generated
  `tsvector`, and document relationship

The `vector` extension, tables, HNSW cosine index, and GIN full-text index are created
automatically. On an existing database, application initialization adds the generated
`tsvector` column and GIN index without re-embedding documents.

Lexical search computes BM25 from `tsvector` position counts, per-term document
frequency, total document count, and average document length. Semantic and lexical
queries each return a larger candidate set. RRF then adds `1 / (60 + rank)` for every
channel containing a chunk and returns the highest fused ranks. Raw cosine similarity
and BM25 values are deliberately not mixed.

## Testing and code quality

Backend checks:

```powershell
.\.venv\Scripts\Activate.ps1
ruff check .
pytest
```

Frontend checks:

```powershell
Set-Location frontend
npm run typecheck
npm run build
```

The current tests cover default model configuration, supported-file discovery, nonempty
and bounded chunks, unique source identifiers, Markdown heading metadata, large plain-text
fallback chunking, repair of common extraction mojibake, RRF overlap behavior, raw-score
independence, and duplicate-result handling.

## Security, privacy, and cost considerations

- Knowledge-file chunks and questions are sent to OpenAI for embedding or generation.
  Do not ingest content that your OpenAI account is not permitted to process.
- The OpenAI key belongs only in the backend `.env` file.
- The local database password in `docker-compose.yml` is intended for development only.
- The API has no authentication. Keep it on a trusted local network during development.
- Ingestion creates billable embedding API usage. Changed files are re-embedded; unchanged
  files are skipped.
- Chat creates a query embedding and a generation request for each submitted question.
- Retrieved excerpts are treated as untrusted reference text in the response prompt to
  reduce prompt-injection risk, but source answers should still be verified.

## Current limitations

- Only Markdown and plain-text input are supported; PDF, Word, HTML, and other formats
  require a preprocessing step.
- The embedding database schema is fixed to 1,536 dimensions.
- File identity uses the base file name rather than its relative path.
- BM25 corpus statistics are computed at query time; very large corpora may benefit from
  precomputed statistics or a dedicated PostgreSQL BM25 extension.
- Browser chat history is local to one browser and is not stored by the backend.
- The included sample data and current response-generator system prompt are AWS-oriented.
  General corpora can be indexed, but `src/impl/response_generator.py` should also be made
  domain-neutral before relying on generated answers for non-AWS content.
- The optional evaluator implementation exists as an extension point but is not wired
  into the API or CLI startup path.

## Troubleshooting

### Docker is not recognized on Windows

Install Docker Desktop, start it, and open a new PowerShell window. If needed, refresh
the current terminal's path:

```powershell
$dockerBin = 'C:\Program Files\Docker\Docker\resources\bin'
$env:Path = "$dockerBin;$env:Path"
docker --version
docker compose version
```

Installing a VS Code extension or the Python `docker` package does not install the Docker
engine.

### PostgreSQL is offline

```powershell
docker compose up -d postgres
docker compose ps
docker compose logs postgres
```

Wait for the health status before retrying ingestion or chat.

### No documents are indexed

Confirm supported files exist under `DATA_DIR`, then run:

```powershell
python main.py validate
python main.py ingest
```

### The browser cannot reach the API

Check that FastAPI is listening on port `8000`, `NEXT_PUBLIC_API_URL` points to that
address, and the frontend origin appears in `CORS_ORIGINS`. Restart both servers after
configuration changes.

### OpenAI requests fail

Confirm `OPENAI_API_KEY` is present in the root `.env`, the configured models are
available to the account, and the machine has network access. The health endpoint reports
whether a key is configured but does not validate the credential.
