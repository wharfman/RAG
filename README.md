# AWS documentation RAG chatbot

A local full-stack RAG application with a Claude-inspired Next.js interface, a FastAPI backend, OpenAI embeddings and generation, and PostgreSQL with pgvector.

## Where the OpenAI API key goes

Copy `.env.example` to a file named `.env` in the repository root and put the key there:

```dotenv
OPENAI_API_KEY=sk-your-real-key-here
```

The Python backend reads this server-side value. Do **not** put the OpenAI key in `frontend/.env.local` or any variable beginning with `NEXT_PUBLIC_`; those values are sent to the browser. The root `.env` is gitignored.

## Windows setup: Docker Desktop and WSL 2

Installing a VS Code Docker extension, a Python `docker` package, or only the Docker
installer is not enough. This application needs **Docker Desktop for Windows** and its
Linux/WSL 2 engine.

1. Open **PowerShell as Administrator** and enable WSL:

   ```powershell
   wsl --install
   ```

2. Restart Windows when prompted. Microsoft documents the restart as part of WSL
   installation.

3. In a new PowerShell window, install Docker Desktop if it is not already present:

   ```powershell
   winget install --exact --id Docker.DockerDesktop --source winget
   ```

4. Close every existing terminal after installation, then open **Docker Desktop** from
   the Start menu. Accept its first-run terms and wait until it reports that the engine
   is running.

5. Open a fresh PowerShell window and verify both the command and engine:

   ```powershell
   docker --version
   docker compose version
   docker info
   ```

If PowerShell still says `docker` is not recognized but Docker Desktop exists, refresh
the current terminal's PATH and retry:

```powershell
$dockerBin = 'C:\Program Files\Docker\Docker\resources\bin'
$env:Path = "$dockerBin;$env:Path"
docker --version
```

If this reports that the path does not exist, Docker Desktop is not fully installed.
Run the `winget install` command above and let the installer finish. The official
[Docker Desktop Windows guide](https://docs.docker.com/desktop/setup/install/windows-install/)
and [Microsoft WSL guide](https://learn.microsoft.com/windows/wsl/install) contain the
current platform requirements.

## First-time project setup

Open a fresh PowerShell window:

```powershell
Set-Location C:\Projects\RAG

# Confirm that PowerShell is in the directory containing docker-compose.yml.
Test-Path .\docker-compose.yml

# Start PostgreSQL 17 with pgvector and verify it becomes healthy.
docker compose -f .\docker-compose.yml up -d postgres
docker compose -f .\docker-compose.yml ps

# Create and populate the Python environment.
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Create the server-only environment file if it does not exist.
if (-not (Test-Path .\.env)) { Copy-Item .\.env.example .\.env }
notepad .\.env
```

In Notepad, replace `sk-your-openai-api-key` with the real key and save the file.
Then continue in the same PowerShell window:

```powershell
# Offline validation: does not call OpenAI or write to PostgreSQL.
python main.py validate

# One-time indexing. This calls the embeddings API for changed files.
python main.py ingest

# Start the API and leave this terminal open.
python -m uvicorn api:app --reload --port 8000
```

Open a **second** PowerShell window for the frontend:

```powershell
Set-Location C:\Projects\RAG\frontend
if (-not (Test-Path .\.env.local)) { Copy-Item .\.env.local.example .\.env.local }
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The API health page is available
at [http://localhost:8000/api/health](http://localhost:8000/api/health).

## Starting the app on later days

Start Docker Desktop first and wait for its engine. Then use three PowerShell windows:

```powershell
# Window 1: database
Set-Location C:\Projects\RAG
docker compose -f .\docker-compose.yml up -d postgres
docker compose -f .\docker-compose.yml ps
```

```powershell
# Window 2: backend
Set-Location C:\Projects\RAG
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --reload --port 8000
```

```powershell
# Window 3: frontend
Set-Location C:\Projects\RAG\frontend
npm run dev
```

Re-run `python main.py ingest` only after adding or changing files under `data/`.
Unchanged files are skipped by SHA-256 hash.

To stop the database after closing the backend and frontend terminals:

```powershell
Set-Location C:\Projects\RAG
docker compose -f .\docker-compose.yml stop postgres
```

## Data processing

The indexer recursively reads `.md`, `.markdown`, and `.txt` files under `data/`.

- Markdown headings are preserved as section breadcrumbs in chunk metadata.
- Plain-text files, including `ec2-api.md`, fall back to bounded line-aware chunks.
- Chunks are at most 3,600 characters with up to 320 characters of overlap.
- UTF-8 is required; common PDF-extraction mojibake is repaired in memory.
- Source files are never rewritten.
- Each document and chunk is hashed, making re-ingestion deterministic and idempotent.

Run `python main.py validate` or `pytest` to check that every current data file is discovered, produces nonempty uniquely identified chunks, and stays within the chunk-size limit. Validation is offline and does not use the OpenAI API.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | Server-side OpenAI credential |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag` | SQLAlchemy PostgreSQL connection |
| `OPENAI_GENERATION_MODEL` | `gpt-5.6-terra` | Answer generation model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 1,536-dimensional embeddings |
| `DATA_DIR` | `./data` | Knowledge-file directory |
| `RAG_TOP_K` | `6` | Retrieved chunks per question |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Browser-to-backend URL; contains no secret |

## Useful commands

```powershell
python main.py validate
python main.py ingest
python main.py query "How does VPC peering work?"
pytest
ruff check .
cd frontend
npm run typecheck
npm run build
```

`python main.py reset` drops and recreates only this app's `documents` and `chunks` tables. The pgvector extension and HNSW cosine index are created automatically during ingestion.
