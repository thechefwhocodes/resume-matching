# Resume Matching Service — `aashish`

Production-shaped take on the resume-matching brief. FastAPI + Postgres +
pgvector + OpenAI, with four pluggable retrieval/rerank approaches and a
3-suite evaluation framework. Everything is wired to the existing Next.js
frontend at `localhost:3000`.

## 5-minute reviewer path

```bash
# 1. From the repo root, copy and (optionally) populate the env file
cp .env.example .env
# Edit .env to add OPENAI_API_KEY (skip to use BM25-only mode)

# 2. Bring up the service (postgres + uvicorn)
cd candidate-service/aashish
make install      # one-time: creates .venv and installs deps
make demo         # docker compose up postgres -> wait healthy -> uvicorn

# 3. In a separate terminal, at the repo root, start the UI
cd ../../          # back to repo root
npm install       # one-time
npm run dev

# 4. Open http://localhost:3000 and upload a resume from data/sample-resumes/
```

That's it. With `OPENAI_API_KEY` set the default approach is `embed+rerank`
(extract → embed → pgvector cosine retrieve → SQL filters + soft scoring →
LLM rerank). Without it, the service falls back to `bm25` and remains fully
functional.

## What this service does

`POST /match` accepts:

```json
{
  "resume": { "content": "...resume text..." },
  "approach": "embed+rerank"   // optional; one of bm25 | bm25+rerank | embed | embed+rerank
}
```

…and returns the contract defined in
[`src/types/index.ts`](../../src/types/index.ts) — top-K matches with
`location`, `salary_range`, `job_category`, `responsibilities`,
`requirements`, `retrieval_score`, `rerank_score`, and `filter_flags`, plus
metadata containing `processing_time_ms`, `cost_usd`, and per-stage token
counts.

## Approaches

| Approach        | Steps                                                                                                | Best for                              |
| --------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `bm25`          | In-memory `rank-bm25` over title + responsibilities + requirements                                   | No-key fallback; lexical baseline     |
| `bm25+rerank`   | BM25 retrieve → extract profile → LLM rerank                                                         | When you trust lexical recall         |
| `embed`         | OpenAI embedding → pgvector cosine retrieve                                                          | Pure semantic baseline                |
| `embed+rerank`  | Embed retrieve → extract profile → SQL hard filters + soft scoring → LLM rerank (default with key)   | Highest fidelity; ~5–7s p95           |

## Layout

```
candidate-service/aashish/
├── README.md             # this file
├── EVALUATION.md         # metrics, risks, what we'd ship next
├── ARCHITECTURE.md       # diagrams + decision log
├── Makefile              # install / demo / run / eval / test / lint
├── Dockerfile            # multi-stage; ~200MB runtime image
├── pyproject.toml, requirements.txt
├── src/
│   ├── main.py           # FastAPI app + lifespan (boot ingest + BM25 build)
│   ├── config.py         # pydantic-settings
│   ├── schemas.py        # Pydantic mirrors of TS contract
│   ├── ingest.py         # ingest_jobs_if_empty (HTML strip + batched embeds)
│   ├── pipeline.py       # approach selector + 4 pipelines
│   ├── llm/client.py     # OpenAI wrapper: embed, chat_structured, cost tracking
│   ├── db/{engine,models,queries}.py
│   ├── extractor/        # resume profile prompts + extractor
│   ├── retrieval/        # BM25 + EmbeddingRetriever
│   ├── rerank/           # rerank prompt + caller
│   └── utils/{html,salary}.py
├── tests/
│   ├── conftest.py       # testcontainers-postgres fixture
│   ├── unit/             # schemas, html, salary, filters, rerank prompts
│   └── integration/      # health, match, BM25 quality, pgvector ordering
└── eval/
    ├── ground_truth/     # 5 hand-labeled profiles + 5 pick scaffolds
    ├── extractor_eval.py
    ├── retriever_eval.py
    ├── reranker_eval.py
    ├── report.py         # writes timestamped markdown + EvalRun row
    └── results/          # generated reports (.gitignored)
```

## Make targets

| Target | Purpose |
| --- | --- |
| `make install` | Create `.venv` and install pinned deps |
| `make demo` | `docker compose up postgres` → wait healthy → `uvicorn` (reviewer command) |
| `make run` | Same as `demo` but assumes postgres is already running |
| `make seed-check` | Print job count from DB |
| `make eval-validate-gt` | Validate ground truth files |
| `make eval` | Run all 3 eval suites; writes `eval/results/<timestamp>.md` |
| `make test` | Pytest (unit + testcontainers integration) |
| `make lint` / `make format` | `ruff check` / `ruff format` |
| `make down` | Stop the postgres container (volume preserved) |
| `make clean` | Remove venv + `__pycache__` |

## Configuration

All env vars (with defaults) live in [`/.env.example`](../../.env.example):

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | _(unset)_ | Without it: `bm25` only, zero-vector embeddings on ingest |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/resumes` | Matches `docker-compose.yml` |
| `EMBED_MODEL` | `text-embedding-3-small` | Hard-coded to 1536 dim |
| `EXTRACT_MODEL` | `gpt-4o-mini` | Used for `ResumeProfile` extraction |
| `RERANK_MODEL` | `gpt-4o-mini` | Configurable per the plan |
| `APPROACH_OVERRIDE` | _(unset)_ | Pin a specific approach for all requests |
| `JOBS_JSON_PATH` | `../../data/jobs.json` | Resolved relative to `candidate-service/aashish/` |
| `RETRIEVAL_TOP_K` | `30` | Candidates fed to filters/rerank |
| `RESULT_TOP_K` | `10` | Final results returned to UI |
| `RERANK_MIN_SCORE` | `50` | Drop reranker scores below this |

## Tests

```bash
make test
```

Spins up an ephemeral pgvector-enabled Postgres via testcontainers per session,
runs 46 tests (38 unit + 8 integration). No live OpenAI calls made in tests.

## Eval

```bash
make eval                              # all 3 suites
.venv/bin/python -m eval.report --suite extractor   # one suite
```

See [EVALUATION.md](EVALUATION.md) for metrics, methodology, and risk register.

## Boundaries

- Ingest happens in-process at startup (idempotent). No Alembic — see EVALUATION.md.
- LLM calls are sequential. p95 budget for `embed+rerank` is ~5–7s; well under
  the 30s Next.js proxy timeout.
- No PDF/DOCX parsing; raw `.txt` only (matches the brief).
- Frontend is untouched apart from extending `src/types/index.ts` with new
  optional fields (the existing `JobCard` already renders them).
