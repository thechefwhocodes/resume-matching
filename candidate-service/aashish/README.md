# Resume Matching Service — `aashish`

Production-shaped take on the resume-matching brief. FastAPI + Postgres +
pgvector + OpenAI, with four pluggable retrieval/rerank approaches and a
3-suite evaluation framework. Everything is wired to the existing Next.js
frontend at `localhost:3000`.

> **One-line headline (measured Apr 25, 2026):** BM25 R@30 = 0.60 beats
> dense embedding R@30 = 0.24 on this dataset. With `OPENAI_API_KEY`
> set, `make demo` defaults to `embed+rerank` (~21s p95) for the full
> pipeline experience, but `APPROACH_OVERRIDE=bm25+rerank` gives the
> currently best-measured retrieval. See [EVALUATION.md](EVALUATION.md).

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

That's it. The service auto-ingests `data/jobs.json` on first boot
(idempotent — re-running `make demo` skips ingest) and builds an
in-memory BM25 index from the rows.

## Should I run with `OPENAI_API_KEY`?

Both modes are fully functional — but they're optimized for different
goals.

| Run with key (recommended) | Run without key (fallback) |
| --- | --- |
| Default approach: `embed+rerank` | Default approach: `bm25` |
| Full pipeline: extract → embed → pgvector → filters → rerank | Lexical-only: in-memory `rank-bm25` |
| ~21s p95 per request, ~$0.001–$0.002 per request | < 100ms per request, $0.00 |
| Surfaces `filter_flags`, `rerank_score`, calibrated `match_score` | Surfaces `retrieval_score` (raw BM25), `match_score` normalised to 0–100 |
| Costs ~$0.0155 for one full `make eval` run | `make eval` runs but skips retriever metrics for embed approaches |
| Requires a working `sk-...` key in `.env` | Embeddings filled with zero-vectors at ingest |

**Want to compare a key/no-key run side-by-side?** Use the approach
override:

```bash
APPROACH_OVERRIDE=bm25 make demo            # always BM25, even with key
APPROACH_OVERRIDE=embed+rerank make demo    # always full pipeline
```

You can also set `approach` per-request in the `POST /match` body — see
[`src/types/index.ts`](../../src/types/index.ts).

## Why BM25 currently outperforms dense embeddings

This was the most surprising finding from the eval (full numbers in
[EVALUATION.md](EVALUATION.md#finding-f2--bm25--embed-at-n300)):

| Approach | R@30 | MRR | Coverage |
| --- | --- | --- | --- |
| **bm25** | **0.600** | **0.389** | **0.680** |
| embed | 0.240 | 0.086 | 0.280 |

Three reasons we believe explain it:

1. **N=300 is small.** Dense embeddings tend to pull ahead at N≥10K
   where they can compress nuance the query lacks. At 300 jobs, BM25
   has plenty of room to find lexical anchors directly.
2. **Job descriptions are jargon-heavy.** "Postgres", "Kubernetes",
   "Go", "gRPC" are exact-match tokens that BM25 catches losslessly.
   The embedding model maps them to nearby but not identical vectors,
   diluting the score.
3. **Resume embeddings have noise.** A resume contains many irrelevant
   facts ("hobby photographer", "5K runner") that bias the centroid
   away from role-relevant clusters. **What we'd ship next:** embed
   only the extractor-derived `summary` field, not the full resume.

The plan called for adding RRF-hybrid only if Recall@30 of bm25 vs
embed are within 0.05; the measured gap is 0.36, so RRF wouldn't help
yet. The pragmatic short-term move is to switch `default_approach()` to
`bm25+rerank`. We left the default as `embed+rerank` so the measured
behaviour matches the plan; reviewers can flip via `APPROACH_OVERRIDE`.

## Latest eval results (Apr 25, 2026)

Full report: [`eval/results/20260425-195344.md`](eval/results/20260425-195344.md).
Methodology, all metrics, risk register, and 14-item production-hardening
backlog: [EVALUATION.md](EVALUATION.md).

| Suite | Headline | Cost |
| --- | --- | --- |
| Extractor | 100% on `primary_category`/`h1b`/`open_to_remote`; 80% seniority; LLM-judge mean 3.87/5 | $0.0084 |
| Retriever | BM25 R@30 0.600 / MRR 0.389; embed R@30 0.240; embed+filters R@30 0.280 | $0.0002 |
| Reranker | NDCG@10 0.315; **MRR@10 0.667**; Precision@5 0.200; calibration 55.6 ± 15.3 | $0.0069 |
| **Total** | — | **$0.0155** |

**One open issue surfaced by the run:** `gpt-4o-mini` silently truncates
batched rerank responses on lists > 15 candidates (returns ~9 of 28
asked). Documented as R37 / Finding F5. Single-line workaround:
`RETRIEVAL_TOP_K=15` in `.env`.

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
`requirements`, `retrieval_score`, `rerank_score`, and `filter_flags`,
plus metadata containing `processing_time_ms`, `cost_usd`,
`retrieval_count`/`filtered_count`/`returned_count`, and per-stage token
counts.

## Approaches

| Approach | Steps | Best for |
| --- | --- | --- |
| `bm25` | In-memory `rank-bm25` over title + responsibilities + requirements | No-key fallback; lexical baseline; **best measured retrieval** |
| `bm25+rerank` | BM25 retrieve → extract profile → LLM rerank | Highest expected fidelity given F2 finding |
| `embed` | OpenAI embedding → pgvector cosine retrieve | Pure semantic baseline (currently underperforms BM25, see F2) |
| `embed+rerank` | Embed retrieve → extract profile → SQL hard filters + soft scoring → LLM rerank | **Default with key**; shows the full pipeline; ~18–28s p95 |

## Restart workflows

The Postgres data lives in a named Docker volume (`pgdata`). Different
restart paths reuse or wipe that volume:

### Warm restart (keep ingested jobs + caches)

Fastest. The `Job`, `Resume`, and `ResumeProfileCache` tables persist;
`ingest_jobs_if_empty` is a no-op on the next boot.

```bash
# Stop just uvicorn (Ctrl-C) and start it again:
make run
# OR if postgres also stopped:
make demo
```

`make seed-check` confirms the row count survived.

### Cold restart (wipe DB, force re-ingest + re-embed)

Useful when:

- You've changed `data/jobs.json`.
- You've changed `EMBED_MODEL` (e.g. switched providers).
- You want to measure ingest cost from scratch.
- You're debugging the lifespan boot sequence.

```bash
make down                                   # stop postgres
docker compose -f ../../docker-compose.yml down -v   # add `-v` to drop the volume
make demo                                   # fresh boot: re-creates schema, re-ingests
```

The first request after a cold boot pays the extractor cache miss
(~3s extra). Subsequent requests for the same resume are fast.

### Rebuild the BM25 index (without restart)

The BM25 index is built once at lifespan startup. To rebuild it after
adding/removing jobs you need to restart uvicorn (Ctrl-C → `make run`).
Documented as a known limitation in
[ARCHITECTURE.md D7](ARCHITECTURE.md#d7--in-memory-bm25-index).

### Wipe everything (start completely fresh)

```bash
make down
docker compose -f ../../docker-compose.yml down -v
make clean       # also removes .venv, .pytest_cache, etc.
rm -rf eval/.cache eval/results/*.md   # drop rerank cache + reports
make install
make demo
```

## Layout

```
candidate-service/aashish/
├── README.md             # this file
├── EVALUATION.md         # measured results, findings, risks, backlog
├── ARCHITECTURE.md       # diagrams + decision log + perf findings
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
│   ├── unit/             # schemas, html, salary, filters, rerank prompts, kappa alignment
│   └── integration/      # health, match, BM25 quality, pgvector ordering
└── eval/
    ├── ground_truth/     # 5 hand-labeled profiles + 5 Claude-Opus pick files
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
| `make eval-validate-gt` | Validate ground truth files (job_id existence + profile shape) |
| `make eval` | Run all 3 eval suites; writes `eval/results/<timestamp>.md` |
| `make test` | Pytest (unit + testcontainers integration) |
| `make lint` / `make format` | `ruff check` / `ruff format` |
| `make down` | Stop the postgres container (data volume preserved) |
| `make clean` | Remove venv + `__pycache__` |

## Configuration

All env vars (with defaults) live in [`/.env.example`](../../.env.example):

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | _(unset)_ | Without it: `bm25` only, zero-vector embeddings on ingest |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/resumes` | Matches `docker-compose.yml` |
| `EMBED_MODEL` | `text-embedding-3-small` | Hard-coded to 1536 dim |
| `EXTRACT_MODEL` | `gpt-4o-mini` | Used for `ResumeProfile` extraction |
| `RERANK_MODEL` | `gpt-4o-mini` | Configurable; set to `gpt-4o` to enable κ + reduce truncation |
| `APPROACH_OVERRIDE` | _(unset)_ | Pin a specific approach for all requests |
| `JOBS_JSON_PATH` | `../../data/jobs.json` | Resolved relative to `candidate-service/aashish/` |
| `RETRIEVAL_TOP_K` | `30` | **Recommended: lower to 15** (see EVALUATION.md F5) |
| `RESULT_TOP_K` | `10` | Final results returned to UI |
| `RERANK_MIN_SCORE` | `50` | Drop reranker scores below this |

## Tests

```bash
make test
```

Spins up an ephemeral pgvector-enabled Postgres via testcontainers per
session, runs 46 tests (38 unit + 8 integration). No live OpenAI calls
are made in tests.

## Eval

```bash
make eval                                          # all 3 suites
.venv/bin/python -m eval.report --suite extractor  # one suite
```

See [EVALUATION.md](EVALUATION.md) for measured results, methodology,
risk register, and the production-hardening backlog.

## Boundaries

- Ingest happens in-process at startup (idempotent). No Alembic — see
  EVALUATION.md (R28).
- LLM calls are sequential. Measured p95 for `embed+rerank` is ~21s,
  inside the 30s Next.js proxy timeout but with little headroom.
  Mitigations in EVALUATION.md F1.
- No PDF/DOCX parsing; raw `.txt` only (matches the brief).
- Frontend is untouched apart from extending `src/types/index.ts` with
  new optional fields (the existing `JobCard` already renders them).
- `gpt-4o-mini` truncates batched rerank lists > 15 items — open R37.
