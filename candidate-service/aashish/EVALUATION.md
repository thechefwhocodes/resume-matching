# Evaluation

This document covers the eval methodology, metric choices, current risk
register, and a short list of what we'd ship next if this graduated past the
take-home.

## Methodology

Evaluation is split into three suites that map to the three meaningful
"learnable" components of the pipeline:

| Suite        | What it scores                                                     | Where it runs              |
| ------------ | ------------------------------------------------------------------ | -------------------------- |
| `extractor`  | Did we extract the right `ResumeProfile`?                          | `eval/extractor_eval.py`   |
| `retriever`  | Does the top-N include the relevant jobs?                          | `eval/retriever_eval.py`   |
| `reranker`   | Within the retrieved top-N, did we rank the relevant jobs first?   | `eval/reranker_eval.py`    |

Run all three:

```bash
make eval
```

Output is written to `eval/results/<timestamp>.md` and an `EvalRun` row is
inserted into Postgres for historical tracking.

## Ground truth

5 of the 20 sample resumes are "locked" for evaluation:

- `experienced/senior-backend-engineer.txt`
- `experienced/ml-engineer-nlp.txt`
- `new-grads/cs-grad-good-internship.txt`
- `remote/india-backend-engineer.txt`
- `less-impressive/job-hopper-with-gaps.txt`

For each we curate two kinds of ground truth:

1. **`expected_job_ids`** — the top-5 jobs from `data/jobs.json` that a
   senior recruiter would shortlist. **Generation method:** Claude Opus 4
   with the prompt in [`eval/ground_truth/README.md`](eval/ground_truth/README.md),
   then human-verified that each `job_id` exists.
2. **`profiles_handlabeled.json`** — a hand-labeled `ResumeProfile` per
   locked resume.

The remaining 15 resumes are scored by an LLM judge (GPT-4o-mini against the
extractor output). Coverage is therefore 25% strict / 75% LLM-judged.

> **Status:** the 5 hand-labeled profiles are committed; the 5 pick files
> are placeholder scaffolds awaiting Claude Opus generation. The validator
> warns (does not fail) so the eval pipeline still runs end-to-end and the
> extractor + LLM-judge metrics are populated. Filling in the picks unlocks
> retriever Recall@K and reranker NDCG@10.

## Metrics

### Extractor

| Metric | What it captures |
| --- | --- |
| Per-field agreement on `primary_category`, `seniority`, `needs_h1b_sponsorship`, `open_to_remote` | Hard categorical accuracy on the 5 hand-labeled set |
| `years_experience` within ±1 yr | Tolerant numeric agreement |
| `secondary_categories` Jaccard | Set-overlap on the secondary list |
| `skills` Jaccard | Set-overlap on top-30 normalised skills |
| LLM-judge mean (0–5) | Plausibility on the 15 LLM-judged resumes |

### Retriever

We compare three approaches on the locked resumes:

- `bm25` — `rank-bm25` over `title + responsibilities + requirements`
- `embed` — pgvector cosine on `text-embedding-3-small`
- `embed+filters` — `embed` then SQL hard filters + soft scoring

Reported per approach: `Recall@5`, `Recall@10`, `Recall@30`, `Recall@50`,
`MRR`, and `Coverage`. We add an `rrf-hybrid` variant only if the
`Recall@30` gap between `bm25` and `embed` is ≤ 0.05 (so it's worth the
complexity).

### Reranker

Within the embed-retrieved + filtered top-30, we score:

- `NDCG@10` against the GT-ranked top-5
- `MRR@10`
- `Precision@5`
- `score_mean` ± `score_std` (calibration check; we want a wide spread, not all 75s)
- `Cohen's κ` versus a second model when `EXTRACT_MODEL != RERANK_MODEL`

Rerank calls are cached on disk by `hash(resume + ranked_job_ids + model)`,
so re-running `make eval` is free.

## Latency & cost budget

Per request, sequential pipeline:

| Stage | Budget (p95) | Tokens (typical) |
| --- | --- | --- |
| Embed resume | 0.2–0.5s | ~400 embed |
| Profile extract (cache miss) | 0.8–1.5s | ~1200 prompt / ~250 completion |
| pgvector cosine + filters | < 0.05s | — |
| Rerank batched (30 candidates) | 3.0–5.0s | ~5000 prompt / ~1500 completion |
| **Total** | **~5–7s** | — |

Well under the 30s Next.js proxy timeout (R11). All numbers reported in
`metadata.cost_usd` and `metadata.tokens` per request.

## Risk register

| ID | Risk | Mitigation | State |
| --- | --- | --- | --- |
| R6 | HTML in `responsibilities` poisons embeddings + BM25 | `utils/html.py` + stripped at ingest | mitigated |
| R11 | 30s proxy timeout vs cold rerank | Sequential budget math; documented above | mitigated |
| R13 | UI expects formatted `salary_range` string | `utils/salary.py` | mitigated |
| R28 | No Alembic = silent schema drift | Documented as production work below | accepted |
| R31 | 25% GT coverage | Supplement with LLM-as-judge; ship CIs in next iteration | partial |
| R32 | Embedding dim drift if `EMBED_MODEL` changes | Hard-coded `Vector(1536)`; only `text-embedding-3-small` supported | accepted |
| R33 | `EvalRun.metrics` JSON cannot store `NaN` | NaN-scrubbed before persist | mitigated |
| R34 | LLM rerank flakiness on bad JSON | `tenacity` retries + structured outputs + cache | mitigated |
| R35 | `embed`/`bm25+rerank` requested without `OPENAI_API_KEY` | Fall back to `bm25` with a logged warning; metadata records the requested approach | mitigated |
| R36 | Cost-tracker pricing drift | Static `_PRICING_PER_1K` map in `llm/client.py` — needs periodic refresh | accepted |

## Production-hardening (next 1–2 sprints)

1. **Alembic migrations** — replace `Base.metadata.create_all` with versioned
   migrations and a `migrate-then-serve` startup contract.
2. **HNSW index on `jobs.embedding`** — currently we do an exact scan; with
   N=300 it's instant, but for ≥10K jobs an `IVFFLAT`/`HNSW` index is needed.
3. **Per-request tracing** — Phoenix or OpenTelemetry spans for ingest /
   extract / embed / retrieve / rerank, exported to a sink the team owns.
4. **Eval as CI gate** — fail the build when `extractor` field agreement
   drops below the previous run's bound (with a tolerance).
5. **Bootstrap CIs** for retriever / reranker metrics — current GT is N=5
   so reported numbers are noisy.
6. **Hybrid retrieval** — RRF fuse of `bm25` and `embed`, gated on the
   recall@30 gap (already a conditional in `retriever_eval.py`).
7. **Eval cost guardrails** — per-suite `cost_usd` cap with a hard stop.
8. **Resume-PDF/DOCX parsing** — extend `Resume` to store the parsed text
   alongside the original blob.
9. **Per-tenant rate limiting** + auth on `/match` (currently CORS-open).
10. **Real LLM observability** — log prompt + completion (with PII redaction)
    so we can replay failed extractions/reranks.

## Out of scope (this take-home)

- Resume PDF/DOCX parsing
- Frontend changes beyond the TS contract additions
- Phoenix / external observability backend
- Alembic migrations
- HNSW index on `jobs.embedding`
- Real-time learning from user feedback
