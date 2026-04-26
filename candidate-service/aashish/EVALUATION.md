# Evaluation

This document covers the eval methodology, **measured results from a real
run**, the metric choices, the current risk register, and a list of what
we'd ship next if this graduated past the take-home.

> **TL;DR** — We ran the full 3-suite eval against the locked ground truth
> on Apr 25, 2026. Total cost $0.0155, total wall time ~90s. The headline
> findings: BM25 meaningfully outperforms dense embeddings at N=300
> (R@30 0.60 vs 0.24), and `gpt-4o-mini` silently truncates rerank output
> on lists of >15 candidates. Both have concrete fixes in the
> production-hardening backlog below.

---

## Measured results — Apr 25, 2026

Full run: [`eval/results/20260425-195344.md`](eval/results/20260425-195344.md).
Models: `text-embedding-3-small`, `gpt-4o-mini` (extract & rerank).

### Extractor

5 hand-labeled resumes scored by field; the other 15 by an LLM judge.

| Field | Agreement |
| --- | --- |
| `primary_category` | **1.000** |
| `seniority` | 0.800 |
| `needs_h1b_sponsorship` | **1.000** |
| `open_to_remote` | **1.000** |
| `years_experience` (within ±1 yr) | **1.000** |
| `secondary_categories` Jaccard | 0.300 |
| `skills` Jaccard | 0.555 |
| LLM-judge mean (other 15, 0–5) | **3.87** |

**Cost:** $0.0084 (42,774 prompt + 3,378 completion tokens).

**Read:** Categorical fields are excellent. The `skills` Jaccard of 0.555
and `secondary_categories` Jaccard of 0.300 are the weakest signals — the
extractor is consistent on intent but lossy on normalisation
("PostgreSQL" vs "Postgres", "ML" vs "Machine Learning"). See
[Finding F3](#finding-f3--extractor-skills-normalisation-is-lossy).

### Retriever

Three approaches compared on the same 5 GT picks (top-5 jobs each).

| Approach | n | R@5 | R@10 | R@30 | R@50 | MRR | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **bm25** | 5 | **0.120** | **0.320** | **0.600** | **0.680** | **0.389** | **0.680** |
| embed | 5 | 0.040 | 0.040 | 0.240 | 0.280 | 0.086 | 0.280 |
| embed+filters | 5 | 0.080 | 0.240 | 0.280 | 0.280 | 0.314 | 0.280 |

**Cost:** $0.0002 (8,228 embedding tokens; no completion tokens).

**Read:** BM25 wins on every metric. Recall@30 gap (0.36) is well above
the 0.05 threshold for adding RRF-hybrid, so it's correctly skipped.
See [Finding F2](#finding-f2--bm25--embed-at-n300).

### Reranker

The reranker runs over `embed+filters` survivors (top-30 by default).

| Metric | Value |
| --- | --- |
| Queries scored | 5 |
| NDCG@10 | 0.315 |
| MRR@10 | **0.667** |
| Precision@5 | 0.200 |
| Score mean ± std | 55.61 ± 15.26 |
| Cohen's κ vs judge | _n/a (same model)_ |

**Cost:** $0.0069 (27,055 prompt + 4,552 completion tokens).

**Read:** MRR@10 of 0.667 means at least one relevant job lands in the
top-2 on average — that's the result the user actually sees. Precision@5
of 0.20 means only 1/5 of the top-5 are in GT — the reranker is good at
finding _the_ best fit but not consistently identifying multiple
relevant jobs. Score std of 15.26 confirms calibration is healthy (not
collapsed to 75s). κ is skipped because we ran with the same model for
extract and rerank — see [Finding F4](#finding-f4--cross-model-κ-needs-different-models).

### Latency (measured from `make demo` + UI uploads)

3 sequential `embed+rerank` requests through the live UI:

| Resume | Wall time | Reranker call | Notes |
| --- | --- | --- | --- |
| Resume 1 (28 cands → 9 valid) | ~28s | ~23s | extractor cache miss |
| Resume 2 (26 cands → 10 valid) | ~18s | ~14s | extractor cache miss |
| Resume 3 (14 cands → 13 valid) | ~21s | ~18s | smaller candidate set |

**The reranker dominates wall time** (75–85% of total). The extractor
adds ~3s on cache misses, the embed call ~0.5s, and pgvector + filters
are sub-millisecond. The original P95 plan budget of 5–7s was wrong —
real numbers are 18–28s, still inside the 30s Next.js proxy ceiling but
without much headroom. See [Finding F1](#finding-f1--latency-is-3-4×-the-planned-budget).

### Total observed cost (one full eval pass)

| Suite | USD | Tokens |
| --- | --- | --- |
| extractor | $0.0084 | 42,774p / 3,378c |
| retriever | $0.0002 | 8,228e |
| reranker | $0.0069 | 27,055p / 4,552c |
| **Total** | **$0.0155** | — |

A single live `embed+rerank` request through the UI costs roughly
**$0.001–$0.002** depending on candidate count. With the disk rerank
cache, repeated `make eval` runs are free.

---

## Findings

### Finding F1 — Latency is 3-4× the planned budget

The original `~5–7s p95` estimate undercounted the reranker call cost on
`gpt-4o-mini` for ~28 structured-output items. Measured 14–23s per
rerank. **What we'd change**:

- **Cap `RETRIEVAL_TOP_K` at 12–15** for `embed+rerank` (also fixes F5).
  Easy `.env` change, no code.
- **Parallelise `extract_resume_profile` + the embed call** with
  `asyncio.gather`. Currently sequential; saves ~1–2s on cache misses.
- **Stream the rerank response** so the UI can render the top-3 before
  the long tail finishes. Requires a small SSE refactor in `main.py` and
  the proxy.
- **Switch to `gpt-4o`** for rerank when latency matters more than cost —
  fewer tokens needed because the model rarely needs reasoning prefix.

### Finding F2 — BM25 > Embed at N=300

`text-embedding-3-small` cosine retrieval substantially underperformed
BM25 on every metric (R@30 0.24 vs 0.60, MRR 0.09 vs 0.39). Hypotheses,
ordered by likelihood:

1. **N is too small for dense embeddings to win.** Job descriptions in
   `data/jobs.json` are jargon-heavy ("gRPC", "Postgres",
   "Kubernetes") and lexical matching catches these directly. Dense
   embeddings tend to win at N≥10K where they can compress nuance the
   query lacks.
2. **Resume embeddings dilute strong signals.** A resume contains many
   irrelevant facts ("Mountaineering", "5K runner") that bias the
   centroid away from the role-relevant cluster. Approach: embed only
   the extractor-derived `summary` field instead of the full resume.
3. **Ground truth bias.** GT was generated by Claude Opus reading the
   same texts, which may favour exact-keyword matches.

**What we'd change**:

- **Default to `bm25+rerank`** when `OPENAI_API_KEY` is set (currently
  defaults to `embed+rerank`). One-line change in
  [`src/config.py`](src/config.py) `default_approach()`.
- **Test embed-of-summary** as `embed_v2` — extract first, embed only
  the structured summary string, re-measure.
- **Hybrid (RRF)** would help if the recall@30 gap were ≤ 0.05; current
  gap of 0.36 says BM25 is doing >2× the work of embed, so RRF would
  mostly mirror BM25.

### Finding F3 — Extractor skills normalisation is lossy

`skills` Jaccard 0.555 and `secondary_categories` Jaccard 0.300 against
the hand-labeled set. The extractor returns "PostgreSQL" while the
hand-label says "Postgres" (etc.). **What we'd change**:

- **Add a skill normaliser** (lowercase + alias dictionary) before
  Jaccard. Easy 30-line change in `eval/_common.py`.
- **Constrain the extractor schema** with `Field(examples=[...])`
  hinting canonical forms.
- **Retrain the hand-labels** to use the same normalised form the
  extractor produces — easier than fixing the LLM.

### Finding F4 — Cross-model κ needs different models

`EXTRACT_MODEL == RERANK_MODEL == gpt-4o-mini`, so `cohens_kappa` is
correctly skipped. To get a real κ:

```bash
RERANK_MODEL=gpt-4o make eval
```

Cost ~3–5× higher but yields a meaningful inter-rater agreement number.

### Finding F5 — Reranker silently truncates long candidate lists ⚠️

The most concerning finding. From the live runs:

```
rerank: calling gpt-4o-mini on 28 candidates
rerank: parsed 9/28 valid scores       ← model returned only 9
```

The system prompt
([`src/rerank/prompts.py:60-66`](src/rerank/prompts.py)) explicitly says
_"Return one entry per candidate id, in any order."_ The model ignores
this on lists of >15 candidates and returns a partial response. Net
effect: ~67% of candidates are silently dropped before the
`min_score=50` filter is even applied. This is the **biggest single
quality issue** in the pipeline today.

**What we'd change** (in priority order):

1. **Cap `RETRIEVAL_TOP_K` at 15** — empirically `gpt-4o-mini` is
   compliant below ~15 items.
2. **Add `min_length` validation** on `RerankResults.items` so a short
   response triggers a `tenacity` retry.
3. **Two-batch fan-out**: split candidates into two batches of 15, run
   reranker twice, merge by `job_id` taking max. Doubles cost but
   restores recall.
4. **Switch to `gpt-4o` or `claude-3-5-sonnet`** for rerank — both
   handle long structured outputs more reliably.

This is **not yet shipped** in the current code; we observed it
post-implementation. It is the #1 backlog item.

---

## Methodology

Evaluation is split into three suites that map to the three meaningful
"learnable" components of the pipeline:

| Suite | What it scores | Where it runs |
| --- | --- | --- |
| `extractor` | Did we extract the right `ResumeProfile`? | `eval/extractor_eval.py` |
| `retriever` | Does the top-N include the relevant jobs? | `eval/retriever_eval.py` |
| `reranker` | Within the retrieved top-N, did we rank the relevant jobs first? | `eval/reranker_eval.py` |

Run all three:

```bash
make eval
```

Output is written to `eval/results/<timestamp>.md` (gitignored — they're
generated artefacts) and an `EvalRun` row is inserted into Postgres for
historical tracking.

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

The remaining 15 resumes are scored by an LLM judge (GPT-4o-mini against
the extractor output). Coverage is therefore 25% strict / 75% LLM-judged.

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
complexity). On Apr 25 the gap was 0.36, so RRF was correctly skipped.

### Reranker

Within the embed-retrieved + filtered top-30, we score:

- `NDCG@10` against the GT-ranked top-5
- `MRR@10`
- `Precision@5`
- `score_mean` ± `score_std` (calibration check; we want a wide spread, not all 75s)
- `Cohen's κ` versus a second model when `EXTRACT_MODEL != RERANK_MODEL`

Rerank calls are cached on disk by `hash(resume + ranked_job_ids + model)`,
so re-running `make eval` is free (no extra tokens).

## Risk register

| ID | Risk | Mitigation | State |
| --- | --- | --- | --- |
| R6 | HTML in `responsibilities` poisons embeddings + BM25 | `utils/html.py` + stripped at ingest | mitigated |
| R11 | 30s proxy timeout vs cold rerank | Measured 18–28s; inside ceiling but tight (see F1) | partial |
| R13 | UI expects formatted `salary_range` string | `utils/salary.py` | mitigated |
| R28 | No Alembic = silent schema drift | Documented as production work below | accepted |
| R31 | 25% GT coverage | Supplement with LLM-as-judge; ship CIs in next iteration | partial |
| R32 | Embedding dim drift if `EMBED_MODEL` changes | Hard-coded `Vector(1536)`; only `text-embedding-3-small` supported | accepted |
| R33 | `EvalRun.metrics` JSON cannot store `NaN` | NaN-scrubbed before persist | mitigated |
| R34 | LLM rerank flakiness on bad JSON | `tenacity` retries + structured outputs + cache | mitigated |
| R35 | `embed`/`bm25+rerank` requested without `OPENAI_API_KEY` | Fall back to `bm25` with a logged warning; metadata records the requested approach | mitigated |
| R36 | Cost-tracker pricing drift | Static `_PRICING_PER_1K` map in `llm/client.py` — needs periodic refresh | accepted |
| **R37** | **Reranker truncates long candidate lists (F5)** | **None yet — biggest open issue** | **open** |
| **R38** | **Dense-embed underperforms BM25 at N=300 (F2)** | Default could move to `bm25+rerank`; needs investigation | open |
| **R39** | **Extractor skills/secondary Jaccard low (F3)** | Add skill normaliser before Jaccard; constrain schema | open |
| R40 | Cross-model κ skipped when EXTRACT_MODEL == RERANK_MODEL (F4) | Documented; toggle by setting `RERANK_MODEL=gpt-4o` | accepted |

## Production-hardening (next 1–2 sprints)

In priority order based on the live findings:

1. **Fix F5 — rerank truncation** (R37). Cap `RETRIEVAL_TOP_K=15`,
   add `min_length` Pydantic validator, retry on short response.
   _Single biggest quality lever._
2. **Investigate F2 — embed underperformance** (R38). Try embedding the
   extractor `summary` instead of full resume text; consider switching
   default approach to `bm25+rerank`.
3. **F1 — parallelise extract+embed** with `asyncio.gather`; saves ~1–2s
   on cache misses. Larger refactor: SSE-stream rerank results.
4. **F3 — skill normaliser** in `eval/_common.py` (lowercase + alias
   dict); constrain extractor schema with canonical examples.
5. **Alembic migrations** — replace `Base.metadata.create_all` with
   versioned migrations and a `migrate-then-serve` startup contract.
6. **HNSW index on `jobs.embedding`** — currently exact scan; with N=300
   it's instant, but for ≥10K jobs an `IVFFLAT`/`HNSW` index is needed.
7. **Per-request tracing** — Phoenix or OpenTelemetry spans for ingest /
   extract / embed / retrieve / rerank, exported to a sink the team
   owns. Would have surfaced F1/F5 earlier.
8. **Bootstrap CIs** for retriever / reranker metrics — current GT is
   N=5 so reported numbers are noisy; bands would say more than the
   point estimates.
9. **Eval as CI gate** — fail the build when `extractor` field
   agreement drops below the previous run's bound (with a tolerance).
10. **Hybrid retrieval** — RRF fuse of `bm25` and `embed`, gated on the
    recall@30 gap (already a conditional in `retriever_eval.py`).
    Currently dormant because gap > 0.05.
11. **Eval cost guardrails** — per-suite `cost_usd` cap with a hard stop.
12. **Resume-PDF/DOCX parsing** — extend `Resume` to store the parsed
    text alongside the original blob.
13. **Per-tenant rate limiting** + auth on `/match` (currently CORS-open).
14. **Real LLM observability** — log prompt + completion (with PII
    redaction) so we can replay failed extractions/reranks.

## Out of scope (this take-home)

- Resume PDF/DOCX parsing
- Frontend changes beyond the TS contract additions
- Phoenix / external observability backend
- Alembic migrations
- HNSW index on `jobs.embedding`
- Real-time learning from user feedback
- Streaming rerank responses (SSE)
