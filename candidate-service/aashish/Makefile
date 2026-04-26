# =============================================================================
# Resume Matching Service - candidate-service/aashish
# =============================================================================
# Reviewer path:   make demo                  (then `npm run dev` at repo root)
# Dev iteration:   make install && make demo
# Validation:      make test
#
# When OPENAI_API_KEY changes, or you suspect bad embeddings:
#   make fresh                                (reset + demo, re-ingests via OpenAI)
#
# Just want to wipe the DB without restarting the server:
#   make reset
# =============================================================================

PYTHON       ?= python3
VENV         ?= .venv
PIP          := $(VENV)/bin/pip
PY           := $(VENV)/bin/python
UVICORN      := $(VENV)/bin/uvicorn
PYTEST       := $(VENV)/bin/pytest
RUFF         := $(VENV)/bin/ruff

REPO_ROOT    := $(shell cd ../../ && pwd)
COMPOSE      := docker compose -f $(REPO_ROOT)/docker-compose.yml

.PHONY: help install venv compose-up compose-wait demo run seed-check eval test lint format down reset fresh clean

help:
	@echo "Targets:"
	@echo "  install       Create venv and install Python deps"
	@echo ""
	@echo "  --- Run the service ---"
	@echo "  demo          Start postgres + uvicorn. Reuses existing DB if present."
	@echo "                Use this for normal runs (review, dev iteration, restarts)."
	@echo "  run           Same as demo but assumes postgres is already running."
	@echo "  fresh         reset + demo. Use when:"
	@echo "                  - you just added/changed OPENAI_API_KEY in .env"
	@echo "                  - you suspect bad embeddings (e.g. all-zero vectors from a"
	@echo "                    previous boot without a key) causing 500s on /match"
	@echo "                  - you changed EMBED_MODEL and want to re-embed all jobs"
	@echo "                Costs ~\$$0.003 in OpenAI embedding calls (300 jobs)."
	@echo ""
	@echo "  --- Stop / wipe ---"
	@echo "  down          Stop the postgres container. Data volume PRESERVED."
	@echo "                Use when you're done for the day but want quick restart later."
	@echo "  reset         Stop postgres AND DELETE its volume (pgdata). Next 'make demo'"
	@echo "                will re-create the schema and re-ingest jobs.json from scratch."
	@echo "                Use when you want a clean slate but aren't ready to run yet."
	@echo ""
	@echo "  --- Diagnostics ---"
	@echo "  seed-check    Print job count from DB (sanity check after ingest)."
	@echo "  eval-validate-gt  Validate eval/ground_truth/ file shape + job_id existence."
	@echo "  eval          Run all 3 eval suites and write a timestamped markdown report."
	@echo "  test          Run pytest (uses testcontainers for integration tests)."
	@echo "  lint          ruff check"
	@echo "  format        ruff format"
	@echo ""
	@echo "  --- Cleanup ---"
	@echo "  clean         Remove venv and pycache (does NOT touch postgres volume)."

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

compose-up:
	$(COMPOSE) up -d postgres

compose-wait: compose-up
	@echo "Waiting for postgres healthcheck..."
	@for i in $$(seq 1 30); do \
	  status=$$($(COMPOSE) ps --format json postgres 2>/dev/null | grep -o '"Health":"[a-z]*"' | head -1 | cut -d'"' -f4); \
	  if [ "$$status" = "healthy" ]; then echo "postgres is healthy"; exit 0; fi; \
	  sleep 1; \
	done; \
	echo "postgres did not become healthy in 30s" >&2; exit 1

demo: compose-wait
	@echo ""
	@echo "============================================================"
	@echo "  Postgres ready. Starting FastAPI on http://localhost:8000"
	@echo "  In a separate terminal at the repo root, run: npm run dev"
	@echo "  Then open http://localhost:3000"
	@echo "============================================================"
	@echo ""
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8000 --reload

run:
	$(UVICORN) src.main:app --host 0.0.0.0 --port 8000 --reload

seed-check:
	$(PY) -c "from src.db.engine import SessionLocal; from src.db.models import Job; \
s = SessionLocal(); print(f'Jobs in DB: {s.query(Job).count()}'); s.close()"

eval-validate-gt:
	$(PY) -m eval.validate_ground_truth

eval: eval-validate-gt
	$(PY) -m eval.report

test:
	$(PYTEST)

lint:
	$(RUFF) check src tests eval

format:
	$(RUFF) format src tests eval

down:
	$(COMPOSE) down

# Stop postgres AND delete the pgdata volume.
# Use this when:
#   - You added OPENAI_API_KEY to .env after a previous boot without it
#     (otherwise the table keeps zero-vector embeddings -> NaN cosine -> 500s)
#   - You changed EMBED_MODEL and want all 300 jobs re-embedded
#   - You want to verify ingest is reproducible from a clean slate
# After this, the next `make demo` will re-create the schema and re-ingest jobs.json.
reset:
	@echo "Stopping postgres and removing pgdata volume..."
	$(COMPOSE) down -v
	@echo "Done. Run 'make demo' (or 'make fresh') to re-ingest."

# Convenience: full clean re-ingest in one command.
# Equivalent to: make reset && make demo
# Costs ~$$0.003 in OpenAI embedding calls (300 jobs at text-embedding-3-small).
fresh: reset demo

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
