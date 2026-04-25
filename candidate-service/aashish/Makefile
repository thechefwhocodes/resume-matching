# =============================================================================
# Resume Matching Service - candidate-service/aashish
# =============================================================================
# Reviewer path:   make demo                  (then `npm run dev` at repo root)
# Dev iteration:   make install && make demo
# Validation:      make test
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

.PHONY: help install venv compose-up compose-wait demo run seed-check eval test lint format down clean

help:
	@echo "Targets:"
	@echo "  install       Create venv and install Python deps"
	@echo "  demo          docker compose up postgres -> wait healthy -> uvicorn (reviewer command)"
	@echo "  run           Same as demo but assumes postgres is already running"
	@echo "  seed-check    Print job count from DB (sanity check after ingest)"
	@echo "  eval-validate-gt  Validate eval/ground_truth/ file shape + job_id existence"
	@echo "  eval          Run all 3 eval suites and write a timestamped markdown report"
	@echo "  test          Run pytest (uses testcontainers for integration tests)"
	@echo "  lint          ruff check"
	@echo "  format        ruff format"
	@echo "  down          Stop the postgres container (data volume preserved)"
	@echo "  clean         Remove venv and pycache"

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

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
