PY=uv run python

.DEFAULT_GOAL := help

.PHONY: help setup env add-deps dev upgrade run lint format test clean \
        phase2 phase3 phase4 phase5 phase6 phase7 phase8 pipeline \
        gate1 gate2 overnight

help: ## Show available targets
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_\-]+:.*?##' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create fresh uv venv and sync prod dependencies
	uv venv --clear
	uv sync

env: setup ## Backward-compatible alias for setup

add-deps: ## Add dependencies from requirements.txt via uv when present
	@if [ -f requirements.txt ]; then uv add -r requirements.txt; else echo "requirements.txt not found"; fi

dev: ## Create fresh uv venv and sync dev dependencies
	uv venv --clear
	uv sync --dev --all-extras

upgrade: ## Upgrade locked dependencies and sync dev environment
	uv lock --upgrade
	uv sync --dev --all-extras

run: ## Run the project entrypoint
	$(PY) main.py

lint: ## Run Ruff checks
	uv run ruff check .

format: ## Run Ruff fix + format
	uv run ruff check --fix .
	uv run ruff format .

test: ## Run the test suite
	uv run pytest

clean: ## Remove local build artifacts and caches
	rm -rf .pytest_cache .ruff_cache .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

# ── Validation gates ───────────────────────────────────────────────────────────

gate1: ## Gate 1 — data loader tests (run after Phase 1)
	uv run pytest tests/test_data.py -v

gate2: ## Gate 2 — unsupervised tests (run after Phase 2)
	uv run pytest tests/test_unsupervised.py -v

# ── Phase scripts ──────────────────────────────────────────────────────────────

phase2: ## Phase 2 — raw clustering sweep (wine + adult)
	bash ml_run.sh "$(PY) scripts/run_phase_2_raw_cluster.py"

phase3: ## Phase 3 — raw dimensionality reduction (PCA, ICA, RP)
	bash ml_run.sh "$(PY) scripts/run_phase_3_raw_reduction.py"

phase4: ## Phase 4 — clustering in reduced spaces
	bash ml_run.sh "$(PY) scripts/run_phase_4_reduced_cluster.py"

phase5: ## Phase 5 — Wine NN on reduced inputs (Step 4)
	bash ml_run.sh "$(PY) scripts/run_phase_5_nn_reduced.py"

phase6: ## Phase 6 — Wine NN with cluster-derived features (Step 5)
	bash ml_run.sh "$(PY) scripts/run_phase_6_nn_cluster_features.py"

phase7: ## Phase 7 — t-SNE visualizations (extra credit)
	bash ml_run.sh "$(PY) scripts/run_phase_7_tsne.py"

phase8: ## Phase 8 — generate report tables from artifacts
	bash ml_run.sh "$(PY) scripts/generate_report_tables.py"

# ── Full pipeline ──────────────────────────────────────────────────────────────

pipeline: gate1 gate2 phase2 phase3 phase4 phase5 phase6 phase8 ## Run full pipeline phase 2→8 (skips t-SNE)

overnight: ## Run full pipeline in background tmux/screen session (safe to close terminal)
	bash ml_run.sh --detach "make pipeline" ul_phases
