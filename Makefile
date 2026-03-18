PY=uv run python

.DEFAULT_GOAL := help

.PHONY: help setup env add-deps dev upgrade run lint format test spec-new spec-list spec-check clean

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

spec-new: ## Create a new spec folder (usage: make spec-new NAME=my-change)
	@if [ -z "$(NAME)" ]; then echo "NAME is required"; exit 1; fi
	$(PY) scripts/new_spec.py "$(NAME)"

spec-list: ## List spec folders
	@find specs -mindepth 1 -maxdepth 1 -type d ! -name templates | sort

spec-check: ## Validate spec folder structure
	$(PY) scripts/check_specs.py

clean: ## Remove local build artifacts and caches
	rm -rf .pytest_cache .ruff_cache .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
