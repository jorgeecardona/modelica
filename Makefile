# modelica -- task runner. uv is the only tool needed.
# One-time setup after cloning:  make install-hooks
.DEFAULT_GOAL := help
.PHONY: help install-hooks format lint format-check typecheck test check version-ahead bump build docs docs-prose

# The scope CI's format-check runs over -- broader than lint's src+tests, so an unformatted
# example (a .py twin) is caught locally before the push rather than in the pipeline.
FMT_SCOPE := src tests examples docs/notebooks

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install-hooks: ## point git at the tracked .githooks/ (pre-commit + pre-push)
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-commit .githooks/pre-push scripts/bump-version scripts/check-version-bump
	@echo "hooks installed: core.hooksPath -> .githooks"

format: ## auto-format + autofix lint (ruff)
	uv run ruff format $(FMT_SCOPE)
	uv run ruff check --fix src tests

lint: ## lint (ruff) -- same as CI
	uv run ruff check src tests

format-check: ## fail if anything in the CI scope is not ruff-formatted -- same as CI
	uv run ruff format --check $(FMT_SCOPE)

typecheck: ## type-check (pyright) -- same as CI
	uv run pyright

test: ## run the test suite in parallel (pytest-parallex, worker count = CPU count)
	uv run pytest -q --parallel=process
# `--parallel=process` runs the suite across one process per core with pytest-parallex, which
# takes the ~2783 independent tests from ~305s serial to ~140s on a 14-core machine and scales
# down cleanly on a smaller CI runner (auto workers = CPU count, so a 2-core box just uses 2).
# Drop the flag -- `uv run pytest -q` -- to run serially when debugging a single failure.

check: lint format-check typecheck test ## the full CI test stage: lint + format + typecheck + test

version-ahead: ## fail unless pyproject version is greater than origin/main (when src/ changed)
	sh scripts/check-version-ahead

docs: ## build the site the way CI does -- notebooks re-executed, cache dropped first
	rm -rf .cache/mkdocs-jupyter
	uv run --extra docs --extra sim --extra verify mkdocs build --strict

docs-prose: ## rebuild fast for a prose edit -- reuses cached notebook output, which may be stale
	uv run --extra docs --extra sim --extra verify mkdocs build --strict

bump: ## bump version in pyproject.toml -- TYPE=patch|minor|major (default patch)
	./scripts/bump-version $(or $(TYPE),patch)

build: ## build sdist + wheel into dist/
	uv build --out-dir dist
