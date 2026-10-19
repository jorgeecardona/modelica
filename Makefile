# modelica -- human task verbs. The CI-graded tasks that match a moon task 1:1
# (lint/format-check/typecheck/build/version-ahead) delegate to it, so the command
# + its inputs/outputs live in ONE place (moon.yml) and `make` runs what CI runs,
# with moon's caching. Kept direct on purpose: `test` (local `--parallel=process`
# dogfood -- moon's :test is serial for CI determinism) and `docs` (drops the
# notebook cache first -- moon's :docs assumes CI's clean checkout). format/bump/
# install-hooks have no moon task.
# One-time setup after cloning:  make install-hooks
.DEFAULT_GOAL := help
.PHONY: help install-hooks format lint format-check typecheck test test-all check version-ahead bump build docs docs-prose

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

lint: ## lint (ruff) -- the CI :lint task
	moon run :lint

format-check: ## fail if anything in the CI scope is not ruff-formatted -- the CI :format-check task
	moon run :format-check

typecheck: ## type-check (pyright) -- the CI :typecheck task
	moon run :typecheck

test: ## fast local loop: everything NOT marked `slow`, one worker per core (~40s/16 cores, ~65s/4)
	uv run pytest -q --parallel=process -m "not slow"
# `make test` is the quick inner loop: it deselects tests marked `slow` (whole-corpus ODE
# integration, model checking, and codegen/compile), leaving ~2400 fast tests that scale with
# cores. Nothing is lost -- the slow tests still run in CI (`moon run :test`) and in
# `make test-all`, which pre-push and `make check` call. Drop `--parallel=process` to run
# serially when debugging one failure; run `make test-all` to include the slow tests locally.

test-all: ## the whole suite in parallel -- what CI's :test runs (serially) and pre-push mirrors
	uv run pytest -q --parallel=process

check: lint format-check typecheck test-all ## the full CI test stage: lint + format + typecheck + ALL tests

version-ahead: ## fail unless __version__ is greater than origin/main (when src/ changed) -- the CI :version-ahead task
	moon run :version-ahead

docs: ## build the site the way CI does -- notebooks re-executed, cache dropped first
	rm -rf .cache/mkdocs-jupyter
	uv run --extra docs --extra sim --extra verify mkdocs build --strict

docs-prose: ## rebuild fast for a prose edit -- reuses cached notebook output, which may be stale
	uv run --extra docs --extra sim --extra verify mkdocs build --strict

bump: ## bump __version__ in src/modelica/__init__.py -- TYPE=patch|minor|major (default patch)
	./scripts/bump-version $(or $(TYPE),patch)

build: ## build sdist + wheel into dist/ -- the CI :build task
	moon run :build
