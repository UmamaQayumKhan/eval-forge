.PHONY: help dev test lint validate power run redteam compare variance baseline gate report clean

CONFIG ?= configs/support-qa.json
BASELINE ?= baselines/support-qa.json

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

dev:
	pip install -r requirements-dev.txt

test: ## 85 tests, offline
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests
	mypy src/evalforge || true

validate: ## dataset integrity check
	PYTHONPATH=src python -m evalforge.cli validate datasets/support-qa.jsonl
	PYTHONPATH=src python -m evalforge.cli validate datasets/redteam.jsonl
	PYTHONPATH=src python -m evalforge.cli validate datasets/extraction.jsonl

power: ## what can an eval of this size detect?
	PYTHONPATH=src python -m evalforge.cli power --n 44

run: ## run the support QA eval
	PYTHONPATH=src python -m evalforge.cli run --config $(CONFIG)

redteam: ## run the red-team suite
	PYTHONPATH=src python -m evalforge.cli run --config configs/redteam.json

compare: ## paired comparison of two prompts
	PYTHONPATH=src python -m evalforge.cli compare --config configs/provider-bakeoff.json

variance: ## measure run-to-run variance (do this first)
	PYTHONPATH=src python -m evalforge.cli variance --config $(CONFIG) --runs 5

baseline: ## record a baseline
	PYTHONPATH=src python -m evalforge.cli baseline --config $(CONFIG) --out $(BASELINE)

gate: ## the CI regression gate
	PYTHONPATH=src python -m evalforge.cli gate --config $(CONFIG) --baseline $(BASELINE)

report: ## write json/md/html reports to reports/
	PYTHONPATH=src python -m evalforge.cli run --config $(CONFIG) --out reports
	@echo "open reports/report.html"

clean:
	rm -rf .pytest_cache .ruff_cache .evalforge_cache reports
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
