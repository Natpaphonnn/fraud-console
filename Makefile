.PHONY: check lint test notebooks kaggle

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

notebooks:
	for f in notebooks/py/*.py; do \
		uv run jupytext --to ipynb --output notebooks/$$(basename $$f .py).ipynb --execute $$f || exit 1; \
	done

kaggle:
	uv run python scripts/build_kaggle.py
