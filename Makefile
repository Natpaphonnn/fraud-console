.PHONY: check lint test notebooks kaggle docs

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q
	uv run python scripts/render_docs.py --check

# 01-06 in order, then 00 (the summary reads their results)
notebooks:
	for f in notebooks/py/0[1-9]*.py notebooks/py/00_summary.py; do \
		uv run jupytext --to ipynb --output notebooks/$$(basename $$f .py).ipynb --execute $$f || exit 1; \
	done
	uv run python scripts/clean_notebooks.py notebooks/*.ipynb

kaggle:
	uv run python scripts/build_kaggle.py
	uv run python scripts/clean_notebooks.py kaggle/*.ipynb

docs:
	uv run python scripts/render_docs.py
