.PHONY: check lint test notebooks kaggle readme

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q
	uv run python scripts/render_readme.py --check

notebooks:
	for f in notebooks/py/*.py; do \
		uv run jupytext --to ipynb --output notebooks/$$(basename $$f .py).ipynb --execute $$f || exit 1; \
	done
	uv run python scripts/clean_notebooks.py notebooks/*.ipynb

kaggle:
	uv run python scripts/build_kaggle.py
	uv run python scripts/clean_notebooks.py kaggle/*.ipynb

readme:
	uv run python scripts/render_readme.py
