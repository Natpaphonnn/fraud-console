.PHONY: check lint test notebooks kaggle docs web-serve web-preview

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

# pytest includes the website tests (and the Node simulator tests when node is installed)
test:
	uv run pytest -q
	uv run python scripts/render_docs.py --check
	uv run python scripts/build_web.py --check

# 01-06 in order, then 00 (the summary reads their results)
notebooks:
	for f in notebooks/py/0[1-9]*.py notebooks/py/00_summary.py; do \
		uv run jupytext --to ipynb --output notebooks/$$(basename $$f .py).ipynb --execute $$f || exit 1; \
	done
	uv run python scripts/clean_notebooks.py notebooks/*.ipynb

kaggle:
	uv run python scripts/build_kaggle.py
	uv run python scripts/clean_notebooks.py kaggle/*.ipynb

# README, model card, Thai interview notes and the website, all from reports/ + configs/links.yaml
docs:
	uv run python scripts/render_docs.py
	uv run python scripts/build_web.py

# local preview of the website (Ctrl+C to stop)
web-serve:
	uv run python -m http.server 8000 --bind 127.0.0.1 --directory web/site

# desktop + mobile screenshots and a smoke test of the simulator (uses the installed Google Chrome)
web-preview:
	uv run --with playwright python scripts/web_preview.py
