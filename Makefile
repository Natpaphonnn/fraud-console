.PHONY: check lint test notebooks kaggle docs web-model web-serve web-preview web-live kaggle-test content-preview

check: lint test

lint:
	uv run ruff check .
	uv run ruff format --check .

# pytest + generated-file freshness + Node tests of the in-browser model and simulator (needs node)
test:
	uv run pytest -q
	node --test web/tests/
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

# README, model card and the website, all from reports/ + configs/links.yaml (+ local-only docs if present)
docs:
	uv run python scripts/render_docs.py
	uv run python scripts/build_web.py

# re-export the logistic pipeline + scored dev sample for the website (dev rows only, deterministic)
web-model:
	uv run python scripts/export_web_model.py

# local preview of the website (Ctrl+C to stop)
web-serve:
	uv run python -m http.server 8000 --bind 127.0.0.1 --directory web/site

# desktop + mobile screenshots and a smoke test of the simulator (uses the installed Google Chrome)
web-preview:
	uv run --with playwright python scripts/web_preview.py

# smoke-test the deployed page (GitHub Pages URL from configs/links.yaml)
web-live:
	cd scripts && uv run --with playwright python check_live.py

# run the Kaggle story + 08/09 editions in a simulated Kaggle layout
kaggle-test:
	uv run python scripts/test_kaggle.py

# low-res content previews (full renders + clips run on GitHub Actions: content.yml)
content-preview:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 19 uv run python scripts/content_figures.py --quick
