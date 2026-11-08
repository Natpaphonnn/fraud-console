"""Strip run-specific noise from executed notebooks so re-runs give clean diffs.

Removes execution timestamps and assigns stable cell ids (cell-000, cell-001, ...).
Outputs are kept.

    uv run python scripts/clean_notebooks.py notebooks/*.ipynb
"""

import sys

import nbformat

for path in sys.argv[1:]:
    nb = nbformat.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        cell["id"] = f"cell-{i:03d}"
        cell.get("metadata", {}).pop("execution", None)
    nbformat.write(nb, path)
    print("cleaned", path)
