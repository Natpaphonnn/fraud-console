"""Build the static project page web/site/ from web/src/ + reports/ (no hand-typed numbers).

uv run python scripts/build_web.py          # write web/site/
uv run python scripts/build_web.py --check  # exit 1 if web/site/ is stale
"""

from __future__ import annotations

import sys

from fraud_e2e.web import SITE, stale_files, write_site


def main(check: bool) -> int:
    if check:
        stale = stale_files()
        if stale:
            print("web/site is stale (run `make docs`):", *stale)
            return 1
        return 0
    for name in write_site():
        print("built", (SITE / name).relative_to(SITE.parents[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
