"""Smoke-test the deployed page (default: the GitHub Pages URL in configs/links.yaml).

Desktop and mobile: no HTTP errors (404s), no console errors, the decision flow animates,
and after pausing every visual matches the counters. Uses the installed Google Chrome.

    uv run --with playwright python scripts/check_live.py [URL]      # (make web-live)
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright
from web_preview import MOBILE, Checks, measure_fps, open_page, visuals_match_counters

from fraud_e2e.summary import load_links


def run(url: str) -> int:
    check = Checks()
    print("checking", url)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        for name, opts in [
            ("live desktop", {"viewport": {"width": 1440, "height": 900}}),
            (
                "live mobile",
                {
                    "viewport": MOBILE,
                    "device_scale_factor": 2,
                    "is_mobile": True,
                    "has_touch": True,
                },
            ),
        ]:
            page, errors = open_page(browser, url, check, name, **opts)
            page.click('.seg [data-speed="4"]')
            page.wait_for_timeout(1500)
            check(
                page.get_attribute("#flow-canvas", "data-animating") == "true",
                f"{name}: decision-flow dots are moving",
            )
            fps, worst = measure_fps(page, 1500)
            print(f"     {name}: ~{fps:.0f} fps (worst frame {worst:.0f} ms)")
            page.click("#s-play")  # pause
            visuals_match_counters(page, check, name)
            check(not errors, f"{name}: no HTTP or console errors {errors}")
            page.close()
        browser.close()
    if check.failures:
        print(f"{len(check.failures)} check(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1] if len(sys.argv) > 1 else load_links()["website"]))
