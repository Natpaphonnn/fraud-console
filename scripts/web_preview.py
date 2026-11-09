"""Screenshot the built website (desktop + mobile) and smoke-test the simulator.

Serves web/site/ on localhost, drives the installed Google Chrome with Playwright,
fails on any console error, horizontal overflow, or a simulator that does not react.

    uv run --with playwright python scripts/web_preview.py      # (make web-preview)

Output: reports/web_preview/*.png
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

from fraud_e2e.summary import build_context

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "web" / "site"
OUT = REPO / "reports" / "web_preview"


def serve() -> tuple[socketserver.TCPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(SITE))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def check(cond: bool, msg: str, failures: list[str]) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    failures: list[str] = []
    httpd, url = serve()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=True)
            shots = [
                ("desktop", {"width": 1366, "height": 900}, 1, "light"),
                ("mobile", {"width": 390, "height": 844}, 2, "light"),
                ("desktop_dark", {"width": 1366, "height": 900}, 1, "dark"),
            ]
            for name, viewport, dpr, scheme in shots:
                page = browser.new_page(
                    viewport=viewport,
                    device_scale_factor=dpr,
                    color_scheme=scheme,
                    is_mobile=name == "mobile",
                    has_touch=name == "mobile",
                )
                errors: list[str] = []
                page.on("console", lambda m, e=errors: m.type == "error" and e.append(m.text))
                page.on("pageerror", lambda exc, e=errors: e.append(str(exc)))
                page.goto(url, wait_until="networkidle")
                page.wait_for_selector("#cost-chart svg")
                check(not errors, f"{name}: no console errors {errors}", failures)
                overflow = page.evaluate(
                    "document.scrollingElement.scrollWidth - document.documentElement.clientWidth"
                )
                check(
                    overflow <= 0, f"{name}: no horizontal scroll (overflow {overflow}px)", failures
                )
                check(
                    page.text_content("#o-recall") == ctx["cv_recall"]
                    and page.text_content("#thr-out") == ctx["threshold"],
                    f"{name}: simulator starts at the project threshold {ctx['threshold']}",
                    failures,
                )
                if name == "desktop_dark":
                    page.screenshot(path=OUT / f"{name}.png")
                    page.close()
                    continue
                page.screenshot(path=OUT / f"{name}.png", full_page=True)
                # element shots scroll the page; keep the sticky nav from covering the section
                page.add_style_tag(content=".toc { position: static !important; }")
                page.locator("#simulator").screenshot(path=OUT / f"{name}_simulator.png")

                if name == "desktop":
                    # interaction smoke test: doubling friction must move the best threshold up
                    page.click("#btn-best")
                    before = page.text_content("#thr-out")
                    page.fill("#c-friction", str(2 * float(ctx["cost_friction"])))
                    page.click("#btn-best")
                    after = page.text_content("#thr-out")
                    check(
                        float(after) > float(before),
                        f"cost change moves the best threshold ({before} -> {after})",
                        failures,
                    )
                    page.fill("#c-friction", "-1")
                    check(page.is_visible("#cost-error"), "invalid cost shows an error", failures)
                    page.click("#btn-reset")
                    check(
                        page.text_content("#o-cost") == ctx["cv_cost_per_1000"],
                        "reset restores the project defaults",
                        failures,
                    )
                page.close()
            browser.close()
    finally:
        httpd.shutdown()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(REPO))
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
