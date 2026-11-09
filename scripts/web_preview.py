"""Screenshot the built website and smoke-test it in a real browser.

Serves web/site/ on localhost and drives the installed Google Chrome with Playwright.
Fails on any console error, horizontal overflow, or a feature that does not react:
live stream (autoplay, reduced-motion pause, step, inspector), shared threshold,
cost simulator, try-it-yourself, command palette, light/dark theme.

    uv run --with playwright python scripts/web_preview.py      # (make web-preview)

Output: reports/web_preview/*.png (v1 of the page is kept in reports/web_preview/v1/).
"""

from __future__ import annotations

import functools
import http.server
import socketserver
import sys
import threading
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from fraud_e2e.summary import build_context

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "web" / "site"
OUT = REPO / "reports" / "web_preview"
DESKTOP = {"width": 1440, "height": 900}
MOBILE = {"width": 390, "height": 844}


def serve() -> tuple[socketserver.TCPServer, str]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(SITE))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def __call__(self, cond: bool, msg: str) -> None:
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            self.failures.append(msg)


def open_page(browser, url, check, name, **ctx) -> tuple[Page, list[str]]:
    page = browser.new_page(**ctx)
    errors: list[str] = []
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(url, wait_until="networkidle")
    page.wait_for_selector("#cost-chart svg")
    overflow = page.evaluate(
        "document.scrollingElement.scrollWidth - document.documentElement.clientWidth"
    )
    check(overflow <= 0, f"{name}: no horizontal scroll (overflow {overflow}px)")
    return page, errors


def played(page: Page) -> int:
    return int(page.text_content("#c-played").split("/")[0].replace(",", ""))


def step_until_catch(page: Page, min_rows: int = 30, max_rows: int = 400) -> None:
    """Deterministic stream state for screenshots: step until a fraud has been caught."""
    for _ in range(max_rows):
        if played(page) >= min_rows and int(page.text_content("#c-tp")) >= 1:
            return
        page.click("#s-step")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    check = Checks()
    httpd, url = serve()
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=True)

            # 1. autoplay (default motion settings)
            page, errors = open_page(browser, url, check, "autoplay", viewport=DESKTOP)
            check(page.get_attribute("html", "data-theme") == "dark", "dark theme is the default")
            page.wait_for_timeout(1800)
            n = played(page)
            check(n > 0, f"stream autoplays ({n} rows in 1.8 s)")
            check(not errors, f"autoplay: no console errors {errors}")
            page.close()

            # 2. desktop, reduced motion: deterministic stream, all interactions
            page, errors = open_page(
                browser, url, check, "desktop", viewport=DESKTOP, reduced_motion="reduce"
            )
            check(played(page) == 0, "reduced motion: stream waits for the user")
            check(page.is_visible("#s-hint"), "reduced motion: hint explains how to start")
            check(
                page.text_content("#thr-out") == ctx["threshold"]
                and page.text_content("#thr-mini-out") == ctx["threshold"]
                and page.text_content("#o-recall") == ctx["cv_recall"],
                f"simulator and stream start at the project threshold {ctx['threshold']}",
            )
            step_until_catch(page)
            rows = page.locator("#s-rows button.srow")
            check(rows.count() > 0, f"stream shows rows ({rows.count()})")
            stepped = page.locator("#s-rows button.srow:has(.dec.step)")
            (stepped.first if stepped.count() else rows.first).click()
            check(page.text_content("#insp-mode") == "pinned", "click pins the inspector")
            n_bars = page.locator("#insp-body .contrib li").count()
            check(n_bars >= 5, f"inspector shows per-feature contributions ({n_bars} lines)")
            check(
                page.text_content("#full-title").startswith("full sample")
                and page.text_content("#played-title").startswith("played so far"),
                "outcomes label 'full sample' and 'played so far' separately",
            )
            check(page.text_content("#f-recall") not in ("", "—"), "full-sample recall shown")
            text = page.text_content("main")
            check("e-" not in page.text_content("#s-rows"), "stream scores use no e-notation")
            check("<0.0001" in text, "scores below 0.0001 show as <0.0001")
            check(page.is_visible(".srow--head .c-scale"), "score column header says 'log scale'")
            page.screenshot(path=OUT / "desktop.png", full_page=True)
            page.screenshot(path=OUT / "desktop_hero.png")

            field = page.locator('#pg-fields [data-f="merchant_risk_score"]')
            field.fill("0")
            low = float(page.text_content("#pg-score").lstrip("<"))
            field.fill("100")
            high = float(page.text_content("#pg-score").lstrip("<"))
            check(high > low, f"try-it score reacts to merchant_risk_score ({low} -> {high})")

            page.click("#btn-best")
            before = page.text_content("#thr-out")
            page.fill("#c-friction", str(2 * float(ctx["cost_friction"])))
            page.click("#btn-best")
            after = page.text_content("#thr-out")
            moved = f"cost change moves the threshold ({before} -> {after})"
            check(float(after) > float(before), moved)
            shared = page.text_content("#thr-mini-out") == after
            check(shared, "stream uses the simulator's threshold")
            page.fill("#c-friction", "-1")
            check(page.is_visible("#cost-error"), "invalid cost shows an error")
            page.click("#btn-reset")
            check(
                page.text_content("#o-cost") == ctx["cv_cost_per_1000"]
                and page.text_content("#thr-out") == ctx["threshold"],
                "reset restores the project defaults",
            )

            page.locator("#metrics h2").click()  # focus outside any input
            page.keyboard.press("Control+k")
            check(page.is_visible("#palette"), "Ctrl+K opens the command palette")
            page.keyboard.type("didn")
            page.screenshot(path=OUT / "desktop_palette.png")
            page.keyboard.press("Enter")
            check(not page.is_visible("#palette"), "Enter runs the command and closes the palette")
            check(
                page.get_attribute("#r-didnt-work", "open") is not None,
                "palette opens a folded report module",
            )
            check(not errors, f"desktop: no console errors {errors}")
            page.close()

            # 3. mobile
            page, errors = open_page(
                browser,
                url,
                check,
                "mobile",
                viewport=MOBILE,
                device_scale_factor=2,
                is_mobile=True,
                has_touch=True,
                reduced_motion="reduce",
            )
            step_until_catch(page)
            check(page.is_visible(".srow--head .c-out"), "mobile: truth column has a header")
            check(page.is_visible("#s-legend"), "mobile: legend for the truth symbols is visible")
            check(page.is_visible(".srow--head .c-scale-s"), "mobile: score header says '(log)'")
            clipped = page.evaluate(
                """[...document.querySelectorAll('#s-legend .lg-item, #s-rows .sbar-val')]
                   .filter(e => e.getBoundingClientRect().right > innerWidth
                             || e.scrollWidth > e.clientWidth + 1).length"""
            )
            check(clipped == 0, f"mobile: legend and scores fit ({clipped} clipped)")
            page.screenshot(path=OUT / "mobile.png", full_page=True)
            page.screenshot(path=OUT / "mobile_hero.png")
            check(not errors, f"mobile: no console errors {errors}")
            page.close()

            # 4. light theme (toggle, persists across reload)
            page, errors = open_page(
                browser, url, check, "light", viewport=DESKTOP, reduced_motion="reduce"
            )
            page.click("#theme-toggle")
            page.reload(wait_until="networkidle")
            page.wait_for_selector("#cost-chart svg")
            check(page.get_attribute("html", "data-theme") == "light", "light theme persists")
            step_until_catch(page)
            stepped = page.locator("#s-rows button.srow:has(.dec.step)")
            if stepped.count():
                stepped.first.click()
            page.screenshot(path=OUT / "light.png", full_page=True)
            page.screenshot(path=OUT / "light_hero.png")
            check(not errors, f"light: no console errors {errors}")
            page.close()
            browser.close()
    finally:
        httpd.shutdown()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(REPO))
    if check.failures:
        print(f"{len(check.failures)} check(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
