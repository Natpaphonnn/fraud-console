"""Screenshot the built website and smoke-test it in a real browser.

Serves web/site/ on localhost under /fraud-console/ (as on GitHub Pages) and drives
the installed Google Chrome with Playwright. Fails on any console error, any HTTP
error (e.g. a 404 from a root-absolute path), horizontal overflow, or a feature that
does not react: live stream (autoplay, reduced-motion pause, step, inspector), shared threshold,
cost simulator, try-it-yourself, command palette, light/dark theme.

    uv run --with playwright python scripts/web_preview.py      # (make web-preview)

Output: reports/web_preview/*.png (v1 of the page is kept in reports/web_preview/v1/).
"""

from __future__ import annotations

import functools
import http.server
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from fraud_e2e.summary import build_context

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "web" / "site"
OUT = REPO / "reports" / "web_preview"
DESKTOP = {"width": 1440, "height": 900}
MOBILE = {"width": 390, "height": 844}


SUBPATH = "fraud-console"  # GitHub Pages serves the site at https://<user>.github.io/fraud-console/


def serve(root: Path) -> tuple[socketserver.TCPServer, str]:
    """Serve web/site/ under /fraud-console/ (like GitHub Pages); anything else 404s."""
    (root / SUBPATH).symlink_to(SITE, target_is_directory=True)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(root))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/{SUBPATH}/"


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
    page.on("response", lambda r: r.status >= 400 and errors.append(f"HTTP {r.status} {r.url}"))
    page.on("requestfailed", lambda r: errors.append(f"request failed {r.url}"))
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


def settle(page: Page) -> None:
    """Wait two animation frames so per-frame visuals have redrawn."""
    page.evaluate("new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


def visuals_match_counters(page: Page, check: Checks, label: str) -> None:
    """Every visual must show what the counters show (same state, no separate numbers)."""
    settle(page)
    v = page.evaluate(
        """() => {
          const ds = (id) => document.getElementById(id).dataset;
          const flow = Object.fromEntries([...document.querySelectorAll('#flow .fl-bucket')]
            .map(b => [b.dataset.outcome, Number(b.dataset.count)]));
          const ctr = Object.fromEntries(['tp', 'fp', 'fn', 'tn']
            .map(k => [k, Number(ds('c-' + k).count)]));
          const num = (x) => (x === '' || x === 'NaN' || x === undefined ? null : Number(x));
          return {
            flow, ctr,
            stepup: Number(document.querySelector('#flow .fl-step').dataset.count),
            approve: Number(document.querySelector('#flow .fl-approve').dataset.count),
            flowN: Number(ds('flow').n), dots: Number(ds('strip').dots),
            played: Number(document.getElementById('c-played').textContent
              .split('/')[0].replace(/,/g, '')),
            convR: num(ds('conv').recallLast), convP: num(ds('conv').precisionLast),
            ctrR: num(ds('c-recall').value), ctrP: num(ds('c-precision').value),
            convFullR: num(ds('conv').recallFull), fullR: num(ds('f-recall').value),
            prR: Number(ds('pr').recall), prP: Number(ds('pr').precision),
            tileR: document.getElementById('o-recall').textContent,
            tileP: document.getElementById('o-precision').textContent,
          };
        }"""
    )
    c = v["ctr"]
    check(v["flow"] == c, f"{label}: flow outcome lanes = counters {c}")
    check(
        v["stepup"] == c["tp"] + c["fp"] and v["approve"] == c["fn"] + c["tn"],
        f"{label}: STEP-UP / APPROVE lanes = counters ({v['stepup']} / {v['approve']})",
    )
    check(
        v["flowN"] == v["dots"] == v["played"],
        f"{label}: flow, strip and counter rows agree ({v['played']})",
    )
    check(
        v["convR"] == v["ctrR"] and v["convP"] == v["ctrP"],
        f"{label}: convergence chart ends at the counters ({v['convR']}, {v['convP']})",
    )
    check(v["convFullR"] == v["fullR"], f"{label}: convergence reference line = full-sample recall")
    pr_ok = f"{100 * v['prR']:.1f}%" == v["tileR"] and f"{100 * v['prP']:.1f}%" == v["tileP"]
    check(pr_ok, f"{label}: PR-curve dot = simulator tiles ({v['tileR']}, {v['tileP']})")


def measure_fps(page: Page, ms: int = 2000) -> tuple[float, float]:
    return tuple(
        page.evaluate(
            """(ms) => new Promise(done => {
                 let n = 0, worst = 0, last = performance.now(); const t0 = last;
                 const f = (t) => { n++; worst = Math.max(worst, t - last); last = t;
                   if (t - t0 < ms) requestAnimationFrame(f);
                   else done([n * 1000 / (t - t0), worst]); };
                 requestAnimationFrame(f);
               })""",
            ms,
        )
    )


def record_video(browser, url: str, seconds: float = 8.0) -> None:
    """The hero at 4x as a short clip: reports/web_preview/stream.webm (+ stream.gif).

    Frames are full-viewport screenshots taken back to back (Chrome's own headless recorder
    emits some undersized frames), assembled by ffmpeg at the measured capture rate.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found: skipping the stream clip")
        return
    page = browser.new_page(viewport={"width": 1280, "height": 720})
    page.goto(url, wait_until="networkidle")
    page.click('.seg [data-speed="4"]')
    page.evaluate("document.getElementById('flow').scrollIntoView({block: 'start'})")
    page.evaluate("window.scrollBy(0, -70)")
    page.wait_for_timeout(800)
    frames = Path(tempfile.mkdtemp())
    t0 = time.perf_counter()
    k = 0
    while time.perf_counter() - t0 < seconds:
        page.screenshot(path=frames / f"f{k:04d}.png")
        k += 1
    fps = k / (time.perf_counter() - t0)
    page.close()
    pattern = str(frames / "f%04d.png")
    rate = f"{fps:.3f}"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-framerate", rate, "-i", pattern,
         "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "38", "-pix_fmt", "yuv420p",
         "-row-mt", "1", str(OUT / "stream.webm")],
        check=True,
    )  # fmt: skip
    palette = "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-framerate", rate, "-i", pattern,
         "-vf", f"scale=800:-1:flags=lanczos,{palette}", str(OUT / "stream.gif")],
        check=True,
    )  # fmt: skip
    shutil.rmtree(frames, ignore_errors=True)
    print(f"wrote reports/web_preview/stream.webm + stream.gif ({k} frames, {fps:.1f} fps capture)")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = build_context()
    check = Checks()
    tmp = tempfile.TemporaryDirectory()
    httpd, url = serve(Path(tmp.name))
    print("serving under", url)
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
            note = page.locator(".scope-note")
            check(
                "not the model evaluation" in " ".join(note.text_content().split())
                and note.locator('a[href="#metrics"]').count() == 1,
                "full-sample note says it is a demo sample and links to Model, measured",
            )
            text = page.text_content("main")
            check("e-" not in page.text_content("#s-rows"), "stream scores use no e-notation")
            check("<0.0001" in text, "scores below 0.0001 show as <0.0001")
            check(page.is_visible(".srow--head .c-scale"), "score column header says 'log scale'")
            visuals_match_counters(page, check, "desktop")
            check(
                page.get_attribute("#flow", "data-static") == "true"
                and page.get_attribute("#flow-canvas", "data-animating") != "true"
                and page.is_visible("#flow-static"),
                "reduced motion: static Sankey, no moving dots, note shown",
            )
            page.evaluate("window.scrollTo(0, 0)")
            settle(page)
            page.screenshot(path=OUT / "desktop.png", full_page=True)
            page.screenshot(path=OUT / "desktop_hero.png")

            # drag the threshold line on the score strip
            page.locator("#strip .strip-plot").scroll_into_view_if_needed()
            box = page.locator("#strip .strip-plot").bounding_box()
            t_before = page.text_content("#thr-out")
            page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.mouse.move(box["x"] + box["width"] * 0.82, box["y"] + box["height"] / 2, steps=6)
            page.mouse.up()
            t_after = page.text_content("#thr-out")
            check(
                float(t_after) > float(t_before)
                and page.text_content("#thr-mini-out") == t_after
                and page.input_value("#thr") == page.input_value("#thr-mini"),
                f"dragging the strip line moves the shared threshold ({t_before} -> {t_after})",
            )
            visuals_match_counters(page, check, "after drag")
            page.click("#btn-project")
            visuals_match_counters(page, check, "back at project threshold")

            # outcome bucket opens the latest transaction of that outcome
            target = "fn" if page.get_attribute("#flow .fl-fn", "data-count") != "0" else "tp"
            page.click(f"#flow .fl-{target}")
            insp = " ".join(page.text_content("#insp-body").split())
            want = ("truth: fraud", "APPROVE") if target == "fn" else ("truth: fraud", "STEP-UP")
            check(
                page.text_content("#insp-mode") == "pinned" and all(w in insp for w in want),
                f"clicking the '{target}' outcome opens such a transaction in the inspector",
            )

            # filtered view: label on screen, counts unchanged
            before_counts = page.get_attribute("#flow .fl-tn", "data-count")
            page.click('[data-view="sample"]')
            check(page.is_visible("#flow-filter"), "filtered dot view shows its label")
            check(
                page.get_attribute("#flow .fl-tn", "data-count") == before_counts,
                "filtered dot view: lanes and counters still count every row",
            )
            page.click('[data-view="all"]')

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
            check(
                page.get_attribute("#flow .flow-box", "class").find("vertical") >= 0,
                "mobile: decision flow is vertical (top to bottom)",
            )
            visuals_match_counters(page, check, "mobile")
            page.evaluate("window.scrollTo(0, 0)")
            settle(page)
            page.screenshot(path=OUT / "mobile.png", full_page=True)
            page.screenshot(path=OUT / "mobile_hero.png")
            check(not errors, f"mobile: no console errors {errors}")
            page.close()

            # 3b. mobile with motion at 16x: smoothness
            page, errors = open_page(
                browser,
                url,
                check,
                "mobile 16x",
                viewport=MOBILE,
                device_scale_factor=2,
                is_mobile=True,
                has_touch=True,
            )
            page.click('.seg [data-speed="16"]')
            page.wait_for_timeout(600)
            animating = page.get_attribute("#flow-canvas", "data-animating") == "true"
            fps, worst = measure_fps(page)
            check(animating, "mobile 16x: dots are moving")
            check(fps >= 50, f"mobile 16x: ~{fps:.0f} fps (worst frame {worst:.0f} ms)")
            page.click("#s-play")  # pause, then compare visuals with counters
            visuals_match_counters(page, check, "mobile 16x paused")
            check(not errors, f"mobile 16x: no console errors {errors}")
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
            page.evaluate("window.scrollTo(0, 0)")
            settle(page)
            page.screenshot(path=OUT / "light.png", full_page=True)
            page.screenshot(path=OUT / "light_hero.png")
            check(not errors, f"light: no console errors {errors}")
            page.close()
            record_video(browser, url)
            browser.close()
    finally:
        httpd.shutdown()
        tmp.cleanup()
    for p in sorted(OUT.glob("*.png")):
        print("wrote", p.relative_to(REPO))
    if check.failures:
        print(f"{len(check.failures)} check(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
