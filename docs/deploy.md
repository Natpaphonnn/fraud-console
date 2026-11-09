# Publishing the project page

The page is a static site: `web/site/` (one `index.html`, CSS, small JS modules, an icon). There is no
backend and no build step on the host, and no third-party scripts or fonts. `web/site/` is
**generated** and committed. Edit `web/src/` (or the templates/reports it reads) and run `make docs`.

- Live URL: <https://natpaphonnn.github.io/fraud-console/> (GitHub Pages)
- Repository: <https://github.com/Natpaphonnn/fraud-console>

## 1. Before every publish (local)

```bash
make check         # lint + tests; fails if web/site/ is stale, paths are not relative, or JS ≠ Python
make web-preview   # browser smoke test served under /fraud-console/ (no 404s, no console errors) + screenshots
git status         # commit web/site/ together with any change to web/src/, reports/ or configs/links.yaml
```

`make web-serve` previews at <http://localhost:8000> (site root); `make web-preview` is the one that
reproduces the GitHub Pages sub-path.

## 2. GitHub Pages (current)

`.github/workflows/pages.yml` uploads `web/site/` as-is with `actions/upload-pages-artifact` and publishes
it with `actions/deploy-pages`. It runs on every push to `main` and can be started by hand.

**One-time setup (repository owner):**

1. Open <https://github.com/Natpaphonnn/fraud-console/settings/pages>.
2. Under **Build and deployment → Source**, choose **GitHub Actions** (not "Deploy from a branch").
3. Open the **Actions** tab → **Deploy project page to GitHub Pages** → **Run workflow** (branch `main`).
   A run triggered before step 2 fails with a Pages "not found / not enabled" error; just run it again.
4. When the run is green, the URL is shown on the run page and in **Settings → Pages**.

**Afterwards:** every push to `main` redeploys. To roll back, revert the commit and push, or open an
earlier successful run of the workflow and choose **Re-run all jobs**. That deploys that commit's
`web/site/`.

**Differences from a header-capable host:** GitHub Pages cannot set response headers, so the
Content-Security-Policy is shipped as a `<meta>` tag generated from `web/src/vercel.json`.
`frame-ancestors` cannot be set that way, so the page can be framed. `vercel.json` is published as an
inert file and ignored by Pages.

**Quick checks on the live site:** loads on a phone without horizontal scrolling. The stream plays,
or waits for Play with reduced motion on. The cost label and the dataset credit (Udit Jain, CC0) are
visible, and the browser console has no errors.

## 3. Alternative: Vercel

Only if a header-capable host is wanted later (real CSP header incl. `frame-ancestors`):

1. <https://vercel.com/new> → **Import Git Repository** → `Natpaphonnn/fraud-console`.
2. **Root Directory:** `web/site` · **Framework Preset:** Other · leave Build/Output/Install empty.
3. **Deploy**. `web/site/vercel.json` adds the security headers and clean URLs.
4. Put the new URL in `configs/links.yaml` (`website`), run `make docs`, `make check`, commit and push.

The Vercel CLI (`cd web/site && npx vercel@latest`, then `--prod`) also works. It writes
`web/site/.vercel/`, which is git-ignored and ignored by the freshness check.
