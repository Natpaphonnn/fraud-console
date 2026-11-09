# Deploying the project page to Vercel

The page is a static site: `web/site/` (one `index.html`, CSS, two small JS modules, an icon and
`vercel.json`). There is no backend, no build step on Vercel, and no third-party scripts or fonts.
`web/site/` is **generated**. Edit `web/src/` (or the templates/reports it reads) and run `make docs`.

> Deployment is done by Nat. Nothing in this repository deploys or pushes automatically.

## 1. Before every deploy (local)

```bash
make check         # lint + tests; fails if web/site/ is stale or the simulator disagrees with reports/
make web-serve     # optional: open http://localhost:8000 and click around (Ctrl+C to stop)
make web-preview   # optional: re-take reports/web_preview/*.png and re-run the browser smoke test
git status         # commit web/site/ together with any change to web/src/, reports/ or configs/links.yaml
```

## 2. Option A — Vercel dashboard + GitHub (recommended)

1. The repository is on GitHub: <https://github.com/Natpaphonnn/fraud-console>.
2. On <https://vercel.com/new>, choose **Import Git Repository** and pick `Natpaphonnn/fraud-console`
   (first time: allow the Vercel GitHub app to see this repository).
3. Configure the project:
   - **Root Directory:** `web/site` (click *Edit* next to Root Directory)
   - **Framework Preset:** `Other`
   - **Build Command / Output Directory / Install Command:** leave empty (no overrides)
4. Click **Deploy**. Vercel serves the files as they are; `vercel.json` adds security headers
   (Content-Security-Policy, nosniff, referrer and permissions policies) and clean URLs.

After this, every push to the default branch redeploys automatically, and other branches get
preview URLs. The site only changes when `web/site/` changes, so always commit it after `make docs`.

## 3. Option B — Vercel CLI (no GitHub needed)

```bash
cd web/site
npx vercel@latest login      # opens the browser; credentials stay in ~/.vercel, never in the repo
npx vercel@latest            # first run: answer the prompts (scope, new project name); gives a preview URL
npx vercel@latest --prod     # promote to the production URL
```

The CLI writes `web/site/.vercel/` (project IDs). It is git-ignored and ignored by the
freshness check. Do not commit it.

## 4. After the first deploy

1. Put the production URL in `configs/links.yaml` → `website: "https://<project>.vercel.app"`,
   plus the GitHub / Kaggle / Medium URLs when they exist (empty values are simply not shown).
2. Run `make docs`. This updates the README links and adds the canonical URL to the page.
3. Run `make check`, commit, and redeploy (push for Option A, `npx vercel@latest --prod` for Option B).

## 5. Quick checks on the live site

- The page loads on a phone without horizontal scrolling. The simulator starts at the project
  threshold and changes when you move the slider or edit a cost.
- The cost label *"Illustrative assumed costs (not bank data)"* and the dataset credit
  (*Credit Card Fraud Detection 2026*, Udit Jain, CC0) are visible.
- The browser console has no errors (a CSP violation would show up there).
- `curl -sI https://<project>.vercel.app | grep -i content-security-policy` prints the policy.

## Rolling back

In the Vercel dashboard, open **Deployments**, pick an earlier deployment, then choose **⋯ → Promote to
Production**.
