# Kaggle notebooks (uploaded by Nat)

All files in `kaggle/` are generated (`make kaggle`) and tested in a simulated Kaggle layout
(`make kaggle-test`: `/kaggle/input/...` and `/kaggle/working` stand-ins, no repository on the path).

| File | What it is | Needs |
|---|---|---|
| `fraud_story.ipynb` | **Main public notebook**: the whole project in a ~10-minute read; re-runs a light version live (5 folds), quotes the pre-registered results | the dataset only |
| `01`–`06`, `08`, `09` | Self-contained editions of the repo notebooks (a first cell writes the project package and any upstream results into `/kaggle/working`) | the dataset only |

## Published

- **Story notebook:** <https://www.kaggle.com/code/natpaphonnukhunthod/fraud-why-logistic-beats-lightgbm-smote-hurts>
  (public; title shortened to fit Kaggle's limit). `kaggle/kernel-metadata.json` holds its settings: CPU only,
  internet off, dataset `uditjain13/credit-card-fraud-detection-2026`.
- New version after `make kaggle && make kaggle-test`: `kaggle kernels push -p kaggle` (uses your own Kaggle
  login; the CLI reads the credentials, nothing is stored in the repo).

## Upload one notebook

1. <https://www.kaggle.com/code> → **New Notebook** → **File → Import Notebook** → choose the `.ipynb`.
2. Right panel → **Add Input** → search *Credit Card Fraud Detection 2026* (Udit Jain) → **Add**. It mounts at
   `/kaggle/input/credit-card-fraud-detection-2026/` (the notebooks also search `/kaggle/input` recursively).
3. **Settings:** Accelerator **None** (CPU is enough); Internet can stay off; Language Python.
4. **Save Version → Save & Run All (Commit)**. The story takes about a minute on CPU (≈10 s on a laptop).
5. Set the title (the first heading is a good one), add tags (e.g. *fraud detection*, *imbalanced data*,
   *logistic regression*, *tutorial*), make it **Public**.
6. Optional: put the Kaggle URL in `configs/links.yaml` (`kaggle:`), run `make docs`, commit (and push when
   you decide), so the README and the website link to it.

Library versions on Kaggle can differ from `uv.lock`, so the last digits of live results may differ; quoted
project results come from the repo and do not change.
