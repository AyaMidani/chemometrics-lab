# Chemometrics Lab — Web Application

A browser-based chemometrics model builder. Upload spectral data (CSV / Excel),
choose the wavelength range and method, and build calibration models with
**MLR, PCR, PLS, ANN, GA-MLR**, plus **wavelength range selection (iPLS)** — all
in the browser. Compare R² / RMSE across methods and view regression and RMSECV plots.

This is the production, deployable version of the original single-file prototype.

## Why there is no backend

Every algorithm and all file parsing run **client-side in the user's browser**.
That means:

- **No server compute cost** no matter how many people use it.
- **Research data never leaves the user's device** — good for unpublished lab data.
- Hosting is just static files: any static host works, and it scales for free.

If accounts, saved analyses, or heavier methods are needed later, a backend can be
added without changing the current UI (see "Possible extensions" below).

## Requirements

- Node.js 18+ and npm

## Run locally

```bash
npm install
npm run dev
```

Open the URL Vite prints (default http://localhost:5173).
Click **Demo data** to try it instantly, or upload `sample-data.csv` (included).

## Build for production

```bash
npm run build      # outputs static files to dist/
npm run preview    # serve the built dist/ locally to verify
```

## Deploy

The output in `dist/` is fully static. Any of these work:

**Vercel** — import the repo, framework preset "Vite", deploy. Zero config.

**Netlify** — build command `npm run build`, publish directory `dist`.

**GitHub Pages** — set `base: "/<repo-name>/"` in `vite.config.js`, then push
`dist/` (e.g. via the `gh-pages` package or an Actions workflow).

**Any static host / nginx** — copy the contents of `dist/` to the web root.
Because it is a single-page app with no client-side routing, no rewrite rules are needed.

## Expected data format

One row per sample. Wavelength columns (the header may be `WL210`, `210`, or
`A_210nm` — any header containing the number works) plus one concentration column.
The app auto-detects the **last numeric column** as the dependent variable (Y);
you can change X/Y selection in the UI.

| WL210 | WL220 | ... | WL350 | Conc |
|-------|-------|-----|-------|------|
| 0.24  | 0.31  | ... | 0.55  | 6.40 |

See `sample-data.csv` for a working example.

## Notes for validation

The in-app R² uses a random 80/20 split per run for quick exploratory comparison.
It is **not** a substitute for full method validation (e.g. ICH Q2). For reporting,
the RMSECV (leave-one-out cross-validation) curves and range-selection results are
the more defensible outputs.

## Possible extensions (not built yet)

- **Login / accounts** and saved analyses (React + Node/Express + MongoDB).
- **Export** results to PDF/Excel for reports and papers.
- **Server-side compute** if larger datasets or heavier ANN/GA runs are needed —
  a small Python (scikit-learn) service is the standard choice in this field and
  can sit behind the existing UI.

## Tech stack

React 18 · Vite 5 · Tailwind CSS 3 · Recharts · PapaParse · SheetJS (xlsx) · lucide-react
