# Morning steps: load the real data (Windows PowerShell)

The overnight build couldn't reach `api.mfapi.in` (the cloud proxy returned 403), so everything currently runs on **synthetic demo data**. Run these on your laptop to replace it with real AMFI NAVs. Paste one block at a time.

## 1. Get the branch and set up Python

```powershell
cd $HOME\Documents          # or wherever you keep code
git clone https://github.com/ajxlv/fund-explained.git
cd fund-explained
git checkout build/v1       # use `main` instead if the PR is already merged

py -3.12 -m venv .venv
# If the next line errors with "running scripts is disabled", run this first:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## 2. Fetch the real NAVs (slow but resumable)

```powershell
python scripts\fetch_navs.py --out data
```

This makes about one request per Direct-Growth candidate scheme, with a polite 0.3 s pause between them, so expect 10 to 30 minutes. If it stops, just run the same command again: finished schemes are cached in `data\raw\`, which git ignores.

When it finishes you should have `data\schemes.csv` and `data\navs.parquet`. The app and tests pick up `data\` over `data\demo\` automatically.

## 3. Regenerate the reports and run the tests

```powershell
python -m fundx.data --summary      # rewrites docs\DATA_SUMMARY.md (should now say "Real data")
python -m fundx.ml --report         # rewrites docs\EVALUATION.md (the "DEMO DATA" header disappears)
ruff check .
pytest                              # the real-data sanity test now runs instead of skipping
```

Open `docs\EVALUATION.md` and copy the real numbers into the "Key findings" section of `README.md` (they're marked as demo numbers right now). Also fill in the resume bullets in `docs\plan.md`.

## 4. Look at the app

```powershell
streamlit run app\streamlit_app.py
```

The amber "Demo mode" banner should be gone. If you'd like a new screenshot for the README, take one and save it over `docs\screenshot.png`.

## 5. Commit and push

```powershell
git add data\schemes.csv data\navs.parquet docs\DATA_SUMMARY.md docs\EVALUATION.md README.md
git commit -m "Load real mfapi.in NAV data and regenerate reports"
git push
```

Check `git status` first: `data\raw\` must **not** be staged (it's in `.gitignore`).
