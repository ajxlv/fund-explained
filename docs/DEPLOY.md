# Deploy to Streamlit Community Cloud (free)

Nothing has been deployed yet. These are the steps for when you're ready. Do step 0 first if you want real funds online instead of the demo data.

## 0. (Recommended) Load real data first

Follow `docs/MORNING_STEPS.md` so that `data/schemes.csv` and `data/navs.parquet` are committed. Without them, the deployed app runs on the synthetic demo data and shows the amber "Demo mode" banner on every page.

## 1. Merge to `main`

Merge the `build/v1` pull request on GitHub (or pick `build/v1` as the branch in step 3).

## 2. Sign in to Streamlit Community Cloud

1. Go to <https://share.streamlit.io>.
2. Click **Continue with GitHub** and authorise Streamlit.
3. If asked, allow it access to the `ajxlv/fund-explained` repository.

## 3. Create the app

1. Click **Create app** (top right).
2. Choose **Deploy a public app from GitHub** ("Yup, I have an app").
3. Fill in:
   - **Repository:** `ajxlv/fund-explained`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** pick a subdomain, e.g. `fund-explained`
4. Open **Advanced settings** and set **Python version** to **3.12**. No secrets are needed.
5. Click **Deploy**.

The first build installs `requirements.txt` (pandas, numpy, scikit-learn, streamlit, plotly, pyarrow) and takes a few minutes. Streamlit picks up the theme from `.streamlit/config.toml` automatically.

## 4. Check it

- The landing page shows "Understand any mutual fund in 30 seconds".
- Pick a fund. The first load computes features and models (a few seconds), then they're cached.
- The amber banner should be **absent** if real data was committed.
- The footer disclaimer is visible at the bottom of each page.

## 5. Updating

Every push to `main` redeploys automatically. To refresh the data, rerun step 2 of `docs/MORNING_STEPS.md` on your laptop, regenerate the reports, commit and push.

## Notes

- The app never calls mfapi.in or any other network service. It only reads the committed files.
- Usage statistics are switched off in `.streamlit/config.toml` (`gatherUsageStats = false`).
- If the app sleeps after a period of inactivity (free tier), the first visitor clicks "Wake up" and waits about 30 seconds.
