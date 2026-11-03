# NBA Totals Lab — Streamlit

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push `app.py` and `requirements.txt` to a GitHub repository. In Streamlit Community Cloud, select that repository, branch, and `app.py` as the entrypoint. Upload your CSVs inside the app.

The required training columns are `Game_Id`, `GAME DATE` (NBSP or ordinary space accepted), `Point Total`, and all 44 columns in `FEATURES`. Historical lines need `Game_Id` and `OU_Line`. Tonight's file needs `HomeTeam`, `AwayTeam`, and those same features.
