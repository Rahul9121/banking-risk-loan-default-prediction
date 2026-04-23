# Banking Risk & Loan Default Prediction System
Credit-risk scoring project that predicts loan default probability, segments portfolio risk bands, and visualizes delinquency trends in a Streamlit dashboard.

## Tech stack
- Python (Logistic Regression + optional XGBoost)
- SQL analytics query templates
- Streamlit dashboard
- FastAPI scoring API (Render-ready)
- Docker

## Project structure
- `streamlit_app.py` — interactive risk dashboard
- `api/main.py` — REST API for model scoring
- `scripts/train_model.py` — train and save model artifacts
- `src/` — data prep, modeling, scoring, and training modules
- `sql/portfolio_queries.sql` — SQL examples for risk segmentation and delinquency trend reporting

## 1) Run locally
### Install dependencies
```bash path=null start=null
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Train model (creates demo dataset automatically if none exists)
```bash path=null start=null
python scripts/train_model.py
```

### Launch dashboard
```bash path=null start=null
streamlit run streamlit_app.py
```

### Run API locally
```bash path=null start=null
uvicorn api.main:app --reload --port 8000
```

## 2) Bring your own dataset
You can train with Kaggle datasets:
- Home Credit Default Risk
- LendingClub Loan Data

Use:
```bash path=null start=null
python scripts/train_model.py --dataset "data/your_dataset.csv"
```

The training logic auto-detects a target using one of:
- `target`, `default`, `loan_default`, `is_default`
- `loan_status` text labels (charged-off/default/late statuses)
- `days_past_due > 30` fallback

## 3) Publish to GitHub
If you already have `gh` CLI authenticated:
```bash path=null start=null
git init
git add .
git commit -m "Initial credit risk dashboard and API"
gh repo create banking-risk-loan-default --public --source . --remote origin --push
```

If you prefer manual GitHub setup:
1. Create a new empty GitHub repository.
2. Run:
```bash path=null start=null
git init
git add .
git commit -m "Initial credit risk dashboard and API"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 4) Deploy Streamlit and share link
1. Open Streamlit Community Cloud: `https://share.streamlit.io`
2. Click **New app**
3. Select your GitHub repo and branch (`main`)
4. Set app file: `streamlit_app.py`
5. Deploy

You will receive a public URL like:
`https://<your-app-name>.streamlit.app`

## 5) Optional: Deploy model API to Render
1. Push this repo to GitHub.
2. In Render, create a new **Web Service** from the repo.
3. Render auto-detects `render.yaml`.
4. Deploy and copy your API URL.

### API example
POST `/score` with:
```json path=null start=null
{
  "records": [
    {
      "loan_amnt": 12000,
      "int_rate": 13.2,
      "annual_inc": 68000,
      "dti": 21.5,
      "fico_score": 640,
      "days_past_due": 18,
      "home_ownership": "RENT",
      "purpose": "debt_consolidation",
      "term_months": 60
    }
  ]
}
```
