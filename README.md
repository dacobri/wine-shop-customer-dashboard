# 🍷 Wine Shop & Delicatessen — Customer Intelligence Dashboard

An interactive Streamlit dashboard turning a 404-customer survey into a
decision tool for the store manager. Live deployment:
**`https://wine-shop-customer-dashboard.streamlit.app`** *(URL filled in once deployed)*.

> Built for the *Advanced Programming with Python* course,
> ESADE MSc Business Analytics — Session 3 (Data Visualization).

---

## What it does

The wine shop has an established customer base but no demographic or
preference data in its billing system. A survey of 404 customers was
conducted; this dashboard turns that survey into a working business tool:

- **Segments customers** using two complementary lenses:
  the **FM (Frequency × Monetary)** framework as the *actionable* view,
  and **K-Prototypes** unsupervised clustering as the *analytical* validation.
- **Quantifies opportunities** with a What-If revenue simulator
  (sliders for conversion campaigns, upsell programmes, loyalty).
- **Surfaces the cross-sell gap** — the store is perceived as a wine shop
  rather than a delicatessen, and the dashboard pinpoints where to act.
- **Delivers a per-segment marketing playbook** the manager can execute.

---

## Tabs

| Tab | What it answers |
|---|---|
| 📊 **Overview** | How is the business doing right now? What would three marketing moves be worth? |
| 👥 **Customer Profile** | Who walks through the door? |
| 🎯 **Market Segments** | Which segment should I focus on? What's each one worth? |
| 🤖 **ML Clustering** | Does an unsupervised model independently validate the FM segments? |
| 🧀 **Product Mix** | Which deli products win, and where's the cross-sell gap? |
| 📋 **Action Plan** | What concrete actions should I take, in what order? |
| 🔎 **Customer Explorer** | Show me individual customers I can target. |

---

## Project structure

```
wine_app/
├── wine_dashboard.py         # Streamlit entrypoint — UI, tabs, render functions
├── wine_data.py              # Data loading, cleaning, FM segmentation
├── wine_clustering.py        # K-Prototypes + K-Means fallback + diagnostics
├── wine_simulator.py         # What-If revenue projection
├── wine_config.py            # Marketing playbook per segment (domain knowledge)
├── wine_theme.py             # Palette, CSS, plotly layout, UI helpers
├── 5. Wine consumption.xlsx  # Source survey data (404 customers)
├── EDA_Wine_Consumption.ipynb# Exploratory notebook (raw EDA)
├── requirements.txt          # Pinned Python dependencies
├── runtime.txt               # Python version for Streamlit Cloud
├── .gitignore
├── LICENSE                   # MIT
└── README.md                 # You are here
```

Each module has a single responsibility — change the marketing playbook
in `wine_config.py`, re-skin the dashboard in `wine_theme.py`, swap the
clustering algorithm in `wine_clustering.py`, all without touching the
others.

---

## Run locally

```bash
git clone https://github.com/<your-username>/wine-shop-customer-dashboard.git
cd wine-shop-customer-dashboard
python -m pip install -r requirements.txt
streamlit run wine_dashboard.py
```

The browser opens automatically at `http://localhost:8501`.

### If you don't have Python set up
On macOS:

```bash
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run wine_dashboard.py
```

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch `main`, file path `wine_dashboard.py`.
4. Python version: 3.11 (auto-detected from `runtime.txt`).
5. Click **Deploy**. First build takes ~3–5 minutes.

The app sleeps after 7 days of inactivity on the free tier — clicking the
URL wakes it up (~30 seconds).

---

## Design notes

### Segmentation methodology

- **FM (Frequency × Monetary):** customers split by median visit frequency
  and median ticket size, producing four segments — Champions,
  Loyal Regulars, Occasion Splurgers, Casual Visitors.
- **K-Prototypes:** the correct unsupervised algorithm for this dataset
  because six of the seven features are categorical (occasion, payment,
  deli choice, age band, gender, frequency). It applies Hamming distance
  on the categoricals and Euclidean distance on the (single) numeric
  feature — Ticket. Falls back to K-Means on one-hot encoded features
  if the `kmodes` package isn't installed.

### Revenue estimate

`annual_revenue = ticket × monthly_visits × 12`, where monthly_visits
is mapped from the verbal frequency category (e.g. "1 to 2 times per week"
→ 6 visits/mo). The mapping is editable in `wine_data.py`.

### Caching

All expensive operations (Excel load, FM computation, K-Prototypes fit,
silhouette diagnostics) are wrapped in `@st.cache_data`. The model is
fit at most once per (filter selection × K) combination within a session.
On Streamlit Cloud, the cache lives in-memory per server process and is
cleared when the app sleeps.

---

## Roadmap

- A client-facing PDF executive summary export, generated from the dashboard
- Time-series analysis once transaction-level data is available
- Connection to a real POS feed for live KPIs

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Case study from *Analytics and Big Data* (ESADE).
Strategic recommendations adapted from the team customer report.
