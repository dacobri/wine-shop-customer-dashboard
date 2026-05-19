# 🍷 Wine Shop & Delicatessen — Customer Intelligence Dashboard

An interactive Streamlit dashboard turning a 404-customer survey into a
decision tool for the store manager.

**Live dashboard:** **https://wine-shop-customer-dashboard-group3.streamlit.app**

**Team:** Sabeena Awan · Brice Da Costa · Lucas Joris Haesaert · Patricia Unger
**Course:** ESADE MSc Business Analytics · Advanced Programming with Python · May 2026

---

## How to read this repo

Two audiences. One repo. No duplicates.

| If you're the… | Start with… |
|---|---|
| **Wine shop manager** (client) | [📄 `deliverables/Wine_Shop_Executive_Report.pdf`](deliverables/Wine_Shop_Executive_Report.pdf) (5-page exec summary), then the [live dashboard](https://wine-shop-customer-dashboard-group3.streamlit.app). A polished `Manager_Guide.pdf` walkthrough ships with the submission zip (`AP_A1_Group3/For_Client/`) rather than living in this repo. |
| **Professor / grader** | This README first (you're already here ✅), then the source code in `app/`, then [`notebooks/EDA_Wine_Consumption.ipynb`](notebooks/EDA_Wine_Consumption.ipynb), then [`docs/`](docs/) for the case brief and the team's preliminary strategy report. |

---

## Repository structure

```
wine_app/
├── README.md                          ← you are here
├── LICENSE                            ← MIT
├── requirements.txt                   ← pinned dependencies
├── runtime.txt                        ← Python 3.11 hint for Streamlit Cloud
│
├── app/                               ← Streamlit application — Python source
│   ├── wine_dashboard.py              ← entrypoint (set this path on Streamlit Cloud)
│   ├── wine_data.py                   ← loading, cleaning, FM segmentation
│   ├── wine_clustering.py             ← behavioral K-Means + K-Prototypes spend tiers
│   ├── wine_simulator.py              ← What-If revenue projection
│   ├── wine_config.py                 ← marketing playbooks as data
│   └── wine_theme.py                  ← palette, CSS, Plotly layout, render helpers
│
├── data/                              ← input data
│   └── wine_consumption_survey.xlsx   ← 404 customers, 9 columns
│
├── deliverables/                      ← polished final outputs
│   └── Wine_Shop_Executive_Report.pdf ← 5-page client-facing exec summary
│
├── notebooks/
│   └── EDA_Wine_Consumption.ipynb     ← exploratory data analysis
│
└── docs/                              ← reference docs (context, not code)
    ├── case_brief.pdf                 ← original ESADE case description
    └── strategy_report.docx           ← preliminary marketing report
```

Each module has a single responsibility. Marketing playbooks live as data in
`app/wine_config.py` so a non-coder can edit them. Both clustering lenses
coexist in `app/wine_clustering.py` (the manager toggles between them).

---

## Run locally

```bash
git clone https://github.com/dacobri/wine-shop-customer-dashboard.git
cd wine-shop-customer-dashboard
python -m pip install -r requirements.txt
streamlit run app/wine_dashboard.py
```

The browser opens at `http://localhost:8501`. The dashboard reads the
dataset from `data/wine_consumption_survey.xlsx` automatically.

### macOS, no virtual env

```bash
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run app/wine_dashboard.py
```

---

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repository: `dacobri/wine-shop-customer-dashboard`.
4. Branch: `main`.
5. **Main file path:** `app/wine_dashboard.py` ← important: must include the `app/` prefix.
6. Python version: 3.11 (auto-detected from `runtime.txt`; set explicitly in Advanced Settings if the cloud ignores it).
7. Click **Deploy**.

The free tier sleeps the app after 7 days of inactivity — clicking the URL wakes it (~30s).

---

## Tabs

| Tab | What it answers |
|---|---|
| 👥 **Customer Profile** | Who walks through the door? (demographics, gender × occasion signal, three spending heatmaps, hidden attribute relationships) |
| 🎯 **Market Segments (FM)** | Which segment is worth what? |
| 🤖 **ML Clustering** | Two togglable lenses: behavioral archetypes (default) and spend tiers |
| 🧀 **Product Mix** | Where's the cross-sell opportunity? |
| 📋 **Action Plan** | Prioritised marketing playbook by segment |
| 🔎 **Customer Explorer** | Filterable table for individual outreach lists (CSV export) |
| 📊 **Strategic Overview** | The full picture + interactive What-If revenue simulator |

---

## Methodology in brief

### Three complementary segmentation lenses

1. **FM (Frequency × Monetary)** — supervised median split on visits/mo and
   Ticket. Four named personas: *Champions*, *Loyal Regulars*, *Occasion
   Splurgers*, *Casual Visitors*. The revenue lens.
2. **Behavioral K-Means** — unsupervised K-Means K=4 on (visits/month,
   sociality score). Ticket is *deliberately excluded* so the segments are
   independent of how much customers pay. Archetypes: *Social Regulars*,
   *Daily Home Drinkers*, *Occasion Celebrants*, *Casual Home Drinkers*.
3. **K-Prototypes spend tiers** — unsupervised mixed-type clustering with
   Hamming distance on categoricals + Euclidean on Ticket. Default K=3 →
   *Entry / Core / Premium Buyers*. A value-anchored sanity check on FM.

### Statistical layer

The Profile tab's "Hidden patterns" chart uses Cramér's V (chi-squared based,
0–1 bounded). The user-facing scale is multiplied by 100 for readability;
a `📐 What's the maths underneath?` expander documents the math and the
Rea & Parker (1992) interpretation thresholds for academic completeness.

### Revenue formula

```
monthly_spend       = Ticket × monthly_visits
est_annual_revenue  = monthly_spend × 12
```

Visits/month converts verbal frequency categories ("1 to 2 times per week"
→ 6 visits/mo) and is documented in-app.

---

## Data caveats

The dataset is the ESADE *Analytics and Big Data* case-study sample. It is
synthetic / case-study data, not real POS records. A handful of rows
contain unrealistic combinations (customers visiting daily at €100 tickets
≈ €36k/year on wine). We surface these as-is rather than clipping — the
segmentation methodology would be re-fit on real POS data when available.

The "Salmond" entry in *Additional products* is preserved verbatim. It is
most likely a typo for *Almond* (deli category fit), possibly *Salmon*. A
footnote disambiguates wherever the term is rendered in the dashboard;
an asterisk marker links every mention to the disambiguation note.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Case study from *Analytics and Big Data* (ESADE).
Strategic recommendations adapted from the team's marketing report
([`docs/strategy_report.docx`](docs/strategy_report.docx)).
Visualization choices follow Prof. Manel Guerris's data-visualization deck
(KISS · sort-by-value · diverging bars · heatmaps over tables · Beginning
& End narrative arc).
