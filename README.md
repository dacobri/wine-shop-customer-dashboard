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
| **Professor / grader** | This README first (you're already here ✅), then the source code in `app/`, then [`notebooks/EDA_Wine_Consumption.ipynb`](notebooks/EDA_Wine_Consumption.ipynb), then [`docs/case_brief.pdf`](docs/case_brief.pdf) for the original ESADE case description. |

---

## Repository structure

```
wine_app/
├── README.md                          ← you are here
├── LICENSE                            ← MIT
├── requirements.txt                   ← pinned dependencies
├── runtime.txt                        ← Python 3.11 hint for Streamlit Cloud
├── wine_dashboard.py                  ← root-level entry shim (runs app/wine_dashboard.py)
│
├── app/                               ← Streamlit application — Python source
│   ├── wine_dashboard.py              ← the real dashboard (UI, sidebar, tabs, renderers)
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
    └── case_brief.pdf                 ← original ESADE case description
```

The root-level `wine_dashboard.py` is a thin **shim** — it adds `app/` to
`sys.path` and delegates to `app/wine_dashboard.py`. It exists so Streamlit
Community Cloud's default entry-file path works out of the box.

Each module has a single responsibility. Marketing playbooks live as data in
`app/wine_config.py` so a non-coder can edit them. Both clustering lenses
coexist in `app/wine_clustering.py` (the manager toggles between them).

---

## Run locally

```bash
git clone https://github.com/dacobri/wine-shop-customer-dashboard.git
cd wine-shop-customer-dashboard
python -m pip install -r requirements.txt
streamlit run wine_dashboard.py
```

The browser opens at `http://localhost:8501`. The dashboard reads the
dataset from `data/wine_consumption_survey.xlsx` automatically. Tested on
Python 3.11.

---

## Tabs

| Tab | What it answers |
|---|---|
| 👥 **Customer Profile** | Who walks through the door? Demographics, gender × occasion signal, three spending heatmaps, hidden attribute relationships, **and an embedded Customer Explorer expander at the bottom** for filterable individual outreach lists (CSV export). |
| 🎯 **FM Segments** | Which segment is worth what? Champions / Loyal Regulars / Occasion Splurgers / Casual Visitors. |
| 🔬 **Behavioral Segments** | Two togglable ML lenses: behavioral archetypes (K-Means, default) and spend tiers (K-Prototypes). |
| 🧀 **Product Mix** | Where's the cross-sell opportunity? The delicatessen gap. |
| 📊 **Strategic Overview** | Synthesis — headline KPIs + the interactive What-If revenue simulator. |
| 📋 **Action Plan** | The close — prioritised marketing playbook by segment. |

---

## How this project maps to the course

| Course pillar | Where it shows up in this repo |
|---|---|
| **Object-oriented programming · modularity** | Six single-responsibility modules in `app/`. Each owns one concern: data loading, clustering, simulation, theming, configuration, presentation. Marketing playbooks (`wine_config.py`) are pure data so a non-coder can edit them. |
| **Data types & pandas** | Loaded from Excel via `pandas.read_excel`. Categorical orderings preserved (frequency, age band, education tier). Derived columns: `monthly_visits`, `monthly_spend`, `est_annual_revenue`, `deli_tier`, `social_score`, `Education_group`, `F_tier`, `M_tier`, `FM_segment`. |
| **Stability & licensing** | MIT-licensed. Deterministic random seeds in clustering (`seed=42`) and in the What-If simulator (`random_state=1, 2`). Version-pinned dependencies in `requirements.txt`. The `kmodes` import soft-fails with a graceful K-Means fallback. |
| **Performance** | `@st.cache_data` on every expensive computation: Excel load, FM segmentation, K-Means fit, K-Prototypes fit, silhouette diagnostics. Each model is fit at most once per (filter × K) combination per session. |
| **Data visualisation** | Follows Prof. Guerris's Session 3 deck: KISS (each tab caps at ~5 high-density charts), sort-by-value never alphabetical, diverging stacked bars for binary survey signals, heatmaps over tables for pattern-finding, beginning-and-end narrative arc, wine-shop palette consistent across the dashboard. |
| **Machine learning** | Two unsupervised clustering lenses on the same base — behavioural K-Means (K=4, freq × sociality) and K-Prototypes (mixed-type, anchored on Ticket) — both validated by silhouette and elbow diagnostics, both cross-tabbed against the supervised FM segmentation. |
| **Statistics** | The "Hidden patterns" chart on the Customer Profile tab uses Cramér's V (chi-squared based, 0–1 bounded). The user-facing scale is multiplied by 100 for readability; an in-app *"What's the maths underneath?"* expander documents the conventional Rea & Parker (1992) interpretation thresholds for academic completeness. |

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
Strategic recommendations developed by the team during the case-study work.
Visualization choices follow Prof. Manel Guerris's data-visualization deck
(KISS · sort-by-value · diverging bars · heatmaps over tables · Beginning
& End narrative arc).
