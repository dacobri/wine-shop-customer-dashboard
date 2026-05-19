# 🍷 Wine Shop & Delicatessen — Customer Intelligence Dashboard

An interactive Streamlit dashboard turning a 404-customer survey into a
decision tool for the store manager. Live deployment:
**`https://wine-shop-customer-dashboard.streamlit.app`**

> Built for the *Advanced Programming with Python* course, ESADE MSc
> Business Analytics — Session 3 (Data Visualization).

**Team:** Sabeena Awan · Brice Da Costa · Lucas Joris Haesaert · Patricia Unger

---

## What it does

The wine shop has an established customer base but no demographic or
preference data in its billing system. A survey of 404 customers was
conducted; this dashboard turns that survey into a working business tool:

- **Segments customers** using three complementary lenses:
  - **FM** (Frequency × Monetary) — the *revenue* lens, four named personas
  - **Behavioral K-Means** — the *lifestyle* lens (freq × sociality), four archetypes
  - **K-Prototypes spend tiers** — the *value* lens, Entry / Core / Premium
- **Quantifies opportunities** with an interactive What-If revenue simulator
  (sliders for conversion campaigns, upsell programmes, loyalty).
- **Surfaces the cross-sell gap** — the store is perceived as a wine shop
  rather than a delicatessen, and the dashboard pinpoints where to act.
- **Delivers a per-segment marketing playbook** the manager can execute,
  with the playbook switchable between the FM and behavioral lenses.

---

## Tabs

| Tab | What it answers |
|---|---|
| 👥 **Customer Profile** | Who walks through the door? Demographics, the gender × occasion signal, monthly spending patterns, hidden attribute relationships. |
| 🎯 **Market Segments (FM)** | Which segment should I focus on? What's each one worth? |
| 🤖 **ML Clustering** | Two complementary lenses: behavioral archetypes (default) and spend tiers (toggle). |
| 🧀 **Product Mix** | Which deli products win, and where's the cross-sell gap? |
| 📋 **Action Plan** | What concrete actions should I take, by FM segment or behavioral archetype? |
| 🔎 **Customer Explorer** | Filterable table for individual outreach lists (CSV export). |
| 📊 **Strategic Overview** | The full picture + What-If revenue simulator with documented assumptions. |

---

## Project structure

```
wine_app/
├── wine_dashboard.py         # Streamlit entrypoint — UI, tabs, render functions
├── wine_data.py              # Data loading, cleaning, FM segmentation, education tier rollup
├── wine_clustering.py        # Behavioral K-Means + K-Prototypes spend tiers (both lenses)
├── wine_simulator.py         # What-If revenue projection
├── wine_config.py            # Marketing playbooks (FM + behavioral) as data
├── wine_theme.py             # Palette, CSS, plotly layout, UI helpers
├── 5. Wine consumption.xlsx  # Source survey data (404 customers)
├── EDA_Wine_Consumption.ipynb# Exploratory notebook (the raw EDA work)
├── requirements.txt          # Pinned Python dependencies
├── runtime.txt               # Python version hint for Streamlit Cloud
├── .gitignore
├── LICENSE                   # MIT
├── MANAGER_GUIDE.md          # Non-technical user guide for the wine shop manager
└── README.md                 # You are here
```

Each module has a single responsibility. Marketing playbooks live in
`wine_config.py` as pure data, so a non-coder can edit them. Clustering
algorithms live in `wine_clustering.py` — both lenses coexist.

---

## Run locally

```bash
git clone https://github.com/<your-username>/wine-shop-customer-dashboard.git
cd wine-shop-customer-dashboard
python -m pip install -r requirements.txt
streamlit run wine_dashboard.py
```

### Without an isolated environment, on macOS

```bash
python3 -m pip install --user -r requirements.txt
python3 -m streamlit run wine_dashboard.py
```

---

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (public).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick the repo, branch `main`, file path `wine_dashboard.py`.
4. Python version: 3.11 (auto-detected from `runtime.txt`; set explicitly in
   Advanced Settings if the cloud ignores it).
5. Click **Deploy**. First build takes ~3–5 minutes.

The app sleeps after 7 days of inactivity on the free tier — clicking the
URL wakes it up (~30 seconds).

---

## Design notes

### Segmentation methodology — three lenses, deliberately

We use three clustering approaches because they answer different
questions and the manager benefits from holding all three in mind:

1. **FM (Frequency × Monetary)** — supervised median split on visits/mo
   and Ticket. Four named personas: *Champions* (high-freq high-spend),
   *Loyal Regulars* (high-freq low-spend), *Occasion Splurgers* (low-freq
   high-spend), *Casual Visitors* (low-freq low-spend). This is the
   revenue lens — protect Champions, upsell Loyal Regulars.

2. **Behavioral K-Means** — unsupervised K-Means K=4 on
   (visits/month, sociality score). Ticket is *deliberately excluded*
   so the segments are independent of how much customers pay. Produces
   archetypes: *Social Regulars*, *Daily Home Drinkers*, *Occasion
   Celebrants*, *Casual Home Drinkers*. This is the lifestyle lens —
   tells you *how* to reach each group.

3. **K-Prototypes spend tiers** — unsupervised mixed-type clustering on
   the full feature set (Hamming on categoricals, Euclidean on Ticket).
   Default K=3 → *Entry / Core / Premium Buyers*. This is the value
   lens — a sanity check that the FM framework is well-anchored.

### Statistical layer

The Profile tab's "Hidden patterns" chart presents customer-attribute
associations on a 0–100 scale for non-statistical readability.
Mathematically, the underlying measure is **Cramér's V × 100** — see
the in-app "📐 What's the maths underneath?" expander for the
academic detail and conventional interpretation thresholds.

### Revenue formula

```
monthly_spend     = Ticket × monthly_visits
est_annual_revenue = monthly_spend × 12
```

The visits/month mapping converts verbal frequency categories
("1 to 2 times per week" → 6 visits/mo) and is documented in the
sidebar and the What-If simulator's assumptions expander.

### Caching

All expensive operations (Excel load, FM computation, behavioral
K-Means, K-Prototypes fit, silhouette diagnostics) are wrapped in
`@st.cache_data`. Each model is fit at most once per (filter selection
× K) combination within a session.

### Data caveats

The dataset is the ESADE *Analytics and Big Data* case-study sample.
It is synthetic / case-study data, not real POS records. Some rows
contain unrealistic combinations (customers visiting daily at €100
tickets = €36k/year). We surface these as-is rather than clipping —
the methodology would be re-fit on real POS data when available.

The "Salmond" entry in the *Additional products* column is preserved as
it appears in the source. It's most likely a typo for *Almond* (deli
category fit), possibly *Salmon*. A footnote explains this wherever
the term is rendered in the dashboard.

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Case study from *Analytics and Big Data* (ESADE).
Strategic recommendations adapted from the team's marketing report.
Visualization choices follow Prof. Manel Guerris's data-visualization
deck (KISS · sort-by-value · diverging bars for survey-style binary
signals · heatmaps over tables for pattern-finding · Beginning & End
narrative arc).
