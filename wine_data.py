"""
wine_data.py
============
Data layer: loading the survey Excel, applying data cleaning,
deriving engineered columns, and computing the FM segmentation.

Corrections vs original:
  - Strips trailing whitespace from all string columns (fixes 'Secondary ')
  - Corrects 'Salmond' typo to 'Salmon' at load time, so PREMIUM_DELI
    uses the corrected spelling throughout
  - Revenue estimate is flagged as a directional projection, not actuals
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "5. Wine consumption.xlsx"

# Midpoint estimates of visits per month for each verbal frequency band.
# Used to compute monthly_visits (the F dimension of FM segmentation)
# and the annual revenue estimate.
FREQ_VISITS_PER_MONTH: dict[str, int] = {
    "Once per month":            1,
    "More than once per month":  2,
    "1 to 2 times per week":     6,
    "3 to 4 times per week":    14,
    "5 to 6 times per week":    22,
    "Once per day":             30,
}
FREQ_ORDER = list(FREQ_VISITS_PER_MONTH.keys())
AGE_ORDER  = ["23 to 30", "31 to 40", "41 to 50", "more than 50"]

# Deli tier classification.
# Based on product category and unit price positioning:
#   Premium — artisan / charcuterie items (Cheese, Spanish ham, Salmon)
#   Entry   — everyday accompaniments    (Olives, Chocolate, Candies, Cookies, Nuts)
# Note: average basket size is nearly identical across tiers (€55.5 Entry vs €55.9 Premium),
# so this classification supports brand positioning decisions, not spend prediction.
PREMIUM_DELI = {"Cheese", "Spanish ham", "Salmon"}
ENTRY_DELI   = {"Olives", "Nuts", "Candies", "Cookies", "Chocolate"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_deli(product: str) -> str:
    if product in PREMIUM_DELI:
        return "Premium"
    if product in ENTRY_DELI:
        return "Entry"
    return "Other"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the survey Excel, clean known data quality issues, and derive
    engineered columns.

    Cleaning applied:
      - Trailing/leading whitespace stripped from all string columns
        (affects 71 rows, notably 'Secondary ' in Education)
      - 'Salmond' corrected to 'Salmon' (typo in 67 survey responses)
      - Education value '1' (encoding error in row 334) set to NaN

    Engineered columns added:
      - monthly_visits      : numeric estimate of visits per month
      - est_annual_revenue  : ticket × monthly_visits × 12
                              (directional estimate only — not actual store revenue)
      - deli_tier           : Premium / Entry classification per product
    """
    df = pd.read_excel(path, sheet_name="Hoja1")

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        # Restore NaN where the original value was blank or 'nan'
        df[col] = df[col].replace({"nan": np.nan, "": np.nan})

    # Fix known data entry errors
    df["Additional products"] = df["Additional products"].replace({"Salmond": "Salmon"})
    df["Education"] = df["Education"].replace({"1": np.nan})

    # Derived columns
    df["monthly_visits"]     = df["Wine frequency consumption"].map(FREQ_VISITS_PER_MONTH).fillna(1)
    df["est_annual_revenue"] = df["Ticket"] * df["monthly_visits"] * 12
    df["deli_tier"]          = df["Additional products"].apply(_classify_deli)

    return df


@st.cache_data(show_spinner=False)
def compute_fm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each customer into one of four FM (Frequency x Monetary) segments
    using a median split on visits/month and average ticket.

    Segment definitions
    -------------------
    Champions          High frequency, high spend   — the revenue engine
    Loyal Regulars     High frequency, low spend    — the upsell opportunity
    Occasion Splurgers Low frequency,  high spend   — gift and occasion driven
    Casual Visitors    Low frequency,  low spend    — conversion pipeline

    Boundary rule: customers exactly on the median are assigned to the High tier
    (consistent with the >= operator). With 404 responses this boundary affects
    172 customers on the frequency dimension (all of whom have 6 visits/month,
    the midpoint of '1 to 2 times per week').

    Returns the input DataFrame with three additional columns:
    F_tier, M_tier, FM_segment.
    """
    df = df.copy()
    f_med = df["monthly_visits"].median()
    m_med = df["Ticket"].median()

    df["F_tier"] = np.where(df["monthly_visits"] >= f_med, "High", "Low")
    df["M_tier"] = np.where(df["Ticket"]         >= m_med, "High", "Low")

    mapping = {
        ("High", "High"): "Champions",
        ("High", "Low"):  "Loyal Regulars",
        ("Low",  "High"): "Occasion Splurgers",
        ("Low",  "Low"):  "Casual Visitors",
    }
    df["FM_segment"] = df.apply(lambda r: mapping[(r["F_tier"], r["M_tier"])], axis=1)
    return df
