"""
wine_data.py
============
Data layer: loading the Excel survey, cleaning known issues, deriving
helper columns (visits/month, annual revenue estimate, deli tier), and
computing the FM (Frequency x Monetary) segmentation.

This module owns the business-domain constants that describe the data:
the frequency-to-visits mapping, the age ordering, and the
premium-vs-entry deli classification. Anything that's "about the data
itself" lives here.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------------------------------
# Constants — domain knowledge about the dataset
# --------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent / "5. Wine consumption.xlsx"

# Map verbal frequency categories to a numeric visits-per-month estimate
# (midpoint of each verbal range). Used for FM segmentation and annual
# revenue estimate (ticket × visits/mo × 12).
FREQ_VISITS_PER_MONTH = {
    "Once per month":            1,
    "More than once per month":  2,
    "1 to 2 times per week":     6,
    "3 to 4 times per week":    14,
    "5 to 6 times per week":    22,
    "Once per day":             30,
}
FREQ_ORDER = list(FREQ_VISITS_PER_MONTH.keys())
AGE_ORDER  = ["23 to 30", "31 to 40", "41 to 50", "more than 50"]

# Sociality score: 1 = private/alone, 7 = large group.
# Used by the behavioral K-Means segmentation (frequency × sociality).
SOCIAL_MAP = {
    "Home":             1,
    "With your couple": 2,
    "Friends home":     3,
    "On holidays":      4,
    "Restaurant":       5,
    "Birthday party":   6,
    "Parties":          7,
}

# Education tier rollup: collapse the 9 raw levels into 4 marketing-relevant
# tiers. The original `Education` column is preserved untouched for the
# raw customer-table view in the Explorer.
EDUCATION_GROUPS = {
    "Primary":                        "Basic",
    "Secondary Unfinished":           "Basic",
    "Secondary":                      "Basic",
    "Technical Superior Unfinished":  "Technical",
    "Technical Superior":             "Technical",
    "Universitary Degree Unfinished": "University",
    "Universitary Degree":            "University",
    "Postgraduate Unfinished":        "Postgraduate",
    "Postgraduate":                   "Postgraduate",
}
EDUCATION_GROUP_ORDER = ["Basic", "Technical", "University", "Postgraduate"]

# Deli tier classification — defines what counts as a "premium" cross-sell
# (the strategic priority is shifting more of the basket into this set).
# Note: "Salmond" is intentional — it matches the (mis)spelling in the survey file.
PREMIUM_DELI = {"Cheese", "Spanish ham", "Salmond"}
ENTRY_DELI   = {"Olives", "Nuts", "Candies", "Cookies", "Chocolate"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _classify_deli(product: str) -> str:
    """Bucket a deli product into Premium / Entry / Other."""
    if product in PREMIUM_DELI:
        return "Premium"
    if product in ENTRY_DELI:
        return "Entry"
    return "Other"


# --------------------------------------------------------------------------
# Public API — loaders & feature engineering
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the survey Excel, clean known issues, and derive helper columns.

    Adds three derived columns:
      - monthly_visits:        numeric visits/month from the frequency category
      - est_annual_revenue:    ticket × monthly_visits × 12
      - deli_tier:             Premium / Entry / Other classification
    """
    df = pd.read_excel(path, sheet_name="Hoja1")

    # Defensive whitespace strip on every text column.
    # The source survey has at least one entry with a trailing space
    # ('Secondary ' instead of 'Secondary'), which would otherwise drop
    # 71 customers out of the Education_group mapping.
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda v: v.strip() if isinstance(v, str) else v)

    # Known data-quality fix: stray '1' value in Education column.
    df["Education"] = df["Education"].replace({1: np.nan, "1": np.nan})

    df["monthly_visits"]     = df["Wine frequency consumption"].map(FREQ_VISITS_PER_MONTH).fillna(1)
    df["est_annual_revenue"] = df["Ticket"] * df["monthly_visits"] * 12
    df["deli_tier"]          = df["Additional products"].apply(_classify_deli)
    df["social_score"]       = df["Place to drink"].map(SOCIAL_MAP).fillna(4)
    df["Education_group"]    = df["Education"].map(EDUCATION_GROUPS)
    return df


@st.cache_data(show_spinner=False)
def compute_fm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each customer to one of 4 FM (Frequency × Monetary) segments.

    Uses a median split on visits/month and avg ticket. Returns the same
    df with three new columns: F_tier, M_tier, FM_segment.

    Segment definitions:
      - Champions          : High frequency, High monetary  (revenue engine)
      - Loyal Regulars     : High frequency, Low  monetary  (upsell opportunity)
      - Occasion Splurgers : Low  frequency, High monetary  (gift-driven)
      - Casual Visitors    : Low  frequency, Low  monetary  (pipeline)
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
