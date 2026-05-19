"""
wine_clustering.py
==================
Two complementary unsupervised lenses on the customer base:

1. **Behavioral K-Means** (Lucas's latest work) — K=4 on
   (visit frequency × sociality score). Produces four named archetypes
   independent of ticket value: Social Regulars, Daily Home Drinkers,
   Occasion Celebrants, Casual Home Drinkers.

2. **K-Prototypes spend tiers** (Lucas's earlier work, restored as a
   complementary view) — clusters the mixed-type feature set with
   Hamming distance on categoricals + Euclidean on Ticket. Produces
   spend-tier names: Entry / Core / Premium Buyers.

Both views coexist in the dashboard — the manager toggles between them.
Behavioral answers "*how* do they buy?", spend tiers answer "*how much*
do they buy?".

Public API
----------
Behavioral lens (default):
    fit_behavioral_clusters(df)       → cluster label array
    name_behavioral_clusters(df, col) → {label_int: archetype name}
    behavioral_diagnostics(df)        → (Ks, inertias, silhouettes)

Spend-tier lens:
    fit_clusters(df, k)               → cluster labels (K-Prototypes or K-Means)
    name_clusters_by_spend(df, col)   → {label_int: 'Entry/Core/Premium ...'}
    spend_tier_diagnostics(df)        → (Ks, inertias, silhouettes) over K=2..6
    KPROTOTYPES_OK                    → True if kmodes is available
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import silhouette_score
from sklearn.cluster       import KMeans

# Try K-Prototypes — degrades gracefully to K-Means on one-hot if absent.
try:
    from kmodes.kprototypes import KPrototypes
    KPROTOTYPES_OK = True
except ImportError:
    KPROTOTYPES_OK = False


# ============================================================================
# Behavioral K-Means (Lucas's latest)
# ============================================================================

FEATURES = ["monthly_visits", "social_score"]


@st.cache_data(show_spinner="Fitting behavioral clusters…")
def fit_behavioral_clusters(df: pd.DataFrame, k: int = 4, seed: int = 42) -> np.ndarray:
    """K-Means K=4 on (frequency, sociality). Returns integer cluster labels."""
    X = StandardScaler().fit_transform(df[FEATURES])
    return KMeans(n_clusters=k, n_init=15, random_state=seed).fit_predict(X)


def name_behavioral_clusters(df: pd.DataFrame, label_col: str) -> dict[int, str]:
    """Rank clusters by freq and sociality, assign archetype names.

    Uses rank-based comparison (clusters vs each other) rather than overall
    data medians — robust against boundary sensitivity when cluster means
    sit close to the population median.
    """
    means = df.groupby(label_col)[FEATURES].mean()
    freq_rank = means["monthly_visits"].rank()
    soc_rank  = means["social_score"].rank()

    def _name(idx: int) -> str:
        hi_f = freq_rank[idx] > len(means) / 2
        hi_s = soc_rank[idx]  > len(means) / 2
        if   hi_f and     hi_s: return "Social Regulars"
        elif hi_f and not hi_s: return "Daily Home Drinkers"
        elif hi_s and not hi_f: return "Occasion Celebrants"
        else:                   return "Casual Home Drinkers"

    return {int(idx): _name(idx) for idx in means.index}


@st.cache_data(show_spinner=False)
def behavioral_diagnostics(df: pd.DataFrame):
    """Elbow inertia + silhouette across K=2..7 for the diagnostic chart."""
    X = StandardScaler().fit_transform(df[FEATURES])
    ks = list(range(2, 8))
    inertias, sils = [], []
    for k in ks:
        m = KMeans(n_clusters=k, n_init=15, random_state=42).fit(X)
        inertias.append(m.inertia_)
        sils.append(silhouette_score(X, m.labels_))
    return ks, inertias, sils


# ============================================================================
# K-Prototypes spend-tier clustering (Lucas's earlier work, restored)
# ============================================================================

CLUSTER_CAT_COLS = ["Wine frequency consumption", "Payment mode", "Place to drink",
                    "Additional products", "Gender", "Age"]
CLUSTER_NUM_COLS = ["Ticket"]


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """One-hot encode categoricals + scale numerics. Used for K-Means fallback
    and for the elbow/silhouette diagnostics regardless of algorithm chosen."""
    X_cat = pd.get_dummies(df[CLUSTER_CAT_COLS].astype(str))
    X_num = StandardScaler().fit_transform(df[CLUSTER_NUM_COLS])
    return np.hstack([X_cat.values, X_num])


@st.cache_data(show_spinner="Fitting K-Prototypes…")
def run_kprototypes(df: pd.DataFrame, k: int = 3, seed: int = 42) -> np.ndarray:
    """Fit K-Prototypes on mixed-type survey data. Returns cluster labels."""
    work = df[CLUSTER_CAT_COLS + CLUSTER_NUM_COLS].copy()
    for c in CLUSTER_CAT_COLS:
        work[c] = work[c].astype(str).fillna("Unknown")
    cat_idx = [work.columns.get_loc(c) for c in CLUSTER_CAT_COLS]
    model = KPrototypes(n_clusters=k, init="Cao", n_init=5,
                        random_state=seed, verbose=0)
    return np.asarray(model.fit_predict(work.values, categorical=cat_idx))


@st.cache_data(show_spinner="Fitting K-Means (fallback)…")
def run_kmeans_fallback(df: pd.DataFrame, k: int = 3, seed: int = 42) -> np.ndarray:
    """K-Means on one-hot encoded features. Used when kmodes is unavailable."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(_design_matrix(df))


def fit_clusters(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    """Convenience wrapper — K-Prototypes if available, else K-Means."""
    if KPROTOTYPES_OK:
        return run_kprototypes(df, k=k)
    return run_kmeans_fallback(df, k=k)


@st.cache_data(show_spinner=False)
def spend_tier_diagnostics(df: pd.DataFrame):
    """Elbow + silhouette across K=2..6 for the spend-tier diagnostic chart.

    Always uses K-Means on the same design matrix regardless of whether
    K-Prototypes is available — silhouette needs a metric space and the
    K-Prototypes inertia is not directly comparable to K-Means inertia.
    """
    X = _design_matrix(df)
    ks = list(range(2, 7))
    inertias, sils = [], []
    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(m.inertia_)
        sils.append(silhouette_score(X, m.labels_))
    return ks, inertias, sils


def name_clusters_by_spend(df: pd.DataFrame, label_col: str) -> dict:
    """Rank clusters by avg Ticket and assign human-readable spend-tier names."""
    avg = df.groupby(label_col)["Ticket"].mean().sort_values()
    n = len(avg)
    presets = {
        2: ["Entry Buyers", "Premium Buyers"],
        3: ["Entry Buyers", "Core Buyers", "Premium Buyers"],
        4: ["Entry Buyers", "Mid-Low Buyers", "Mid-High Buyers", "Premium Buyers"],
    }
    names = presets.get(n, [f"Tier {i+1}" for i in range(n)])
    return dict(zip(avg.index, names))
