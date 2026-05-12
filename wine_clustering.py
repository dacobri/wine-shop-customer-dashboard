"""
wine_clustering.py
==================
Unsupervised market segmentation.

Primary algorithm: **K-Prototypes** (from the third-party `kmodes` package).
It's the correct choice for this dataset because seven of our eight feature
columns are categorical (occasion, payment, deli choice, age band, etc.)
and only Ticket is numeric. K-Prototypes uses Hamming distance on the
categoricals and Euclidean on the numerics — properly weighting both.

Fallback: K-Means on one-hot encoded data, used if `kmodes` isn't
installed. It works but distorts the categorical distance geometry, so
it's strictly a "show something rather than nothing" backstop.

Public API:
    fit_clusters(df, k)       → cluster labels (auto-picks algorithm)
    cluster_diagnostics(df)   → (Ks, inertias, silhouettes) for elbow plot
    name_clusters_by_spend()  → rename numeric labels as Entry/Core/Premium
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import silhouette_score
from sklearn.cluster       import KMeans

# K-Prototypes is the preferred algorithm. Soft-fail so the app still
# runs (with the K-Means fallback) on machines without kmodes installed.
try:
    from kmodes.kprototypes import KPrototypes
    KPROTOTYPES_OK = True
except ImportError:
    KPROTOTYPES_OK = False


# --------------------------------------------------------------------------
# Features used for clustering
# --------------------------------------------------------------------------

CLUSTER_CAT_COLS = [
    "Wine frequency consumption",
    "Payment mode",
    "Place to drink",
    "Additional products",
    "Gender",
    "Age",
]
CLUSTER_NUM_COLS = ["Ticket"]


# --------------------------------------------------------------------------
# Internal: design matrix builder for K-Means / silhouette
# --------------------------------------------------------------------------

def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """One-hot encode categoricals + scale numerics. Used by K-Means routes."""
    X_cat = pd.get_dummies(df[CLUSTER_CAT_COLS].astype(str))
    X_num = StandardScaler().fit_transform(df[CLUSTER_NUM_COLS])
    return np.hstack([X_cat.values, X_num])


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Fitting K-Prototypes…")
def run_kprototypes(df: pd.DataFrame, k: int = 3, seed: int = 42) -> np.ndarray:
    """Fit K-Prototypes on the mixed-type data and return cluster labels.

    Streamlit caches the result keyed on (df contents, k, seed). The model
    is therefore fit at most once per (filter selection × K) combination
    within a session — subsequent calls return the cached labels instantly.
    """
    work = df[CLUSTER_CAT_COLS + CLUSTER_NUM_COLS].copy()
    for c in CLUSTER_CAT_COLS:
        work[c] = work[c].astype(str).fillna("Unknown")
    cat_idx = [work.columns.get_loc(c) for c in CLUSTER_CAT_COLS]

    model = KPrototypes(n_clusters=k, init="Cao", n_init=5, random_state=seed, verbose=0)
    labels = model.fit_predict(work.values, categorical=cat_idx)
    return np.asarray(labels)


@st.cache_data(show_spinner="Fitting K-Means…")
def run_kmeans_fallback(df: pd.DataFrame, k: int = 3, seed: int = 42) -> np.ndarray:
    """K-Means on one-hot encoded features. Fallback when kmodes is unavailable."""
    X = _design_matrix(df)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


@st.cache_data(show_spinner=False)
def cluster_diagnostics(df: pd.DataFrame):
    """Elbow inertia + silhouette score across K=2..6 for the K-selection chart.

    We always use K-Means here regardless of whether K-Prototypes is
    available: silhouette needs a metric space, and the inertia from
    K-Prototypes isn't directly comparable. This is purely diagnostic.
    """
    X = _design_matrix(df)
    ks = list(range(2, 7))
    inertias, sils = [], []
    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(m.inertia_)
        sils.append(silhouette_score(X, m.labels_))
    return ks, inertias, sils


def fit_clusters(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    """Convenience wrapper — runs K-Prototypes if available, else K-Means."""
    if KPROTOTYPES_OK:
        return run_kprototypes(df, k=k)
    return run_kmeans_fallback(df, k=k)


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
