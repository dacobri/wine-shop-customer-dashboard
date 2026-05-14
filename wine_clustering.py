"""
wine_clustering.py
==================
Unsupervised market segmentation as an independent validation of the
FM framework.

Primary algorithm: K-Prototypes (kmodes package).
The survey data is mixed-type: six categorical columns (occasion,
payment, deli product, age, gender, frequency) and one numeric (Ticket).
K-Prototypes applies Hamming distance on categoricals and Euclidean on
numerics — the correct approach for this data structure.

Fallback: K-Means on one-hot encoded features, used when kmodes is not
installed. It distorts categorical distance geometry but produces a
usable approximation for exploratory purposes.

Public API
----------
    fit_clusters(df, k)         → cluster labels (np.ndarray)
    cluster_diagnostics(df)     → (Ks, inertias, silhouettes) for elbow plot
    name_clusters_by_spend()    → dict mapping label → spend-tier name
    KPROTOTYPES_OK              → bool, whether kmodes is available
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics        import silhouette_score
from sklearn.cluster        import KMeans

try:
    from kmodes.kprototypes import KPrototypes
    KPROTOTYPES_OK = True
except ImportError:
    KPROTOTYPES_OK = False


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------

CLUSTER_CAT_COLS = [
    "Wine frequency consumption",
    "Payment mode",
    "Place to drink",
    "Additional products",
    "Gender",
    "Age",
]
CLUSTER_NUM_COLS = ["Ticket"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """One-hot encode categoricals and scale numerics. Used by K-Means routes."""
    X_cat = pd.get_dummies(df[CLUSTER_CAT_COLS].astype(str))
    X_num = StandardScaler().fit_transform(df[CLUSTER_NUM_COLS])
    return np.hstack([X_cat.values, X_num])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Fitting K-Prototypes model...")
def run_kprototypes(df: pd.DataFrame, k: int = 3, seed: int = 42) -> np.ndarray:
    """
    Fit K-Prototypes on the mixed-type survey data and return cluster labels.

    Results are cached per (df content, k, seed) — the model fits at most
    once per filter selection within a session.
    """
    work = df[CLUSTER_CAT_COLS + CLUSTER_NUM_COLS].copy()
    for c in CLUSTER_CAT_COLS:
        work[c] = work[c].astype(str).fillna("Unknown")
    cat_idx = [work.columns.get_loc(c) for c in CLUSTER_CAT_COLS]

    model = KPrototypes(n_clusters=k, init="Cao", n_init=5, random_state=seed, verbose=0)
    return np.asarray(model.fit_predict(work.values, categorical=cat_idx))


@st.cache_data(show_spinner="Fitting K-Means model...")
def run_kmeans_fallback(df: pd.DataFrame, k: int = 3, seed: int = 42) -> np.ndarray:
    """K-Means fallback on one-hot encoded features. Used when kmodes is unavailable."""
    X = _design_matrix(df)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


@st.cache_data(show_spinner=False)
def cluster_diagnostics(df: pd.DataFrame):
    """
    Compute elbow inertia and silhouette score for K = 2..6.

    K-Means is used here regardless of whether K-Prototypes is available,
    because silhouette score requires a metric space and K-Prototypes
    inertia is not directly comparable. This is a diagnostic tool only.

    Returns: (ks, inertias, silhouettes)
    """
    X   = _design_matrix(df)
    ks  = list(range(2, 7))
    inertias, sils = [], []
    for k in ks:
        m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertias.append(m.inertia_)
        sils.append(silhouette_score(X, m.labels_))
    return ks, inertias, sils


def fit_clusters(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    """Run K-Prototypes if available, otherwise K-Means."""
    if KPROTOTYPES_OK:
        return run_kprototypes(df, k=k)
    return run_kmeans_fallback(df, k=k)


def name_clusters_by_spend(df: pd.DataFrame, label_col: str) -> dict:
    """
    Rank clusters by average Ticket and return human-readable spend-tier labels.

    Preset names for K = 2, 3, 4. Fallback for other K values uses Tier N.
    """
    avg = df.groupby(label_col)["Ticket"].mean().sort_values()
    n   = len(avg)
    presets = {
        2: ["Entry Buyers",  "Premium Buyers"],
        3: ["Entry Buyers",  "Core Buyers",     "Premium Buyers"],
        4: ["Entry Buyers",  "Mid-Low Buyers",  "Mid-High Buyers", "Premium Buyers"],
    }
    names = presets.get(n, [f"Tier {i + 1}" for i in range(n)])
    return dict(zip(avg.index, names))
