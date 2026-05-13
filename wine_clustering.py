"""
wine_clustering.py
==================
Behavioral K-Means segmentation: clusters on visit frequency × sociality score.

This replaces the earlier K-Prototypes approach. Analysis in the research
notebooks showed that K-Prototypes on the full mixed-type feature set merely
rediscovers ticket price tiers (because the categorical features are uniformly
distributed). Clustering on frequency × sociality finds four meaningful
behavioral archetypes that are independent of ticket price, making them
genuinely complementary to the FM (revenue) segmentation.

Fixed K=4, validated by silhouette/DB/CH analysis (see cluster_insights.ipynb).

Public API:
    fit_behavioral_clusters(df)          → cluster label array
    name_behavioral_clusters(df, col)    → {label_int: segment_name} mapping
    behavioral_diagnostics(df)           → (Ks, inertias, silhouettes)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import silhouette_score
from sklearn.cluster       import KMeans


FEATURES = ["monthly_visits", "social_score"]


@st.cache_data(show_spinner="Fitting behavioral clusters…")
def fit_behavioral_clusters(df: pd.DataFrame, k: int = 4, seed: int = 42) -> np.ndarray:
    """K-Means K=4 on (frequency, sociality). Returns integer cluster labels."""
    X = StandardScaler().fit_transform(df[FEATURES])
    return KMeans(n_clusters=k, n_init=15, random_state=seed).fit_predict(X)


def name_behavioral_clusters(df: pd.DataFrame, label_col: str) -> dict[int, str]:
    """Rank clusters by freq and sociality, assign names by archetype.

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
