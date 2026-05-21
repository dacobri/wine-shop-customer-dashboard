"""
wine_dashboard.py
=================
Wine Shop & Delicatessen — Strategic Customer Dashboard.

Tab order (story flow — May 2026 reorder):
  1. Customer Profile   — who walks through the door (+ Customer Explorer
                          embedded as a collapsed expander at the bottom
                          for individual-level outreach lists)
  2. FM Segments        — who is worth what? (revenue lens)
  3. Behavioral Segs    — how do they buy? Two ML lenses (behavioural + spend)
  4. Product Mix        — where is the deli opportunity?
  5. Strategic Overview — synthesis + What-If revenue simulator
  6. Action Plan        — the close: prioritised marketing playbook

The dashboard ships with a root-level shim at wine_app/wine_dashboard.py
that delegates to this file, so the Streamlit Cloud default entry path
works without manual configuration.

INSTALL
-------
    pip install streamlit plotly pandas numpy scikit-learn kmodes openpyxl

RUN
---
    streamlit run wine_dashboard.py        # via the shim (preferred)
    streamlit run app/wine_dashboard.py    # directly (also works)

Course: Advanced Programming with Python (ESADE MSc Business Analytics)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wine_theme      import (PALETTE, SEQ, SEGMENT_COLORS, BEHAVIORAL_COLORS,
                              inject_css, base_layout, kpi_card, callout,
                              salmond_footnote, with_salmond_marker)
from wine_data       import (load_data, compute_fm_segments,
                              FREQ_ORDER, FREQ_VISITS_PER_MONTH, AGE_ORDER,
                              EDUCATION_GROUP_ORDER, SOCIAL_MAP)
from wine_clustering import (
    # Behavioral lens — K-Means K=4 on (frequency, sociality)
    fit_behavioral_clusters, name_behavioral_clusters, behavioral_diagnostics,
    # Value lens — K-Prototypes spend tiers (Entry/Core/Premium)
    fit_clusters, name_clusters_by_spend, spend_tier_diagnostics, KPROTOTYPES_OK,
)
from wine_simulator  import simulate_revenue
from wine_config     import SEGMENT_META, BEHAVIORAL_META


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Wine Shop Customer Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ============================================================================
# TAB RENDERERS
# ============================================================================

def _next_tab_hint(text: str) -> None:
    """Small italic hand-off line at the bottom of each tab, so each view
    introduces the next one and the narrative feels continuous rather than
    seven disconnected sections."""
    st.markdown(
        f"<div style='margin-top:24px; padding:10px 14px; "
        f"border-left:3px solid {PALETTE['gold']}; "
        f"background:{PALETTE['cream']}; "
        f"color:{PALETTE['midgray']}; font-style:italic; font-size:13px; "
        f"border-radius:4px;'>→ {text}</div>",
        unsafe_allow_html=True,
    )


def render_profile(df_f: pd.DataFrame, df_all: pd.DataFrame) -> None:
    """Customer Profile tab — 4-section narrative arc.

    Design follows Prof. Guerris's visualization principles:
    KISS (5 charts only), Z-pattern info priority, sort-by-value never
    alphabetical, color for emphasis, diverging bars for survey-style
    binary signals, heatmap for relationship patterns. Beginning &
    End: hero up top, big-picture relationships at the bottom.

    Filter behavior (per user spec):
      - Hero, pyramid, education funnel: filter-INDEPENDENT (always
        show the true customer base). Note appears if filters active.
      - Diverging bar, boxplot, Cramér's V: filter-AWARE.
      - The diverging bar self-falls back to df_all if a gender filter
        would collapse the chart.
    """
    filters_active = len(df_f) != len(df_all)

    # ── Hero ───────────────────────────────────────────────────────────────
    _profile_hero(df_all, filters_active)

    st.markdown("---")

    # ── Section A: Demographics (filter-independent) ───────────────────────
    st.markdown("### A. Who they are — demographic composition")
    if filters_active:
        st.caption(f"ℹ️ Sidebar filters are active. These composition views "
                   f"always show all {len(df_all)} customers — they would "
                   f"lose meaning if filtered.")
    else:
        st.caption("The structural make-up of the customer base — age, gender, education tier.")
    pa, pb = st.columns([3, 2])
    with pa: _profile_population_pyramid(df_all)
    with pb: _profile_education_funnel(df_all)

    st.markdown("---")

    # ── Section B: Where & When they drink (THE marquee chart) ────────────
    st.markdown("### B. Where they drink — the gender signal")
    st.caption("The strongest behavioral signal in the dataset. "
               "Sort order is by Male/Female skew per occasion.")
    _profile_gender_place_diverging(df_f, df_all)

    st.markdown("---")

    # ── Section C: Monthly spending patterns (filter-aware) ────────────────
    st.markdown("### C. How they spend — monthly spending patterns")
    st.caption("Monthly spend = ticket × visits per month. A daily €20 drinker "
               "(€600/mo) is worth more than a monthly €80 drinker (€80/mo). "
               "This is the dimension that matters for revenue.")
    _profile_spending_patterns(df_f)

    st.markdown("---")

    # ── Section D: Hidden patterns / transition to segmentation ────────────
    st.markdown("### D. Hidden patterns — which customer attributes go together?")
    st.caption(
        "Each bar shows how strongly two customer attributes predict each "
        "other on a 0–100 scale. A high score means knowing one attribute "
        "(say, gender) tells you something useful about the other (drinking "
        "occasion). These hidden links are exactly why we segment customers "
        "by behaviour and revenue on the next tabs rather than by any single "
        "demographic."
    )
    _profile_relationship_strengths(df_f)


# ── Hero ──────────────────────────────────────────────────────────────────

def _profile_hero(df: pd.DataFrame, filters_active: bool) -> None:
    """Big-number persona card. Always reflects the full customer base."""
    avg_ticket = df["Ticket"].mean()
    modal = {
        "Gender":      df["Gender"].mode().iloc[0],
        "Age":         df["Age"].mode().iloc[0],
        "Edu":         df["Education_group"].dropna().mode().iloc[0],
        "Freq":        df["Wine frequency consumption"].mode().iloc[0],
        "Place":       df["Place to drink"].mode().iloc[0],
        "Product":     df["Additional products"].mode().iloc[0],
    }
    # Salmond marker for the product chip
    product_disp = modal["Product"] + ("*" if modal["Product"] == "Salmond" else "")

    avatar_emoji = "👨" if modal["Gender"] == "Male" else "👩"
    chip_style = (f"display:inline-block; padding:6px 14px; margin:4px 4px 0 0; "
                  f"background:{PALETTE['cream']}; color:{PALETTE['charcoal']}; "
                  f"border-radius:16px; font-size:12px; font-weight:500;")

    st.markdown(f"""
    <div style='background:white; padding:28px 32px; border-radius:14px;
                border:1px solid {PALETTE["border"]};
                box-shadow:0 2px 12px rgba(0,0,0,0.04); margin-bottom:8px;'>
        <div style='display:flex; align-items:center; gap:28px; flex-wrap:wrap;'>
            <div style='flex-shrink:0;'>
                <div style='font-size:13px; color:{PALETTE["midgray"]}; letter-spacing:1px;
                            text-transform:uppercase; margin-bottom:4px;'>
                    Typical basket
                </div>
                <div style='font-size:72px; font-weight:800; color:{PALETTE["burgundy"]};
                            line-height:1; font-family:Georgia, serif;'>
                    €{avg_ticket:.0f}
                </div>
            </div>
            <div style='flex:1; min-width:300px;'>
                <p style='font-size:16px; color:{PALETTE["charcoal"]}; line-height:1.5; margin:0 0 12px 0;'>
                    {avatar_emoji} The typical wine-shop customer is
                    <b>{modal["Gender"].lower()}</b>, aged <b>{modal["Age"]}</b>,
                    with a <b>{modal["Edu"]}</b>-level education.
                    Visits <b>{modal["Freq"].lower()}</b> and most often drinks
                    at <b>{modal["Place"].lower()}</b>.
                </p>
                <div>
                    <span style='{chip_style}'>👤 {modal["Gender"]}</span>
                    <span style='{chip_style}'>🎂 {modal["Age"]}</span>
                    <span style='{chip_style}'>🎓 {modal["Edu"]}</span>
                    <span style='{chip_style}'>📅 {modal["Freq"]}</span>
                    <span style='{chip_style}'>📍 {modal["Place"]}</span>
                    <span style='{chip_style}'>🧀 {product_disp}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if filters_active:
        st.caption(f"ℹ️ Sidebar filters active — this hero always reflects "
                   f"the full {len(df)} customer base, not the filtered slice.")


# ── Section A.1: Population pyramid ───────────────────────────────────────

def _profile_population_pyramid(df: pd.DataFrame) -> None:
    """Classic horizontal age × gender pyramid. Male on left (negative), Female on right."""
    pivot = (df.pivot_table(index="Age", columns="Gender", aggfunc="size", fill_value=0)
               .reindex(AGE_ORDER))
    if "Male"   not in pivot.columns: pivot["Male"]   = 0
    if "Female" not in pivot.columns: pivot["Female"] = 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=pivot.index, x=-pivot["Male"], orientation="h",
        name="Male", marker_color=PALETTE["burgundy"],
        text=pivot["Male"], textposition="auto",
        hovertemplate="Age: <b>%{y}</b><br>Male: %{text} customers<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=pivot.index, x=pivot["Female"], orientation="h",
        name="Female", marker_color=PALETTE["teal"],
        text=pivot["Female"], textposition="auto",
        hovertemplate="Age: <b>%{y}</b><br>Female: %{text} customers<extra></extra>",
    ))

    # Symmetric X-axis range with absolute-value tick labels
    max_abs = int(max(pivot["Male"].max(), pivot["Female"].max())) + 5
    tick_step = 20 if max_abs > 40 else 10
    ticks = list(range(-max_abs - (-max_abs % tick_step), max_abs + 1, tick_step))
    tick_text = [str(abs(t)) for t in ticks]

    fig.update_layout(base_layout(
        title="Age × Gender — population pyramid",
        height=380, barmode="overlay", bargap=0.15,
        xaxis=dict(tickvals=ticks, ticktext=tick_text,
                   zeroline=True, zerolinewidth=2,
                   zerolinecolor=PALETTE["charcoal"], title="Customers"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ── Section A.2: Education funnel ─────────────────────────────────────────

# What the 9 survey levels collapse into. Used in tooltips so the manager
# can see exactly which raw responses ended up in each bucket.
EDUCATION_BUCKET_CONTENTS = {
    "Basic":        "Primary · Secondary Unfinished · Secondary",
    "Technical":    "Technical Superior Unfinished · Technical Superior",
    "University":   "Universitary Degree Unfinished · Universitary Degree",
    "Postgraduate": "Postgraduate Unfinished · Postgraduate",
}


def _profile_education_funnel(df: pd.DataFrame) -> None:
    """Horizontal bar of the 4 education tiers, sorted by count (largest first).

    Hovering a bar reveals the raw survey levels that roll up into the tier,
    so the manager can verify what 'Basic' or 'Technical' really mean.
    """
    counts = (df["Education_group"].dropna()
                .value_counts()
                .reindex(EDUCATION_GROUP_ORDER).fillna(0)
                .astype(int)
                .sort_values(ascending=True))   # ascending → largest at top
    total = counts.sum() or 1
    pct = (counts / total * 100).round(0).astype(int)

    # Colour the largest bar in burgundy, the rest in muted teal — focal emphasis
    colors = [PALETTE["burgundy"] if v == counts.max() else PALETTE["teal"]
              for v in counts.values]

    hover = [
        f"<b>{tier}</b><br>{c} customers ({p}%)"
        f"<br><i>Includes:</i> {EDUCATION_BUCKET_CONTENTS.get(tier, '')}"
        for tier, c, p in zip(counts.index, counts.values, pct.values)
    ]

    fig = go.Figure(go.Bar(
        y=counts.index, x=counts.values, orientation="h",
        marker_color=colors,
        text=[f"{c}  ({p}%)" for c, p in zip(counts.values, pct.values)],
        textposition="outside",
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(base_layout(
        title="Education tiers — 4 grouped levels",
        height=380, showlegend=False,
        xaxis=dict(title="Customers", range=[0, counts.max() * 1.25]),
        yaxis=dict(title=""),
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🛈 Hover any bar to see the raw survey levels that roll up into the tier.")


# ── Section B: Diverging Gender × Place ───────────────────────────────────

def _profile_gender_place_diverging(df_f: pd.DataFrame, df_all: pd.DataFrame) -> None:
    """Diverging horizontal bars — Male left, Female right, sorted by skew.

    Falls back to df_all (with a notice) if a Gender filter has collapsed
    the chart to one side.
    """
    df = df_f
    fallback_note = None
    if df["Gender"].nunique() < 2:
        df = df_all
        fallback_note = ("Gender filter active — this chart needs both genders, "
                         f"so it shows all {len(df_all)} customers instead.")

    ct = pd.crosstab(df["Place to drink"], df["Gender"])
    if "Male"   not in ct.columns: ct["Male"]   = 0
    if "Female" not in ct.columns: ct["Female"] = 0
    ct["total"]  = ct["Male"] + ct["Female"]
    ct["m_pct"]  = ct["Male"]   / ct["total"]
    ct["f_pct"]  = ct["Female"] / ct["total"]
    ct = ct.sort_values("m_pct")   # most-female at top, most-male at bottom

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=ct.index, x=-ct["Male"], orientation="h",
        name="Male", marker_color=PALETTE["burgundy"],
        text=[f"{n} ({p*100:.0f}%)" for n, p in zip(ct["Male"], ct["m_pct"])],
        textposition="inside", insidetextanchor="end",
        hovertemplate="<b>%{y}</b><br>Male: %{text}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=ct.index, x=ct["Female"], orientation="h",
        name="Female", marker_color=PALETTE["teal"],
        text=[f"{n} ({p*100:.0f}%)" for n, p in zip(ct["Female"], ct["f_pct"])],
        textposition="inside", insidetextanchor="start",
        hovertemplate="<b>%{y}</b><br>Female: %{text}<extra></extra>",
    ))
    max_abs = int(max(ct["Male"].max(), ct["Female"].max())) + 10
    fig.update_layout(base_layout(
        title="Where customers drink — Male / Female split",
        height=470, barmode="relative",
        xaxis=dict(range=[-max_abs, max_abs],
                   zeroline=True, zerolinewidth=2,
                   zerolinecolor=PALETTE["charcoal"], title="Customers"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Top-2 skew insights for the prose callout (only if both genders present)
    insights = []
    male_skewed = ct.nlargest(2, "m_pct")
    female_skewed = ct.nlargest(2, "f_pct")
    for place, row in male_skewed.iterrows():
        if row["m_pct"] >= 0.60 and row["total"] >= 5:
            insights.append(f"<b>{row['m_pct']*100:.0f}%</b> of <i>{place}</i> drinkers are men")
    for place, row in female_skewed.iterrows():
        if row["f_pct"] >= 0.60 and row["total"] >= 5:
            insights.append(f"<b>{row['f_pct']*100:.0f}%</b> of <i>{place}</i> drinkers are women")

    if insights:
        st.markdown(callout(
            "📊", "The gender signal",
            " · ".join(insights) + ". Use this to design gender-tailored "
            "in-store displays and bundle promotions for each occasion."
        ), unsafe_allow_html=True)

    if fallback_note:
        st.info(f"ℹ️ {fallback_note}")


# ── Section C: Monthly spending patterns ──────────────────────────────────

def _profile_spending_patterns(df: pd.DataFrame) -> None:
    """Three-part view of monthly spending: KPI strip + 2 demographic heatmaps.

    Why monthly (not per-visit ticket)? A daily €20 drinker generates
    €600/mo of revenue; a once-a-month €80 drinker generates €80/mo. The
    same per-visit ticket can hide a 7× difference in revenue contribution.
    """
    work = df.dropna(subset=["Education_group"]).copy()
    if len(work) < 10:
        st.info("Too few customers in current filter to compute reliable spending patterns.")
        return

    # ── KPI strip ──
    avg_monthly = work["monthly_spend"].mean()
    median_monthly = work["monthly_spend"].median()
    sorted_spend = work["monthly_spend"].sort_values(ascending=False)
    n_top20 = max(1, int(len(sorted_spend) * 0.2))
    top20_share = sorted_spend.iloc[:n_top20].sum() / sorted_spend.sum() * 100
    top_decile_threshold = sorted_spend.iloc[max(1, int(len(sorted_spend) * 0.1)) - 1]

    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_card("Avg monthly spend", f"{avg_monthly:.0f}", "€"),
                unsafe_allow_html=True)
    k2.markdown(kpi_card("Top 20% of customers", f"{top20_share:.0f}% of revenue"),
                unsafe_allow_html=True)
    k3.markdown(kpi_card("Top decile threshold", f"{top_decile_threshold:.0f}+/mo", "€"),
                unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # Education-bucket reminder (kept close to the two heatmaps that use it)
    with st.expander("🛈 What's in each education tier?"):
        for tier in EDUCATION_GROUP_ORDER:
            st.caption(f"**{tier}** — {EDUCATION_BUCKET_CONTENTS[tier]}")

    # ── Heatmap 1: Gender × Education ──
    _spending_heatmap(
        work, row_col="Gender", col_col="Education_group",
        row_order=sorted(work["Gender"].unique()),
        col_order=EDUCATION_GROUP_ORDER,
        title="Avg monthly spend per customer · Gender × Education tier",
        height=320,
    )

    # Compute insight for the callout
    means = (work.groupby(["Gender", "Education_group"], observed=True)["monthly_spend"]
                  .mean().round(0))
    if len(means) >= 2:
        top_idx = means.idxmax(); top_val = int(means.max())
        bot_idx = means.idxmin(); bot_val = int(means.min())
        ratio = means.max() / max(means.min(), 1)
        st.markdown(callout(
            "💡", "The most-valuable customer profile may surprise you",
            f"<b>{top_idx[0]} · {top_idx[1]}-tier</b> customers spend on average "
            f"<b>€{top_val}/month</b> — {ratio:.1f}× more than the lowest cell, "
            f"<b>{bot_idx[0]} · {bot_idx[1]}-tier</b> at <b>€{bot_val}/month</b>. "
            f"Education does <b>not</b> linearly correlate with monthly spend — "
            f"in fact the relationship is roughly inverted, especially for women. "
            f"Marketing copy assuming 'university-educated = high-value' is wrong "
            f"for this customer base."
        ), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Heatmap 2: Age × Education ──
    # Reveals life-stage × class spending patterns. Combined with Heatmap 1,
    # gives the manager both the gender and age dimension of the education
    # signal — useful for cohort-targeted email / WhatsApp campaigns.
    _spending_heatmap(
        work, row_col="Age", col_col="Education_group",
        row_order=AGE_ORDER, col_order=EDUCATION_GROUP_ORDER,
        title="Avg monthly spend per customer · Age × Education tier",
        height=360,
    )

    age_edu = (work.groupby(["Age", "Education_group"], observed=True)["monthly_spend"]
                    .mean().round(0))
    if len(age_edu) >= 2:
        top_idx = age_edu.idxmax()
        top_val = int(round(age_edu.max()))
        st.markdown(callout(
            "🎯", "Life-stage matters more than diplomas",
            f"The highest-value (age × education) cohort is "
            f"<b>{top_idx[0]} · {top_idx[1]}-tier</b> at <b>€{top_val}/month</b>. "
            f"Use this matrix when designing cohort-specific direct marketing — "
            f"combine age band and education tier to target individual cells, "
            f"not whole demographic dimensions."
        ), unsafe_allow_html=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Heatmap 3: Age × Drinking occasion ──
    # Occasion is the most marketing-actionable dimension — easy to design
    # themed campaigns around (party packs, restaurant pairings, festive
    # bundles). Combined with age band, the cells become directly targetable
    # by channel (Instagram for parties / WhatsApp for couples / etc.).
    place_order = sorted(SOCIAL_MAP, key=SOCIAL_MAP.get)  # private → social
    _spending_heatmap(
        work, row_col="Age", col_col="Place to drink",
        row_order=AGE_ORDER, col_order=place_order,
        title="Avg monthly spend per customer · Age × Drinking occasion",
        height=380,
    )

    # Find the highest-value WELL-POPULATED cell (n ≥ 10) so the callout
    # doesn't celebrate a fluke driven by 1–2 outlier customers.
    age_occ_mean  = work.groupby(["Age", "Place to drink"], observed=True)["monthly_spend"].mean()
    age_occ_count = work.groupby(["Age", "Place to drink"], observed=True).size()
    reliable = age_occ_mean[age_occ_count >= 10]
    if len(reliable) >= 1:
        top_idx = reliable.idxmax()
        top_val = int(round(reliable.max()))
        top_n   = int(age_occ_count.loc[top_idx])
        st.markdown(callout(
            "🎯", "Occasion is the most actionable lens for campaigns",
            f"The highest-spending well-populated cohort is "
            f"<b>{top_idx[0]}-year-olds drinking at {top_idx[1].lower()}</b> "
            f"at <b>€{top_val}/month</b> per customer (n = {top_n}). "
            f"Unlike education or age alone, occasion translates directly into "
            f"campaign mechanics — themed party packs for the <b>Parties</b> cell, "
            f"gift-ready bundles for <b>Birthday party</b>, restaurant wine cards "
            f"for the <b>Restaurant</b> cell, romantic boxes for <b>With your couple</b>. "
            f"Build the in-store displays around the strongest cells in this "
            f"matrix, not around demographic averages."
        ), unsafe_allow_html=True)


def _spending_heatmap(df: pd.DataFrame, row_col: str, col_col: str,
                      row_order: list, col_order: list, title: str,
                      height: int = 360) -> None:
    """Render a single monthly-spend heatmap with cell counts in hover."""
    means = (df.pivot_table(values="monthly_spend", index=row_col,
                            columns=col_col, aggfunc="mean")
                .reindex(index=row_order, columns=col_order))
    counts = (df.pivot_table(values="monthly_spend", index=row_col,
                             columns=col_col, aggfunc="count")
                 .reindex(index=row_order, columns=col_order))

    # Cell text: "€NNN" when populated; counts go to hover.
    # IMPORTANT: use round() not int() — int() truncates, which would make
    # cell labels disagree with the rounded averages used in the callouts.
    text = means.map(lambda v: f"€{int(round(v)):,}" if pd.notna(v) else "").values
    extra_hint = ""
    if col_col == "Education_group":
        extra_hint = "<br><i>Education tiers: hover the funnel above for tier contents.</i>"
    hover = []
    for i, r in enumerate(means.index):
        row = []
        for j, c in enumerate(means.columns):
            mv = means.iloc[i, j]
            cnt = counts.iloc[i, j]
            if pd.notna(mv):
                row.append(f"<b>{r}</b> × <b>{c}</b><br>"
                           f"Avg monthly spend: €{int(round(mv)):,}<br>"
                           f"n = {int(cnt)} customer{'s' if cnt != 1 else ''}"
                           f"{extra_hint}")
            else:
                row.append(f"<b>{r}</b> × <b>{c}</b><br>(no customers)")
        hover.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=means.values,
        x=list(means.columns),
        y=list(means.index),
        text=text,
        texttemplate="%{text}",
        textfont={"size": 13, "color": PALETTE["charcoal"]},
        hovertext=hover,
        hoverinfo="text",
        colorscale=[
            [0.0, PALETTE["cream"]],
            [0.5, PALETTE["gold"]],
            [1.0, PALETTE["burgundy"]],
        ],
        colorbar=dict(title="€/month"),
        zmin=0,
    ))
    fig.update_layout(base_layout(
        title=title, height=height,
        xaxis=dict(side="bottom", title=""),
        yaxis=dict(title="", autorange="reversed"),  # keep natural top-to-bottom order
    ))
    st.plotly_chart(fig, use_container_width=True)


# ── Section D: Ranked categorical relationships ───────────────────────────

def _profile_relationship_strengths(df: pd.DataFrame) -> None:
    """Top categorical attribute pairs, scored 0–100 for non-statisticians.

    The score under the hood is Cramér's V × 100 — we keep the math
    but never expose the term. The 0–100 framing is what the manager
    will actually use. The chart's purpose is to set up the segmentation
    tabs: 'no single attribute explains everything → that's why we segment'.
    """
    from scipy.stats import chi2_contingency

    def _cramers_v(x, y):
        ct = pd.crosstab(x, y)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return np.nan
        chi2 = chi2_contingency(ct, correction=False)[0]
        n = ct.values.sum()
        if n == 0:
            return np.nan
        r, k = ct.shape
        return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))

    cats = ["Gender", "Age", "Education_group", "Wine frequency consumption",
            "Place to drink", "Additional products", "Payment mode"]
    short = {
        "Wine frequency consumption": "Frequency",
        "Place to drink":              "Occasion",
        "Additional products":         "Deli product",
        "Education_group":             "Education",
        "Payment mode":                "Payment",
    }

    # Marketing translations of what each top pair means in plain English.
    pair_meanings = {
        ("Gender", "Place to drink"):
            "Men and women drink in different settings — design in-store displays "
            "and bundle promotions that reflect each gender's preferred occasions.",
        ("Age", "Additional products"):
            "Each age cohort has a different favourite deli pairing — bundle by "
            "age band, not by spend alone.",
        ("Education_group", "Payment mode"):
            "Education tier predicts payment habits — useful when calibrating "
            "loyalty-card incentives.",
        ("Education_group", "Additional products"):
            "Education tier shapes deli choice — informs which products to "
            "showcase in targeted promotions.",
        ("Age", "Education_group"):
            "Age and education co-vary in this base — expected; combining the "
            "two adds little extra signal.",
        ("Place to drink", "Payment mode"):
            "Occasion shapes how customers pay — useful when planning point-of-sale "
            "experience for high-traffic occasions.",
        ("Gender", "Age"):
            "Mild gender skew across age bands — moderate input for cohort-specific "
            "campaigns.",
        ("Age", "Place to drink"):
            "Each age cohort drinks in different settings — calibrate occasion "
            "campaigns by life stage.",
        ("Wine frequency consumption", "Place to drink"):
            "Frequent and casual drinkers go to different occasions — useful for "
            "matching loyalty perks to behaviour.",
        ("Place to drink", "Additional products"):
            "Each occasion has a different deli companion — the basis for "
            "'tonight's pairing' shelf-talkers.",
    }

    work = df.dropna(subset=["Education_group"])
    if len(work) < 10:
        st.info("Too few customers in current filter to compute reliable patterns.")
        return

    pairs = []   # list of (display_label, score_0_100, key_tuple)
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            v = _cramers_v(work[a], work[b])
            if not np.isnan(v):
                pairs.append((
                    f"{short.get(a, a)} × {short.get(b, b)}",
                    v * 100,
                    (a, b),
                ))
    pairs.sort(key=lambda p: -p[1])

    top_n = min(10, len(pairs))
    top = pairs[:top_n]
    labels = [p[0] for p in top][::-1]   # reversed → strongest sits at the top
    values = [p[1] for p in top][::-1]

    # Burgundy emphasis on the strongest pair, teal for the rest
    max_v = max(values) if values else 1
    colors = [PALETTE["burgundy"] if v == max_v else PALETTE["teal"] for v in values]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Pattern score: %{x:.0f}/100<extra></extra>",
    ))
    fig.update_layout(base_layout(
        title=f"Strongest hidden patterns · {len(work)} customers",
        height=460, showlegend=False,
        xaxis=dict(title="Pattern score (0 = no connection, 100 = perfectly linked)",
                   range=[0, max_v * 1.30]),
        yaxis=dict(title=""),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Build a small explanatory callout: top 2 pairs translated into
    # marketing actions, plus the transition to segmentation.
    if len(pairs) >= 2:
        top1_label, top1_score, top1_key = pairs[0]
        top2_label, top2_score, top2_key = pairs[1]
        def _meaning(k):
            return (pair_meanings.get(k)
                    or pair_meanings.get((k[1], k[0]))
                    or "Worth investigating with targeted campaigns.")
        bullet1 = f"<b>{top1_label}</b> ({top1_score:.0f}/100) — {_meaning(top1_key)}"
        bullet2 = f"<b>{top2_label}</b> ({top2_score:.0f}/100) — {_meaning(top2_key)}"

        body = (
            f"<b>What this means in plain English:</b><br>"
            f"&bull; {bullet1}<br>"
            f"&bull; {bullet2}<br><br>"
            f"Even the strongest pair scores only ~{int(round(top1_score))}/100 — "
            f"<b>no single customer attribute predicts everything.</b> That's the "
            f"whole reason this dashboard segments customers on the next tabs: "
            f"combining several signals at once (frequency, spend, occasion) "
            f"produces customer groups that single attributes never could."
        )
        st.markdown(callout("🧭", "How to read this chart", body),
                    unsafe_allow_html=True)

    # Academic transparency expander — what the 0–100 scale actually is.
    # Keeps the manager-friendly facade above, but lets a marker/professor see
    # the underlying statistics are sound.
    with st.expander("📐 What's the maths underneath? · for the academically curious"):
        st.markdown(f"""
The 0–100 *Pattern score* is **Cramér's V × 100**.

**Cramér's V** is a standard measure of association between two
categorical variables, derived from the chi-squared statistic of their
cross-tabulation. It is bounded between 0 and 1:

- **0** means the two variables are statistically independent
- **1** means perfect dependence (knowing one tells you the other exactly)

We multiply by 100 in this dashboard because *"27 out of 100"* reads as a
more concrete signal of strength than *"V = 0.27"* for a non-statistical
audience — but the underlying values are unchanged. Conventional
interpretation thresholds in applied statistics (Rea & Parker, 1992):

| Cramér's V | 0–100 scale | Conventional label |
|---|---|---|
| < 0.10 | < 10 | Negligible |
| 0.10 – 0.20 | 10 – 20 | Weak |
| 0.20 – 0.40 | 20 – 40 | Moderate |
| 0.40 – 0.60 | 40 – 60 | Relatively strong |
| > 0.60 | > 60 | Very strong |

Our top pair scores ~**27/100** (V ≈ 0.27), which lands solidly in the
*moderate* band — strong enough to inform campaign design, but not so
strong that a single attribute fully predicts another. That asymmetry is
exactly why segmenting on multiple signals (FM × Behavioral) is more
informative than slicing the base on one demographic alone.

The score is computed only from the rows currently in view (filter-aware).
        """)


def render_segments(df_f: pd.DataFrame) -> None:
    st.subheader("FM Segmentation — the revenue framework")
    st.caption(
        "Frequency × Monetary: customers classified by how often they visit and "
        "how much they spend per visit. A median split produces four actionable segments."
    )

    seg_summary = df_f.groupby("FM_segment").agg(
        customers=("ID", "count"),
        avg_ticket=("Ticket", "mean"),
        avg_visits=("monthly_visits", "mean"),
        revenue=("est_annual_revenue", "sum"),
    ).reset_index()
    seg_summary["share_customers"] = (
        seg_summary["customers"] / seg_summary["customers"].sum() * 100
    )
    total_rev = seg_summary["revenue"].sum() or 1
    seg_summary["share_revenue"] = seg_summary["revenue"] / total_rev * 100

    seg_order = ["Champions", "Loyal Regulars", "Occasion Splurgers", "Casual Visitors"]
    prio_class_map = {"High": "priority-high", "Medium": "priority-med", "Low": "priority-low"}
    cols = st.columns(4)

    for i, seg in enumerate(seg_order):
        row = seg_summary[seg_summary["FM_segment"] == seg]
        if len(row) == 0:
            with cols[i]:
                st.info(f"No **{seg}** in current filters.")
            continue
        row = row.iloc[0]
        meta = SEGMENT_META[seg]
        with cols[i]:
            st.markdown(f"""
            <div class='segment-card' style='border-top-color:{SEGMENT_COLORS[seg]};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <h4 style='margin:0;'>{seg}</h4>
                    <span class='{prio_class_map[meta["priority"]]}'>{meta["priority"]}</span>
                </div>
                <p style='font-size:12px; color:{PALETTE["charcoal"]}; margin:8px 0; min-height:50px;'>{meta["summary"]}</p>
                <hr style='margin:6px 0; border:none; border-top:1px solid #EEE;'>
                <p style='margin:4px 0; font-size:13px;'><b>{int(row["customers"])}</b> customers ({row["share_customers"]:.0f}%)</p>
                <p style='margin:4px 0; font-size:13px;'>Avg ticket: <b>€{row["avg_ticket"]:.0f}</b></p>
                <p style='margin:4px 0; font-size:13px;'>Visits/mo: <b>{row["avg_visits"]:.1f}</b></p>
                <p style='margin:4px 0; font-size:13px;'>Revenue share: <b>{row["share_revenue"]:.0f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(" ")

    f_med = df_f["monthly_visits"].median()
    m_med = df_f["Ticket"].median()
    rng_fm = np.random.default_rng(42)
    df_fm = df_f.copy()
    df_fm["visits_j"] = df_f["monthly_visits"] + rng_fm.uniform(-0.40, 0.40, len(df_f))
    df_fm["ticket_j"] = df_f["Ticket"]         + rng_fm.uniform(-0.80, 0.80, len(df_f))
    fig = px.scatter(
        df_fm, x="visits_j", y="ticket_j", color="FM_segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_data={"visits_j": False, "ticket_j": False,
                    "monthly_visits": True, "Ticket": True,
                    "Age": True, "Gender": True,
                    "Place to drink": True, "Additional products": True},
        labels={"visits_j": "Visits / month (jittered)", "ticket_j": "Avg ticket (€)"},
        opacity=0.70,
    )
    fig.add_hline(y=m_med, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.4,
                  annotation_text=f"Median ticket €{m_med:.0f}",
                  annotation_position="bottom right")
    fig.add_vline(x=f_med, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.4,
                  annotation_text=f"Median {f_med:.0f} visits/mo",
                  annotation_position="top left")
    fig.update_layout(base_layout(
        title="FM quadrant — every dot is one customer (jitter applied)",
        height=520,
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Product mix per segment**")
    prod = (df_f.groupby(["FM_segment", "Additional products"]).size()
                  .reset_index(name="customers"))
    fig = px.bar(prod, x="customers", y="FM_segment", color="Additional products",
                 orientation="h", color_discrete_sequence=SEQ,
                 category_orders={"FM_segment": seg_order})
    fig.update_layout(base_layout(title="Deli product mix by FM segment", height=380,
                                  barmode="stack", xaxis_title="Customers", yaxis_title=""))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Demographic mix per segment**")
    dc1, dc2 = st.columns(2)
    age_mix = pd.crosstab(df_f["FM_segment"], df_f["Age"], normalize="index") * 100
    fig = px.bar(
        age_mix.reset_index().melt(id_vars="FM_segment", var_name="Age", value_name="pct"),
        x="pct", y="FM_segment", color="Age", orientation="h",
        color_discrete_sequence=SEQ,
        category_orders={"FM_segment": seg_order, "Age": AGE_ORDER},
    )
    fig.update_layout(base_layout(title="Age mix per segment (%)", height=360,
                                  barmode="stack", xaxis_title="% of segment", yaxis_title=""))
    dc1.plotly_chart(fig, use_container_width=True)

    gen_mix = pd.crosstab(df_f["FM_segment"], df_f["Gender"], normalize="index") * 100
    fig = px.bar(
        gen_mix.reset_index().melt(id_vars="FM_segment", var_name="Gender", value_name="pct"),
        x="pct", y="FM_segment", color="Gender", orientation="h",
        color_discrete_sequence=[PALETTE["merlot"], PALETTE["rose"]],
        category_orders={"FM_segment": seg_order},
    )
    fig.update_layout(base_layout(title="Gender mix per segment (%)", height=360,
                                  barmode="stack", xaxis_title="% of segment", yaxis_title=""))
    dc2.plotly_chart(fig, use_container_width=True)


def _render_spend_tiers_view(df_f: pd.DataFrame) -> None:
    """Render the K-Prototypes spend-tier clustering view.

    Mixed-type clustering: Hamming distance on six categorical features
    (frequency, payment, occasion, deli, gender, age) + Euclidean on
    Ticket. Default K=3 → Entry / Core / Premium spend tiers. Falls back
    to K-Means on one-hot encoded features if the `kmodes` package is
    unavailable, with a visible warning to the user.
    """
    if KPROTOTYPES_OK:
        st.caption(
            "K-Prototypes is the proper algorithm for this mixed-type data: "
            "Hamming distance on the six categorical attributes, Euclidean on "
            "the single numeric attribute (Ticket). Default K=3 gives the "
            "**Entry / Core / Premium** spend tiers."
        )
    else:
        st.warning(
            "`kmodes` is not installed — falling back to K-Means on one-hot "
            "encoded features (distorts categorical distances but still useful "
            "as a directional view). For the proper algorithm: "
            "`pip install kmodes`."
        )

    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Number of spend tiers (K)", 2, 6, 3,
                      help="K=3 produces the canonical Entry / Core / Premium "
                           "framing used in the executive report.")
        st.info(f"Algorithm: **{'K-Prototypes' if KPROTOTYPES_OK else 'K-Means (fallback)'}**")

    # Elbow + silhouette diagnostics
    ks, inertias, sils = spend_tier_diagnostics(df_f)
    diag = make_subplots(specs=[[{"secondary_y": True}]])
    diag.add_trace(go.Scatter(x=ks, y=inertias, name="Inertia (elbow)",
                              line=dict(color=PALETTE["burgundy"], width=3),
                              mode="lines+markers"), secondary_y=False)
    diag.add_trace(go.Scatter(x=ks, y=sils, name="Silhouette",
                              line=dict(color=PALETTE["gold"], width=3, dash="dot"),
                              mode="lines+markers"), secondary_y=True)
    diag.update_layout(base_layout(title="K-selection diagnostics for spend tiers",
                                   height=340))
    diag.update_xaxes(title="K (clusters)")
    diag.update_yaxes(title="Inertia ↓ (tighter)", secondary_y=False)
    diag.update_yaxes(title="Silhouette ↑ (cleaner)", secondary_y=True)
    c2.plotly_chart(diag, use_container_width=True)

    # Fit at chosen K, name by spend, summarise
    labels = fit_clusters(df_f, k=k)
    df_c = df_f.copy()
    df_c["cluster_id"]   = labels
    df_c["cluster_name"] = df_c["cluster_id"].map(name_clusters_by_spend(df_c, "cluster_id"))

    csum = (df_c.groupby("cluster_name").agg(
                Customers=("ID", "count"),
                Avg_ticket=("Ticket", "mean"),
                Min_ticket=("Ticket", "min"),
                Max_ticket=("Ticket", "max"),
                Avg_visits=("monthly_visits", "mean"),
                Annual_rev=("est_annual_revenue", "sum"),
            )
            .reset_index()
            .sort_values("Avg_ticket"))
    csum["Avg_ticket"] = csum["Avg_ticket"].round(1)
    csum["Avg_visits"] = csum["Avg_visits"].round(1)
    csum["Annual_rev"] = (csum["Annual_rev"] / 1000).round(0).astype(int).astype(str) + "K"
    csum.columns = ["Tier", "Customers", "Avg ticket (€)", "Min ticket",
                    "Max ticket", "Avg visits/mo", "Annual revenue (€)"]
    st.markdown("**Spend-tier summary**")
    st.dataframe(csum, hide_index=True, use_container_width=True)

    # Scatter + size pie side-by-side
    cv1, cv2 = st.columns(2)
    fig = px.scatter(
        df_c, x="monthly_visits", y="Ticket", color="cluster_name",
        color_discrete_sequence=SEQ,
        hover_data=["Age", "Gender", "Additional products", "Place to drink"],
        labels={"monthly_visits": "Visits / month", "Ticket": "Avg ticket (€)"},
        opacity=0.75,
    )
    fig.update_layout(base_layout(
        title="Spend tiers projected onto Frequency × Ticket", height=420))
    cv1.plotly_chart(fig, use_container_width=True)

    sizes = df_c["cluster_name"].value_counts().reset_index()
    sizes.columns = ["Tier", "Customers"]
    fig = px.pie(sizes, values="Customers", names="Tier", hole=0.5,
                 color_discrete_sequence=SEQ)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Spend-tier sizes", height=420,
                                  showlegend=False))
    cv2.plotly_chart(fig, use_container_width=True)

    # FM × spend-tier cross-tab
    st.markdown("**Do the spend tiers validate the FM revenue segments?**")
    fm_order = ["Champions", "Loyal Regulars", "Occasion Splurgers", "Casual Visitors"]
    ct = pd.crosstab(df_c["FM_segment"], df_c["cluster_name"])
    ct = ct.reindex(index=fm_order, fill_value=0)
    fig = px.imshow(ct, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["burgundy"]]],
                    labels=dict(x="Spend tier", y="FM segment", color="Customers"))
    fig.update_layout(base_layout(title="FM segment × spend tier cross-tab", height=340))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(callout(
        "🧭", "How to read this view",
        "<b>Spend tiers</b> are a value-anchored sanity check on the FM segmentation. "
        "If a tier maps cleanly onto Champions+Occasion Splurgers (the high-monetary "
        "FM quadrants), spend is doing the heavy lifting in FM and the framework "
        "is solid. A messy cross-tab — like the one above — means FM combines spend "
        "AND frequency, while the spend-tier model only sees spend. <br><br>"
        "<i>Methodological note:</i> on this dataset K-Prototypes mostly "
        "rediscovers ticket-anchored tiers, because the categorical features are "
        "near-uniformly distributed across customers and Ticket is the dominant "
        "differentiating signal. The behavioural K-Means lens (toggle above) "
        "was therefore added to find lifestyle archetypes that are independent "
        "of how much each customer pays per visit. "
        "We keep both because they answer different questions."
    ), unsafe_allow_html=True)


def render_behavioral(df_f: pd.DataFrame) -> None:
    """Render the ML Clustering tab — two complementary unsupervised lenses.

    A radio toggle at the top selects between:
      - Behavioural archetypes (K-Means K=4 on frequency × sociality),
        which answer *how* customers buy and are independent of ticket
        value (default view).
      - Spend tiers (K-Prototypes on the full mixed-type feature set),
        which answer *how much* customers buy and serve as a
        value-anchored sanity check on the FM segmentation.

    Both lenses coexist because they answer different business questions
    and the manager benefits from holding both in mind.
    """
    st.subheader("ML Clustering — two complementary lenses on the customer base")
    st.caption(
        "Two unsupervised approaches on the same survey data. "
        "**Behavioral archetypes** capture *how* customers buy (frequency × sociality, "
        "ignoring ticket); **spend tiers** capture *how much* they buy (clusters across "
        "the full mixed-type feature set, anchored by ticket). Toggle below."
    )

    view = st.radio(
        "Clustering lens",
        ["🧑‍🤝‍🧑 Behavioral archetypes (K-Means · freq × sociality)",
         "💰 Spend tiers (K-Prototypes · mixed feature set)"],
        horizontal=True, label_visibility="collapsed",
    )
    if view.startswith("💰"):
        _render_spend_tiers_view(df_f)
        return

    # ===== Behavioural archetypes view (default) ===========================
    labels = fit_behavioral_clusters(df_f)
    df_b = df_f.copy()
    df_b["beh_id"] = labels
    name_map = name_behavioral_clusters(df_b, "beh_id")
    df_b["beh_segment"] = df_b["beh_id"].map(name_map)

    beh_order = ["Social Regulars", "Daily Home Drinkers",
                 "Occasion Celebrants", "Casual Home Drinkers"]
    prio_class_map = {"High": "priority-high", "Medium": "priority-med", "Low": "priority-low"}

    # ---- Segment cards ----
    bsummary = df_b.groupby("beh_segment").agg(
        customers=("ID", "count"),
        avg_ticket=("Ticket", "mean"),
        avg_visits=("monthly_visits", "mean"),
        avg_social=("social_score", "mean"),
        revenue=("est_annual_revenue", "sum"),
    ).reset_index()
    bsummary["rev_share"] = bsummary["revenue"] / (bsummary["revenue"].sum() or 1) * 100

    cols = st.columns(4)
    for i, seg in enumerate(beh_order):
        row = bsummary[bsummary["beh_segment"] == seg]
        if len(row) == 0:
            with cols[i]:
                st.info(f"No **{seg}** in current filters.")
            continue
        row = row.iloc[0]
        meta = BEHAVIORAL_META[seg]
        with cols[i]:
            st.markdown(f"""
            <div class='segment-card' style='border-top-color:{BEHAVIORAL_COLORS[seg]};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <h4 style='margin:0; font-size:14px;'>{seg}</h4>
                    <span class='{prio_class_map[meta["priority"]]}'>{meta["priority"]}</span>
                </div>
                <p style='font-size:11px; color:{PALETTE["charcoal"]}; margin:8px 0; min-height:44px;'>{meta["summary"]}</p>
                <hr style='margin:6px 0; border:none; border-top:1px solid #EEE;'>
                <p style='margin:3px 0; font-size:12px;'><b>{int(row["customers"])}</b> customers</p>
                <p style='margin:3px 0; font-size:12px;'>Avg ticket: <b>€{row["avg_ticket"]:.0f}</b></p>
                <p style='margin:3px 0; font-size:12px;'>Visits/mo: <b>{row["avg_visits"]:.1f}</b></p>
                <p style='margin:3px 0; font-size:12px;'>Sociality: <b>{row["avg_social"]:.1f} / 7</b></p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(" ")

    # ---- Scatter: frequency × sociality (jittered — both axes are discrete) ----
    freq_mid   = df_b["monthly_visits"].median()
    social_mid = df_b["social_score"].median()

    rng = np.random.default_rng(42)
    df_plot = df_b.copy()
    df_plot["visits_j"]  = df_b["monthly_visits"] + rng.uniform(-0.40, 0.40, len(df_b))
    df_plot["social_j"]  = df_b["social_score"]   + rng.uniform(-0.25, 0.25, len(df_b))

    fig = px.scatter(
        df_plot, x="visits_j", y="social_j",
        color="beh_segment", color_discrete_map=BEHAVIORAL_COLORS,
        hover_data={"visits_j": False, "social_j": False,
                    "monthly_visits": True, "social_score": True,
                    "Age": True, "Gender": True, "Ticket": True,
                    "Additional products": True},
        labels={"visits_j": "Visits / month (jittered)", "social_j": "Sociality (1=home, 7=parties)"},
        opacity=0.65,
    )
    fig.add_hline(y=social_mid, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.35,
                  annotation_text="Median sociality", annotation_position="bottom right")
    fig.add_vline(x=freq_mid, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.35,
                  annotation_text=f"Median {freq_mid:.0f} visits/mo", annotation_position="top left")
    fig.update_layout(base_layout(
        title="Behavioral map — frequency × sociality · every dot is one customer (jitter applied)",
        height=500,
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(callout(
        "💡", "Why ticket is NOT on this chart",
        "Ticket price is deliberately excluded. A Kruskal-Wallis test confirms it "
        "is <b>not</b> significantly different across these four groups (p ≈ 0.58). "
        "That's the whole point: FM tells you who's worth the most, "
        "behavioral tells you <i>how</i> to reach them."
    ), unsafe_allow_html=True)

    # ---- Diagnostics (collapsed) ----
    with st.expander("📐 K-selection diagnostics — elbow & silhouette"):
        ks, inertias, sils = behavioral_diagnostics(df_f)
        diag = make_subplots(specs=[[{"secondary_y": True}]])
        diag.add_trace(go.Scatter(x=ks, y=inertias, name="Inertia (elbow)",
                                  line=dict(color=PALETTE["merlot"], width=3),
                                  mode="lines+markers"), secondary_y=False)
        diag.add_trace(go.Scatter(x=ks, y=sils, name="Silhouette",
                                  line=dict(color=PALETTE["teal"], width=3, dash="dot"),
                                  mode="lines+markers"), secondary_y=True)
        diag.add_vline(x=4, line_dash="dash", line_color=PALETTE["rose"], opacity=0.7,
                       annotation_text="Chosen K=4", annotation_position="top right")
        diag.update_layout(base_layout(title="K selection — behavioral clustering", height=340))
        diag.update_xaxes(title="K")
        diag.update_yaxes(title="Inertia ↓", secondary_y=False)
        diag.update_yaxes(title="Silhouette ↑", secondary_y=True)
        st.plotly_chart(diag, use_container_width=True)

    # ---- Cross-tab: FM × Behavioral ----
    st.markdown("**How do the two segmentations overlap?**")
    st.caption(
        "A dense diagonal would mean they measure the same thing. "
        "A scattered heatmap means they're genuinely complementary."
    )
    fm_order = ["Champions", "Loyal Regulars", "Occasion Splurgers", "Casual Visitors"]
    ct = pd.crosstab(df_b["FM_segment"], df_b["beh_segment"])
    ct = ct.reindex(index=fm_order, columns=beh_order, fill_value=0)
    fig = px.imshow(
        ct, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["merlot"]]],
        labels=dict(x="Behavioral segment", y="FM segment", color="Customers"),
    )
    fig.update_layout(base_layout(title="FM segment × Behavioral segment — customer overlap",
                                  height=340))
    st.plotly_chart(fig, use_container_width=True)

    # ---- Deli mix per behavioral segment ----
    st.markdown("**Deli product mix per behavioral segment**")
    prod = (df_b.groupby(["beh_segment", "Additional products"]).size()
                 .reset_index(name="customers"))
    fig = px.bar(prod, x="customers", y="beh_segment", color="Additional products",
                 orientation="h", color_discrete_sequence=SEQ,
                 category_orders={"beh_segment": beh_order})
    fig.update_layout(base_layout(title="What each behavioral group buys from the deli",
                                  height=380, barmode="stack",
                                  xaxis_title="Customers", yaxis_title=""))
    st.plotly_chart(fig, use_container_width=True)

    # ---- Gender split ----
    gen_mix = pd.crosstab(df_b["beh_segment"], df_b["Gender"], normalize="index") * 100
    fig = px.bar(
        gen_mix.reset_index().melt(id_vars="beh_segment", var_name="Gender", value_name="pct"),
        x="pct", y="beh_segment", color="Gender", orientation="h",
        color_discrete_sequence=[PALETTE["merlot"], PALETTE["rose"]],
        category_orders={"beh_segment": beh_order},
    )
    fig.update_layout(base_layout(title="Gender mix per behavioral segment (%)",
                                  height=340, barmode="stack",
                                  xaxis_title="% of segment", yaxis_title=""))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(callout(
        "📌", "Daily Home Drinkers — the hidden opportunity",
        "62 % female, near-daily visits, average ticket identical to the base — "
        "yet almost entirely absent from the deli. They're buying wine on autopilot. "
        "A shelf-talker pairing suggestion and a midweek 'tonight's pairing' message "
        "could shift this without any discount."
    ), unsafe_allow_html=True)


def render_products(df_f: pd.DataFrame) -> None:
    st.subheader("Customer Behaviour")
    st.markdown(
        "<p style='font-size:16px; color:#2B2118; line-height:1.7;'>"
        "Wine drives footfall, but the deli is where margin and loyalty are built. "
        "Customers already buy food alongside their wine — the opportunity is in "
        "making the right products visible to the right people."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ── Shared data ────────────────────────────────────────────────────────
    seg_order = ["Champions", "Loyal Regulars", "Occasion Splurgers", "Casual Visitors"]
    PREMIUM   = {"Cheese", "Salmon", "Spanish ham"}
    ENTRY     = {"Olives", "Chocolate", "Candies", "Cookies", "Nuts"}

    df_f = df_f.copy()
    df_f["deli_tier"] = df_f["Additional products"].apply(
        lambda x: "Premium" if x in PREMIUM else ("Entry" if x in ENTRY else "Other")
    )
    prod = (
        df_f.groupby("Additional products")
        .agg(n=("ID", "count"))
        .reset_index()
    )
    prod["penetration_pct"] = prod["n"] / len(df_f) * 100
    prod["tier"] = prod["Additional products"].apply(
        lambda x: "Premium" if x in PREMIUM else "Entry"
    )

    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 1 — Pie + classification cards
    # ═══════════════════════════════════════════════════════════════════════
    col_pie, col_key = st.columns([1, 1.4], gap="large")

    with col_pie:
        tier_counts = df_f["deli_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Customers"]
        fig_pie = px.pie(
            tier_counts, values="Customers", names="Tier",
            hole=0.58, color="Tier",
            color_discrete_map={"Premium": PALETTE["burgundy"], "Entry": PALETTE["gold"]},
        )
        fig_pie.update_traces(
            textinfo="percent+label",
            textfont=dict(size=15, family="Georgia, serif"),
            pull=[0.03, 0.03],
        )
        fig_pie.update_layout(
            base_layout(title="Premium vs Entry deli split", height=360, showlegend=False),
            annotations=[dict(
                text=f"<b>{len(df_f)}</b><br>customers",
                x=0.5, y=0.5,
                font=dict(size=15, color=PALETTE["charcoal"]),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_key:
        prem_n    = int((df_f["deli_tier"] == "Premium").sum())
        entry_n   = int((df_f["deli_tier"] == "Entry").sum())
        prem_pct  = prem_n  / len(df_f) * 100
        entry_pct = entry_n / len(df_f) * 100

        st.markdown(
            f"""
            <div style='margin-top:20px; display:flex; flex-direction:column; gap:12px;'>
              <div style='background:white; border-radius:12px; border:1px solid #E2D8D6;
                          border-left:5px solid {PALETTE["burgundy"]}; padding:18px 22px;'>
                <p style='margin:0 0 4px; font-size:13px; color:#4A3F35;
                          font-weight:600; letter-spacing:.5px;'>PREMIUM DELI</p>
                <p style='margin:0 0 8px; font-size:16px; font-weight:700;
                          color:{PALETTE["charcoal"]};'>Cheese · Salmon · Spanish ham</p>
                <p style='margin:0; font-size:14px; color:{PALETTE["charcoal"]};'>
                  <b>{prem_n} customers ({prem_pct:.0f}%)</b> currently choose
                  a premium deli item alongside their wine.
                </p>
              </div>
              <div style='background:white; border-radius:12px; border:1px solid #E2D8D6;
                          border-left:5px solid {PALETTE["gold"]}; padding:18px 22px;'>
                <p style='margin:0 0 4px; font-size:13px; color:#4A3F35;
                          font-weight:600; letter-spacing:.5px;'>ENTRY DELI</p>
                <p style='margin:0 0 8px; font-size:16px; font-weight:700;
                          color:{PALETTE["charcoal"]};'>
                  Olives · Chocolate · Candies · Cookies · Nuts</p>
                <p style='margin:0; font-size:14px; color:{PALETTE["charcoal"]};'>
                  <b>{entry_n} customers ({entry_pct:.0f}%)</b> choose
                  an entry-level deli item alongside their wine.
                </p>
              </div>
              <div style='background:{PALETTE["ivory"]}; border-radius:12px;
                          border:1px solid #E2D8D6; padding:18px 22px;'>
                <p style='margin:0 0 6px; font-size:15px; font-weight:700;
                          color:{PALETTE["charcoal"]};'>What this tells us</p>
                <p style='margin:0; font-size:14px; color:{PALETTE["charcoal"]}; line-height:1.7;'>
                  Nearly half the store's customers already choose premium deli —
                  without being prompted. The opportunity is not in converting
                  entry customers into premium ones. It is in making premium
                  products <b>more visible</b> so that the other half
                  encounters them naturally.
                </p>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:48px;'></div>", unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 2 — Product reach bar
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<p style='font-size:19px; font-weight:700; color:{PALETTE['charcoal']};'>"
        "Which products are customers actually buying?</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:15px; color:#4A3F35; margin-bottom:4px;'>"
        "Every customer surveyed buys exactly one deli product per visit. "
        "The bars show how many of the 404 surveyed customers chose each product.</p>",
        unsafe_allow_html=True,
    )
    by_reach = prod.sort_values("n", ascending=False).copy()

    fig_reach = go.Figure()
    # Split into two traces so Plotly can show a legend
    for tier_name, tier_color in [("Premium deli", PALETTE["burgundy"]),
                                   ("Entry deli",   PALETTE["gold"])]:
        mask = by_reach["tier"] == tier_name.split()[0]
        sub  = by_reach[mask]
        fig_reach.add_trace(go.Bar(
            name=tier_name,
            y=sub["Additional products"],
            x=sub["n"],
            orientation="h",
            marker=dict(color=tier_color, opacity=0.88),
            text=sub["n"].apply(lambda v: f"{v} customers"),
            textposition="outside",
            textfont=dict(size=13, color=PALETTE["charcoal"]),
            width=0.55,
            hovertemplate="<b>%{y}</b><br>%{x} of 404 surveyed customers<extra></extra>",
        ))
    fig_reach.update_layout(
        base_layout(
            title="Surveyed customers who chose each deli product (n=404)",
            height=400,
        ),
        xaxis=dict(
            range=[0, 150],
            title="Number of customers (out of 404 surveyed)",
            title_font=dict(size=14),
            tickfont=dict(size=13),
            dtick=25,
        ),
        yaxis=dict(tickfont=dict(size=15), autorange="reversed"),
        barmode="overlay",
        bargap=0.35,
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            font=dict(size=13),
        ),
        margin=dict(t=60, b=50, l=120, r=140),
    )
    st.plotly_chart(fig_reach, use_container_width=True)

    st.markdown(callout(
        "", "The reach problem",
        "Olives was chosen by 106 customers — Nuts by just 8. "
        "That is a 13x difference across the same product range, "
        "in the same store, bought by the same customers. "
        "The lower products on this chart are not disliked — "
        "they are simply not being seen or suggested."
    ), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:44px;'></div>", unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 3 — Occasion drives ticket and premium choice
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:44px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:19px; font-weight:700; color:{PALETTE['charcoal']};'>"
        "Where customers drink determines what they spend</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:15px; color:#4A3F35; margin-bottom:20px;'>"
        "The occasion a customer is buying for is one of the strongest predictors "
        "of their basket size — more so than age or gender. "
        "Customers buying for a restaurant or holiday spend significantly more "
        "and choose premium deli at a higher rate.</p>",
        unsafe_allow_html=True,
    )

    occ_stats = (
        df_f.groupby("Place to drink")
        .agg(
            customers   = ("ID",       "count"),
            avg_ticket  = ("Ticket",   "mean"),
            pct_premium = ("deli_tier",
                           lambda x: (x == "Premium").mean() * 100),
        )
        .reset_index()
        .sort_values("avg_ticket", ascending=False)
    )

    avg_ticket_all = df_f["Ticket"].mean()

    fig_occ = go.Figure()
    fig_occ.add_trace(go.Bar(
        y=occ_stats["Place to drink"],
        x=occ_stats["avg_ticket"],
        orientation="h",
        marker=dict(color=PALETTE["burgundy"], opacity=0.82),
        text=[
            f"  €{row['avg_ticket']:.0f} avg · {row['pct_premium']:.0f}% choose premium deli"
            for _, row in occ_stats.iterrows()
        ],
        textposition="inside",
        textfont=dict(size=13, color="white"),
        insidetextanchor="start",
        width=0.55,
        showlegend=False,
        customdata=occ_stats[["pct_premium", "customers"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg ticket: €%{x:.0f}<br>"
            "Premium deli: %{customdata[0]:.0f}%<br>"
            "Customers: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
    ))

    fig_occ.add_vline(
        x=avg_ticket_all,
        line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.25,
        annotation_text=f"Store avg €{avg_ticket_all:.0f}",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color=PALETTE["charcoal"]),
    )

    fig_occ.update_layout(
        base_layout(
            title="Average ticket and premium deli choice by drinking occasion",
            height=420,
        ),
        xaxis=dict(
            title="Average ticket (€)",
            title_font=dict(size=13),
            tickfont=dict(size=12),
            tickprefix="€",
            range=[30, 95],
        ),
        yaxis=dict(tickfont=dict(size=14), autorange="reversed"),
        showlegend=False,
        margin=dict(t=60, b=50, l=150, r=20),
    )
    st.plotly_chart(fig_occ, use_container_width=True)

    rest_ticket = occ_stats[occ_stats["Place to drink"]=="Restaurant"]["avg_ticket"].values[0]
    party_ticket = occ_stats[occ_stats["Place to drink"]=="Parties"]["avg_ticket"].values[0]
    rest_prem   = occ_stats[occ_stats["Place to drink"]=="Restaurant"]["pct_premium"].values[0]
    holiday_ticket = occ_stats[occ_stats["Place to drink"]=="On holidays"]["avg_ticket"].values[0]

    st.markdown(callout(
        "", "Occasion is the clearest signal",
        f"Restaurant customers spend <b>€{rest_ticket:.0f} on average</b> — "
        f"€{rest_ticket - party_ticket:.0f} more than party buyers — "
        f"and <b>{rest_prem:.0f}% of them choose premium deli</b>. "
        f"Holiday buyers follow at €{holiday_ticket:.0f}. "
        f"These customers are already in a premium mindset when they walk in. "
        f"A targeted display for restaurant and holiday occasions — "
        f"pairing premium wine with premium deli — would meet them exactly where they are."
    ), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 4 — Age predicts premium preference (linear trend)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:44px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:19px; font-weight:700; color:{PALETTE['charcoal']};'>"
        "Older customers consistently choose premium deli</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:15px; color:#4A3F35; margin-bottom:20px;'>"
        "Premium deli adoption rises steadily with age. "
        "Over-50s choose premium products at nearly double the rate of the youngest group. "
        "This is a direct guide for how to staff and stock for different customer profiles.</p>",
        unsafe_allow_html=True,
    )

    AGE_ORDER = ["23 to 30", "31 to 40", "41 to 50", "more than 50"]
    age_labels= ["23–30", "31–40", "41–50", "50+"]

    age_prem = []
    age_entry= []
    age_n    = []
    for age in AGE_ORDER:
        sub = df_f[df_f["Age"] == age]
        age_n.append(len(sub))
        age_prem.append((sub["deli_tier"]=="Premium").mean()*100)
        age_entry.append((sub["deli_tier"]=="Entry").mean()*100)

    col_age1, col_age2 = st.columns(2, gap="large")

    with col_age1:
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            name="Premium deli",
            x=age_labels,
            y=age_prem,
            marker_color=PALETTE["burgundy"],
            opacity=0.88,
            text=[f"{v:.0f}%" for v in age_prem],
            textposition="outside",
            textfont=dict(size=13),
            width=0.45,
        ))
        fig_age.add_trace(go.Bar(
            name="Entry deli",
            x=age_labels,
            y=age_entry,
            marker_color=PALETTE["gold"],
            opacity=0.88,
            text=[f"{v:.0f}%" for v in age_entry],
            textposition="outside",
            textfont=dict(size=13),
            width=0.45,
        ))
        fig_age.update_layout(
            base_layout(
                title="Premium vs Entry deli choice by age group",
                height=380,
                barmode="group",
            ),
            xaxis=dict(tickfont=dict(size=14), title="Age group",
                       title_font=dict(size=13)),
            yaxis=dict(range=[0, 75], ticksuffix="%",
                       title="% of age group", title_font=dict(size=13),
                       tickfont=dict(size=12)),
            legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            font=dict(size=13),
        ),
            margin=dict(t=50, b=50, l=60, r=120),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_age2:
        age_ticket = [df_f[df_f["Age"]==a]["Ticket"].mean() for a in AGE_ORDER]
        age_counts  = [df_f[df_f["Age"]==a]["Ticket"].count() for a in AGE_ORDER]

        fig_age_tkt = go.Figure()
        fig_age_tkt.add_trace(go.Scatter(
            x=age_labels,
            y=age_ticket,
            mode="lines+markers+text",
            line=dict(color=PALETTE["burgundy"], width=3),
            marker=dict(size=12, color=PALETTE["burgundy"],
                        line=dict(width=2, color="white")),
            text=[f"€{v:.0f}" for v in age_ticket],
            textposition="top center",
            textfont=dict(size=13, color=PALETTE["charcoal"]),
            showlegend=False,
        ))
        fig_age_tkt.update_layout(
            base_layout(
                title="Average ticket by age group",
                height=380,
            ),
            xaxis=dict(tickfont=dict(size=14), title="Age group",
                       title_font=dict(size=13)),
            yaxis=dict(range=[50, 62], tickprefix="€",
                       title="Average ticket (€)", title_font=dict(size=13),
                       tickfont=dict(size=12)),
            margin=dict(t=60, b=50, l=60, r=20),
        )
        st.plotly_chart(fig_age_tkt, use_container_width=True)

    youngest_prem = age_prem[0]
    oldest_prem   = age_prem[-1]
    st.markdown(callout(
        "", "Age is a reliable predictor of premium preference",
        f"Premium deli choice rises from <b>{youngest_prem:.0f}%</b> among 23–30 year olds "
        f"to <b>{oldest_prem:.0f}%</b> among over-50s — a consistent increase at every age band. "
        f"This gives the store a practical guide: when an older customer comes in, "
        f"a premium deli suggestion is more likely to land. "
        f"Staff awareness of this pattern costs nothing to implement."
    ), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 5 — The loyalty blind spot: cash dominates every segment
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:44px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:19px; font-weight:700; color:{PALETTE['charcoal']};'>"
        "64% of customers are invisible to the store</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:15px; color:#4A3F35; margin-bottom:20px;'>"
        "Cash purchases leave no record. The store cannot identify who its best "
        "customers are, track whether they are returning, or reach them with any offer. "
        "This is not a marginal issue — it affects every segment, including the most valuable ones.</p>",
        unsafe_allow_html=True,
    )

    seg_order_pay = ["Champions","Loyal Regulars","Occasion Splurgers","Casual Visitors"]
    pay_data = (
        df_f.groupby(["FM_segment","Payment mode"])
        .size().reset_index(name="n")
    )
    pay_totals = df_f.groupby("FM_segment").size().reset_index(name="total")
    pay_data   = pay_data.merge(pay_totals, on="FM_segment")
    pay_data["pct"] = pay_data["n"] / pay_data["total"] * 100

    pay_colors = {
        "Cash":        PALETTE["burgundy"],
        "Credit card": PALETTE["gold"],
        "Debit card":  "#B0A09A",
    }
    fig_pay = go.Figure()
    for mode in ["Cash", "Credit card", "Debit card"]:
        sub  = pay_data[pay_data["Payment mode"]==mode].set_index("FM_segment")
        vals = [sub.loc[s,"pct"] if s in sub.index else 0 for s in seg_order_pay]
        fig_pay.add_trace(go.Bar(
            name=mode,
            x=seg_order_pay,
            y=vals,
            marker_color=pay_colors[mode],
            opacity=0.88,
            text=[f"{v:.0f}%" for v in vals],
            textposition="inside",
            textfont=dict(size=13, color="white"),
            width=0.55,
        ))
    fig_pay.update_layout(
        base_layout(
            title="Payment method by segment — share of customers",
            height=400,
            barmode="stack",
        ),
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(
            range=[0, 105],
            ticksuffix="%",
            title="% of segment",
            title_font=dict(size=13),
            tickfont=dict(size=12),
        ),
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.02,
            font=dict(size=14),
        ),
        margin=dict(t=50, b=50, l=60, r=130),
    )
    st.plotly_chart(fig_pay, use_container_width=True)

    champ_cash_pct = pay_data[
        (pay_data["FM_segment"]=="Champions") &
        (pay_data["Payment mode"]=="Cash")
    ]["pct"].values[0]

    st.markdown(callout(
        "", "The loyalty blind spot",
        f"<b>{champ_cash_pct:.0f}% of Champions — the store's highest-frequency customers "
        f"— pay in cash</b>, leaving no purchase trail. "
        f"The store cannot know how often they visit, what they buy over time, "
        f"or whether they are at risk of leaving. "
        f"Cash dominance is consistent across all segments, which means "
        f"any loyalty programme, targeted offer, or personalised service "
        f"is impossible to deliver without first solving the data capture problem. "
        f"A simple stamp card or app-based loyalty scheme would begin to change this "
        f"without requiring customers to change how they pay."
    ), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # KEY TAKEAWAYS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:52px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:22px; font-weight:700; color:{PALETTE['burgundy']};'>"
        "Key Takeaways</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:15px; color:#4A3F35; margin-bottom:24px;'>"
        "Three findings from this analysis that translate directly into action.</p>",
        unsafe_allow_html=True,
    )

    takeaways = [
        (
            PALETTE["burgundy"],
            "01",
            "Target the occasion, not just the customer",
            f"Restaurant and holiday buyers spend €{rest_ticket:.0f} and €{holiday_ticket:.0f} "
            f"on average — the highest in the store. They also choose premium deli at the highest rate. "
            f"A dedicated premium display for these occasions would require no new stock "
            f"and no new customers — just better placement for the ones already walking in.",
        ),
        (
            PALETTE["gold"],
            "02",
            "Older customers are the premium deli audience",
            f"Premium deli choice rises from {youngest_prem:.0f}% at age 23–30 to "
            f"{oldest_prem:.0f}% at over 50, consistently at every age band. "
            f"Staff can use age as a simple, reliable signal for when a premium "
            f"suggestion is most likely to resonate — no data system required.",
        ),
        (
            PALETTE["merlot"],
            "03",
            "The store cannot see its best customers",
            f"{champ_cash_pct:.0f}% of Champions pay cash. "
            f"Without a loyalty mechanism, the store cannot identify who they are, "
            f"track retention, or personalise any offer. "
            f"Everything else in this analysis — better stocking, smarter suggestions, "
            f"targeted promotions — depends on first solving this data capture gap.",
        ),
    ]

    for color, number, title, body in takeaways:
        st.markdown(
            f"""
            <div style='background:white; border-radius:12px; border:1px solid #E2D8D6;
                        border-left:6px solid {color}; padding:24px 28px;
                        margin-bottom:16px; display:flex; gap:24px; align-items:flex-start;'>
              <div style='flex-shrink:0;'>
                <p style='margin:0; font-size:32px; font-weight:700;
                           color:{color}; opacity:0.25; line-height:1;'>{number}</p>
              </div>
              <div>
                <p style='margin:0 0 8px; font-size:17px; font-weight:700;
                           color:{PALETTE["charcoal"]};'>{title}</p>
                <p style='margin:0; font-size:15px; color:#4A3F35;
                           line-height:1.7;'>{body}</p>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

def render_actions(df_f: pd.DataFrame) -> None:
    st.subheader("Strategic Action Plan")
    st.caption("Concrete marketing playbook per segment — your prioritised to-do list.")

    lens = st.radio("Segmentation lens", ["FM (revenue)", "Behavioral (how they buy)"],
                    horizontal=True)

    if lens == "FM (revenue)":
        meta_dict = SEGMENT_META
        seg_col = "FM_segment"
        color_map = SEGMENT_COLORS
    else:
        labels = fit_behavioral_clusters(df_f)
        df_tmp = df_f.copy()
        df_tmp["beh_id"] = labels
        df_tmp["beh_segment"] = df_tmp["beh_id"].map(name_behavioral_clusters(df_tmp, "beh_id"))
        df_f = df_tmp
        meta_dict = BEHAVIORAL_META
        seg_col = "beh_segment"
        color_map = BEHAVIORAL_COLORS

    selected = st.multiselect(
        "Filter by segment",
        options=list(meta_dict.keys()),
        default=list(meta_dict.keys()),
    )

    rev_by_seg = df_f.groupby(seg_col)["est_annual_revenue"].sum().to_dict()
    prio_rank = {"High": 0, "Medium": 1, "Low": 2}
    ordered = sorted(
        selected,
        key=lambda s: (prio_rank[meta_dict[s]["priority"]], -rev_by_seg.get(s, 0)),
    )
    prio_class_map = {"High": "priority-high", "Medium": "priority-med", "Low": "priority-low"}

    for seg in ordered:
        meta = meta_dict[seg]
        n = int((df_f[seg_col] == seg).sum())
        rev = df_f.loc[df_f[seg_col] == seg, "est_annual_revenue"].sum()
        actions_html = "".join([f"<li style='margin-bottom:6px;'>{a}</li>"
                                for a in meta["actions"]])
        profile_html = (f"<p style='font-size:12px; color:{PALETTE['charcoal']}; "
                        f"font-style:italic; margin:4px 0;'>{meta.get('profile', '')}</p>"
                        if meta.get("profile") else "")
        st.markdown(f"""
        <div class='segment-card' style='border-top-color:{color_map[seg]};'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h3 style='margin:0;'>{seg}</h3>
                <span class='{prio_class_map[meta["priority"]]}'>Priority: {meta["priority"]}</span>
            </div>
            <p style='color:{PALETTE["charcoal"]}; font-style:italic; margin:8px 0;'>{meta["summary"]}</p>
            {profile_html}
            <p style='color:{PALETTE["charcoal"]}; margin:4px 0;'>
                <b>{n}</b> customers · Estimated revenue: <b>€{rev/1000:.1f}K / year</b>
            </p>
            <hr style='border:none; border-top:1px solid #EEE; margin:10px 0;'>
            <strong style='color:{color_map[seg]};'>Recommended actions</strong>
            <ul style='color:{PALETTE["charcoal"]}; margin-top:8px; padding-left:20px;'>{actions_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background: {PALETTE["merlot"]}; color: white; padding: 18px 22px;
                border-radius: 8px; margin-top: 14px;'>
        <strong>🍷 Cross-cutting priority</strong><br>
        Address the delicatessen perception gap. Olives dominate the deli mix and premium
        adoption is uneven across all segments. A simple <b>wine + deli pairing display
        at eye level</b> and staff trained to suggest one product match per purchase could
        shift the average basket within one quarter and reposition the store away from
        "wine-only".
    </div>
    """, unsafe_allow_html=True)

    st.markdown(salmond_footnote(), unsafe_allow_html=True)


def render_explorer(df_f: pd.DataFrame) -> None:
    st.subheader("Customer Explorer")
    st.caption(
        "Search, sort, and filter individual customers. Build personalised outreach lists "
        "(e.g. all Champions aged 31–40 who pay in cash) and export as CSV."
    )

    e1, e2 = st.columns(2)
    seg_filter = e1.multiselect(
        "FM Segment", options=df_f["FM_segment"].unique().tolist(),
        default=df_f["FM_segment"].unique().tolist(),
    )
    product_filter = e2.multiselect(
        "Deli product", options=df_f["Additional products"].unique().tolist(),
        default=df_f["Additional products"].unique().tolist(),
    )

    df_x = df_f[
        df_f["FM_segment"].isin(seg_filter)
        & df_f["Additional products"].isin(product_filter)
    ].copy()

    if len(df_x) == 0:
        st.info("No customers match the current selection.")
        return

    # Display copy with 'Salmond' marked — see footnote below the table
    df_x = with_salmond_marker(df_x)
    df_show = df_x[[
        "ID", "Gender", "Age", "Education",
        "Wine frequency consumption", "Place to drink", "Additional products",
        "Payment mode", "Ticket", "monthly_visits", "est_annual_revenue", "FM_segment",
    ]].rename(columns={
        "Wine frequency consumption": "Frequency",
        "Place to drink":             "Occasion",
        "Additional products":        "Deli item",
        "Payment mode":               "Payment",
        "monthly_visits":             "Visits/mo",
        "est_annual_revenue":         "Est. annual rev (€)",
        "FM_segment":                 "FM Segment",
    })
    df_show["Est. annual rev (€)"] = df_show["Est. annual rev (€)"].round(0).astype(int)

    st.dataframe(df_show, use_container_width=True, hide_index=True, height=520)

    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download as CSV",
        data=csv,
        file_name="wine_shop_customers_filtered.csv",
        mime="text/csv",
    )

    st.markdown(salmond_footnote(), unsafe_allow_html=True)


def render_overview(df_f: pd.DataFrame) -> None:
    """Strategic conclusion — the full business picture in one view."""
    st.subheader("The full picture")
    st.markdown(
        f"<p style='color:{PALETTE['charcoal']}; font-size:15px; max-width:760px;'>"
        "A survey of 404 customers reveals a business with a loyal core, an underused "
        "deli, and a clear path to meaningful revenue growth — without needing a single "
        "new customer. Two complementary lenses frame the opportunity: "
        "<b>FM segments</b> tell you who is worth protecting and growing; "
        "<b>behavioral segments</b> tell you how to reach each group in a way that resonates."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(" ")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Customers surveyed", f"{len(df_f):,}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Avg ticket", f"{df_f['Ticket'].mean():.0f}", "€"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Est. annual revenue", f"{df_f['est_annual_revenue'].sum()/1000:.0f}K", "€"), unsafe_allow_html=True)
    c4.markdown(kpi_card("FM segments", f"{df_f['FM_segment'].nunique()}"), unsafe_allow_html=True)

    st.markdown(" ")
    col_a, col_b = st.columns(2)

    seg = df_f["FM_segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Customers"]
    fig = px.pie(seg, values="Customers", names="Segment", hole=0.55,
                 color="Segment", color_discrete_map=SEGMENT_COLORS)
    fig.update_traces(textinfo="percent+label", textfont_size=12)
    fig.update_layout(base_layout(title="Customer share by FM segment",
                                  showlegend=False, height=400))
    col_a.plotly_chart(fig, use_container_width=True)

    rev = (df_f.groupby("FM_segment")["est_annual_revenue"].sum()
                .sort_values().reset_index())
    rev["share"] = rev["est_annual_revenue"] / rev["est_annual_revenue"].sum() * 100
    fig2 = px.bar(rev, x="est_annual_revenue", y="FM_segment", orientation="h",
                  color="FM_segment", color_discrete_map=SEGMENT_COLORS,
                  text=rev["share"].apply(lambda x: f"{x:.0f}%"))
    fig2.update_traces(textposition="outside")
    fig2.update_layout(base_layout(title="Estimated annual revenue contribution",
                                   xaxis_title="€ / year", yaxis_title="",
                                   showlegend=False, height=400))
    col_b.plotly_chart(fig2, use_container_width=True)

    top_seg = rev.iloc[-1]
    pct_customers = (df_f["FM_segment"] == top_seg["FM_segment"]).mean() * 100
    st.markdown(callout(
        "🍇", "Headline finding",
        f"<b>{top_seg['FM_segment']}</b> represent {pct_customers:.0f}% of customers "
        f"but contribute <b>{top_seg['share']:.0f}%</b> of estimated annual revenue. "
        f"Protecting this group is the single highest-priority action."
    ), unsafe_allow_html=True)

    # ---- What-If simulator ----
    st.markdown("---")
    st.subheader("What-If Revenue Simulator")
    st.caption("Move the levers to project the revenue impact of three strategic actions.")

    s1, s2, s3 = st.columns(3)
    pct_convert_casual = s1.slider(
        "Convert Casual Visitors → Loyal Regulars (%)", 0, 100, 20, 5,
        help="Share of casual visitors given loyal-regular-level visit frequency",
    )
    ticket_lift_loyal = s2.slider(
        "Loyal Regulars ticket lift (€)", 0, 30, 5, 1,
        help="Incremental basket value per Loyal Regular from the 'Pair It' deli upsell",
    )
    pct_convert_loyal = s3.slider(
        "Convert Loyal Regulars → Champions (%)", 0, 50, 10, 5,
        help="Share of loyal regulars elevated to champions via the loyalty programme",
    )

    sim_rev  = simulate_revenue(df_f, pct_convert_casual, ticket_lift_loyal, pct_convert_loyal)
    base_rev = df_f["est_annual_revenue"].sum()
    delta    = sim_rev - base_rev
    delta_pct = (delta / base_rev * 100) if base_rev else 0

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Baseline annual revenue", f"€{base_rev/1000:.0f}K")
    rc2.metric("Simulated annual revenue", f"€{sim_rev/1000:.0f}K", f"€{delta/1000:+.0f}K")
    rc3.metric("Uplift", f"{delta_pct:+.1f}%")

    with st.expander("📐 How the simulator works · assumptions & formula"):
        st.markdown(f"""
**Core formula:** `annual_revenue = ticket × visits_per_month × 12`

**Three combined interventions** (mechanics applied in order):

1. **Convert Casual Visitors → Loyal Regulars** *(top slider)*
   A randomly-selected share of customers currently in the *Casual Visitors* segment
   has their `visits_per_month` raised to the **median** visit frequency of the
   *Loyal Regulars* segment. Their ticket is unchanged. Represents the effect of
   activation marketing (Instagram, in-store tastings, neighbourhood promotion).

2. **Lift Loyal Regulars' basket** *(middle slider)*
   Every customer in the *Loyal Regulars* segment has €N added to their
   `Ticket` value, where N is the slider's value. Represents the "Pair It"
   deli upsell at the till.

3. **Elevate Loyal Regulars → Champions** *(bottom slider)*
   A randomly-selected share of customers currently in the *Loyal Regulars*
   segment has their `Ticket` raised to the **median** ticket of the
   *Champions* segment. Represents the impact of the loyalty programme
   (Cave Club membership, premium hamper subscriptions).

**Visits-per-month mapping** (verbal → numeric, midpoint of each band):
{', '.join(f"*{label}* → {visits}" for label, visits in FREQ_VISITS_PER_MONTH.items())}

**Random seeds** are fixed (seeds 1 and 2 for the two random samples) so the
projection is deterministic and reproducible for any slider combination.

**This is a directional sensitivity model, not a financial forecast.** It shows
the order-of-magnitude impact of each intervention rather than a precise
revenue projection. Read the output as *"if X% of Casual Visitors matched
Loyal-Regular visit cadence, estimated annual revenue would change by
approximately €Y."*
        """)


# ============================================================================
# APP MAIN
# ============================================================================

def main() -> None:
    df_raw = load_data()
    df     = compute_fm_segments(df_raw)

    # ---- Sidebar: global filters ----
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{PALETTE['burgundy']};'>🍷 Wine Shop</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color:{PALETTE['charcoal']}; font-style:italic; margin-top:-12px;'>"
            "Customer Intelligence</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**Global filters**")

        gender_sel = st.multiselect(
            "Gender", options=sorted(df["Gender"].unique()),
            default=sorted(df["Gender"].unique()),
        )
        age_sel = st.multiselect(
            "Age band",
            options=[a for a in AGE_ORDER if a in df["Age"].unique()],
            default=[a for a in AGE_ORDER if a in df["Age"].unique()],
        )
        ticket_range = st.slider(
            "Ticket range (€)",
            int(df["Ticket"].min()), int(df["Ticket"].max()),
            (int(df["Ticket"].min()), int(df["Ticket"].max())),
        )

        st.markdown("---")
        with st.expander("ℹ️ Visits/month assumptions"):
            st.caption("Used for the annual revenue estimate.")
            for label in FREQ_ORDER:
                st.caption(f"• {label} → ~{FREQ_VISITS_PER_MONTH[label]} visits/mo")

        with st.expander("📖 About"):
            st.caption(
                "Built on a 404-customer survey for the Wine Shop & Delicatessen case "
                "(Advanced Programming with Python — ESADE MSc Business Analytics). "
                "FM segmentation is the primary revenue lens; behavioral K-Means "
                "reveals how each group actually buys; K-Prototypes spend tiers "
                "provide a secondary value-anchored validation."
            )
            st.caption(
                "**Team:** Sabeena Awan · Brice Da Costa · Lucas Joris Haesaert · "
                "Patricia Unger"
            )

        with st.expander("⚠️ Data caveats"):
            st.caption(
                "**Ticket = value per visit (assumed).** The survey gives one ticket "
                "value per customer; we assume that figure is the customer's *typical* "
                "basket size on each visit, not a cumulative lifetime total."
            )
            st.caption(
                "**Synthetic / case-study data.** The dataset is the ESADE *Analytics "
                "and Big Data* case-study sample. Some records contain unrealistic "
                "combinations (e.g. customers visiting daily and spending €100 every "
                "time — that's €36,000/year on wine and deli, which is implausible "
                "for an individual). We surface these as-is rather than clipping, "
                "but the methodology and segmentation framework are unaffected — "
                "they would be re-fit on real POS data when available."
            )
            st.caption(
                "**Monthly spend** = `Ticket × visits/month`. **Annual revenue** = "
                "`monthly_spend × 12`. The visits/month mapping is documented above."
            )

    mask = (
        df["Gender"].isin(gender_sel)
        & df["Age"].isin(age_sel)
        & df["Ticket"].between(ticket_range[0], ticket_range[1])
    )
    df_f = df[mask].copy()

    st.markdown(f"""
    <div style='border-bottom: 3px solid {PALETTE["burgundy"]}; padding-bottom: 10px;
                margin-bottom: 18px;'>
        <h1 style='margin: 0;'>🍷 Wine Shop & Delicatessen</h1>
        <p style='margin: 4px 0 0 0; color: {PALETTE["charcoal"]}; font-style: italic;'>
            Strategic Customer Intelligence · {len(df_f)} of {len(df)} customers in view
        </p>
    </div>
    """, unsafe_allow_html=True)

    if len(df_f) == 0:
        st.warning("No customers match the current filters — widen them in the sidebar.")
        st.stop()

    # ─── New 6-tab narrative arc (May 2026 reorder) ───────────────────────
    #
    # The journey: who they are → how they group by revenue → how they group
    # by behaviour and spend → what they buy → synthesis + projection →
    # prioritised action.
    #
    # Customer Explorer is no longer a top-level tab — it lives as a
    # collapsed expander at the bottom of the Customer Profile tab (since
    # drilling from aggregate profile down to individuals is the natural
    # continuation of that tab's content).
    tabs = st.tabs([
        "Customer Profile",      # who they are (+ explorer)
        "FM Segments",           # group by revenue value
        "Behavioral Segments",   # two ML lenses (behavioral + spend tiers)
        "Customer Behaviour",           # what they buy
        "Strategic Overview",    # synthesis + What-If simulator
        "Action Plan",           # the close — what to do
    ])

    with tabs[0]:
        render_profile(df_f, df)
        # Customer Explorer absorbed here as a collapsed bottom expander
        # (drilling from aggregate profile down to individuals is a natural
        # continuation of this tab's content).
        st.markdown("---")
        with st.expander("🔎 Drill into individual customers — Customer Explorer"):
            render_explorer(df_f)
        _next_tab_hint(
            "Now that we know who walks through the door, the next tab — "
            "<b>FM Segments</b> — groups these customers by revenue value "
            "(frequency × monetary)."
        )
    with tabs[1]:
        render_segments(df_f)
        _next_tab_hint(
            "FM uses a deterministic median split. The next tab — "
            "<b>Behavioral Segments</b> — lets the data group customers "
            "without any rules, using two complementary unsupervised lenses "
            "(behavioral archetypes + spend tiers)."
        )
    with tabs[2]:
        render_behavioral(df_f)
        _next_tab_hint(
            "Three lenses on the same 404 customers. The next tab — "
            "<b>Customer Behaviour</b> — steps away from segmentation to look at "
            "what customers actually buy, and where the cross-sell gap sits."
        )
    with tabs[3]:
        render_products(df_f)
        _next_tab_hint(
            "With the customer base understood from every angle, the next "
            "tab — <b>Strategic Overview</b> — pulls back to the headline "
            "numbers and lets you project the revenue impact of strategic "
            "actions via the What-If simulator."
        )
    with tabs[4]:
        render_overview(df_f)
        _next_tab_hint(
            "The simulator shows what <i>could</i> change. The final tab — "
            "<b>Action Plan</b> — translates these levers into specific, "
            "prioritised actions per segment."
        )
    with tabs[5]:
        render_actions(df_f)
        # Last tab — no further hand-off; render_actions already ends with its
        # own cross-cutting priority callout.


if __name__ == "__main__":
    main()
