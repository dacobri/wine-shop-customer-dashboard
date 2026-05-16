"""
wine_dashboard.py
=================
Wine Shop & Delicatessen — Strategic Customer Dashboard.

Tab order (story flow):
  1. Customer Profile   — who walks through the door?
  2. FM Segments        — who is worth what? (revenue lens)
  3. Behavioral Segs    — how do they actually buy? (behaviour lens)
  4. Product Mix        — where is the deli opportunity?
  5. Action Plan        — what to do, in what order
  6. Customer Explorer  — individual-level targeting
  7. Strategic Overview — the full picture + What-If simulator (conclusion)

INSTALL
-------
    pip install streamlit plotly pandas numpy scikit-learn openpyxl

RUN
---
    streamlit run wine_dashboard.py

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
                              EDUCATION_GROUP_ORDER)
from wine_clustering import (fit_behavioral_clusters, name_behavioral_clusters,
                              behavioral_diagnostics)
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

    # ── Section C: How they spend (filter-aware) ───────────────────────────
    st.markdown("### C. How they spend — basket size by education tier")
    _profile_ticket_by_education(df_f)

    st.markdown("---")

    # ── Section D: What connects (filter-aware) ────────────────────────────
    st.markdown("### D. What connects — relationship strength map")
    st.caption("Cramér's V quantifies association between categorical variables. "
               "Higher values (darker) = stronger relationship.")
    _profile_cramers_v_heatmap(df_f)


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

def _profile_education_funnel(df: pd.DataFrame) -> None:
    """Horizontal bar of the 4 education tiers, sorted by count (largest first).

    Per the professor's "sort by value, never alphabetical" principle.
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

    fig = go.Figure(go.Bar(
        y=counts.index, x=counts.values, orientation="h",
        marker_color=colors,
        text=[f"{c}  ({p}%)" for c, p in zip(counts.values, pct.values)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x} customers<extra></extra>",
    ))
    fig.update_layout(base_layout(
        title="Education tiers — 4 grouped levels",
        height=380, showlegend=False,
        xaxis=dict(title="Customers", range=[0, counts.max() * 1.25]),
        yaxis=dict(title=""),
    ))
    st.plotly_chart(fig, use_container_width=True)


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


# ── Section C: Ticket × Education boxplot ─────────────────────────────────

def _profile_ticket_by_education(df: pd.DataFrame) -> None:
    """Boxplot of Ticket grouped by 4-tier Education. Filter-aware."""
    work = df.dropna(subset=["Education_group"]).copy()
    if len(work) == 0:
        st.info("No customers with Education data in current filter.")
        return

    fig = px.box(
        work, x="Education_group", y="Ticket",
        category_orders={"Education_group": EDUCATION_GROUP_ORDER},
        color="Education_group",
        color_discrete_map={
            "Basic":        PALETTE["burgundy"],
            "Technical":    PALETTE["merlot"],
            "University":   PALETTE["gold"],
            "Postgraduate": PALETTE["teal"],
        },
        points="outliers",
    )
    fig.update_layout(base_layout(
        title="Basket distribution by education tier",
        height=380, showlegend=False,
        xaxis=dict(title=""), yaxis=dict(title="Ticket (€)"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    means = work.groupby("Education_group", observed=True)["Ticket"].mean().round(0)
    means = means.reindex(EDUCATION_GROUP_ORDER).dropna()
    if len(means) >= 2:
        highest_grp, highest_val = means.idxmax(), int(means.max())
        lowest_grp,  lowest_val  = means.idxmin(), int(means.min())
        st.markdown(callout(
            "💸", "Education does not predict basket size",
            f"<b>{highest_grp}</b>-tier customers average <b>€{highest_val}</b> · "
            f"<b>{lowest_grp}</b>-tier customers average <b>€{lowest_val}</b>. "
            f"Education predicts <i>what</i> people buy, not <i>how much</i>. "
            f"Segmenting offers by spend (FM segments) is more actionable than "
            f"segmenting by education."
        ), unsafe_allow_html=True)


# ── Section D: Cramér's V heatmap ─────────────────────────────────────────

def _profile_cramers_v_heatmap(df: pd.DataFrame) -> None:
    """Pairwise Cramér's V across categorical variables. Filter-aware."""
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
    work = df.dropna(subset=["Education_group"])
    if len(work) < 10:
        st.info("Too few customers in current filter to compute reliable associations.")
        return

    mat = pd.DataFrame(index=cats, columns=cats, dtype=float)
    for a in cats:
        for b in cats:
            mat.loc[a, b] = np.nan if a == b else _cramers_v(work[a], work[b])

    # Friendlier short labels
    short = {
        "Wine frequency consumption": "Frequency",
        "Place to drink":              "Occasion",
        "Additional products":         "Deli product",
        "Education_group":             "Education",
        "Payment mode":                "Payment",
    }
    pretty = [short.get(c, c) for c in cats]
    mat.index = pretty
    mat.columns = pretty

    fig = px.imshow(
        mat, text_auto=".2f", aspect="auto",
        color_continuous_scale=[
            [0.0, PALETTE["cream"]],
            [0.5, PALETTE["gold"]],
            [1.0, PALETTE["burgundy"]],
        ],
        zmin=0, zmax=0.4,
        labels=dict(color="Cramér's V"),
    )
    fig.update_layout(base_layout(
        title=f"Relationship strength map · {len(work)} customers",
        height=470, xaxis=dict(side="bottom"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Surface the top-3 strongest off-diagonal pairs
    upper = mat.where(np.triu(np.ones(mat.shape, dtype=bool), k=1))
    top3 = upper.stack().dropna().sort_values(ascending=False).head(3)
    pair_strings = [f"<b>{a}</b> × <b>{b}</b> (V={v:.2f})"
                    for (a, b), v in top3.items()]
    if pair_strings:
        st.markdown(callout(
            "🧭", "Strongest relationships in the data",
            " · ".join(pair_strings) + ". Darker cells = stronger association. "
            "Payment mode shows the weakest links overall — confirming the "
            "decision to drop it as a filter dimension."
        ), unsafe_allow_html=True)


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


def render_behavioral(df_f: pd.DataFrame) -> None:
    st.subheader("Behavioral Segmentation — how customers actually buy")
    st.caption(
        "K-Means K=4 on visit frequency × sociality score. "
        "Ticket price is **not** used — so these segments are genuinely complementary "
        "to FM: they tell you *how* people buy, FM tells you *how much*."
    )

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
    st.subheader("Product Mix & Cross-Sell Opportunity")
    st.caption(
        "The store is perceived as a wine shop rather than a delicatessen. "
        "This view diagnoses where the deli is winning and where the gaps are."
    )

    # Replace 'Salmond' with 'Salmond*' in displayed copies — see footnote at bottom.
    df_display = with_salmond_marker(df_f)

    c1, c2 = st.columns(2)
    sold = df_display["Additional products"].value_counts().reset_index()
    sold.columns = ["Product", "Customers"]
    fig = px.bar(sold, x="Customers", y="Product", orientation="h",
                 color="Customers",
                 color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["merlot"]]])
    fig.update_layout(base_layout(title="Most popular deli products", height=440,
                                  coloraxis_showscale=False, yaxis=dict(autorange="reversed")))
    c1.plotly_chart(fig, use_container_width=True)

    tier = df_f["deli_tier"].value_counts().reset_index()
    tier.columns = ["Tier", "Customers"]
    fig = px.pie(tier, values="Customers", names="Tier", hole=0.55,
                 color="Tier",
                 color_discrete_map={
                     "Premium": PALETTE["merlot"],
                     "Entry":   PALETTE["gold"],
                     "Other":   PALETTE["teal"],
                 })
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Premium vs Entry deli split", height=440))
    c2.plotly_chart(fig, use_container_width=True)

    premium_share = (df_f["deli_tier"] == "Premium").mean() * 100
    st.markdown(callout(
        "🧀", "Cross-sell gap — the strategic priority",
        f"<b>{premium_share:.0f}%</b> of customers buy premium deli "
        f"(cheese, Spanish ham, salmon*), but no single premium product reaches "
        f"majority share in any segment — most still default to olives. "
        f"A \"Pair It\" display at eye level and trained staff suggesting one deli match "
        f"per wine could lift premium adoption within a quarter."
    ), unsafe_allow_html=True)

    st.markdown(" ")
    ct = pd.crosstab(df_display["FM_segment"], df_display["Additional products"], normalize="index") * 100
    fig = px.imshow(ct, text_auto=".0f", aspect="auto",
                    color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["merlot"]]],
                    labels=dict(x="Product", y="Segment", color="% of segment"))
    fig.update_layout(base_layout(title="Product preference by FM segment (%)", height=380))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Drinking occasion × Product purchased — pairing intelligence for in-store displays**")
    ct2 = pd.crosstab(df_display["Place to drink"], df_display["Additional products"])
    fig = px.imshow(ct2, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["merlot"]]],
                    labels=dict(x="Product", y="Occasion", color="Customers"))
    fig.update_layout(base_layout(title="Occasion × Product co-occurrence", height=440))
    st.plotly_chart(fig, use_container_width=True)

    top = (df_display.groupby(["FM_segment", "Additional products"]).size()
                 .reset_index(name="customers")
                 .sort_values(["FM_segment", "customers"], ascending=[True, False]))
    top["rank"] = top.groupby("FM_segment").cumcount() + 1
    top1 = top[top["rank"] == 1][["FM_segment", "Additional products", "customers"]]
    top1.columns = ["Segment", "Top deli choice", "Customers"]
    st.markdown("**Top deli choice per FM segment** — natural anchor product for each group's bundles")
    st.dataframe(top1, hide_index=True, use_container_width=True)

    # ── Salmond footnote — single source of truth, referenced by every * in this tab ──
    st.markdown(salmond_footnote(), unsafe_allow_html=True)


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

    st.caption(
        "Mechanics: a slice of Casual Visitors is assigned loyal-regular-level visit "
        "frequency; Loyal Regulars' ticket gets the lift you set; a slice of Loyal "
        "Regulars adopts Champions-level ticket. Annual revenue = ticket × visits/mo × 12."
    )


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
                "reveals how each group actually buys."
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

    tabs = st.tabs([
        "👥 Customer Profile",
        "🎯 FM Segments",
        "🔬 Behavioral Segments",
        "🧀 Product Mix",
        "📋 Action Plan",
        "🔎 Explorer",
        "📊 Strategic Overview",
    ])

    with tabs[0]: render_profile(df_f, df)
    with tabs[1]: render_segments(df_f)
    with tabs[2]: render_behavioral(df_f)
    with tabs[3]: render_products(df_f)
    with tabs[4]: render_actions(df_f)
    with tabs[5]: render_explorer(df_f)
    with tabs[6]: render_overview(df_f)


if __name__ == "__main__":
    main()
