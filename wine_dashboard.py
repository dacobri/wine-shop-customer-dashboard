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
                              inject_css, base_layout, kpi_card, callout)
from wine_data       import (load_data, compute_fm_segments,
                              FREQ_ORDER, FREQ_VISITS_PER_MONTH, AGE_ORDER)
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

def render_profile(df_f: pd.DataFrame) -> None:
    st.subheader("Who walks through the door?")
    st.caption("Demographic and behavioral profile of the (filtered) customer base.")

    c1, c2, c3 = st.columns(3)

    g = df_f["Gender"].value_counts().reset_index()
    g.columns = ["Gender", "Count"]
    fig = px.pie(g, values="Count", names="Gender", hole=0.5,
                 color_discrete_sequence=[PALETTE["merlot"], PALETTE["rose"]])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Gender", height=320, showlegend=False))
    c1.plotly_chart(fig, use_container_width=True)

    a = df_f["Age"].value_counts().reindex(AGE_ORDER).fillna(0).reset_index()
    a.columns = ["Age", "Count"]
    fig = px.bar(a, x="Age", y="Count", color_discrete_sequence=[PALETTE["merlot"]])
    fig.update_layout(base_layout(title="Age band", height=320, showlegend=False))
    c2.plotly_chart(fig, use_container_width=True)

    p = df_f["Payment mode"].value_counts().reset_index()
    p.columns = ["Payment", "Count"]
    fig = px.pie(p, values="Count", names="Payment", hole=0.5,
                 color_discrete_sequence=[PALETTE["merlot"], PALETTE["teal"], PALETTE["gold"]])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Payment mode", height=320, showlegend=False))
    c3.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    freq = (df_f["Wine frequency consumption"].value_counts()
              .reindex(FREQ_ORDER).fillna(0).reset_index())
    freq.columns = ["Frequency", "Count"]
    fig = px.bar(freq, x="Count", y="Frequency", orientation="h",
                 color_discrete_sequence=[PALETTE["merlot"]])
    fig.update_layout(base_layout(title="Wine consumption frequency", height=360,
                                  showlegend=False))
    c4.plotly_chart(fig, use_container_width=True)

    place = df_f["Place to drink"].value_counts().reset_index()
    place.columns = ["Place", "Count"]
    fig = px.bar(place, x="Count", y="Place", orientation="h",
                 color_discrete_sequence=[PALETTE["teal"]])
    fig.update_layout(base_layout(title="Where customers drink wine", height=360,
                                  showlegend=False))
    c5.plotly_chart(fig, use_container_width=True)

    edu = df_f["Education"].dropna().value_counts().reset_index()
    edu.columns = ["Education", "Count"]
    fig = px.bar(edu, x="Count", y="Education", orientation="h",
                 color_discrete_sequence=[PALETTE["merlot"]])
    fig.update_layout(base_layout(title="Education level", height=380, showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df_f, x="Ticket", nbins=25,
                       color_discrete_sequence=[PALETTE["teal"]])
    fig.update_layout(base_layout(title="Ticket distribution (€)", height=320,
                                  showlegend=False, xaxis_title="Ticket (€)",
                                  yaxis_title="Customers"))
    st.plotly_chart(fig, use_container_width=True)


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

    c1, c2 = st.columns(2)
    sold = df_f["Additional products"].value_counts().reset_index()
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
        f"(cheese, Spanish ham, salmon), but no single premium product reaches "
        f"majority share in any segment — most still default to olives. "
        f"A \"Pair It\" display at eye level and trained staff suggesting one deli match "
        f"per wine could lift premium adoption within a quarter."
    ), unsafe_allow_html=True)

    st.markdown(" ")
    ct = pd.crosstab(df_f["FM_segment"], df_f["Additional products"], normalize="index") * 100
    fig = px.imshow(ct, text_auto=".0f", aspect="auto",
                    color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["merlot"]]],
                    labels=dict(x="Product", y="Segment", color="% of segment"))
    fig.update_layout(base_layout(title="Product preference by FM segment (%)", height=380))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Drinking occasion × Product purchased — pairing intelligence for in-store displays**")
    ct2 = pd.crosstab(df_f["Place to drink"], df_f["Additional products"])
    fig = px.imshow(ct2, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["merlot"]]],
                    labels=dict(x="Product", y="Occasion", color="Customers"))
    fig.update_layout(base_layout(title="Occasion × Product co-occurrence", height=440))
    st.plotly_chart(fig, use_container_width=True)

    top = (df_f.groupby(["FM_segment", "Additional products"]).size()
                 .reset_index(name="customers")
                 .sort_values(["FM_segment", "customers"], ascending=[True, False]))
    top["rank"] = top.groupby("FM_segment").cumcount() + 1
    top1 = top[top["rank"] == 1][["FM_segment", "Additional products", "customers"]]
    top1.columns = ["Segment", "Top deli choice", "Customers"]
    st.markdown("**Top deli choice per FM segment** — natural anchor product for each group's bundles")
    st.dataframe(top1, hide_index=True, use_container_width=True)


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
        payment_sel = st.multiselect(
            "Payment mode", options=sorted(df["Payment mode"].unique()),
            default=sorted(df["Payment mode"].unique()),
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
        & df["Payment mode"].isin(payment_sel)
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

    with tabs[0]: render_profile(df_f)
    with tabs[1]: render_segments(df_f)
    with tabs[2]: render_behavioral(df_f)
    with tabs[3]: render_products(df_f)
    with tabs[4]: render_actions(df_f)
    with tabs[5]: render_explorer(df_f)
    with tabs[6]: render_overview(df_f)


if __name__ == "__main__":
    main()
