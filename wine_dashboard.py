"""
wine_dashboard.py
=================
Wine Shop & Delicatessen — Strategic Customer Intelligence Dashboard.

Entrypoint: owns the sidebar, tab structure, and per-tab render functions.
Business logic lives in sibling modules:

    wine_data.py        data loading, cleaning, FM segmentation
    wine_clustering.py  K-Prototypes / K-Means clustering and diagnostics
    wine_simulator.py   what-if revenue projection
    wine_config.py      strategic playbook per segment
    wine_theme.py       palette, CSS, Plotly layout helpers

Install
-------
    pip install streamlit plotly pandas numpy scikit-learn kmodes openpyxl

Run
---
    streamlit run wine_dashboard.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from wine_theme      import (PALETTE, SEQ, SEGMENT_COLORS,
                             inject_css, base_layout, kpi_card, callout)
from wine_data       import (load_data, compute_fm_segments,
                             FREQ_ORDER, FREQ_VISITS_PER_MONTH, AGE_ORDER)
from wine_clustering import (fit_clusters, cluster_diagnostics,
                             name_clusters_by_spend, KPROTOTYPES_OK)
from wine_config     import SEGMENT_META


# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="Wine Shop & Delicatessen — Customer Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ============================================================================
# Tab renderers
# ============================================================================

def render_overview(df_f: pd.DataFrame) -> None:
    st.subheader("Business at a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Customers surveyed",  f"{len(df_f):,}"),
                unsafe_allow_html=True)
    c2.markdown(kpi_card("Average ticket",       f"{df_f['Ticket'].mean():.0f}", "€"),
                unsafe_allow_html=True)
    c3.markdown(kpi_card("Median ticket",        f"{df_f['Ticket'].median():.0f}", "€"),
                unsafe_allow_html=True)
    c4.markdown(kpi_card("Active segments",      f"{df_f['FM_segment'].nunique()}"),
                unsafe_allow_html=True)

    st.markdown(" ")

    col_a, col_b = st.columns(2)

    # Customer share by segment
    seg = df_f["FM_segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Customers"]
    fig = px.pie(seg, values="Customers", names="Segment", hole=0.55,
                 color="Segment", color_discrete_map=SEGMENT_COLORS)
    fig.update_traces(textinfo="percent+label", textfont_size=12)
    fig.update_layout(base_layout(title="Customer share by segment",
                                  showlegend=False, height=400))
    col_a.plotly_chart(fig, use_container_width=True)

    # Avg ticket per segment — directly from data, no fabrication
    tkt = (df_f.groupby("FM_segment")["Ticket"]
               .mean().sort_values().reset_index())
    tkt.columns = ["FM_segment", "avg_ticket"]
    fig2 = px.bar(tkt, x="avg_ticket", y="FM_segment", orientation="h",
                  color="FM_segment", color_discrete_map=SEGMENT_COLORS,
                  text=tkt["avg_ticket"].apply(lambda x: f"€{x:.0f}"))
    fig2.update_traces(textposition="outside")
    fig2.update_layout(base_layout(
        title="Average ticket per segment (€ per visit)",
        xaxis_title="Average ticket (€)", yaxis_title="",
        showlegend=False, height=400))
    fig2.update_xaxes(range=[0, tkt["avg_ticket"].max() * 1.25])
    col_b.plotly_chart(fig2, use_container_width=True)

    # Callout: based only on customer counts and ticket — no revenue
    top_seg   = tkt.iloc[-1]
    n_top     = int((df_f["FM_segment"] == top_seg["FM_segment"]).sum())
    pct_cust  = n_top / len(df_f) * 100
    low_seg   = tkt.iloc[0]
    st.markdown(callout(
        "Key observation",
        f"<b>{top_seg['FM_segment']}</b> have the highest average ticket "
        f"at <b>€{top_seg['avg_ticket']:.0f} per visit</b> and represent "
        f"{pct_cust:.0f}% of surveyed customers. "
        f"<b>{low_seg['FM_segment']}</b> have the lowest at "
        f"€{low_seg['avg_ticket']:.0f}. "
        f"The difference between segments is driven by visit frequency, "
        f"not by how much they spend when they come in."
    ), unsafe_allow_html=True)

    # Ticket distribution
    st.markdown("---")
    st.markdown(
        f"<p style='font-size:17px; font-weight:700; color:{PALETTE['charcoal']};'>"
        "How ticket size is distributed across all customers</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='font-size:14px; color:{PALETTE['sage']}; margin-bottom:8px;'>"
        "Each bar shows how many customers reported a basket in that price range. "
        "This is the actual survey data — one ticket value per customer.</p>",
        unsafe_allow_html=True,
    )
    fig3 = px.histogram(df_f, x="Ticket", nbins=20,
                        color_discrete_sequence=[PALETTE["burgundy"]])
    fig3.update_layout(base_layout(
        title="Ticket distribution across all surveyed customers",
        height=340, showlegend=False,
        xaxis_title="Ticket size (€)", yaxis_title="Number of customers"))
    fig3.update_xaxes(tickprefix="€")
    st.plotly_chart(fig3, use_container_width=True)


def render_profile(df_f: pd.DataFrame) -> None:
    st.subheader("Who visits the store?")
    st.caption("Demographic and behavioural profile of the current customer selection.")

    c1, c2, c3 = st.columns(3)

    g = df_f["Gender"].value_counts().reset_index()
    g.columns = ["Gender", "Count"]
    fig = px.pie(g, values="Count", names="Gender", hole=0.5,
                 color_discrete_sequence=[PALETTE["burgundy"], PALETTE["gold"]])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Gender split", height=320, showlegend=False))
    c1.plotly_chart(fig, use_container_width=True)

    a = df_f["Age"].value_counts().reindex(AGE_ORDER).fillna(0).reset_index()
    a.columns = ["Age", "Count"]
    fig = px.bar(a, x="Age", y="Count", color_discrete_sequence=[PALETTE["burgundy"]])
    fig.update_layout(base_layout(title="Age distribution", height=320, showlegend=False))
    c2.plotly_chart(fig, use_container_width=True)

    p = df_f["Payment mode"].value_counts().reset_index()
    p.columns = ["Payment", "Count"]
    fig = px.pie(p, values="Count", names="Payment", hole=0.5,
                 color_discrete_sequence=[PALETTE["burgundy"], PALETTE["gold"], PALETTE["sage"]])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Payment method", height=320, showlegend=False))
    c3.plotly_chart(fig, use_container_width=True)

    st.markdown(callout(
        "Data gap",
        f"<b>{(df_f['Payment mode'] == 'Cash').mean()*100:.0f}% of customers pay in cash</b>, "
        f"leaving no purchase history. Without a loyalty mechanism, it is impossible to "
        f"identify who the store's best customers are, track retention, or personalise any offer."
    ), unsafe_allow_html=True)

    c4, c5 = st.columns(2)

    freq = (df_f["Wine frequency consumption"].value_counts()
            .reindex(FREQ_ORDER).fillna(0).reset_index())
    freq.columns = ["Frequency", "Count"]
    fig = px.bar(freq, x="Count", y="Frequency", orientation="h",
                 color_discrete_sequence=[PALETTE["burgundy"]])
    fig.update_layout(base_layout(title="Purchase frequency", height=360, showlegend=False))
    c4.plotly_chart(fig, use_container_width=True)

    place = df_f["Place to drink"].value_counts().reset_index()
    place.columns = ["Occasion", "Count"]
    fig = px.bar(place, x="Count", y="Occasion", orientation="h",
                 color_discrete_sequence=[PALETTE["gold"]])
    fig.update_layout(base_layout(title="Where customers consume wine", height=360, showlegend=False))
    c5.plotly_chart(fig, use_container_width=True)

    edu = df_f["Education"].dropna().value_counts().reset_index()
    edu.columns = ["Education", "Count"]
    fig = px.bar(edu, x="Count", y="Education", orientation="h",
                 color_discrete_sequence=[PALETTE["merlot"]])
    fig.update_layout(base_layout(title="Education level", height=380, showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df_f, x="Ticket", nbins=25,
                       color_discrete_sequence=[PALETTE["burgundy"]])
    fig.update_layout(base_layout(title="Ticket distribution (€)", height=320,
                                  showlegend=False, xaxis_title="Ticket (€)",
                                  yaxis_title="Customers"))
    st.plotly_chart(fig, use_container_width=True)


def render_segments(df_f: pd.DataFrame) -> None:
    st.subheader("FM Segmentation — the strategic framework")
    st.caption(
        "Customers are classified by how often they visit (Frequency) and how much "
        "they spend per visit (Monetary). A median split on both dimensions produces "
        "four segments, each with a distinct strategic role."
    )

    seg_summary = df_f.groupby("FM_segment").agg(
        customers  = ("ID",     "count"),
        avg_ticket = ("Ticket", "mean"),
    ).reset_index()
    seg_summary["share_customers"] = seg_summary["customers"] / seg_summary["customers"].sum() * 100

    cols      = st.columns(4)
    seg_order = ["Champions", "Loyal Regulars", "Occasion Splurgers", "Casual Visitors"]
    prio_map  = {"High": "priority-high", "Medium": "priority-med", "Low": "priority-low"}

    for i, seg in enumerate(seg_order):
        row = seg_summary[seg_summary["FM_segment"] == seg]
        if row.empty:
            cols[i].info(f"No {seg} customers match the current filters.")
            continue
        row  = row.iloc[0]
        meta = SEGMENT_META[seg]
        with cols[i]:
            st.markdown(f"""
            <div class='segment-card' style='border-top-color:{SEGMENT_COLORS[seg]};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <h4 style='margin:0;'>{seg}</h4>
                    <span class='{prio_map[meta["priority"]]}'>{meta["priority"]}</span>
                </div>
                <p style='font-size:12px; color:{PALETTE["charcoal"]}; margin:8px 0;
                          min-height:50px;'>{meta["summary"][:120]}...</p>
                <hr style='margin:6px 0; border:none; border-top:1px solid #EEE;'>
                <p style='margin:4px 0; font-size:13px;'>
                    <b>{int(row["customers"])}</b> customers ({row["share_customers"]:.0f}%)
                </p>
                <p style='margin:4px 0; font-size:13px;'>
                    Avg ticket: <b>€{row["avg_ticket"]:.0f}</b>
                </p>

            </div>
            """, unsafe_allow_html=True)

    st.markdown(" ")

    f_med = df_f["monthly_visits"].median()
    m_med = df_f["Ticket"].median()
    fig = px.scatter(
        df_f, x="monthly_visits", y="Ticket",
        color="FM_segment", color_discrete_map=SEGMENT_COLORS,
        hover_data=["Age", "Gender", "Place to drink", "Additional products"],
        labels={"monthly_visits": "Stated visit frequency (survey category)", "Ticket": "Avg ticket (€)"},
        opacity=0.75,
    )
    fig.add_hline(y=m_med, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.4,
                  annotation_text=f"Median ticket €{m_med:.0f}",
                  annotation_position="bottom right")
    fig.add_vline(x=f_med, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.4,
                  annotation_text=f"Median {f_med:.0f} visits/month",
                  annotation_position="top left")
    fig.update_layout(base_layout(title="FM quadrant — each point represents one customer",
                                  height=520))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(callout(
        "Methodology note",
        "The median split means customers exactly on the boundary (172 customers have "
        "exactly 6 visits/month) are assigned to the High-frequency tier. "
        "This is a deliberate and consistent rule, not an error."
    ), unsafe_allow_html=True)
    st.markdown(" ")

    st.markdown("**Deli product mix per segment**")
    prod = (df_f.groupby(["FM_segment", "Additional products"]).size()
                .reset_index(name="customers"))
    fig = px.bar(prod, x="customers", y="FM_segment", color="Additional products",
                 orientation="h", color_discrete_sequence=SEQ,
                 category_orders={"FM_segment": seg_order})
    fig.update_layout(base_layout(title="Deli product mix by segment",
                                  height=380, barmode="stack",
                                  xaxis_title="Customers", yaxis_title=""),
                      legend=dict(orientation="v", yanchor="middle", y=0.5,
                                  xanchor="left", x=1.02, font=dict(size=12)),
                      margin=dict(t=50, b=40, l=40, r=120))
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
                                  barmode="stack", xaxis_title="% of segment", yaxis_title=""),
                      legend=dict(orientation="v", yanchor="middle", y=0.5,
                                  xanchor="left", x=1.02, font=dict(size=12)),
                      margin=dict(t=50, b=40, l=40, r=110))
    dc1.plotly_chart(fig, use_container_width=True)

    gen_mix = pd.crosstab(df_f["FM_segment"], df_f["Gender"], normalize="index") * 100
    fig = px.bar(
        gen_mix.reset_index().melt(id_vars="FM_segment", var_name="Gender", value_name="pct"),
        x="pct", y="FM_segment", color="Gender", orientation="h",
        color_discrete_sequence=[PALETTE["burgundy"], PALETTE["gold"]],
        category_orders={"FM_segment": seg_order},
    )
    fig.update_layout(base_layout(title="Gender mix per segment (%)", height=360,
                                  barmode="stack", xaxis_title="% of segment", yaxis_title=""),
                      legend=dict(orientation="v", yanchor="middle", y=0.5,
                                  xanchor="left", x=1.02, font=dict(size=12)),
                      margin=dict(t=50, b=40, l=40, r=100))
    dc2.plotly_chart(fig, use_container_width=True)


def render_clustering(df_f: pd.DataFrame) -> None:
    st.subheader("Unsupervised clustering — validating the FM framework")
    if KPROTOTYPES_OK:
        st.caption(
            "K-Prototypes is used as the primary algorithm. It applies Hamming distance "
            "on the six categorical features and Euclidean distance on ticket value — "
            "the correct approach for this mixed-type dataset."
        )
    else:
        st.warning(
            "The kmodes package is not installed. Falling back to K-Means on one-hot "
            "encoded features. For the correct algorithm: pip install kmodes"
        )

    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Number of clusters (K)", 2, 6, 3)
        st.caption(
            "K = 3 produces Entry, Core and Premium spend tiers, which map "
            "cleanly onto the FM framework for interpretation."
        )
        st.info(f"Algorithm: {'K-Prototypes' if KPROTOTYPES_OK else 'K-Means (fallback)'}")

    ks, inertias, sils = cluster_diagnostics(df_f)
    diag = make_subplots(specs=[[{"secondary_y": True}]])
    diag.add_trace(
        go.Scatter(x=ks, y=inertias, name="Inertia (elbow)",
                   line=dict(color=PALETTE["burgundy"], width=3), mode="lines+markers"),
        secondary_y=False,
    )
    diag.add_trace(
        go.Scatter(x=ks, y=sils, name="Silhouette score",
                   line=dict(color=PALETTE["gold"], width=3, dash="dot"), mode="lines+markers"),
        secondary_y=True,
    )
    diag.update_layout(base_layout(title="Selecting K — elbow and silhouette diagnostics",
                                   height=340),
                       legend=dict(orientation="v", yanchor="middle", y=0.5,
                                   xanchor="left", x=1.08, font=dict(size=12)),
                       margin=dict(t=50, b=40, l=40, r=130))
    diag.update_xaxes(title="K (number of clusters)")
    diag.update_yaxes(title="Inertia — lower is tighter",  secondary_y=False)
    diag.update_yaxes(title="Silhouette — higher is cleaner", secondary_y=True)
    c2.plotly_chart(diag, use_container_width=True)

    labels   = fit_clusters(df_f, k=k)
    df_c     = df_f.copy()
    df_c["cluster_id"]   = labels
    df_c["cluster_name"] = df_c["cluster_id"].map(name_clusters_by_spend(df_c, "cluster_id"))

    csum = (
        df_c.groupby("cluster_name").agg(
            Customers   = ("ID",               "count"),
            Avg_ticket  = ("Ticket",           "mean"),
            Min_ticket  = ("Ticket",           "min"),
            Max_ticket  = ("Ticket",           "max"),
            Avg_visits  = ("monthly_visits",   "mean"),
        )
        .reset_index()
        .sort_values("Avg_ticket")
    )
    csum["Avg_ticket"] = csum["Avg_ticket"].round(1)
    csum["Avg_visits"] = csum["Avg_visits"].round(1)
    csum.columns = ["Cluster", "Customers", "Avg ticket (€)", "Min ticket",
                    "Max ticket", "Avg visits/month (est.)"]
    st.markdown("**Cluster summary**")
    st.dataframe(csum, hide_index=True, use_container_width=True)

    cv1, cv2 = st.columns(2)
    fig = px.scatter(
        df_c, x="monthly_visits", y="Ticket", color="cluster_name",
        color_discrete_sequence=SEQ,
        hover_data=["Age", "Gender", "Additional products", "Place to drink"],
        labels={"monthly_visits": "Visits / month", "Ticket": "Avg ticket (€)"},
        opacity=0.75,
    )
    fig.update_layout(base_layout(title="Clusters on Frequency × Ticket space", height=420),
                      legend=dict(orientation="v", yanchor="middle", y=0.5,
                                  xanchor="left", x=1.02, font=dict(size=12)),
                      margin=dict(t=50, b=40, l=40, r=120))
    cv1.plotly_chart(fig, use_container_width=True)

    sizes = df_c["cluster_name"].value_counts().reset_index()
    sizes.columns = ["Cluster", "Customers"]
    fig = px.pie(sizes, values="Customers", names="Cluster", hole=0.5,
                 color_discrete_sequence=SEQ)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Cluster size distribution", height=420,
                                  showlegend=False))
    cv2.plotly_chart(fig, use_container_width=True)

    st.markdown("**Do the unsupervised clusters validate the FM segments?**")
    ct = pd.crosstab(df_c["FM_segment"], df_c["cluster_name"])
    fig = px.imshow(
        ct, text_auto=True, aspect="auto",
        color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["burgundy"]]],
        labels=dict(x="Cluster (ML)", y="Segment (FM)", color="Customers"),
    )
    fig.update_layout(base_layout(title="FM segment × ML cluster cross-tabulation", height=340))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(callout(
        "Interpretation note",
        "Because the FM segmentation uses both ticket and visit frequency, while "
        "K-Means clusters primarily on ticket value (the dominant numeric signal), "
        "the cross-tabulation diagonal will not be perfectly dense. Champions and "
        "Loyal Regulars share the same ticket threshold and are separated only by "
        "frequency — which one-hot encoding partially loses. This is expected "
        "behaviour, not a flaw in either method."
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
                <p style='margin:0 0 4px; font-size:13px; color:{PALETTE["sage"]};
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
                <p style='margin:0 0 4px; font-size:13px; color:{PALETTE["sage"]};
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
        f"<p style='font-size:15px; color:{PALETTE['sage']}; margin-bottom:4px;'>"
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
        "The reach problem",
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
        f"<p style='font-size:15px; color:{PALETTE['sage']}; margin-bottom:20px;'>"
        "The occasion a customer is buying for is one of the strongest predictors "
        "of their basket size — more so than age or gender. "
        "Customers buying for a restaurant or holiday spend significantly more "
        "and choose premium deli at a higher rate.</p>",
        unsafe_allow_html=True,
    )

    occ_stats = (
        df_f.groupby("Place to drink")
        .agg(
            customers = ("ID",         "count"),
            avg_ticket= ("Ticket",     "mean"),
        )
        .reset_index()
        .sort_values("avg_ticket", ascending=False)
    )
    occ_stats["pct_premium"] = occ_stats["Place to drink"].map(
        df_f.groupby("Place to drink").apply(
            lambda x: (x["deli_tier"] == "Premium").mean() * 100
        )
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
    # Store average reference line
    fig_occ.add_vline(
        x=avg_ticket_all,
        line_dash="dash",
        line_color=PALETTE["charcoal"],
        opacity=0.35,
        annotation_text=f"Store avg €{avg_ticket_all:.0f}",
        annotation_position="top",
        annotation_font=dict(size=12, color=PALETTE["charcoal"]),
    )
    fig_occ.update_layout(
        base_layout(
            title="Average ticket and premium deli choice by drinking occasion",
            height=400,
        ),
        xaxis=dict(
            range=[35, 90],
            title="Average ticket (€)",
            title_font=dict(size=13),
            tickfont=dict(size=12),
            tickprefix="€",
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
        "Occasion is the clearest signal",
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
        f"<p style='font-size:15px; color:{PALETTE['sage']}; margin-bottom:20px;'>"
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
        "Age is a reliable predictor of premium preference",
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
        f"<p style='font-size:15px; color:{PALETTE['sage']}; margin-bottom:20px;'>"
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
        "The loyalty blind spot",
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
        f"<p style='font-size:15px; color:{PALETTE['sage']}; margin-bottom:24px;'>"
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
                <p style='margin:0; font-size:15px; color:{PALETTE["sage"]};
                           line-height:1.7;'>{body}</p>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

def render_actions(df_f: pd.DataFrame) -> None:
    st.subheader("Strategic Action Plan")
    st.caption(
        "Concrete marketing recommendations per segment, ordered by priority. "
        "Each action is grounded in the survey data."
    )

    selected = st.multiselect(
        "Filter by segment",
        options=list(SEGMENT_META.keys()),
        default=list(SEGMENT_META.keys()),
    )

    rev_by_seg = df_f.groupby("FM_segment")["Ticket"].mean().to_dict()
    prio_rank  = {"High": 0, "Medium": 1, "Low": 2}
    ordered    = sorted(
        selected,
        key=lambda s: (prio_rank[SEGMENT_META[s]["priority"]], -rev_by_seg.get(s, 0)),
    )
    prio_map = {"High": "priority-high", "Medium": "priority-med", "Low": "priority-low"}

    for seg in ordered:
        meta    = SEGMENT_META[seg]
        n       = int((df_f["FM_segment"] == seg).sum())
        avg_tkt = df_f.loc[df_f["FM_segment"] == seg, "Ticket"].mean()
        act_html = "".join(
            [f"<li style='margin-bottom:6px;'>{a}</li>" for a in meta["actions"]]
        )
        st.markdown(f"""
        <div class='segment-card' style='border-top-color:{SEGMENT_COLORS[seg]};'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h3 style='margin:0;'>{seg}</h3>
                <span class='{prio_map[meta["priority"]]}'>Priority: {meta["priority"]}</span>
            </div>
            <p style='color:{PALETTE["charcoal"]}; font-style:italic; margin:8px 0;'>
                {meta["summary"]}
            </p>
            <p style='color:{PALETTE["charcoal"]}; margin:4px 0;'>
                <b>{n}</b> customers &nbsp;·&nbsp;
                Avg ticket: <b>€{avg_tkt:.0f} per visit</b>
            </p>
            <hr style='border:none; border-top:1px solid #EEE; margin:10px 0;'>
            <strong style='color:{PALETTE["burgundy"]};'>Recommended actions</strong>
            <ul style='color:{PALETTE["charcoal"]}; margin-top:8px;
                       padding-left:20px;'>{act_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:{PALETTE["burgundy"]}; color:white; padding:18px 22px;
                border-radius:8px; margin-top:14px;'>
        <strong>Cross-cutting priority</strong><br><br>
        Address the deli perception gap. Olives dominate the product mix and premium
        adoption is uneven across all segments. A wine-and-deli pairing display at
        eye level, combined with staff trained to suggest one product match per
        purchase, could shift the average basket within a quarter and begin to
        reposition the store away from wine-only.
    </div>
    """, unsafe_allow_html=True)


def render_explorer(df_f: pd.DataFrame) -> None:
    st.subheader("Customer Explorer")
    st.caption(
        "Filter and export individual customer records. "
        "Use this to build targeted outreach lists — for example, all Champions "
        "aged 31–40 who pay in cash — and download as CSV."
    )

    e1, e2 = st.columns(2)
    seg_filter = e1.multiselect(
        "Segment",
        options=df_f["FM_segment"].unique().tolist(),
        default=df_f["FM_segment"].unique().tolist(),
    )
    product_filter = e2.multiselect(
        "Deli product",
        options=df_f["Additional products"].unique().tolist(),
        default=df_f["Additional products"].unique().tolist(),
    )

    df_x = df_f[
        df_f["FM_segment"].isin(seg_filter)
        & df_f["Additional products"].isin(product_filter)
    ].copy()

    if df_x.empty:
        st.info("No customers match the current selection.")
        return

    df_show = df_x[[
        "ID", "Gender", "Age", "Education",
        "Wine frequency consumption", "Place to drink", "Additional products",
        "Payment mode", "Ticket", "FM_segment",
    ]].rename(columns={
        "Wine frequency consumption": "Purchase frequency",
        "Place to drink":             "Primary occasion",
        "Additional products":        "Deli product",
        "Payment mode":               "Payment",
        "FM_segment":                 "Segment",
    })

    st.dataframe(df_show, use_container_width=True, hide_index=True, height=520)

    csv = df_show.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV",
        data=csv,
        file_name="wine_shop_customers_filtered.csv",
        mime="text/csv",
    )


# ============================================================================
# Application entry point
# ============================================================================

def main() -> None:
    df_raw = load_data()
    df     = compute_fm_segments(df_raw)

    # Sidebar — global filters
    with st.sidebar:
        st.markdown(
            f"<h2 style='color:{PALETTE['burgundy']};'>Wine Shop</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color:{PALETTE['charcoal']}; font-style:italic; margin-top:-12px;'>"
            f"Customer Intelligence Dashboard</p>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("**Filters**")

        gender_sel = st.multiselect(
            "Gender",
            options=sorted(df["Gender"].unique()),
            default=sorted(df["Gender"].unique()),
        )
        age_sel = st.multiselect(
            "Age band",
            options=[a for a in AGE_ORDER if a in df["Age"].unique()],
            default=[a for a in AGE_ORDER if a in df["Age"].unique()],
        )
        payment_sel = st.multiselect(
            "Payment method",
            options=sorted(df["Payment mode"].unique()),
            default=sorted(df["Payment mode"].unique()),
        )
        ticket_range = st.slider(
            "Ticket range (€)",
            int(df["Ticket"].min()), int(df["Ticket"].max()),
            (int(df["Ticket"].min()), int(df["Ticket"].max())),
        )

        st.markdown("---")
        with st.expander("Visit frequency assumptions"):
            st.caption("Midpoint estimates used for the annual revenue calculation.")
            for label, visits in FREQ_VISITS_PER_MONTH.items():
                st.caption(f"· {label} → {visits} visits / month")

        with st.expander("About this dashboard"):
            st.caption(
                "Built on a 404-customer survey for the Wine Shop & Delicatessen "
                "strategic marketing case study. FM segmentation is the primary "
                "analytical lens; K-Prototypes clustering provides independent validation. "
                "Revenue figures are directional estimates, not actual store data."
            )

    mask = (
        df["Gender"].isin(gender_sel)
        & df["Age"].isin(age_sel)
        & df["Payment mode"].isin(payment_sel)
        & df["Ticket"].between(ticket_range[0], ticket_range[1])
    )
    df_f = df[mask].copy()

    st.markdown(
        f"""
        <div style='border-bottom:3px solid {PALETTE["burgundy"]};
                    padding-bottom:10px; margin-bottom:18px;'>
            <h1 style='margin:0;'>Wine Shop & Delicatessen</h1>
            <p style='margin:4px 0 0 0; color:{PALETTE["charcoal"]}; font-style:italic;'>
                Strategic Customer Intelligence &nbsp;·&nbsp;
                {len(df_f)} of {len(df)} customers in view
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_f.empty:
        st.warning("No customers match the current filters — please widen the selection.")
        st.stop()

    tabs = st.tabs([
        "Overview",
        "Customer Profile",
        "Market Segments",
        "ML Clustering",
        "Customer Behaviour",
        "Action Plan",
        "Customer Explorer",
    ])

    with tabs[0]: render_overview(df_f)
    with tabs[1]: render_profile(df_f)
    with tabs[2]: render_segments(df_f)
    with tabs[3]: render_clustering(df_f)
    with tabs[4]: render_products(df_f)
    with tabs[5]: render_actions(df_f)
    with tabs[6]: render_explorer(df_f)


if __name__ == "__main__":
    main()
