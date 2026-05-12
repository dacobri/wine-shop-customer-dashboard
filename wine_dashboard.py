"""
wine_dashboard.py
=================
Wine Shop & Delicatessen — Strategic Customer Dashboard.

This is the Streamlit app entrypoint. It owns the sidebar, the tab
structure, and the per-tab render functions. All business logic lives
in the sibling modules:

    wine_data.py         data loading, cleaning, FM segmentation
    wine_clustering.py   K-Prototypes / K-Means clustering + diagnostics
    wine_simulator.py    what-if revenue projection
    wine_config.py       strategic recommendations per segment (playbook)
    wine_theme.py        palette, CSS, plotly layout, render helpers

INSTALL
-------
    pip install streamlit plotly pandas numpy scikit-learn kmodes openpyxl

RUN
---
    streamlit run wine_dashboard.py

Course: Advanced Programming with Python (ESADE MSc Business Analytics)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Local modules ---------------------------------------------------------
from wine_theme      import (PALETTE, SEQ, SEGMENT_COLORS,
                             inject_css, base_layout, kpi_card, callout)
from wine_data       import (load_data, compute_fm_segments,
                             FREQ_ORDER, FREQ_VISITS_PER_MONTH, AGE_ORDER)
from wine_clustering import (fit_clusters, cluster_diagnostics,
                             name_clusters_by_spend, KPROTOTYPES_OK)
from wine_simulator  import simulate_revenue
from wine_config     import SEGMENT_META


# ============================================================================
# 1. PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Wine Shop Customer Dashboard",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ============================================================================
# 2. TAB RENDERERS
# ============================================================================

def render_overview(df_f: pd.DataFrame):
    st.subheader("Business at a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Customers", f"{len(df_f):,}"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Avg ticket", f"{df_f['Ticket'].mean():.1f}", "€"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Est. annual revenue", f"{df_f['est_annual_revenue'].sum()/1000:.0f}K", "€"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Active segments", f"{df_f['FM_segment'].nunique()}"), unsafe_allow_html=True)

    st.markdown(" ")
    col_a, col_b = st.columns(2)

    # Customers per segment
    seg = df_f["FM_segment"].value_counts().reset_index()
    seg.columns = ["Segment", "Customers"]
    fig = px.pie(seg, values="Customers", names="Segment", hole=0.55,
                 color="Segment", color_discrete_map=SEGMENT_COLORS)
    fig.update_traces(textinfo="percent+label", textfont_size=12)
    fig.update_layout(base_layout(title="Customer share by segment",
                                  showlegend=False, height=400))
    col_a.plotly_chart(fig, use_container_width=True)

    # Revenue per segment
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

    # Headline insight callout
    top_seg = rev.iloc[-1]
    pct_customers = (df_f["FM_segment"] == top_seg["FM_segment"]).mean() * 100
    st.markdown(callout(
        "🍇", "Headline insight",
        f"<b>{top_seg['FM_segment']}</b> represent {pct_customers:.0f}% of customers "
        f"but contribute <b>{top_seg['share']:.0f}%</b> of estimated annual revenue. "
        f"Protecting and growing this group is the highest-priority action."
    ), unsafe_allow_html=True)

    # ---- What-If simulator ----
    st.markdown("---")
    st.subheader("📈 What-If Simulator")
    st.caption("Move the levers to project the revenue impact of strategic actions.")

    s1, s2, s3 = st.columns(3)
    pct_convert_casual = s1.slider(
        "Convert Casual Visitors → Loyal Regulars (%)", 0, 100, 20, 5,
        help="Share of casual visitors converted to loyal regulars via activation campaigns",
    )
    ticket_lift_loyal = s2.slider(
        "Loyal Regulars ticket lift (€)", 0, 30, 5, 1,
        help="Incremental basket value per Loyal Regular from the 'Pair It' deli upsell",
    )
    pct_convert_loyal = s3.slider(
        "Convert Loyal Regulars → Champions (%)", 0, 50, 10, 5,
        help="Share of loyal regulars elevated to champions via the loyalty programme",
    )

    sim_rev = simulate_revenue(df_f, pct_convert_casual, ticket_lift_loyal, pct_convert_loyal)
    base_rev = df_f["est_annual_revenue"].sum()
    delta = sim_rev - base_rev
    delta_pct = (delta / base_rev * 100) if base_rev else 0

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Baseline annual revenue", f"€{base_rev/1000:.0f}K")
    rc2.metric("Simulated annual revenue", f"€{sim_rev/1000:.0f}K", f"€{delta/1000:+.0f}K")
    rc3.metric("Uplift", f"{delta_pct:+.1f}%")

    st.caption(
        "Mechanics: a slice of Casual Visitors is given Loyal-Regulars-level visit frequency; "
        "Loyal Regulars' ticket gets the lift you set; a slice of Loyal Regulars adopts "
        "Champions-level ticket. Annual revenue = ticket × visits/month × 12."
    )


def render_profile(df_f: pd.DataFrame):
    st.subheader("Who walks through the door?")
    st.caption("Demographic and behavioral profile of the (filtered) customer base.")

    c1, c2, c3 = st.columns(3)

    g = df_f["Gender"].value_counts().reset_index()
    g.columns = ["Gender", "Count"]
    fig = px.pie(g, values="Count", names="Gender", hole=0.5,
                 color_discrete_sequence=[PALETTE["burgundy"], PALETTE["gold"]])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Gender", height=320, showlegend=False))
    c1.plotly_chart(fig, use_container_width=True)

    a = df_f["Age"].value_counts().reindex(AGE_ORDER).fillna(0).reset_index()
    a.columns = ["Age", "Count"]
    fig = px.bar(a, x="Age", y="Count", color_discrete_sequence=[PALETTE["burgundy"]])
    fig.update_layout(base_layout(title="Age band", height=320, showlegend=False))
    c2.plotly_chart(fig, use_container_width=True)

    p = df_f["Payment mode"].value_counts().reset_index()
    p.columns = ["Payment", "Count"]
    fig = px.pie(p, values="Count", names="Payment", hole=0.5,
                 color_discrete_sequence=[PALETTE["burgundy"], PALETTE["gold"], PALETTE["sage"]])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Payment mode", height=320, showlegend=False))
    c3.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    freq = df_f["Wine frequency consumption"].value_counts().reindex(FREQ_ORDER).fillna(0).reset_index()
    freq.columns = ["Frequency", "Count"]
    fig = px.bar(freq, x="Count", y="Frequency", orientation="h",
                 color_discrete_sequence=[PALETTE["burgundy"]])
    fig.update_layout(base_layout(title="Wine consumption frequency", height=360, showlegend=False))
    c4.plotly_chart(fig, use_container_width=True)

    place = df_f["Place to drink"].value_counts().reset_index()
    place.columns = ["Place", "Count"]
    fig = px.bar(place, x="Count", y="Place", orientation="h",
                 color_discrete_sequence=[PALETTE["gold"]])
    fig.update_layout(base_layout(title="Where customers drink wine", height=360, showlegend=False))
    c5.plotly_chart(fig, use_container_width=True)

    edu = df_f["Education"].dropna().value_counts().reset_index()
    edu.columns = ["Education", "Count"]
    fig = px.bar(edu, x="Count", y="Education", orientation="h",
                 color_discrete_sequence=[PALETTE["merlot"]])
    fig.update_layout(base_layout(title="Education level", height=380, showlegend=False))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df_f, x="Ticket", nbins=25, color_discrete_sequence=[PALETTE["burgundy"]])
    fig.update_layout(base_layout(title="Ticket distribution (€)", height=320, showlegend=False,
                                  xaxis_title="Ticket (€)", yaxis_title="Customers"))
    st.plotly_chart(fig, use_container_width=True)


def render_segments(df_f: pd.DataFrame):
    st.subheader("🎯 FM Segmentation — the strategic framework")
    st.caption(
        "Frequency × Monetary: customers are classified by how often they visit and "
        "how much they spend per visit. A median split produces four actionable segments."
    )

    seg_summary = df_f.groupby("FM_segment").agg(
        customers=("ID", "count"),
        avg_ticket=("Ticket", "mean"),
        avg_visits=("monthly_visits", "mean"),
        revenue=("est_annual_revenue", "sum"),
    ).reset_index()
    seg_summary["share_customers"] = seg_summary["customers"] / seg_summary["customers"].sum() * 100
    total_rev = seg_summary["revenue"].sum() or 1
    seg_summary["share_revenue"] = seg_summary["revenue"] / total_rev * 100

    cols = st.columns(4)
    seg_order = ["Champions", "Loyal Regulars", "Occasion Splurgers", "Casual Visitors"]
    prio_class_map = {"High": "priority-high", "Medium": "priority-med", "Low": "priority-low"}

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

    # FM quadrant scatter
    f_med = df_f["monthly_visits"].median()
    m_med = df_f["Ticket"].median()
    fig = px.scatter(
        df_f, x="monthly_visits", y="Ticket", color="FM_segment",
        color_discrete_map=SEGMENT_COLORS,
        hover_data=["Age", "Gender", "Place to drink", "Additional products"],
        labels={"monthly_visits": "Visits / month (estimated)", "Ticket": "Avg ticket (€)"},
        opacity=0.75,
    )
    fig.add_hline(y=m_med, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.4,
                  annotation_text=f"Median ticket €{m_med:.0f}", annotation_position="bottom right")
    fig.add_vline(x=f_med, line_dash="dash", line_color=PALETTE["charcoal"], opacity=0.4,
                  annotation_text=f"Median visits {f_med:.1f}/mo", annotation_position="top left")
    fig.update_layout(base_layout(title="FM quadrant — every dot is a customer", height=520))
    st.plotly_chart(fig, use_container_width=True)

    # Product mix per segment
    st.markdown("**Product mix per segment**")
    prod = (df_f.groupby(["FM_segment", "Additional products"]).size()
                  .reset_index(name="customers"))
    fig = px.bar(prod, x="customers", y="FM_segment", color="Additional products",
                 orientation="h", color_discrete_sequence=SEQ,
                 category_orders={"FM_segment": seg_order})
    fig.update_layout(base_layout(title="Deli product mix by segment", height=380,
                                  barmode="stack", xaxis_title="Customers", yaxis_title=""))
    st.plotly_chart(fig, use_container_width=True)

    # Demographic mix per segment
    st.markdown("**Demographic mix per segment**")
    dc1, dc2 = st.columns(2)
    age_mix = (pd.crosstab(df_f["FM_segment"], df_f["Age"], normalize="index") * 100)
    fig = px.bar(age_mix.reset_index().melt(id_vars="FM_segment", var_name="Age", value_name="pct"),
                 x="pct", y="FM_segment", color="Age", orientation="h",
                 color_discrete_sequence=SEQ,
                 category_orders={"FM_segment": seg_order, "Age": AGE_ORDER})
    fig.update_layout(base_layout(title="Age mix per segment (%)", height=360, barmode="stack",
                                  xaxis_title="% of segment", yaxis_title=""))
    dc1.plotly_chart(fig, use_container_width=True)

    gen_mix = (pd.crosstab(df_f["FM_segment"], df_f["Gender"], normalize="index") * 100)
    fig = px.bar(gen_mix.reset_index().melt(id_vars="FM_segment", var_name="Gender", value_name="pct"),
                 x="pct", y="FM_segment", color="Gender", orientation="h",
                 color_discrete_sequence=[PALETTE["burgundy"], PALETTE["gold"]],
                 category_orders={"FM_segment": seg_order})
    fig.update_layout(base_layout(title="Gender mix per segment (%)", height=360, barmode="stack",
                                  xaxis_title="% of segment", yaxis_title=""))
    dc2.plotly_chart(fig, use_container_width=True)


def render_clustering(df_f: pd.DataFrame):
    st.subheader("🤖 Unsupervised Clustering — spend tiers")
    if KPROTOTYPES_OK:
        st.caption(
            "**K-Prototypes** handles the mixed-type survey data correctly: Hamming distance "
            "on the six categorical features, Euclidean distance on Ticket."
        )
    else:
        st.warning(
            "`kmodes` is not installed — falling back to K-Means on one-hot features. "
            "For the proper K-Prototypes algorithm: `pip install kmodes`."
        )

    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.slider("Number of clusters (K)", 2, 6, 3)
        st.caption("Default K=3 → Entry / Core / Premium spend tiers (per the report).")
        algo_label = "K-Prototypes" if KPROTOTYPES_OK else "K-Means (fallback)"
        st.info(f"**Algorithm:** {algo_label}")

    ks, inertias, sils = cluster_diagnostics(df_f)
    diag = make_subplots(specs=[[{"secondary_y": True}]])
    diag.add_trace(go.Scatter(x=ks, y=inertias, name="Inertia (elbow)",
                              line=dict(color=PALETTE["burgundy"], width=3),
                              mode="lines+markers"), secondary_y=False)
    diag.add_trace(go.Scatter(x=ks, y=sils, name="Silhouette",
                              line=dict(color=PALETTE["gold"], width=3, dash="dot"),
                              mode="lines+markers"), secondary_y=True)
    diag.update_layout(base_layout(title="Choosing K — elbow & silhouette diagnostics", height=340))
    diag.update_xaxes(title="K (clusters)")
    diag.update_yaxes(title="Inertia (lower=tighter)", secondary_y=False)
    diag.update_yaxes(title="Silhouette (higher=cleaner)", secondary_y=True)
    c2.plotly_chart(diag, use_container_width=True)

    labels = fit_clusters(df_f, k=k)
    df_c = df_f.copy()
    df_c["cluster_id"] = labels
    df_c["cluster_name"] = df_c["cluster_id"].map(name_clusters_by_spend(df_c, "cluster_id"))

    csum = df_c.groupby("cluster_name").agg(
        Customers=("ID", "count"),
        Avg_ticket=("Ticket", "mean"),
        Min_ticket=("Ticket", "min"),
        Max_ticket=("Ticket", "max"),
        Avg_visits=("monthly_visits", "mean"),
        Revenue=("est_annual_revenue", "sum"),
    ).reset_index().sort_values("Avg_ticket")
    csum["Avg_ticket"] = csum["Avg_ticket"].round(1)
    csum["Avg_visits"] = csum["Avg_visits"].round(1)
    csum["Revenue"] = (csum["Revenue"] / 1000).round(0).astype(int).astype(str) + "K"
    csum.columns = ["Cluster", "Customers", "Avg ticket (€)", "Min ticket", "Max ticket",
                    "Avg visits/mo", "Annual revenue (€)"]
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
    fig.update_layout(base_layout(title="Clusters projected onto Frequency × Ticket", height=420))
    cv1.plotly_chart(fig, use_container_width=True)

    sizes = df_c["cluster_name"].value_counts().reset_index()
    sizes.columns = ["Cluster", "Customers"]
    fig = px.pie(sizes, values="Customers", names="Cluster", hole=0.5,
                 color_discrete_sequence=SEQ)
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(base_layout(title="Cluster sizes", height=420, showlegend=False))
    cv2.plotly_chart(fig, use_container_width=True)

    st.markdown("**Do unsupervised clusters validate the FM segments?**")
    ct = pd.crosstab(df_c["FM_segment"], df_c["cluster_name"])
    fig = px.imshow(ct, text_auto=True, aspect="auto",
                    color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["burgundy"]]],
                    labels=dict(x="ML cluster", y="FM segment", color="Customers"))
    fig.update_layout(base_layout(title="FM segment × ML cluster", height=340))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(callout(
        "✅", "What this tells us",
        "If the diagonal of the heatmap is dense, the unsupervised model independently "
        "rediscovers the FM grouping — confirming spend is the primary differentiator and "
        "the FM framework is a sound basis for marketing actions."
    ), unsafe_allow_html=True)


def render_products(df_f: pd.DataFrame):
    st.subheader("🧀 Product Mix & Cross-Sell Opportunity")
    st.caption(
        "The store is perceived as a wine shop rather than a delicatessen. "
        "This view diagnoses where the deli is winning and where the gaps are."
    )

    c1, c2 = st.columns(2)
    sold = df_f["Additional products"].value_counts().reset_index()
    sold.columns = ["Product", "Customers"]
    fig = px.bar(sold, x="Customers", y="Product", orientation="h",
                 color="Customers",
                 color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["burgundy"]]])
    fig.update_layout(base_layout(title="Most popular deli products", height=440,
                                  coloraxis_showscale=False, yaxis=dict(autorange="reversed")))
    c1.plotly_chart(fig, use_container_width=True)

    tier = df_f["deli_tier"].value_counts().reset_index()
    tier.columns = ["Tier", "Customers"]
    fig = px.pie(tier, values="Customers", names="Tier", hole=0.55,
                 color="Tier",
                 color_discrete_map={
                     "Premium": PALETTE["burgundy"],
                     "Entry":   PALETTE["gold"],
                     "Other":   PALETTE["sage"],
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
                    color_continuous_scale=[[0, PALETTE["cream"]], [1, PALETTE["burgundy"]]],
                    labels=dict(x="Product", y="Segment", color="% of segment"))
    fig.update_layout(base_layout(title="Product preference by segment (%)", height=380))
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
    st.markdown("**Top deli choice per segment** — the natural anchor product for each group's bundles")
    st.dataframe(top1, hide_index=True, use_container_width=True)


def render_actions(df_f: pd.DataFrame):
    st.subheader("📋 Strategic Action Plan")
    st.caption("Concrete marketing playbook per segment — your prioritised to-do list.")

    selected = st.multiselect(
        "Filter by segment",
        options=list(SEGMENT_META.keys()),
        default=list(SEGMENT_META.keys()),
    )

    rev_by_seg = df_f.groupby("FM_segment")["est_annual_revenue"].sum().to_dict()
    prio_rank = {"High": 0, "Medium": 1, "Low": 2}
    ordered = sorted(
        selected,
        key=lambda s: (prio_rank[SEGMENT_META[s]["priority"]], -rev_by_seg.get(s, 0)),
    )
    prio_class_map = {"High": "priority-high", "Medium": "priority-med", "Low": "priority-low"}

    for seg in ordered:
        meta = SEGMENT_META[seg]
        n = int((df_f["FM_segment"] == seg).sum())
        rev = df_f.loc[df_f["FM_segment"] == seg, "est_annual_revenue"].sum()
        actions_html = "".join([f"<li style='margin-bottom:6px;'>{a}</li>" for a in meta["actions"]])
        st.markdown(f"""
        <div class='segment-card' style='border-top-color:{SEGMENT_COLORS[seg]};'>
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h3 style='margin:0;'>{seg}</h3>
                <span class='{prio_class_map[meta["priority"]]}'>Priority: {meta["priority"]}</span>
            </div>
            <p style='color:{PALETTE["charcoal"]}; font-style:italic; margin:8px 0;'>{meta["summary"]}</p>
            <p style='color:{PALETTE["charcoal"]}; margin:4px 0;'>
                <b>{n}</b> customers · Estimated revenue: <b>€{rev/1000:.1f}K / year</b>
            </p>
            <hr style='border:none; border-top:1px solid #EEE; margin:10px 0;'>
            <strong style='color:{PALETTE["burgundy"]};'>Recommended actions</strong>
            <ul style='color:{PALETTE["charcoal"]}; margin-top:8px; padding-left:20px;'>{actions_html}</ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background: {PALETTE["burgundy"]}; color: white; padding: 18px 22px; border-radius: 8px; margin-top: 14px;'>
        <strong>🍷 Cross-cutting priority</strong><br>
        Address the delicatessen perception gap. Olives dominate the deli mix and premium
        adoption is uneven across segments. A simple <b>wine + deli pairing display at eye level</b>
        and staff trained to suggest one product match per purchase could shift the average basket
        within one quarter and reposition the store away from "wine-only".
    </div>
    """, unsafe_allow_html=True)


def render_explorer(df_f: pd.DataFrame):
    st.subheader("🔎 Customer Explorer")
    st.caption(
        "Search, sort, and filter individual customers. Build personalised outreach lists "
        "(e.g. all Champions aged 31–40 who pay in cash) and export as CSV."
    )

    e1, e2 = st.columns(2)
    seg_filter = e1.multiselect(
        "Segment", options=df_f["FM_segment"].unique().tolist(),
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
        "Place to drink": "Occasion",
        "Additional products": "Deli item",
        "Payment mode": "Payment",
        "monthly_visits": "Visits/mo",
        "est_annual_revenue": "Est. annual rev (€)",
        "FM_segment": "Segment",
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


# ============================================================================
# 3. APP MAIN
# ============================================================================

def main():
    df_raw = load_data()
    df = compute_fm_segments(df_raw)

    # ----- Sidebar: global filters -----
    with st.sidebar:
        st.markdown(f"<h2 style='color:{PALETTE['burgundy']};'>🍷 Wine Shop</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{PALETTE['charcoal']}; font-style:italic; margin-top:-12px;'>Customer Intelligence</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Global filters**")

        gender_sel = st.multiselect(
            "Gender", options=sorted(df["Gender"].unique()),
            default=sorted(df["Gender"].unique()),
        )
        age_sel = st.multiselect(
            "Age band", options=[a for a in AGE_ORDER if a in df["Age"].unique()],
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

        with st.expander("📖 About this dashboard"):
            st.caption(
                "Built on a 404-customer survey for the Wine Shop & Delicatessen case "
                "(Advanced Programming with Python — Session 3). "
                "FM segmentation is the primary actionable lens; K-Prototypes clustering "
                "validates it independently."
            )

    mask = (
        df["Gender"].isin(gender_sel)
        & df["Age"].isin(age_sel)
        & df["Payment mode"].isin(payment_sel)
        & df["Ticket"].between(ticket_range[0], ticket_range[1])
    )
    df_f = df[mask].copy()

    st.markdown(f"""
    <div style='border-bottom: 3px solid {PALETTE["burgundy"]}; padding-bottom: 10px; margin-bottom: 18px;'>
        <h1 style='margin: 0;'>🍷 Wine Shop & Delicatessen</h1>
        <p style='margin: 4px 0 0 0; color: {PALETTE["charcoal"]}; font-style: italic;'>
            Strategic Customer Intelligence Dashboard · {len(df_f)} of {len(df)} customers in view
        </p>
    </div>
    """, unsafe_allow_html=True)

    if len(df_f) == 0:
        st.warning("No customers match the current filters — widen them in the sidebar.")
        st.stop()

    tabs = st.tabs([
        "📊 Overview",
        "👥 Customer Profile",
        "🎯 Market Segments",
        "🤖 ML Clustering",
        "🧀 Product Mix",
        "📋 Action Plan",
        "🔎 Customer Explorer",
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
