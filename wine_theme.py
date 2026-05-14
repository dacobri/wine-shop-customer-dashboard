"""
wine_theme.py
=============
Visual identity: colour palette, custom CSS, Plotly layout defaults,
and HTML render helpers (KPI cards, insight callouts).

All styling is centralised here. To re-skin the dashboard, edit this
file only — no other module contains presentational logic.
"""

from __future__ import annotations
import streamlit as st

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE = {
    "burgundy": "#6B2737",
    "gold":     "#C9A961",
    "cream":    "#F4E8D0",
    "charcoal": "#2B2118",
    "merlot":   "#8B1A1A",
    "rose":     "#D4A5A5",
    "sage":     "#8B9474",
    "ivory":    "#FAF6EE",
}

# Default categorical sequence for Plotly charts
SEQ = [
    PALETTE["burgundy"], PALETTE["gold"],    PALETTE["merlot"],
    PALETTE["sage"],     PALETTE["rose"],    PALETTE["charcoal"],
]

# Segment colours — consistent across every chart and card
SEGMENT_COLORS = {
    "Champions":          PALETTE["burgundy"],
    "Loyal Regulars":     PALETTE["gold"],
    "Occasion Splurgers": PALETTE["merlot"],
    "Casual Visitors":    PALETTE["sage"],
}


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject dashboard CSS. Call once at application startup."""
    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, {PALETTE['ivory']} 0%, #FFFFFF 100%);
        }}
        h1, h2, h3, h4 {{
            color: {PALETTE['burgundy']} !important;
            font-family: 'Georgia', 'Garamond', serif;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: {PALETTE['cream']};
            padding: 6px;
            border-radius: 10px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {PALETTE['charcoal']};
            font-weight: 500;
            border-radius: 6px;
            padding: 8px 14px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {PALETTE['burgundy']};
            color: white !important;
        }}
        .kpi-card {{
            background: white;
            padding: 18px 20px;
            border-radius: 10px;
            border-left: 4px solid {PALETTE['burgundy']};
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }}
        .kpi-label {{
            color: {PALETTE['charcoal']};
            font-size: 13px;
            opacity: 0.75;
            margin: 0;
        }}
        .kpi-value {{
            color: {PALETTE['burgundy']};
            font-size: 30px;
            font-weight: 700;
            margin: 4px 0 0 0;
            font-family: 'Georgia', serif;
        }}
        .segment-card {{
            background: white;
            padding: 16px 18px;
            border-radius: 10px;
            border-top: 4px solid {PALETTE['gold']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 14px;
        }}
        .priority-high {{
            background: {PALETTE['burgundy']};
            color: white; padding: 3px 12px; border-radius: 12px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
        }}
        .priority-med {{
            background: {PALETTE['gold']};
            color: white; padding: 3px 12px; border-radius: 12px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
        }}
        .priority-low {{
            background: {PALETTE['sage']};
            color: white; padding: 3px 12px; border-radius: 12px;
            font-size: 11px; font-weight: 600; letter-spacing: 0.5px;
        }}
        div[data-testid="stMetricValue"] {{
            color: {PALETTE['burgundy']};
            font-family: 'Georgia', serif;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #F0E8DA;
        }}
        section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] {{
            background-color: {PALETTE['gold']} !important;
            border-color: {PALETTE['gold']} !important;
            border-radius: 4px !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="tag"] span {{
            color: white !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {{
            background-color: {PALETTE['gold']} !important;
            border-color: {PALETTE['gold']} !important;
        }}
        section[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div:first-child {{
            background-color: {PALETTE['gold']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly layout defaults
# ---------------------------------------------------------------------------

def base_layout(**overrides) -> dict:
    """
    Return a Plotly layout dict with consistent branding.

    Usage:
        fig.update_layout(base_layout(title="Chart title", height=400))
    """
    layout = dict(
        font=dict(family="Georgia, serif", color=PALETTE["charcoal"]),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=SEQ,
        title=dict(font=dict(size=15, color=PALETTE["burgundy"])),
        xaxis=dict(gridcolor="#EEE", linecolor=PALETTE["charcoal"]),
        yaxis=dict(gridcolor="#EEE", linecolor=PALETTE["charcoal"]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, prefix: str = "") -> str:
    """HTML for a single KPI metric card."""
    return f"""
    <div class='kpi-card'>
        <p class='kpi-label'>{label}</p>
        <p class='kpi-value'>{prefix}{value}</p>
    </div>
    """


def callout(title: str, body: str, color: str | None = None) -> str:
    """
    Coloured insight callout box.

    Usage:
        st.markdown(callout("Finding", "Body text."), unsafe_allow_html=True)
    """
    color = color or PALETTE["burgundy"]
    return f"""
    <div style='background:{PALETTE["ivory"]}; padding:16px 20px;
                border-left:4px solid {color}; border-radius:6px; margin:8px 0;'>
        <strong style='color:{color}; font-size:13px;'>{title}</strong><br>
        <span style='color:{PALETTE["charcoal"]}; font-size:13px;'>{body}</span>
    </div>
    """
