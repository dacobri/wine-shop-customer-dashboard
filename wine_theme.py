"""
wine_theme.py
=============
Visual identity: palette, CSS, Plotly layout defaults, HTML helpers.
"""

from __future__ import annotations
import streamlit as st

# ── Structural palette (UI chrome) ──────────────────────────────────────────
PALETTE = {
    "burgundy": "#7B2D3E",   # brand — headers, tab bar, KPI values
    "charcoal": "#1A1A1A",   # primary text
    "midgray":  "#6B7280",   # secondary text / captions
    "cream":    "#F5EEE4",   # sidebar, card backgrounds
    "ivory":    "#FAFAF8",   # page background
    "border":   "#E5DDD0",   # dividers
    # kept for backwards-compat references in non-chart code
    "gold":     "#D4872A",
    "merlot":   "#B5303F",
    "sage":     "#4E7C59",
    "teal":     "#2980B9",
    "rose":     "#D4606A",
}

# ── Segment colour maps ──────────────────────────────────────────────────────
# Four clearly distinct, readable hues — consistent across both segmentations
# so the same archetype always uses the same colour (Champions = Daily Home
# Drinkers = wine-red; Loyal Regulars = Daily Home Drinkers = steel-blue, etc.)

_C1 = "#B5303F"   # wine-red   — high-freq / high-priority
_C2 = "#2980B9"   # steel-blue — high-freq / lower-ticket
_C3 = "#D4872A"   # amber-gold — low-freq  / high-ticket / occasion
_C4 = "#4E7C59"   # forest     — low-freq  / low-engagement

SEGMENT_COLORS = {
    "Champions":          _C1,
    "Loyal Regulars":     _C2,
    "Occasion Splurgers": _C3,
    "Casual Visitors":    _C4,
}

BEHAVIORAL_COLORS = {
    "Social Regulars":      _C1,
    "Daily Home Drinkers":  _C2,
    "Occasion Celebrants":  _C3,
    "Casual Home Drinkers": _C4,
}

# Default categorical sequence for charts that aren't segment-coloured
SEQ = [_C1, _C2, _C3, _C4, "#8E44AD", "#2C3E50"]


# ── CSS ─────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(f"""
    <style>
        /* ── Page background ── */
        .stApp {{
            background-color: {PALETTE['ivory']};
        }}

        /* ── Typography ── */
        body, p, span, div {{
            color: {PALETTE['charcoal']};
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        }}
        h1, h2, h3, h4 {{
            color: {PALETTE['burgundy']} !important;
            font-family: 'Georgia', 'Garamond', serif;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: {PALETTE['cream']};
            padding: 5px 6px;
            border-radius: 10px;
            border: 1px solid {PALETTE['border']};
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            color: {PALETTE['charcoal']};
            font-weight: 500;
            font-size: 13px;
            border-radius: 6px;
            padding: 7px 13px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {PALETTE['burgundy']} !important;
            color: white !important;
        }}

        /* ── KPI cards ── */
        .kpi-card {{
            background: white;
            padding: 16px 18px;
            border-radius: 8px;
            border: 1px solid {PALETTE['border']};
            border-left: 4px solid {PALETTE['burgundy']};
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }}
        .kpi-label {{
            color: {PALETTE['midgray']};
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            margin: 0;
        }}
        .kpi-value {{
            color: {PALETTE['burgundy']};
            font-size: 28px;
            font-weight: 700;
            margin: 6px 0 0 0;
            font-family: 'Georgia', serif;
            line-height: 1;
        }}

        /* ── Segment cards ── */
        .segment-card {{
            background: white;
            padding: 16px 18px;
            border-radius: 8px;
            border: 1px solid {PALETTE['border']};
            border-top: 3px solid {PALETTE['burgundy']};
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
            margin-bottom: 14px;
        }}
        .segment-card h4 {{
            color: {PALETTE['charcoal']} !important;
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif !important;
        }}
        .segment-card p {{
            color: {PALETTE['charcoal']};
        }}

        /* ── Priority badges ── */
        .priority-high {{
            background: {_C1};
            color: white; padding: 2px 10px; border-radius: 10px;
            font-size: 11px; font-weight: 600;
        }}
        .priority-med {{
            background: {_C3};
            color: white; padding: 2px 10px; border-radius: 10px;
            font-size: 11px; font-weight: 600;
        }}
        .priority-low {{
            background: {_C4};
            color: white; padding: 2px 10px; border-radius: 10px;
            font-size: 11px; font-weight: 600;
        }}

        /* ── Metric widgets ── */
        div[data-testid="stMetricValue"] {{
            color: {PALETTE['burgundy']};
            font-family: 'Georgia', serif;
        }}

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {{
            background-color: {PALETTE['cream']};
            border-right: 1px solid {PALETTE['border']};
        }}
    </style>
    """, unsafe_allow_html=True)


# ── Plotly layout ────────────────────────────────────────────────────────────

def base_layout(**overrides) -> dict:
    layout = dict(
        font=dict(family="Inter, Helvetica Neue, Arial, sans-serif",
                  color=PALETTE["charcoal"], size=12),
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=SEQ,
        title=dict(font=dict(size=14, color=PALETTE["burgundy"],
                             family="Georgia, serif")),
        xaxis=dict(gridcolor="#F0EBE3", linecolor="#D0C8C0",
                   tickfont=dict(color=PALETTE["charcoal"])),
        yaxis=dict(gridcolor="#F0EBE3", linecolor="#D0C8C0",
                   tickfont=dict(color=PALETTE["charcoal"])),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(color=PALETTE["charcoal"])),
        margin=dict(t=48, b=36, l=40, r=20),
    )
    layout.update(overrides)
    return layout


# ── HTML helpers ─────────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, prefix: str = "") -> str:
    return f"""
    <div class='kpi-card'>
        <p class='kpi-label'>{label}</p>
        <p class='kpi-value'>{prefix}{value}</p>
    </div>
    """


def callout(emoji: str, title: str, body: str, color: str | None = None) -> str:
    color = color or PALETTE["burgundy"]
    return f"""
    <div style='background: white; padding: 14px 18px;
                border: 1px solid {PALETTE["border"]};
                border-left: 4px solid {color};
                border-radius: 6px; margin: 10px 0;'>
        <strong style='color: {color}; font-size:13px;'>{emoji} {title}</strong><br>
        <span style='color: {PALETTE["charcoal"]}; font-size:13px;'>{body}</span>
    </div>
    """
