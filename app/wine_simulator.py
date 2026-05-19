"""
wine_simulator.py
=================
What-If revenue projection — a small scenario engine used by the
Overview tab's interactive sliders.

The simulator quantifies the revenue impact of three combined
strategic interventions:

  1. Convert a share of Casual Visitors to Loyal-Regulars-level
     visit frequency (activation / community marketing).
  2. Lift Loyal Regulars' average ticket by € N (the "Pair It"
     deli upsell programme).
  3. Convert a share of Loyal Regulars to Champions-level ticket
     (the loyalty / membership programme).

Annual revenue = ticket × monthly_visits × 12 (assumed constant).

This is deliberately a simple back-of-envelope model — the goal is
to give the manager directional sensitivity, not financial accuracy.
"""

from __future__ import annotations
import pandas as pd


def simulate_revenue(
    df: pd.DataFrame,
    pct_casual: float,
    ticket_lift_loyal: float,
    pct_loyal: float,
) -> float:
    """Return projected annual revenue under the three combined interventions.

    Parameters
    ----------
    df : DataFrame with at least ['FM_segment', 'Ticket', 'monthly_visits']
    pct_casual : 0–100, share of Casual Visitors converted to Loyal-Regular visit cadence
    ticket_lift_loyal : € added to every Loyal Regular's ticket
    pct_loyal : 0–100, share of Loyal Regulars adopting Champions-level ticket
    """
    sim = df.copy()

    # 1) Convert Casuals → Loyal-Regular visit cadence
    casual_mask = sim["FM_segment"] == "Casual Visitors"
    n_casual = int(casual_mask.sum() * pct_casual / 100)
    if n_casual > 0:
        loyal_freq = sim.loc[sim["FM_segment"] == "Loyal Regulars", "monthly_visits"]
        target_freq = loyal_freq.median() if len(loyal_freq) else sim["monthly_visits"].median()
        convert_idx = sim[casual_mask].sample(n=n_casual, random_state=1).index
        sim.loc[convert_idx, "monthly_visits"] = target_freq

    # 2) Lift Loyal Regulars' ticket
    loyal_mask = sim["FM_segment"] == "Loyal Regulars"
    sim.loc[loyal_mask, "Ticket"] = sim.loc[loyal_mask, "Ticket"] + ticket_lift_loyal

    # 3) Convert Loyal Regulars → Champions ticket level
    n_loyal = int(loyal_mask.sum() * pct_loyal / 100)
    if n_loyal > 0:
        champion_ticket = sim.loc[sim["FM_segment"] == "Champions", "Ticket"]
        target_ticket = (
            champion_ticket.median() if len(champion_ticket)
            else sim["Ticket"].quantile(0.75)
        )
        convert_idx2 = sim[loyal_mask].sample(n=n_loyal, random_state=2).index
        sim.loc[convert_idx2, "Ticket"] = target_ticket

    return float((sim["Ticket"] * sim["monthly_visits"] * 12).sum())
