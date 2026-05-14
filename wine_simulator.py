"""
wine_simulator.py
=================
What-if revenue projection for the Overview tab sliders.

Models three combined strategic interventions:

  1. Activate Casual Visitors — convert a share to Loyal-Regulars-level
     visit frequency via community and acquisition marketing.
  2. Upsell Loyal Regulars — lift average basket by €N through the
     deli pairing programme.
  3. Elevate Loyal Regulars — convert a share to Champions-level spend
     via the loyalty membership programme.

Revenue formula: ticket × monthly_visits × 12

This is a directional sensitivity model, not a financial forecast.
It shows the order-of-magnitude impact of each intervention, not
a precise revenue projection. Results should be read as "if X% of
Casual Visitors matched Loyal-Regular visit frequency, estimated
annual revenue would increase by approximately €Y."
"""

from __future__ import annotations
import pandas as pd


def simulate_revenue(
    df: pd.DataFrame,
    pct_casual: float,
    ticket_lift_loyal: float,
    pct_loyal: float,
) -> float:
    """
    Return projected annual revenue under the three combined interventions.

    Parameters
    ----------
    df                 : DataFrame with FM_segment, Ticket, monthly_visits columns
    pct_casual         : 0–100. Share of Casual Visitors converted to
                         Loyal-Regulars-level visit frequency.
    ticket_lift_loyal  : Euro amount added to every Loyal Regular's ticket.
    pct_loyal          : 0–100. Share of Loyal Regulars who adopt
                         Champions-level ticket spend.

    Returns
    -------
    float : projected annual revenue (€)
    """
    sim = df.copy()

    # Intervention 1: raise visit frequency for a share of Casual Visitors
    casual_mask = sim["FM_segment"] == "Casual Visitors"
    n_casual = int(casual_mask.sum() * pct_casual / 100)
    if n_casual > 0:
        loyal_visits = sim.loc[sim["FM_segment"] == "Loyal Regulars", "monthly_visits"]
        target_freq  = loyal_visits.median() if len(loyal_visits) else sim["monthly_visits"].median()
        convert_idx  = sim[casual_mask].sample(n=n_casual, random_state=1).index
        sim.loc[convert_idx, "monthly_visits"] = target_freq

    # Intervention 2: lift basket for all Loyal Regulars
    loyal_mask = sim["FM_segment"] == "Loyal Regulars"
    sim.loc[loyal_mask, "Ticket"] = sim.loc[loyal_mask, "Ticket"] + ticket_lift_loyal

    # Intervention 3: elevate a share of Loyal Regulars to Champions-level ticket
    n_loyal = int(loyal_mask.sum() * pct_loyal / 100)
    if n_loyal > 0:
        champion_ticket = sim.loc[sim["FM_segment"] == "Champions", "Ticket"]
        target_ticket   = (
            champion_ticket.median() if len(champion_ticket)
            else sim["Ticket"].quantile(0.75)
        )
        convert_idx2 = sim[loyal_mask].sample(n=n_loyal, random_state=2).index
        sim.loc[convert_idx2, "Ticket"] = target_ticket

    return float((sim["Ticket"] * sim["monthly_visits"] * 12).sum())
