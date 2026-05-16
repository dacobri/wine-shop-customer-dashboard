"""
wine_config.py
==============
Domain knowledge: the strategic marketing playbook for both segmentation lenses.

  SEGMENT_META      — FM (Frequency × Monetary) segments
  BEHAVIORAL_META   — Behavioral K-Means segments (frequency × sociality)

Keeping recommendations as data means a marketing manager can update the
playbook by editing this one file — without touching the dashboard code.

Schema per segment:
    priority : "High" | "Medium" | "Low"   — drives the badge colour
    summary  : one-line description of the segment
    profile  : two-sentence behavioral description for the manager
    actions  : ordered list of concrete tactical actions
"""


SEGMENT_META = {
    "Champions": {
        "priority": "High",
        "summary": "High-frequency, high-spend buyers — the revenue engine.",
        "profile": "Come in several times a week and spend above the median. "
                   "They know the range, trust the shop, and will pay for quality.",
        "actions": [
            "Launch a named loyalty programme (e.g. \"Cave Club\") with digital membership",
            "Invite to private tastings, new-arrival previews, and producer evenings",
            "Offer premium hamper bundles (wine + Spanish ham or salmon*) as default upsell at checkout",
            "Ensure a consistent, high-quality in-store experience — this group is retention-sensitive",
        ],
    },
    "Loyal Regulars": {
        "priority": "High",
        "summary": "Frequent visitors with modest baskets — the biggest upsell opportunity.",
        "profile": "Visit as often as Champions but keep their basket small. "
                   "They're loyal by habit; unlocking deli spend is the lever.",
        "actions": [
            "Introduce a \"Pair It\" card at the till suggesting one deli item per wine",
            "Trade-up mechanic: spend €10 more, receive a complimentary deli taster",
            "Develop a monthly curated box (wine + 2–3 deli items) at a slight discount vs individual pricing",
            "Highlight premium deli provenance in-store with eye-level signage",
        ],
    },
    "Occasion Splurgers": {
        "priority": "Medium",
        "summary": "Infrequent visitors who spend big when they come — gift-driven.",
        "profile": "Come rarely but when they do, the basket is above median. "
                   "Occasions (birthdays, gifts, holidays) are the trigger.",
        "actions": [
            "Build a seasonal hamper range (Christmas, Easter, summer) priced €60–100",
            "Create shareable, social-media-ready gift packaging to drive word-of-mouth",
            "Run email/WhatsApp campaigns before key occasions (Valentine's, Mother's Day, festive season)",
            "Position the store as a one-stop gift destination — not just a wine shop",
        ],
    },
    "Casual Visitors": {
        "priority": "Low",
        "summary": "Low frequency, low spend — pipeline for future Loyal Regulars.",
        "profile": "Drop in occasionally, spend little. "
                   "Worth activating via low-barrier discovery events.",
        "actions": [
            "Activate on Instagram with accessible entry-level content (wine tips, easy pairings)",
            "Offer a \"first deli taster\" with any wine purchase to lower the barrier to cross-category trial",
            "Run a monthly Saturday tasting event — low ticket, social, discovery-focused",
            "Track conversion: if no migration to Loyal Regulars within 6 months, re-evaluate spend",
        ],
    },
}


# Behavioral segment playbook — derived from the K-Means clustering on
# visit frequency × sociality (see cluster_insights.ipynb).
# These segments describe HOW customers buy, not how much —
# complementary to the FM revenue lens above.
BEHAVIORAL_META = {
    "Social Regulars": {
        "priority": "High",
        "summary": "Frequent visitors who drink with others — the social ambassadors.",
        "profile": "Come several times a week and nearly always drink at restaurants, "
                   "parties, or with groups. They introduce the shop to new people.",
        "actions": [
            "Create a 'bring a friend' referral card — €5 off their next purchase when the friend buys",
            "Feature prominently in event and tasting communications; they will share on social media",
            "Offer party packs (3+ bottles + deli platter) at a bundle price for hosting occasions",
            "Invite to 'first look' evenings for new arrivals — they spread word-of-mouth faster than any advert",
        ],
    },
    "Daily Home Drinkers": {
        "priority": "High",
        "summary": "Near-daily buyers who drink privately — loyal but invisible to the deli.",
        "profile": "Visit very frequently but drink alone or with a partner at home. "
                   "62 % female. They buy wine on autopilot and rarely consider the deli.",
        "actions": [
            "\"Tonight's pairing\" shelf-talker next to bestselling everyday wines — one deli suggestion per label",
            "Introduce a subscription basket (weekly wine + one deli item) at a slight discount",
            "WhatsApp/email nudge mid-week: 'New arrival that pairs well with your usual red'",
            "Seasonal hamper tailored for two (wine + cheese or salmon*) positioned near the checkout",
        ],
    },
    "Occasion Celebrants": {
        "priority": "Medium",
        "summary": "Infrequent buyers who drink socially — motivated by special occasions.",
        "profile": "Come less often but almost always for a celebration, birthday, or holiday. "
                   "Largest group (41 % of customers) with high occasion-driven spend potential.",
        "actions": [
            "Seasonal campaigns (Valentine's, Mother's Day, Christmas) with curated gift sets €40–80",
            "In-store 'celebration station' display — champagnes, premium deli, and gift wrapping in one spot",
            "Capture email at purchase; send occasion reminder 3 weeks before key dates",
            "Promote the deli story on social media ahead of holiday weekends when this group is most active",
        ],
    },
    "Casual Home Drinkers": {
        "priority": "Low",
        "summary": "Infrequent, private drinkers — the largest conversion opportunity.",
        "profile": "Come rarely and drink alone or with a partner at home. "
                   "Largest group by count alongside Occasion Celebrants. "
                   "The lowest-engagement segment — need a reason to return.",
        "actions": [
            "Monthly discovery evening: €10 ticket, 4 wines, easy deli bites — low barrier, social hook",
            "Loyalty stamp card: every 5 visits earns a free deli taster",
            "Entry-level pairing guide available at the door — educates without pressure",
            "Re-engagement SMS/email after 8 weeks of inactivity with a low-friction offer",
        ],
    },
}
