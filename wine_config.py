"""
wine_config.py
==============
Domain knowledge: the strategic marketing playbook per FM segment.

Pulled directly from the customer analysis report
(Wine_Shop_Customer_Report.docx).

Keeping these recommendations as data (rather than buried inside
Streamlit render functions) means a marketing manager can update the
playbook by editing this one file — without touching the dashboard
code.

Schema per segment:
    priority : "High" | "Medium" | "Low"   — drives the badge colour
    summary  : one-line description of the segment
    actions  : ordered list of concrete tactical actions
"""


SEGMENT_META = {
    "Champions": {
        "priority": "High",
        "summary": "High-frequency, high-spend buyers — the revenue engine.",
        "actions": [
            "Launch a named loyalty programme (e.g. \"Cave Club\") with digital membership",
            "Invite to private tastings, new-arrival previews, and producer evenings",
            "Offer premium hamper bundles (wine + Spanish ham or salmon) as default upsell at checkout",
            "Ensure a consistent, high-quality in-store experience — this group is retention-sensitive",
        ],
    },
    "Loyal Regulars": {
        "priority": "High",
        "summary": "Frequent visitors with modest baskets — the biggest upsell opportunity.",
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
        "actions": [
            "Activate on Instagram with accessible entry-level content (wine tips, easy pairings)",
            "Offer a \"first deli taster\" with any wine purchase to lower the barrier to cross-category trial",
            "Run a monthly Saturday tasting event — low ticket, social, discovery-focused",
            "Track conversion: if no migration to Loyal Regulars within 6 months, re-evaluate spend",
        ],
    },
}
