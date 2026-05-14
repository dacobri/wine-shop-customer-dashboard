"""
wine_config.py
==============
Strategic marketing playbook per FM segment.

Keeping recommendations as data (rather than inside render functions)
means the playbook can be updated by editing this file alone,
without touching dashboard code.

Schema per segment:
    priority : "High" | "Medium" | "Low"
    summary  : one-line description of the segment's strategic role
    actions  : ordered list of concrete tactical recommendations
"""

SEGMENT_META: dict[str, dict] = {
    "Champions": {
        "priority": "High",
        "summary": (
            "High-frequency, high-spend customers. They generate the largest share of "
            "estimated annual revenue and are retention-sensitive — losing even a small "
            "number has a disproportionate revenue impact."
        ),
        "actions": [
            "Introduce a named membership programme (e.g. Cave Club) with a digital card "
            "to make loyalty visible and trackable — currently 77% of this group pay cash, "
            "making their purchase history invisible.",
            "Invite members to private tastings, new-arrival previews and producer evenings. "
            "These cost little to run and significantly raise switching costs.",
            "Offer premium pairing hampers (wine + salmon or Spanish ham) as a default "
            "checkout suggestion. Salmon is already the top deli choice for this segment.",
            "Maintain consistent product availability. A stockout of a preferred product "
            "is a material churn risk for the highest-value customers.",
        ],
    },
    "Loyal Regulars": {
        "priority": "High",
        "summary": (
            "Frequent visitors with below-median basket sizes. The largest segment by "
            "customer count and the primary upsell opportunity: small increases in "
            "average ticket compound quickly given their visit frequency."
        ),
        "actions": [
            "Place a 'Pair It' recommendation card at the till — one suggested deli item "
            "per wine type. A €5 average basket lift across this segment adds roughly "
            "€5,000 to estimated monthly revenue.",
            "Introduce a spend-to-earn mechanic: spend €10 above the usual basket and "
            "receive a complimentary deli taster. This introduces premium products at "
            "zero barrier for the customer.",
            "Offer a monthly curated box (wine + 2–3 deli items) at a slight discount "
            "to individual pricing. Builds habit and grows the average transaction.",
            "Use eye-level deli display to feature provenance — origin, producer, "
            "suggested pairing. Informed customers buy up more reliably than prompted ones.",
        ],
    },
    "Occasion Splurgers": {
        "priority": "Medium",
        "summary": (
            "Infrequent visitors with above-median spend. They arrive with a specific "
            "purpose — typically a gift or celebration — and leave with the highest "
            "average basket of any segment (€81). The opportunity is to be "
            "front-of-mind when those occasions arise."
        ),
        "actions": [
            "Develop a seasonal hamper range (Christmas, Easter, summer) priced €60–100. "
            "Spanish ham is the top deli choice for this segment — anchor the range on it.",
            "Create gift-ready packaging with a card insert. The purchase decision is "
            "social and presentational, not purely about the product.",
            "Run pre-occasion outreach (WhatsApp, email) before Valentine's Day, "
            "Mother's Day and the festive period — these are the trigger moments.",
            "Position the store explicitly as a one-stop gifting destination. "
            "In-store signage and the website (if any) should reflect this.",
        ],
    },
    "Casual Visitors": {
        "priority": "Low",
        "summary": (
            "Low-frequency, low-spend customers. Small in revenue contribution but "
            "represent the conversion pipeline into Loyal Regulars. Investment here "
            "should be proportionate — high-reach, low-cost activities only."
        ),
        "actions": [
            "Offer a complimentary deli taster with any wine purchase to lower the "
            "barrier to cross-category trial. The goal is a first deli experience, "
            "not immediate revenue.",
            "Run a monthly Saturday tasting event — low ticket price, social format, "
            "discovery focused. Converts one-time visitors into repeat customers.",
            "Use social media (Instagram) for accessible, entry-level content: "
            "wine tips, easy food pairings, behind-the-scenes. Keeps the store "
            "present between visits.",
            "Track segment migration: if fewer than 10% of Casual Visitors move to "
            "Loyal Regulars within six months, reassess the activation investment.",
        ],
    },
}
