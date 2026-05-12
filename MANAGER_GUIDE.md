# 🍷 Manager's Guide to the Dashboard

A one-page, non-technical walkthrough. The dashboard lives at
**`https://wine-shop-customer-dashboard.streamlit.app`** *(URL once deployed)*
and opens in any modern browser. No login required.

---

## In 30 seconds

The dashboard turns the 404-customer survey into a working tool for
day-to-day marketing decisions. It groups customers into segments,
shows what each segment is worth, and gives you concrete actions for each.

Two big questions it answers:

1. **Where is the money?** — which customers drive revenue today?
2. **Where is the next euro?** — which interventions would lift sales fastest?

---

## How to use it

### 1. Set the lens (sidebar, left side)

The filters at the top-left act on every chart at once. Use them when
you want to ask focused questions — e.g. *"How big is the women's
31–40 segment that pays in cash?"*

| Filter | What it does |
|---|---|
| Gender | Show only Male / Female / both |
| Age band | Restrict to one or more age ranges |
| Payment mode | Cash / credit / debit |
| Ticket range (€) | Restrict to a basket-value band |

Whenever a filter changes, every number on every tab recalculates.

### 2. Read the tabs (top, left to right)

#### 📊 **Overview** — start here every morning
- Four headline numbers at the top: customers, average basket,
  estimated annual revenue, number of segments.
- Two charts: who they are by segment, and what each segment is worth.
- **The What-If simulator** at the bottom — your most important tool.
  Move the three sliders and see how the annual revenue would change:
  - *"If 20% of casual visitors become regulars, what's that worth?"*
  - *"If we lift loyal regulars' basket by €5, what's that worth?"*
  - *"If 10% of loyal regulars become champions, what's that worth?"*

#### 👥 **Customer Profile**
Who walks through your door. Age, gender, payment mix, where customers
drink the wine (parties, home, etc.), education, basket size. Useful
for marketing copy and channel decisions.

#### 🎯 **Market Segments**
The four segments at a glance — Champions (high freq + high spend),
Loyal Regulars (high freq + modest spend), Occasion Splurgers
(rare visitors + big baskets), Casual Visitors (rare + small).
Each has a coloured card with size, average ticket, and revenue share.
Below: the FM quadrant scatter chart shows every customer as a dot.

#### 🤖 **ML Clustering**
A second, independent way of grouping customers using machine learning.
It groups them by **spend tier** — Entry / Core / Premium buyers.
Useful as a sanity check: if it agrees with the FM segments,
you can trust the marketing strategy is built on a solid base.

#### 🧀 **Product Mix**
This is where the **delicatessen perception gap** is diagnosed.
Olives and Spanish ham dominate; cheese and salmon are under-bought.
Use the "Occasion × Product" heatmap to design in-store displays —
e.g. if "parties" customers tend to buy olives, place an olives display
next to the party wines.

#### 📋 **Action Plan**
Your prioritised marketing to-do list per segment. Each segment card
has 4 concrete actions, e.g.:
- Champions → loyalty programme, VIP tastings, premium hamper bundles
- Loyal Regulars → "Pair It" cards at the till, monthly curated boxes
- Occasion Splurgers → seasonal hampers, gift-ready packaging
- Casual Visitors → social media activation, first-time-buyer tasters

#### 🔎 **Customer Explorer**
Look up individual customers. Filter to "all Champions aged 31–40
who pay in cash" and click **⬇️ Download as CSV** for a personalised
outreach list.

---

## The single most important insight

> **Champions are ~37% of customers but generate ~68% of estimated annual revenue.**
> Protecting and growing them is the single highest-priority action.
> The Loyal Regulars segment is the biggest upsell opportunity — they
> visit often but spend modestly, so the right deli-pairing programme
> could materially lift basket value.

---

## Reading the numbers honestly

- **Revenue is estimated**, not measured. We multiply the survey ticket
  by an assumed visits/month for each frequency category. The exact
  number is less important than the *relative size* between segments.
- **The survey is a snapshot of 404 customers.** It's representative,
  not exhaustive. Connecting the dashboard to live POS data is on the
  roadmap.
- **Filters change everything.** A "headline insight" reads with the
  current filters applied — clear them in the sidebar to see the
  whole-store view.

---

## Need help?

Hover over any 🛈 icon in the dashboard for a quick tooltip. For
technical changes (new tabs, different segmentation rules, connection
to your POS), contact the development team.
