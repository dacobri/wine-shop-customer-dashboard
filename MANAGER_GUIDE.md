# 🍷 Manager's Guide to the Dashboard

A one-page, non-technical walkthrough. The dashboard lives at
**`https://wine-shop-customer-dashboard-group3.streamlit.app`** and opens in
any modern browser. No login required.

**Built by:** Sabeena Awan · Brice Da Costa · Lucas Joris Haesaert · Patricia Unger
*ESADE MSc Business Analytics — Advanced Programming with Python*

---

## In 30 seconds

The dashboard turns the 404-customer survey into a working tool for
day-to-day marketing decisions. It groups customers using three
complementary lenses, shows what each group is worth, and gives you
concrete actions for each.

Two big questions it answers:

1. **Where is the money today?** — which customers drive revenue?
2. **Where is the next euro?** — which interventions would lift sales fastest?

---

## How to use it

### 1. Set the lens (sidebar, left side)

The filters at the top-left act on every chart at once. Use them when
you want to ask focused questions — e.g. *"How big is the women's
31–40 segment?"*

| Filter | What it does |
|---|---|
| Gender | Show only Male / Female / both |
| Age band | Restrict to one or more age ranges |
| Ticket range (€) | Restrict to a basket-value band |

Whenever a filter changes, every number on every tab recalculates.
*Exception:* the Customer Profile tab's hero card and demographic
composition views always show all 404 customers — that's the "true"
profile of the base regardless of what's filtered.

### 2. Read the tabs (top, left to right)

#### 👥 **Customer Profile** — who walks through the door

- **Hero card** — the typical basket and persona at a glance
- **A. Demographics** — population pyramid (Age × Gender) + education tiers
- **B. The gender × occasion signal** — the strongest behavioral pattern in
  the data (80% of birthday-party drinkers are men, 73% of restaurant
  drinkers are women)
- **C. Monthly spending heatmaps** — three views:
  *Gender × Education* (counter-intuitive: women with Basic education
  spend the most), *Age × Education* (life-stage × class),
  *Age × Occasion* (the marketing-actionable matrix)
- **D. Hidden patterns** — which customer attributes go together?
  Translates each top pair into a concrete marketing action. The
  expander below shows the academic statistics for the curious.

#### 🎯 **Market Segments (FM)** — who's worth what

The four FM personas:

| Segment | Profile | Strategy |
|---|---|---|
| **Champions** | High frequency, high spend | Protect — loyalty programme, VIP tastings |
| **Loyal Regulars** | High frequency, modest spend | Upsell — "Pair It" deli card at the till |
| **Occasion Splurgers** | Low frequency, big baskets | Capture occasions — seasonal hampers |
| **Casual Visitors** | Low frequency, small spend | Activate — social media, first-time tasters |

#### 🤖 **ML Clustering** — two lenses

A radio toggle at the top picks between:

- **Behavioral archetypes** — *Social Regulars*, *Daily Home Drinkers*,
  *Occasion Celebrants*, *Casual Home Drinkers*. Independent of ticket
  price — tells you *how* people buy.
- **Spend tiers** — *Entry / Core / Premium Buyers*. Anchored on ticket
  value — tells you *how much* people buy.

#### 🧀 **Product Mix** — the delicatessen gap

Olives dominate; cheese, salmon, and ham are under-bought. This view
diagnoses where the deli is winning and where the cross-sell
opportunity sits.

#### 📋 **Action Plan** — your prioritised to-do list

A radio at the top toggles between the *FM* and *behavioral* playbooks.
Each segment shows priority, customer count, estimated revenue, and a
list of 4 concrete actions.

#### 🔎 **Customer Explorer** — outreach lists

Filter to "all Champions aged 31–40 who pay in cash", export as CSV,
hand to your marketing team.

#### 📊 **Strategic Overview** — the headline + the simulator

The full picture in one view plus the **What-If Revenue Simulator** —
three sliders that project the revenue impact of conversion campaigns,
deli upsells, and loyalty programmes. An expander documents exactly
how the calculations work.

---

## The single most important insight

> **Champions are ~37% of customers but generate ~68% of estimated
> annual revenue.** Protecting and growing them is the single
> highest-priority action.
>
> The **Loyal Regulars** segment is the biggest upsell opportunity —
> they visit often but spend modestly, so the right deli-pairing
> programme could materially lift basket value.
>
> The data also flags a **delicatessen perception gap** — most
> customers default to olives, and premium deli adoption is uneven
> across segments.

---

## Reading the numbers honestly

- **Revenue is estimated**, not measured. We multiply the survey ticket
  by an assumed visits/month for each frequency category. The exact
  number is less important than the *relative size* between segments.
- **The survey is a snapshot of 404 customers.** It's representative
  of the case-study sample, not exhaustive. Connecting the dashboard
  to a live POS feed is on the roadmap.
- **Filters change everything.** A "headline" reads with the current
  filters applied — clear them to see the whole-store view.
- **Some data combinations are unrealistic.** A few customers in the
  dataset visit daily and spend €100 every time. We surface these as
  the survey reports them rather than clipping; the methodology would
  be re-fit on real POS data when available.

---

## Need help?

Hover any 🛈 icon in the dashboard for tooltips. For technical changes
(new tabs, different segmentation rules, connection to your POS),
contact the dashboard team.
