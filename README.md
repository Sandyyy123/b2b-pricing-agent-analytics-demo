# B2B Pricing Agent Analytics Demo

> **Proprietary — All Rights Reserved.** This repository is publicly visible for code-review and architectural-inspection purposes only. No license is granted to copy, modify, distribute, or use this code without prior written permission from Sandeep Grover. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).

> **Synthetic Data Disclaimer:** All accounts, billing rows, price elasticities, CLV figures, and realization bridges in this repository are computed on synthetic B2B SaaS transactions (`pricing_analytics.py`). This pre-existing illustrative prototype contains zero client data and does not claim pre-verified causal uplift on any external customer base.

## Overview

An AI Pricing Agent requires **deterministic, unit-tested quantitative modules** with strict input/output contracts, explicit confidence/uncertainty intervals, commercial guardrails, and loud failure warnings when sample sizes are thin or reconciliation checks fail.

This repository implements a callable Python reference architecture aligned with a 5-stage B2B pricing roadmap:
1. **Milestone 1 — Foundations (`validate_baseline_dataset`):** Builds and validates the customer × month transaction panel (`64` accounts, `768` monthly rows), verifies invoice-to-ledger revenue tie-out (`reconciliation_diff_usd = 0.00`), checks unit cost floors, and emits structured warnings (`RECONCILIATION_MISMATCH`, `BELOW_COST_PRICING`) on corrupted inputs.
2. **Milestone 2 — Customer Analytics (`compute_rfm_clv_and_segments`):** Computes B2B account Recency, Frequency, Monthly ARPA, gross margin, cohort retention, and discounted 24-month Customer Lifetime Value (`CLV`) with `[p10, p50, p90]` uncertainty bands across interpretable segments (`Strategic_Enterprise`, `MidMarket_Expansion`, `PriceSensitive_SMB`, `AtRisk_Renewal`).
3. **Milestone 3 — Price Response & Guardrailed Optimization (`estimate_price_response_and_optimize`):** Estimates segment-level price elasticity (`elasticity_hat` ± 95% CI) from historical renewal adjustments and optimizes price recommendations on conservative lower-bound (`p10`) gross profit while enforcing hard commercial guardrails (`min_gross_margin_pct >= 68%`, `max_increase_pct_strategic <= 6%`, `AtRisk_Renewal` freeze at `<= 0%`, and `INSUFFICIENT_SAMPLE_FALLBACK` when `n < min_segment_sample_size`).
4. **Milestones 4 & 5 — Forecasting & Price Realization (`decompose_price_realization`):** Generates 90-day and 12-month `[p10, p50, p90]` net revenue lift forecasts and decomposes expected vs. realized monthly revenue into **Gross List-Price Effect**, **Unapproved Discount Leakage**, and **Volume/Retention Offset**.

## Quick Start

```bash
python3 main.py --seed 20261105 --out results.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v test_pricing_analytics.py
```

See [`README_SCOPE_NOTE.md`](./README_SCOPE_NOTE.md) for explicit synthetic fixture, between-account CLV variation, OLS slope, and sensitivity-multiplier disclosures.
