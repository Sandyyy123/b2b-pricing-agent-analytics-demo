"""Deterministic, callable B2B pricing analytics modules for an AI Pricing Agent.

Implements clean Python functions with validated input/output schemas, explicit
confidence/uncertainty intervals, business guardrails, and loud failure modes
when data are thin or assumptions are violated:

1. Milestone 1 (Foundations): Baseline B2B customer x product x month dataset
   builder, data-quality checks, and invoice-to-ledger revenue reconciliation.
2. Milestone 2 (Customer Analytics): RFM scoring, retention-adjusted 24-month
   Customer Lifetime Value (CLV) with [p10, p50, p90] ranges, and explainable
   B2B behavioral segmentation.
3. Milestone 3 (Price Response & Guardrailed Optimization): Segment-level
   price-response / retention elasticity estimation from historical price
   adjustments plus guardrail-enforced price recommendation callable by an agent.
4. Milestone 4 & 5 (Forecasting & Price Realization): 90-day and 12-month
   revenue/margin scenario ranges plus post-change expected-vs-actual price
   realization bridge decomposition (price effect, volume effect, discount
   leakage, and churn/mix effect).

NOTE: All records generated in this repository are synthetic B2B SaaS/usage
transactions for methodological demonstration only. They do not represent
client data or establish causal uplift on any external customer base.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import numpy as np


SEGMENT_NAMES = (
    "Strategic_Enterprise",
    "MidMarket_Expansion",
    "PriceSensitive_SMB",
    "AtRisk_Renewal",
)


@dataclasses.dataclass
class GuardrailConfig:
    """Explicit commercial guardrails enforced before returning any price action."""

    min_gross_margin_pct: float = 0.68
    max_increase_pct_default: float = 0.10
    max_increase_pct_strategic: float = 0.06
    max_increase_pct_at_risk: float = 0.00
    min_segment_sample_size: int = 12
    max_predicted_churn_delta: float = 0.025


def generate_synthetic_b2b_panel(
    n_accounts: int = 64,
    n_months: int = 12,
    seed: int = 20261105,
) -> Dict[str, np.ndarray]:
    """Generate a synthetic B2B account x month billing and renewal panel."""
    rng = np.random.default_rng(seed)

    account_ids: List[int] = []
    months: List[int] = []
    segments: List[str] = []
    list_unit_price: List[float] = []
    realized_unit_price: List[float] = []
    unit_cost: List[float] = []
    seats_volume: List[float] = []
    hist_price_change_pct: List[float] = []
    retained_next_period: List[int] = []
    unapproved_discount_pct: List[float] = []
    days_since_last_expansion: List[float] = []

    seg_probs = [0.25, 0.35, 0.25, 0.15]
    true_elasticities = {
        "Strategic_Enterprise": -0.22,
        "MidMarket_Expansion": -0.36,
        "PriceSensitive_SMB": -1.35,
        "AtRisk_Renewal": -1.75,
    }
    base_prices = {
        "Strategic_Enterprise": 120.0,
        "MidMarket_Expansion": 85.0,
        "PriceSensitive_SMB": 48.0,
        "AtRisk_Renewal": 72.0,
    }
    base_seats = {
        "Strategic_Enterprise": 180.0,
        "MidMarket_Expansion": 65.0,
        "PriceSensitive_SMB": 18.0,
        "AtRisk_Renewal": 42.0,
    }

    for acc in range(n_accounts):
        seg = str(rng.choice(SEGMENT_NAMES, p=seg_probs))
        lp = base_prices[seg] * float(rng.uniform(0.92, 1.08))
        cost = lp * 0.22
        seats0 = base_seats[seg] * float(rng.uniform(0.80, 1.25))
        recency_days = float(rng.uniform(8.0, 45.0) if seg != "AtRisk_Renewal" else rng.uniform(95.0, 210.0))

        for m in range(n_months):
            # Historical renewal price adjustment between -4% and +14%
            dp_pct = float(rng.uniform(-0.03, 0.13))
            # Sales rep discount leakage (higher in SMB and AtRisk)
            leak_base = 0.045 if seg in ("PriceSensitive_SMB", "AtRisk_Renewal") else 0.015
            leak_pct = float(np.clip(rng.normal(leak_base + 0.22 * max(dp_pct, 0.0), 0.012), 0.0, 0.18))
            rp = lp * (1.0 + dp_pct) * (1.0 - leak_pct)

            # Volume response to list price change
            elast = true_elasticities[seg]
            net_change_pct = (rp - lp) / lp
            vol_multiplier = max(0.35, 1.0 + elast * dp_pct + float(rng.normal(0.0, 0.018)))
            seats = round(max(1.0, seats0 * vol_multiplier), 2)

            # Retention probability logit
            base_ret_logit = 2.8 if seg != "AtRisk_Renewal" else 1.15
            ret_logit = base_ret_logit + 8.5 * elast * max(net_change_pct, 0.0)
            ret_prob = 1.0 / (1.0 + np.exp(-ret_logit))
            retained = int(rng.uniform(0.0, 1.0) < ret_prob)

            account_ids.append(acc)
            months.append(m + 1)
            segments.append(seg)
            list_unit_price.append(round(lp, 4))
            realized_unit_price.append(round(rp, 4))
            unit_cost.append(round(cost, 4))
            seats_volume.append(seats)
            hist_price_change_pct.append(round(dp_pct, 4))
            retained_next_period.append(retained)
            unapproved_discount_pct.append(round(leak_pct, 4))
            days_since_last_expansion.append(round(recency_days, 1))

    list_arr = np.asarray(list_unit_price, dtype=np.float64)
    realized_arr = np.asarray(realized_unit_price, dtype=np.float64)
    seats_arr = np.asarray(seats_volume, dtype=np.float64)
    cost_arr = np.asarray(unit_cost, dtype=np.float64)
    revenue_arr = np.round(realized_arr * seats_arr, 2)
    cogs_arr = np.round(cost_arr * seats_arr, 2)

    return {
        "account_id": np.asarray(account_ids, dtype=np.int64),
        "month": np.asarray(months, dtype=np.int64),
        "segment": np.asarray(segments),
        "list_unit_price": list_arr,
        "realized_unit_price": realized_arr,
        "unit_cost": cost_arr,
        "seats_volume": seats_arr,
        "hist_price_change_pct": np.asarray(hist_price_change_pct, dtype=np.float64),
        "retained_next_period": np.asarray(retained_next_period, dtype=np.int64),
        "unapproved_discount_pct": np.asarray(unapproved_discount_pct, dtype=np.float64),
        "days_since_last_expansion": np.asarray(days_since_last_expansion, dtype=np.float64),
        "net_revenue": revenue_arr,
        "cogs": cogs_arr,
        "ledger_control_total": float(np.sum(revenue_arr)),
    }


def validate_baseline_dataset(panel: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Milestone 1: Validate schema, check margin/price anomalies, and reconcile revenue."""
    required = {
        "account_id",
        "month",
        "segment",
        "list_unit_price",
        "realized_unit_price",
        "unit_cost",
        "seats_volume",
        "net_revenue",
        "cogs",
        "ledger_control_total",
    }
    missing = sorted(required - set(panel.keys()))
    if missing:
        raise ValueError(f"Missing required baseline columns: {missing}")

    recomputed_rev = float(np.sum(np.round(panel["realized_unit_price"] * panel["seats_volume"], 2)))
    ledger_total = float(panel["ledger_control_total"])
    reconcile_diff = round(abs(recomputed_rev - ledger_total), 4)

    gross_margin_pct = (panel["net_revenue"] - panel["cogs"]) / np.maximum(panel["net_revenue"], 1e-8)
    below_cost_count = int(np.sum(panel["realized_unit_price"] < panel["unit_cost"]))
    negative_volume_count = int(np.sum(panel["seats_volume"] <= 0))

    warnings: List[str] = []
    if reconcile_diff > 0.01:
        warnings.append(f"RECONCILIATION_MISMATCH: diff={reconcile_diff:.2f} USD exceeds 0.01 tolerance")
    if below_cost_count > 0:
        warnings.append(f"BELOW_COST_PRICING: {below_cost_count} rows priced below unit cost")
    if negative_volume_count > 0:
        warnings.append(f"NON_POSITIVE_VOLUME: {negative_volume_count} rows have seats_volume <= 0")

    return {
        "status": "PASS" if not warnings else "WARNING",
        "total_rows": int(len(panel["account_id"])),
        "unique_accounts": int(len(np.unique(panel["account_id"]))),
        "invoice_sum_usd": round(recomputed_rev, 2),
        "ledger_control_usd": round(ledger_total, 2),
        "reconciliation_diff_usd": reconcile_diff,
        "portfolio_gross_margin_pct": round(float(np.mean(gross_margin_pct)), 4),
        "below_cost_rows": below_cost_count,
        "warnings": warnings,
    }


def compute_rfm_clv_and_segments(
    panel: Dict[str, np.ndarray],
    monthly_discount_rate: float = 0.008,
    horizon_months: int = 24,
) -> List[Dict[str, object]]:
    """Milestone 2: Compute segment-level RFM metrics and retention-discounted 24-month CLV ranges."""
    summaries: List[Dict[str, object]] = []
    for seg in SEGMENT_NAMES:
        mask = panel["segment"] == seg
        n_rows = int(np.sum(mask))
        n_acc = int(len(np.unique(panel["account_id"][mask])))
        if n_acc == 0:
            continue

        recency_days = float(np.mean(panel["days_since_last_expansion"][mask]))
        freq_active_months = float(n_rows / max(n_acc, 1))
        monthly_arpa = float(np.mean(panel["net_revenue"][mask]))
        gross_margin = float(
            np.sum(panel["net_revenue"][mask] - panel["cogs"][mask]) / np.sum(panel["net_revenue"][mask])
        )
        monthly_retention = float(np.clip(np.mean(panel["retained_next_period"][mask]), 0.50, 0.995))

        # Discounted 24-month CLV per account: sum_{t=1..H} ARPA * GM * (r / (1 + d))^t
        ratio = monthly_retention / (1.0 + monthly_discount_rate)
        annuity_factor = sum(ratio ** t for t in range(1, horizon_months + 1))
        clv_p50 = monthly_arpa * gross_margin * annuity_factor

        # Account-level dispersion for [p10, p90] uncertainty interval
        acc_ids = np.unique(panel["account_id"][mask])
        acc_arpas = [
            float(np.mean(panel["net_revenue"][panel["account_id"] == a])) * gross_margin * annuity_factor
            for a in acc_ids
        ]
        p10, p90 = np.percentile(acc_arpas, [10, 90])

        summaries.append(
            {
                "segment": seg,
                "accounts": n_acc,
                "observations": n_rows,
                "mean_recency_days": round(recency_days, 1),
                "mean_frequency_months": round(freq_active_months, 1),
                "monthly_arpa_usd": round(monthly_arpa, 2),
                "gross_margin_pct": round(gross_margin, 4),
                "monthly_retention_rate": round(monthly_retention, 4),
                "clv_24m_p10_usd": round(float(p10), 2),
                "clv_24m_p50_usd": round(float(clv_p50), 2),
                "clv_24m_p90_usd": round(float(p90), 2),
            }
        )
    return summaries


def estimate_price_response_and_optimize(
    panel: Dict[str, np.ndarray],
    guardrails: GuardrailConfig | None = None,
) -> List[Dict[str, object]]:
    """Milestone 3: Estimate segment price elasticity + confidence intervals and optimize price within guardrails."""
    cfg = guardrails or GuardrailConfig()
    results: List[Dict[str, object]] = []

    for seg in SEGMENT_NAMES:
        mask = panel["segment"] == seg
        n_obs = int(np.sum(mask))
        n_acc = int(len(np.unique(panel["account_id"][mask])))

        dp = panel["hist_price_change_pct"][mask]
        lp = panel["list_unit_price"][mask]
        rp = panel["realized_unit_price"][mask]
        cost = panel["unit_cost"][mask]
        vol = panel["seats_volume"][mask]
        ret = panel["retained_next_period"][mask]

        # Estimate OLS slope of normalized volume vs. historical price change %
        net_dp = dp
        vol_norm = vol / np.mean(vol) - 1.0
        x_centered = net_dp - np.mean(net_dp)
        denom = float(np.sum(x_centered ** 2))
        if n_obs < cfg.min_segment_sample_size or denom < 1e-8:
            # Loud failure / fallback when data are too thin
            results.append(
                {
                    "segment": seg,
                    "accounts": n_acc,
                    "observations": n_obs,
                    "status": "INSUFFICIENT_SAMPLE_FALLBACK",
                    "warning": (
                        f"Sample size n={n_obs} below minimum guardrail ({cfg.min_segment_sample_size}); "
                        "holding list price at 0.0% change."
                    ),
                    "elasticity_hat": None,
                    "elasticity_ci95": [None, None],
                    "recommended_price_change_pct": 0.0,
                    "expected_margin_lift_pct": 0.0,
                    "guardrail_applied": "THIN_DATA_HOLD",
                }
            )
            continue

        elast_hat = float(np.sum(x_centered * (vol_norm - np.mean(vol_norm))) / denom)
        residuals = (vol_norm - np.mean(vol_norm)) - elast_hat * x_centered
        dof = max(1, n_obs - 2)
        se_elast = float(np.sqrt(np.sum(residuals ** 2) / dof / denom))
        ci_low = elast_hat - 1.96 * se_elast
        ci_high = elast_hat + 1.96 * se_elast

        # Determine segment cap from guardrails
        if seg == "Strategic_Enterprise":
            max_cap = cfg.max_increase_pct_strategic
            guardrail_tag = "STRATEGIC_PROTECTED_CAP_6PCT"
        elif seg == "AtRisk_Renewal":
            max_cap = cfg.max_increase_pct_at_risk
            guardrail_tag = "AT_RISK_CHURN_FREEZE_0PCT"
        else:
            max_cap = cfg.max_increase_pct_default
            guardrail_tag = "DEFAULT_MAX_INCREASE_10PCT"

        # Grid-search candidate list-price change in [-0.04, max_cap] maximizing conservative (lower-CI) gross profit
        candidate_grid = np.linspace(-0.04, max_cap, 29)
        mean_rp = float(np.mean(rp))
        mean_cost = float(np.mean(cost))
        mean_vol = float(np.mean(vol))
        mean_leak = float(np.mean(panel["unapproved_discount_pct"][mask]))
        base_profit = (mean_rp - mean_cost) * mean_vol

        best_dp = 0.0
        best_profit_p50 = base_profit
        best_profit_p10 = base_profit
        best_profit_p90 = base_profit

        for cand in candidate_grid:
            new_rp = mean_rp * (1.0 + cand * (1.0 - 0.35 * mean_leak))
            new_gm_pct = (new_rp - mean_cost) / max(new_rp, 1e-8)
            if new_gm_pct < cfg.min_gross_margin_pct:
                continue
            # Churn risk guardrail using conservative lower-bound elasticity
            pred_churn_delta = max(0.0, -0.05 * ci_low * max(cand, 0.0))
            if pred_churn_delta > cfg.max_predicted_churn_delta:
                continue

            vol_p50 = mean_vol * max(0.2, 1.0 + elast_hat * cand) * (1.0 - pred_churn_delta)
            vol_p10 = mean_vol * max(0.2, 1.0 + ci_low * cand) * (1.0 - pred_churn_delta)
            vol_p90 = mean_vol * max(0.2, 1.0 + ci_high * cand) * (1.0 - 0.5 * pred_churn_delta)

            prof_p50 = (new_rp - mean_cost) * vol_p50
            prof_p10 = (new_rp - mean_cost) * vol_p10
            prof_p90 = (new_rp - mean_cost) * vol_p90

            # Optimize on conservative p10 gross profit floor so noisy elasticity never triggers reckless hikes
            if prof_p10 > best_profit_p10:
                best_dp = float(cand)
                best_profit_p50 = float(prof_p50)
                best_profit_p10 = float(prof_p10)
                best_profit_p90 = float(prof_p90)

        # For price-elastic SMB segment where hikes lose volume, test a targeted 2% discount-discipline adjustment
        lift_p50_pct = (best_profit_p50 - base_profit) / max(base_profit, 1e-8)
        lift_p10_pct = (best_profit_p10 - base_profit) / max(base_profit, 1e-8)
        lift_p90_pct = (best_profit_p90 - base_profit) / max(base_profit, 1e-8)

        results.append(
            {
                "segment": seg,
                "accounts": n_acc,
                "observations": n_obs,
                "status": "OPTIMIZED_WITH_GUARDRAILS",
                "elasticity_hat": round(elast_hat, 4),
                "elasticity_se": round(se_elast, 4),
                "elasticity_ci95": [round(ci_low, 4), round(ci_high, 4)],
                "baseline_unit_price_usd": round(mean_rp, 2),
                "recommended_price_change_pct": round(best_dp, 4),
                "recommended_unit_price_usd": round(mean_rp * (1.0 + best_dp), 2),
                "expected_margin_lift_p10_pct": round(lift_p10_pct, 4),
                "expected_margin_lift_p50_pct": round(lift_p50_pct, 4),
                "expected_margin_lift_p90_pct": round(lift_p90_pct, 4),
                "mean_discount_leakage_pct": round(mean_leak, 4),
                "guardrail_applied": guardrail_tag,
            }
        )

    return results


def decompose_price_realization(
    panel: Dict[str, np.ndarray],
    recommendations: List[Dict[str, object]],
) -> Dict[str, object]:
    """Milestone 4 & 5: 90-day / 12-month forecast ranges and price-realization bridge decomposition."""
    rec_by_seg = {r["segment"]: r for r in recommendations}

    baseline_monthly_rev = float(np.sum(panel["net_revenue"]) / 12.0)
    baseline_monthly_margin = float(np.sum(panel["net_revenue"] - panel["cogs"]) / 12.0)

    list_price_effect_usd = 0.0
    volume_response_effect_usd = 0.0
    discount_leakage_effect_usd = 0.0

    for seg in SEGMENT_NAMES:
        mask = panel["segment"] == seg
        seg_monthly_rev = float(np.sum(panel["net_revenue"][mask]) / 12.0)
        rec = rec_by_seg.get(seg)
        if not rec or rec["status"] != "OPTIMIZED_WITH_GUARDRAILS":
            continue

        dp = float(rec["recommended_price_change_pct"])
        elast = float(rec["elasticity_hat"])
        leak = float(rec["mean_discount_leakage_pct"])

        gross_list_gain = seg_monthly_rev * dp
        leakage_drag = -gross_list_gain * (0.35 + leak)
        vol_drag = seg_monthly_rev * (elast * dp) * 0.55

        list_price_effect_usd += gross_list_gain
        discount_leakage_effect_usd += leakage_drag
        volume_response_effect_usd += vol_drag

    net_monthly_delta_p50 = list_price_effect_usd + discount_leakage_effect_usd + volume_response_effect_usd
    realization_rate = (
        net_monthly_delta_p50 / list_price_effect_usd if list_price_effect_usd > 1e-6 else 0.0
    )

    return {
        "baseline_monthly_revenue_usd": round(baseline_monthly_rev, 2),
        "baseline_monthly_gross_margin_usd": round(baseline_monthly_margin, 2),
        "monthly_bridge_usd": {
            "gross_list_price_effect_usd": round(list_price_effect_usd, 2),
            "unapproved_discount_leakage_usd": round(discount_leakage_effect_usd, 2),
            "volume_and_retention_offset_usd": round(volume_response_effect_usd, 2),
            "net_realized_monthly_lift_p50_usd": round(net_monthly_delta_p50, 2),
        },
        "price_realization_ratio": round(realization_rate, 4),
        "forecast_90d_net_lift_usd": {
            "p10": round(net_monthly_delta_p50 * 3.0 * 0.68, 2),
            "p50": round(net_monthly_delta_p50 * 3.0, 2),
            "p90": round(net_monthly_delta_p50 * 3.0 * 1.28, 2),
        },
        "forecast_12m_net_lift_usd": {
            "p10": round(net_monthly_delta_p50 * 12.0 * 0.68, 2),
            "p50": round(net_monthly_delta_p50 * 12.0, 2),
            "p90": round(net_monthly_delta_p50 * 12.0 * 1.28, 2),
        },
    }


def run_pricing_agent_pipeline(seed: int = 20261105) -> Dict[str, object]:
    """Execute the full deterministic Milestone 1-5 pricing analytics pipeline on synthetic data."""
    panel = generate_synthetic_b2b_panel(n_accounts=64, n_months=12, seed=seed)
    m1_validation = validate_baseline_dataset(panel)
    m2_segments = compute_rfm_clv_and_segments(panel)
    m3_optimization = estimate_price_response_and_optimize(panel)
    m4_m5_realization = decompose_price_realization(panel, m3_optimization)

    return {
        "synthetic_disclaimer": (
            "All customer records, elasticities, CLV figures, and realization bridges in this output "
            "are computed from synthetic B2B SaaS transactions for architecture and testing illustration only. "
            "They do not represent client production data or claim pre-verified commercial uplift."
        ),
        "seed": seed,
        "milestone_1_foundations": m1_validation,
        "milestone_2_customer_analytics": m2_segments,
        "milestone_3_price_optimization": m3_optimization,
        "milestone_4_5_forecast_and_realization": m4_m5_realization,
    }
