"""Unit tests for the deterministic B2B pricing analytics modules."""

from __future__ import annotations

import numpy as np

from pricing_analytics import (
    GuardrailConfig,
    compute_rfm_clv_and_segments,
    decompose_price_realization,
    estimate_price_response_and_optimize,
    generate_synthetic_b2b_panel,
    run_pricing_agent_pipeline,
    validate_baseline_dataset,
)


def test_milestone1_reconciliation_and_anomaly_detection() -> None:
    panel = generate_synthetic_b2b_panel(n_accounts=32, n_months=12, seed=20261105)
    res = validate_baseline_dataset(panel)
    assert res["status"] == "PASS"
    assert res["reconciliation_diff_usd"] == 0.0
    assert res["below_cost_rows"] == 0

    # Inject a ledger tie-out discrepancy and below-cost unit price to verify loud failure warnings
    corrupted = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in panel.items()}
    corrupted["ledger_control_total"] = float(panel["ledger_control_total"]) + 250.0
    corrupted["realized_unit_price"][0] = 1.0
    res_bad = validate_baseline_dataset(corrupted)
    assert res_bad["status"] == "WARNING"
    assert any("RECONCILIATION_MISMATCH" in w for w in res_bad["warnings"])
    assert any("BELOW_COST_PRICING" in w for w in res_bad["warnings"])


def test_milestone2_clv_ordering_and_uncertainty_intervals() -> None:
    panel = generate_synthetic_b2b_panel(n_accounts=64, n_months=12, seed=20261105)
    m2 = compute_rfm_clv_and_segments(panel)
    by_seg = {r["segment"]: r for r in m2}
    assert set(by_seg.keys()) == {
        "Strategic_Enterprise",
        "MidMarket_Expansion",
        "PriceSensitive_SMB",
        "AtRisk_Renewal",
    }
    for row in m2:
        assert row["clv_24m_p10_usd"] <= row["clv_24m_p50_usd"] <= row["clv_24m_p90_usd"]
        assert 0.50 <= row["monthly_retention_rate"] <= 1.0
    assert by_seg["Strategic_Enterprise"]["clv_24m_p50_usd"] > by_seg["PriceSensitive_SMB"]["clv_24m_p50_usd"]


def test_milestone3_guardrails_and_thin_data_loud_fallback() -> None:
    panel = generate_synthetic_b2b_panel(n_accounts=64, n_months=12, seed=20261105)
    cfg = GuardrailConfig(
        min_gross_margin_pct=0.68,
        max_increase_pct_default=0.10,
        max_increase_pct_strategic=0.06,
        max_increase_pct_at_risk=0.00,
        min_segment_sample_size=12,
    )
    recs = estimate_price_response_and_optimize(panel, guardrails=cfg)
    by_seg = {r["segment"]: r for r in recs}

    # Strategic Enterprise increase never exceeds 6% cap
    assert by_seg["Strategic_Enterprise"]["recommended_price_change_pct"] <= 0.06 + 1e-9
    # At-Risk Renewal segment is frozen at <= 0% increase
    assert by_seg["AtRisk_Renewal"]["recommended_price_change_pct"] <= 0.0 + 1e-9

    # Verify loud fallback when minimum sample size guardrail is triggered
    strict_cfg = GuardrailConfig(min_segment_sample_size=5000)
    thin_recs = estimate_price_response_and_optimize(panel, guardrails=strict_cfg)
    assert all(r["status"] == "INSUFFICIENT_SAMPLE_FALLBACK" for r in thin_recs)
    assert all(r["recommended_price_change_pct"] == 0.0 for r in thin_recs)


def test_milestone4_5_realization_bridge_identities() -> None:
    summary = run_pricing_agent_pipeline(seed=20261105)
    m45 = summary["milestone_4_5_forecast_and_realization"]
    bridge = m45["monthly_bridge_usd"]
    expected_net = round(
        bridge["gross_list_price_effect_usd"]
        + bridge["unapproved_discount_leakage_usd"]
        + bridge["volume_and_retention_offset_usd"],
        2,
    )
    assert abs(expected_net - bridge["net_realized_monthly_lift_p50_usd"]) <= 0.02
    assert m45["forecast_90d_net_lift_usd"]["p10"] < m45["forecast_90d_net_lift_usd"]["p50"]
    assert m45["forecast_12m_net_lift_usd"]["p50"] > m45["forecast_90d_net_lift_usd"]["p50"]
