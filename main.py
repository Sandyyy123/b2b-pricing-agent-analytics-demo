"""Entry point for running the deterministic B2B pricing analytics pipeline (Milestones 1-5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pricing_analytics import run_pricing_agent_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic B2B pricing analytics pipeline for an AI Pricing Agent."
    )
    parser.add_argument("--seed", type=int, default=20261105, help="Deterministic random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="results.json",
        help="Path to write JSON analytics output",
    )
    args = parser.parse_args()

    summary = run_pricing_agent_pipeline(seed=args.seed)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    m1 = summary["milestone_1_foundations"]
    print("=== B2B Pricing Agent Analytics Pipeline (Synthetic Demonstration) ===")
    print(
        f"Milestone 1 Foundations: status={m1['status']} | accounts={m1['unique_accounts']} | "
        f"rows={m1['total_rows']} | invoice_sum=${m1['invoice_sum_usd']:,.2f} | "
        f"reconciliation_diff=${m1['reconciliation_diff_usd']:.2f} | gross_margin={m1['portfolio_gross_margin_pct']*100:.2f}%"
    )
    print("\nMilestone 2 & 3 Segment Analytics & Guardrailed Price Optimization:")
    print(
        f"{'Segment':<22} {'Accts':>5} {'ARPA ($)':>10} {'CLV p50 ($)':>13} "
        f"{'Elast_hat':>10} {'Rec dP%':>9} {'Lift p50%':>10} {'Guardrail':<28}"
    )
    print("-" * 114)
    m2_map = {r["segment"]: r for r in summary["milestone_2_customer_analytics"]}
    for r3 in summary["milestone_3_price_optimization"]:
        seg = r3["segment"]
        r2 = m2_map[seg]
        print(
            f"{seg:<22} {r3['accounts']:>5} {r2['monthly_arpa_usd']:>10.2f} {r2['clv_24m_p50_usd']:>13.2f} "
            f"{r3['elasticity_hat']:>10.4f} {r3['recommended_price_change_pct']*100:>8.2f}% "
            f"{r3['expected_margin_lift_p50_pct']*100:>9.2f}% {r3['guardrail_applied']:<28}"
        )

    m45 = summary["milestone_4_5_forecast_and_realization"]
    bridge = m45["monthly_bridge_usd"]
    print(
        f"\nMilestone 4 & 5 Realization Bridge (Monthly): "
        f"List Effect=+${bridge['gross_list_price_effect_usd']:,.2f} | "
        f"Discount Leakage=${bridge['unapproved_discount_leakage_usd']:,.2f} | "
        f"Volume Offset=${bridge['volume_and_retention_offset_usd']:,.2f} | "
        f"Net Realized Lift=+${bridge['net_realized_monthly_lift_p50_usd']:,.2f}/mo "
        f"(Realization Ratio={m45['price_realization_ratio']*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
