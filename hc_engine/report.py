from __future__ import annotations

import math
from typing import Any

import pandas as pd


def summarize(result: dict[str, Any]) -> dict[str, float]:
    eq_rows = result.get("equity_curve") or []
    fills = result.get("fills") or []
    state = result.get("state") or {}
    if not eq_rows:
        return {
            "ending_equity": 0.0,
            "net_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_like": 0.0,
            "calmar_like": 0.0,
            "fills": float(len(fills)),
            "fee_paid_total": float(state.get("fee_paid_total", 0.0) or 0.0),
            "fee_paid_maker": float(state.get("fee_paid_maker", 0.0) or 0.0),
            "fee_paid_taker": float(state.get("fee_paid_taker", 0.0) or 0.0),
            "funding_pnl": float(state.get("funding_pnl", 0.0) or 0.0),
            "margin_reject_count": float(state.get("margin_reject_count", 0.0) or 0.0),
            "liquidation_count": float(state.get("liquidation_count", 0.0) or 0.0),
            "liquidation_fee_paid": float(state.get("liquidation_fee_paid", 0.0) or 0.0),
        }

    eq = pd.DataFrame(eq_rows)
    eq["equity"] = eq["equity"].astype(float)
    start = float(eq["equity"].iloc[0])
    end = float(eq["equity"].iloc[-1])
    ret = ((end - start) / start * 100.0) if start > 0 else 0.0

    peak = eq["equity"].cummax()
    dd = (peak - eq["equity"]) / peak.replace(0.0, pd.NA)
    max_dd_pct = float(dd.fillna(0.0).max() * 100.0)

    rets = eq["equity"].pct_change().replace([math.inf, -math.inf], pd.NA).fillna(0.0)
    vol = float(rets.std())
    sharpe_like = float((rets.mean() / vol) * math.sqrt(max(len(rets), 1))) if vol > 0 else 0.0
    first_ts = int(eq["ts_ms"].iloc[0]) if "ts_ms" in eq.columns else 0
    last_ts = int(eq["ts_ms"].iloc[-1]) if "ts_ms" in eq.columns else 0
    years = max(0.0, float(last_ts - first_ts) / (365.2425 * 24 * 3600 * 1000.0))
    if start > 0 and end > 0 and years > 0:
        annualized_return_pct = float((pow(end / start, 1.0 / years) - 1.0) * 100.0)
    else:
        annualized_return_pct = float(ret)
    max_dd_dec = max(0.0, max_dd_pct / 100.0)
    calmar_like = float((annualized_return_pct / 100.0) / max_dd_dec) if max_dd_dec > 0 else 0.0

    return {
        "starting_equity": float(start),
        "ending_equity": float(end),
        "net_return_pct": float(ret),
        "annualized_return_pct": float(annualized_return_pct),
        "max_drawdown_pct": float(max_dd_pct),
        "sharpe_like": float(sharpe_like),
        "calmar_like": float(calmar_like),
        "fills": float(len(fills)),
        "fee_paid_total": float(state.get("fee_paid_total", 0.0) or 0.0),
        "fee_paid_maker": float(state.get("fee_paid_maker", 0.0) or 0.0),
        "fee_paid_taker": float(state.get("fee_paid_taker", 0.0) or 0.0),
        "funding_pnl": float(state.get("funding_pnl", 0.0) or 0.0),
        "trade_notional_cum": float(state.get("trade_notional_cum", 0.0) or 0.0),
        "margin_reject_count": float(state.get("margin_reject_count", 0.0) or 0.0),
        "liquidation_count": float(state.get("liquidation_count", 0.0) or 0.0),
        "liquidation_fee_paid": float(state.get("liquidation_fee_paid", 0.0) or 0.0),
    }
