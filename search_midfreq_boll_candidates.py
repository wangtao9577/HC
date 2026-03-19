from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import run_gp3_hc_backtest as hcmod
import search_gp3_hc_candidates as gp3scan
from hc_engine import BacktestConfig, BacktestEngine, IntrabarPathModel, StopTriggerSource, summarize


def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()
    avg_down = down.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()


def _build_feature_frame(primary_raw: pd.DataFrame, filter_raw: pd.DataFrame, cand: dict) -> pd.DataFrame:
    out = primary_raw.copy()
    boll_period = int(cand["boll_period"])
    boll_std = float(cand["boll_std"])
    kc_mult = float(cand["kc_mult"])
    rsi_period = int(cand["rsi_period"])
    filter_ema_fast = int(cand["filter_ema_fast"])
    filter_ema_slow = int(cand["filter_ema_slow"])
    filter_slope_lookback = int(cand["filter_slope_lookback"])

    out["basis"] = out["close"].rolling(boll_period).mean()
    std = out["close"].rolling(boll_period).std(ddof=0)
    out["std"] = std
    out["upper"] = out["basis"] + boll_std * std
    out["lower"] = out["basis"] - boll_std * std
    out["bandwidth"] = (out["upper"] - out["lower"]) / out["basis"].replace(0.0, np.nan)
    out["zscore"] = (out["close"] - out["basis"]) / out["std"].replace(0.0, np.nan)
    out["atr"] = _calc_atr(out, int(cand["atr_period"]))
    out["rsi"] = _calc_rsi(out["close"], rsi_period)
    out["kc_basis"] = out["close"].ewm(span=boll_period, adjust=False).mean()
    out["kc_upper"] = out["kc_basis"] + kc_mult * out["atr"]
    out["kc_lower"] = out["kc_basis"] - kc_mult * out["atr"]
    out["z_prev"] = out["zscore"].shift(1)
    out["close_prev"] = out["close"].shift(1)
    out["lower_prev"] = out["lower"].shift(1)
    out["upper_prev"] = out["upper"].shift(1)
    out["rsi_prev"] = out["rsi"].shift(1)

    tf = filter_raw.copy()
    tf["filter_ema_fast"] = tf["close"].ewm(span=filter_ema_fast, adjust=False).mean()
    tf["filter_ema_slow"] = tf["close"].ewm(span=filter_ema_slow, adjust=False).mean()
    tf["filter_slope"] = tf["filter_ema_fast"] - tf["filter_ema_fast"].shift(filter_slope_lookback)
    tf = tf[["close_time", "filter_ema_fast", "filter_ema_slow", "filter_slope", "close"]].rename(
        columns={"close": "filter_close"}
    )

    merged = pd.merge_asof(
        out.sort_values("close_time"),
        tf.sort_values("close_time"),
        on="close_time",
        direction="backward",
    )
    merged["trend_up"] = (
        (merged["filter_close"] > merged["filter_ema_fast"])
        & (merged["filter_ema_fast"] > merged["filter_ema_slow"])
        & (merged["filter_slope"] > 0)
    )
    merged["trend_down"] = (
        (merged["filter_close"] < merged["filter_ema_fast"])
        & (merged["filter_ema_fast"] < merged["filter_ema_slow"])
        & (merged["filter_slope"] < 0)
    )

    use_kc = bool(int(cand["use_kc_filter"]))
    entry_mode = int(cand["entry_mode"])
    long_stretch = merged["close"] <= merged["lower"]
    short_stretch = merged["close"] >= merged["upper"]
    if use_kc:
        long_stretch = long_stretch & (merged["close"] <= merged["kc_lower"])
        short_stretch = short_stretch & (merged["close"] >= merged["kc_upper"])

    entry_z = float(cand["entry_z"])
    if entry_mode == 0:
        merged["long_signal"] = (
            merged["trend_up"]
            & (merged["z_prev"] > -entry_z)
            & (merged["zscore"] <= -entry_z)
            & (merged["rsi"] <= float(cand["rsi_long_max"]))
            & (merged["bandwidth"] >= float(cand["bandwidth_min"]))
            & long_stretch
        )
        merged["short_signal"] = (
            merged["trend_down"]
            & (merged["z_prev"] < entry_z)
            & (merged["zscore"] >= entry_z)
            & (merged["rsi"] >= float(cand["rsi_short_min"]))
            & (merged["bandwidth"] >= float(cand["bandwidth_min"]))
            & short_stretch
        )
    else:
        merged["long_signal"] = (
            merged["trend_up"]
            & (merged["close_prev"] <= merged["lower_prev"])
            & (merged["close"] > merged["lower"])
            & (merged["zscore"] <= 0.0)
            & (merged["rsi_prev"] <= float(cand["rsi_long_max"]))
            & (merged["rsi"] > merged["rsi_prev"])
            & (merged["bandwidth"] >= float(cand["bandwidth_min"]))
        )
        merged["short_signal"] = (
            merged["trend_down"]
            & (merged["close_prev"] >= merged["upper_prev"])
            & (merged["close"] < merged["upper"])
            & (merged["zscore"] >= 0.0)
            & (merged["rsi_prev"] >= float(cand["rsi_short_min"]))
            & (merged["rsi"] < merged["rsi_prev"])
            & (merged["bandwidth"] >= float(cand["bandwidth_min"]))
        )

    # Reuse the GP3 HC shell. These columns drive entry anchor and trend-flip exit.
    merged["ema_fast"] = merged["filter_ema_fast"]
    merged["ema_slow"] = merged["filter_ema_slow"]

    need = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "basis",
        "upper",
        "lower",
        "bandwidth",
        "zscore",
        "atr",
        "rsi",
        "ema_fast",
        "ema_slow",
        "long_signal",
        "short_signal",
    ]
    return merged[need].dropna().reset_index(drop=True)


def _build_grid(args) -> list[dict]:
    field_order = [
        "boll_period",
        "boll_std",
        "kc_mult",
        "use_kc_filter",
        "rsi_period",
        "entry_z",
        "rsi_long_max",
        "rsi_short_min",
        "bandwidth_min",
        "entry_mode",
        "filter_ema_fast",
        "filter_ema_slow",
        "filter_slope_lookback",
        "entry_pullback_bps",
        "entry_timeout_bars",
        "tp_atr_mult",
        "stop_atr_mult",
        "trail_atr_mult",
        "breakeven_trigger_atr",
        "position_mult",
        "cooldown_bars",
        "max_hold_bars",
        "atr_period",
    ]
    grids = {
        "boll_period": gp3scan._split_ints(args.boll_periods),
        "boll_std": gp3scan._split_floats(args.boll_stds),
        "kc_mult": gp3scan._split_floats(args.kc_mults),
        "use_kc_filter": gp3scan._split_ints(args.use_kc_filter),
        "rsi_period": gp3scan._split_ints(args.rsi_periods),
        "entry_z": gp3scan._split_floats(args.entry_zs),
        "rsi_long_max": gp3scan._split_floats(args.rsi_long_maxs),
        "rsi_short_min": gp3scan._split_floats(args.rsi_short_mins),
        "bandwidth_min": gp3scan._split_floats(args.bandwidth_mins),
        "entry_mode": gp3scan._split_ints(args.entry_modes),
        "filter_ema_fast": gp3scan._split_ints(args.filter_ema_fasts),
        "filter_ema_slow": gp3scan._split_ints(args.filter_ema_slows),
        "filter_slope_lookback": gp3scan._split_ints(args.filter_slope_lookbacks),
        "entry_pullback_bps": gp3scan._split_floats(args.entry_pullbacks),
        "entry_timeout_bars": gp3scan._split_ints(args.entry_timeout_bars),
        "tp_atr_mult": gp3scan._split_floats(args.tp_mults),
        "stop_atr_mult": gp3scan._split_floats(args.stop_mults),
        "trail_atr_mult": gp3scan._split_floats(args.trail_mults),
        "breakeven_trigger_atr": gp3scan._split_floats(args.breakeven_trigger_atrs),
        "position_mult": gp3scan._split_floats(args.position_mults),
        "cooldown_bars": gp3scan._split_ints(args.cooldown_bars),
        "max_hold_bars": gp3scan._split_ints(args.max_hold_bars),
        "atr_period": gp3scan._split_ints(args.atr_periods),
    }
    for name, values in grids.items():
        if not values:
            raise RuntimeError(f"grid field is empty: {name}")

    out: list[dict] = []
    for combo in product(*(grids[name] for name in field_order)):
        cand = {name: combo[idx] for idx, name in enumerate(field_order)}
        if int(cand["filter_ema_fast"]) >= int(cand["filter_ema_slow"]):
            continue
        if float(cand["tp_atr_mult"]) <= 0 or float(cand["stop_atr_mult"]) <= 0 or float(cand["trail_atr_mult"]) <= 0:
            continue
        if int(cand["entry_timeout_bars"]) <= 0 or int(cand["max_hold_bars"]) <= 0:
            continue
        out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Search 5m/15m BOLL-Keltner mid-frequency candidates on HC")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--klines-file", default="d:/project/hc/cache/klines_ETHUSDC_1741904460000_1773440460000_1m.pkl")
    ap.add_argument("--out-dir", default="d:/project/hc/output/midfreq_boll_search")
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--interval", default="5m", choices=["5m", "15m"])
    ap.add_argument("--filter-interval", default="15m", choices=["15m", "30m", "1h"])
    ap.add_argument("--stop-trigger-source", default="mark", choices=["last", "mark"])
    ap.add_argument("--tick-size", type=float, default=0.0)
    ap.add_argument("--step-size", type=float, default=0.0)
    ap.add_argument("--min-qty", type=float, default=0.0)
    ap.add_argument("--boll-periods", default="18,24")
    ap.add_argument("--boll-stds", default="1.8,2.2")
    ap.add_argument("--kc-mults", default="1.2,1.6")
    ap.add_argument("--use-kc-filter", default="1")
    ap.add_argument("--rsi-periods", default="7")
    ap.add_argument("--entry-zs", default="1.6,2.0")
    ap.add_argument("--rsi-long-maxs", default="42")
    ap.add_argument("--rsi-short-mins", default="58")
    ap.add_argument("--bandwidth-mins", default="0.002")
    ap.add_argument("--entry-modes", default="0,1")
    ap.add_argument("--filter-ema-fasts", default="8")
    ap.add_argument("--filter-ema-slows", default="21")
    ap.add_argument("--filter-slope-lookbacks", default="1")
    ap.add_argument("--entry-pullbacks", default="2")
    ap.add_argument("--entry-timeout-bars", default="1")
    ap.add_argument("--tp-mults", default="0.8,1.2")
    ap.add_argument("--stop-mults", default="1.0,1.4")
    ap.add_argument("--trail-mults", default="1.0")
    ap.add_argument("--breakeven-trigger-atrs", default="0.5")
    ap.add_argument("--position-mults", default="24")
    ap.add_argument("--cooldown-bars", default="0,1")
    ap.add_argument("--max-hold-bars", default="6,10")
    ap.add_argument("--atr-periods", default="14")
    ap.add_argument("--min-round-trips", type=int, default=360)
    ap.add_argument("--max-round-trips", type=int, default=600)
    args = ap.parse_args()

    gp3 = hcmod._load_gp3_module()
    hcmod._load_env_compat(gp3, Path("d:/project/.env"))
    hcmod._load_env_compat(gp3, Path(args.profile))
    base_cfg = gp3.build_config_from_env()
    base_cfg.symbol = "ETHUSDC"
    base_cfg.skip_entry_bar_exit_checks = False
    if not hasattr(base_cfg, "initial_capital"):
        base_cfg.initial_capital = 100.0
    if not hasattr(base_cfg, "maker_fee"):
        base_cfg.maker_fee = 0.0
    if not hasattr(base_cfg, "taker_fee"):
        base_cfg.taker_fee = 0.0004
    if not hasattr(base_cfg, "stop_slippage_bps"):
        base_cfg.stop_slippage_bps = 2.0

    rules = hcmod._build_rules(gp3, base_cfg, args)
    primary_minutes = gp3scan._parse_interval_minutes(args.interval)
    filter_minutes = {"15m": 15, "30m": 30, "1h": 60}[str(args.filter_interval)]
    if filter_minutes <= primary_minutes:
        raise RuntimeError("filter interval must be greater than primary interval")

    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=float(args.days))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    raw = hcmod._load_klines_local(Path(args.klines_file), start_ms=start_ms, end_ms=end_ms)
    primary_raw = gp3scan._resample_klines(raw, primary_minutes)
    filter_raw = gp3scan._resample_klines(raw, filter_minutes)
    candidates = _build_grid(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    progress_path = out_dir / f"midfreq_boll_progress_{stamp}.csv"
    result_csv = out_dir / f"midfreq_boll_results_{stamp}.csv"
    summary_path = out_dir / f"midfreq_boll_summary_{stamp}.json"
    results: list[dict] = []

    bars_cache = None
    last_feat = None
    for idx, cand in enumerate(candidates, start=1):
        cfg = gp3scan._clone_cfg(base_cfg)
        cfg.entry_pullback_bps = float(cand["entry_pullback_bps"])
        cfg.entry_timeout_bars = int(cand["entry_timeout_bars"])
        cfg.tp_atr_mult = float(cand["tp_atr_mult"])
        cfg.stop_atr_mult = float(cand["stop_atr_mult"])
        cfg.trail_atr_mult = float(cand["trail_atr_mult"])
        cfg.breakeven_trigger_atr = float(cand["breakeven_trigger_atr"])
        cfg.cooldown_bars = int(cand["cooldown_bars"])
        cfg.max_hold_bars = int(cand["max_hold_bars"])
        cfg.position_mult = float(cand["position_mult"])

        feat = _build_feature_frame(primary_raw, filter_raw, cand)
        if feat.empty:
            continue
        bars = gp3scan._build_bars(feat)
        last_feat = feat
        bars_cache = bars

        engine_cfg = BacktestConfig(
            symbol=cfg.symbol,
            initial_cash=float(cfg.initial_capital),
            leverage=float(getattr(cfg, "leverage", 1.0) or 1.0),
            maker_fee_rate=float(cfg.maker_fee),
            taker_fee_rate=float(cfg.taker_fee),
            market_slippage_bps=0.5,
            stop_slippage_bps=float(cfg.stop_slippage_bps),
            tick_size=float(rules.tick_size),
            maker_queue_delay_ms=800,
            maker_buffer_ticks=0,
            allow_same_bar_entry_exit=True,
            stop_trigger_source=StopTriggerSource(str(args.stop_trigger_source)),
            partial_fill_enabled=False,
            partial_fill_ratio=0.35,
            min_partial_qty=0.0,
            partial_fill_scope="maker",
            funding_interval_ms=8 * 60 * 60 * 1000,
            funding_rate=0.0,
            maker_fee_tiers=[],
            taker_fee_tiers=[],
        )
        engine = BacktestEngine(engine_cfg, path_model=IntrabarPathModel(mode="oc_aware"))
        strategy = gp3scan.IntervalAwareGp3HcStrategy(feat, cfg, rules, gp3, bar_ms=primary_minutes * 60_000)
        result = engine.run(bars, strategy, events_by_bar=None, external_events_by_ts=None)
        stats = summarize(result)
        round_trips = int(len(hcmod._extract_round_trips(result["fills"])))
        row = {
            **cand,
            "interval": str(args.interval),
            "filter_interval": str(args.filter_interval),
            "net_return_pct": float(stats.get("net_return_pct", 0.0)),
            "max_drawdown_pct": float(stats.get("max_drawdown_pct", 0.0)),
            "sharpe_like": float(stats.get("sharpe_like", 0.0)),
            "ending_equity": float(stats.get("ending_equity", 0.0)),
            "fills": int(stats.get("fills", 0.0)),
            "round_trips": int(round_trips),
            "candidate_index": int(idx),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(progress_path, index=False)
        print(
            f"[{idx}/{len(candidates)}] ret={row['net_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% "
            f"rt={row['round_trips']} bb={row['boll_period']}/{row['boll_std']:.2f} "
            f"kc={row['kc_mult']:.2f} z={row['entry_z']:.2f} tp={row['tp_atr_mult']:.2f} "
            f"stop={row['stop_atr_mult']:.2f} hold={row['max_hold_bars']} cd={row['cooldown_bars']}"
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df["target_score"] = [
            gp3scan._candidate_score(
                row,
                min_round_trips=int(args.min_round_trips or 0),
                max_round_trips=int(args.max_round_trips or 0),
            )[1]
            for row in df.to_dict("records")
        ]
        df = df.sort_values(
            by=["target_score", "net_return_pct", "max_drawdown_pct", "sharpe_like"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)
    df.to_csv(result_csv, index=False)

    feasible = df[
        (df["net_return_pct"] >= 2000.0)
        & (df["max_drawdown_pct"] <= 30.0)
        & ((int(args.min_round_trips or 0) <= 0) | (df["round_trips"] >= int(args.min_round_trips or 0)))
        & ((int(args.max_round_trips or 0) <= 0) | (df["round_trips"] <= int(args.max_round_trips or 0)))
    ].copy() if not df.empty else pd.DataFrame()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_start_utc": start_dt.isoformat(),
        "window_end_utc": end_dt.isoformat(),
        "profile": str(Path(args.profile)),
        "symbol": "ETHUSDC",
        "interval": str(args.interval),
        "filter_interval": str(args.filter_interval),
        "source_bars_1m": int(len(raw)),
        "bars_after_primary_resample": int(len(primary_raw)),
        "bars_after_filter_resample": int(len(filter_raw)),
        "bars_after_last_features": int(len(last_feat)) if last_feat is not None else 0,
        "tested_candidates": int(len(df)),
        "feasible_count": int(len(feasible)),
        "min_round_trips": int(args.min_round_trips or 0),
        "max_round_trips": int(args.max_round_trips or 0),
        "progress_csv": str(progress_path),
        "result_csv": str(result_csv),
        "best_feasible": None if feasible.empty else feasible.iloc[0].to_dict(),
        "best_overall": None if df.empty else df.iloc[0].to_dict(),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_json: {summary_path}")
    print(f"results_csv: {result_csv}")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
