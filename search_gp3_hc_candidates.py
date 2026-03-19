from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

import run_gp3_hc_backtest as hcmod
from hc_engine import BacktestConfig, Bar, BacktestEngine, IntrabarPathModel, StopTriggerSource, summarize


def _parse_interval_minutes(raw: str) -> int:
    text = str(raw or "1m").strip().lower()
    if not text.endswith("m"):
        raise ValueError(f"unsupported interval: {raw}")
    minutes = int(text[:-1] or "0")
    if minutes not in {1, 5, 15}:
        raise ValueError(f"unsupported interval minutes: {minutes}")
    return minutes


def _split_floats(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _split_ints(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(float(p)))
    return out


def _clone_cfg(cfg):
    return copy.deepcopy(cfg)


def _resample_klines(raw: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    if int(interval_minutes) <= 1:
        return raw.reset_index(drop=True).copy()

    out = raw.copy()
    out["open_time"] = out["open_time"].astype("int64")
    out["close_time"] = out["close_time"].astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = out[c].astype(float)

    interval_ms = int(interval_minutes) * 60_000
    out["bucket_open_time"] = (out["open_time"] // interval_ms) * interval_ms
    grouped = (
        out.groupby("bucket_open_time", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            first_open_time=("open_time", "min"),
            last_open_time=("open_time", "max"),
            rows=("open_time", "count"),
        )
        .reset_index()
    )

    expected_rows = int(interval_minutes)
    grouped = grouped[
        (grouped["rows"] == expected_rows)
        & (grouped["first_open_time"] == grouped["bucket_open_time"])
        & (grouped["last_open_time"] == grouped["bucket_open_time"] + (expected_rows - 1) * 60_000)
    ].copy()
    if grouped.empty:
        raise RuntimeError(f"no complete {interval_minutes}m bars after resample")

    grouped["open_time"] = grouped["bucket_open_time"].astype("int64")
    grouped["close_time"] = grouped["open_time"] + interval_ms - 1
    return grouped[["open_time", "open", "high", "low", "close", "volume", "close_time"]].reset_index(drop=True)


def _build_indicator_frame(raw: pd.DataFrame, cfg) -> pd.DataFrame:
    gp3 = hcmod._load_gp3_module()
    base = hcmod._build_gp3_features_server_like(gp3, raw, cfg).copy()
    need = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "ema_fast",
        "ema_slow",
        "tr",
        "atr",
        "adx",
        "rsi",
        "rsi_prev",
        "vol_ma",
        "vol_ok",
        "ema_fast_slope",
        "bar_range_bps",
        "spread_ok",
    ]
    cols = [c for c in need if c in base.columns]
    return base[cols].dropna().reset_index(drop=True)


def _apply_signal_rules(ind: pd.DataFrame, cfg) -> pd.DataFrame:
    out = ind.copy()
    rsi_buf = float(getattr(cfg, "rsi_trigger_buffer", 0.0) or 0.0)
    long_rsi_rebound = (out["rsi_prev"] <= (cfg.rsi_long_reset - rsi_buf)) & (
        out["rsi"] >= (cfg.rsi_long_trigger + rsi_buf)
    )
    short_rsi_rebound = (out["rsi_prev"] >= (cfg.rsi_short_reset + rsi_buf)) & (
        out["rsi"] <= (cfg.rsi_short_trigger - rsi_buf)
    )
    range_ok = out["bar_range_bps"] >= float(getattr(cfg, "range_min_bps", 0.0) or 0.0)
    vol_gate = out["vol_ok"] if bool(getattr(cfg, "use_volume_filter", False)) else True

    out["long_signal"] = (
        (out["ema_fast"] > out["ema_slow"])
        & (out["ema_fast_slope"] > 0)
        & (out["close"] >= out["ema_fast"])
        & long_rsi_rebound
        & range_ok
        & (out["adx"] >= float(getattr(cfg, "adx_min", 0.0) or 0.0))
        & out["spread_ok"]
        & vol_gate
    )
    out["short_signal"] = (
        (out["ema_fast"] < out["ema_slow"])
        & (out["ema_fast_slope"] < 0)
        & (out["close"] <= out["ema_fast"])
        & short_rsi_rebound
        & range_ok
        & (out["adx"] >= float(getattr(cfg, "adx_min", 0.0) or 0.0))
        & out["spread_ok"]
        & vol_gate
    )
    return out.reset_index(drop=True)


def _build_bars(feat: pd.DataFrame) -> list[Bar]:
    return [
        Bar(
            open_time_ms=int(r.open_time),
            close_time_ms=int(r.close_time),
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(r.volume),
        )
        for r in feat.itertuples(index=False)
    ]


def _read_optional_float(row: dict, key: str) -> float | None:
    raw = row.get(key)
    if raw in (None, "", "nan"):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _read_optional_int(row: dict, key: str) -> int | None:
    val = _read_optional_float(row, key)
    if val is None:
        return None
    return int(val)


def _load_seed_candidates(csv_paths: Iterable[Path]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple] = set()
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            cand = {
                "ema_fast": _read_optional_int(row, "ema_fast"),
                "ema_slow": _read_optional_int(row, "ema_slow"),
                "rsi_trigger_buffer": float(row.get("rsi_trigger_buffer", 0.0) or 0.0),
                "rsi_long_reset": _read_optional_float(row, "rsi_long_reset"),
                "rsi_long_trigger": _read_optional_float(row, "rsi_long_trigger"),
                "rsi_short_reset": _read_optional_float(row, "rsi_short_reset"),
                "rsi_short_trigger": _read_optional_float(row, "rsi_short_trigger"),
                "tp_atr_mult": float(row.get("tp_atr_mult", 0.0) or 0.0),
                "entry_pullback_bps": float(row.get("entry_pullback_bps", 8.0) or 8.0),
                "entry_timeout_bars": int(float(row.get("entry_timeout_bars", 2) or 2)),
                "adx_min": _read_optional_float(row, "adx_min"),
                "range_min_bps": _read_optional_float(row, "range_min_bps"),
                "stop_atr_mult": _read_optional_float(row, "stop_atr_mult"),
                "trail_atr_mult": _read_optional_float(row, "trail_atr_mult"),
                "cooldown_bars": _read_optional_int(row, "cooldown_bars"),
                "max_hold_bars": _read_optional_int(row, "max_hold_bars"),
                "breakeven_trigger_atr": _read_optional_float(row, "breakeven_trigger_atr"),
                "position_mult": _read_optional_float(row, "position_mult"),
            }
            key = (
                cand["ema_fast"],
                cand["ema_slow"],
                cand["rsi_trigger_buffer"],
                cand["tp_atr_mult"],
                cand["entry_pullback_bps"],
                cand["entry_timeout_bars"],
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


def _build_base_candidate(cfg) -> dict:
    return {
        "ema_fast": int(getattr(cfg, "ema_fast", 18)),
        "ema_slow": int(getattr(cfg, "ema_slow", 55)),
        "rsi_trigger_buffer": float(getattr(cfg, "rsi_trigger_buffer", 0.0) or 0.0),
        "rsi_long_reset": float(getattr(cfg, "rsi_long_reset", 48.0) or 48.0),
        "rsi_long_trigger": float(getattr(cfg, "rsi_long_trigger", 50.0) or 50.0),
        "rsi_short_reset": float(getattr(cfg, "rsi_short_reset", 52.0) or 52.0),
        "rsi_short_trigger": float(getattr(cfg, "rsi_short_trigger", 50.0) or 50.0),
        "tp_atr_mult": float(getattr(cfg, "tp_atr_mult", 0.45) or 0.45),
        "entry_pullback_bps": float(getattr(cfg, "entry_pullback_bps", 8.0) or 8.0),
        "entry_timeout_bars": int(getattr(cfg, "entry_timeout_bars", 2) or 2),
        "adx_min": float(getattr(cfg, "adx_min", 11.0) or 11.0),
        "range_min_bps": float(getattr(cfg, "range_min_bps", 0.3) or 0.3),
        "stop_atr_mult": float(getattr(cfg, "stop_atr_mult", 1.5) or 1.5),
        "trail_atr_mult": float(getattr(cfg, "trail_atr_mult", 1.8) or 1.8),
        "cooldown_bars": int(getattr(cfg, "cooldown_bars", 2) or 2),
        "max_hold_bars": int(getattr(cfg, "max_hold_bars", 30) or 30),
        "breakeven_trigger_atr": float(getattr(cfg, "breakeven_trigger_atr", 0.25) or 0.25),
        "position_mult": float(getattr(cfg, "position_mult", 1.0) or 1.0),
    }


def _candidate_value(cand: dict, key: str, default):
    val = cand.get(key)
    return default if val is None else val


def _candidate_score(result: dict, *, min_round_trips: int, max_round_trips: int) -> tuple:
    ret = float(result["net_return_pct"])
    dd = float(result["max_drawdown_pct"])
    rt = float(result.get("round_trips", 0.0) or 0.0)
    dd_penalty = max(0.0, dd - 30.0) * 200.0
    if min_round_trips > 0 and rt < float(min_round_trips):
        dd_penalty += float(min_round_trips - rt) * 8.0
    if max_round_trips > 0 and rt > float(max_round_trips):
        dd_penalty += float(rt - max_round_trips) * 8.0
    feasible = int(
        ret >= 2000.0
        and dd <= 30.0
        and (min_round_trips <= 0 or rt >= float(min_round_trips))
        and (max_round_trips <= 0 or rt <= float(max_round_trips))
    )
    mid_rt = ((min_round_trips + max_round_trips) / 2.0) if max_round_trips > 0 else rt
    return (feasible, ret - dd_penalty, -dd, -abs(rt - mid_rt))


class IntervalAwareGp3HcStrategy(hcmod.Gp3HcStrategy):
    def __init__(self, feat: pd.DataFrame, cfg, rules, gp3_mod, *, bar_ms: int):
        super().__init__(feat, cfg, rules, gp3_mod)
        bar_sec = max(float(bar_ms) / 1000.0, 1.0)
        self.order_timeout_bars = max(1, math.ceil(float(cfg.order_timeout_sec) / bar_sec))


def main() -> None:
    ap = argparse.ArgumentParser(description="Search GP3 candidates on HC annual no-agg backtest")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--klines-file", default="d:/project/hc/cache/klines_ETHUSDC_1741904460000_1773440460000_1m.pkl")
    ap.add_argument("--out-dir", default="d:/project/hc/output/gp3_hc_search_20260315")
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--interval", default="1m", choices=["1m", "5m", "15m"])
    ap.add_argument("--stop-trigger-source", default="mark", choices=["last", "mark"])
    ap.add_argument("--tick-size", type=float, default=0.0)
    ap.add_argument("--step-size", type=float, default=0.0)
    ap.add_argument("--min-qty", type=float, default=0.0)
    ap.add_argument("--seed-csv", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0, help="limit number of seed candidates")
    ap.add_argument("--ema-fasts", default="")
    ap.add_argument("--ema-slows", default="")
    ap.add_argument("--rsi-buffers", default="")
    ap.add_argument("--rsi-long-resets", default="")
    ap.add_argument("--rsi-long-triggers", default="")
    ap.add_argument("--rsi-short-resets", default="")
    ap.add_argument("--rsi-short-triggers", default="")
    ap.add_argument("--tp-mults", default="")
    ap.add_argument("--entry-pullbacks", default="")
    ap.add_argument("--entry-timeout-bars", default="")
    ap.add_argument("--adx-mins", default="")
    ap.add_argument("--range-min-bps", default="")
    ap.add_argument("--stop-mults", default="")
    ap.add_argument("--trail-mults", default="")
    ap.add_argument("--breakeven-trigger-atrs", default="")
    ap.add_argument("--position-mults", default="")
    ap.add_argument("--cooldown-bars", default="")
    ap.add_argument("--max-hold-bars", default="")
    ap.add_argument("--min-round-trips", type=int, default=0)
    ap.add_argument("--max-round-trips", type=int, default=0)
    args = ap.parse_args()

    seed_paths = [Path(p) for p in (args.seed_csv or [])]
    if not seed_paths:
        seed_paths = [
            Path("d:/project/gp3.0/results/gp3_opt/gp3_rsi_sweep_20260311_021748.csv"),
            Path("d:/project/gp3.0/results/gp3_opt/gp3_refine_sweep_20260311_022627.csv"),
        ]

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

    candidates = _load_seed_candidates(seed_paths)
    base_candidate = _build_base_candidate(base_cfg)
    if args.limit > 0:
        candidates = candidates[: int(args.limit)]
    if not candidates:
        candidates = [base_candidate]
    else:
        ordered_keys = list(base_candidate.keys())
        seed_keys = {tuple(c.get(k) for k in ordered_keys) for c in candidates}
        base_key = tuple(base_candidate.get(k) for k in ordered_keys)
        if base_key not in seed_keys:
            candidates.insert(0, base_candidate)

    rules = hcmod._build_rules(gp3, base_cfg, args)
    interval_minutes = _parse_interval_minutes(args.interval)
    interval_ms = interval_minutes * 60_000

    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=float(args.days))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    raw = hcmod._load_klines_local(Path(args.klines_file), start_ms=start_ms, end_ms=end_ms)
    raw_tf = _resample_klines(raw, interval_minutes)
    ind = _build_indicator_frame(raw_tf, base_cfg)
    if ind.empty:
        raise RuntimeError("indicator frame is empty after resample/features")
    bars = _build_bars(ind)

    grid_ema_fast = _split_ints(args.ema_fasts) or None
    grid_ema_slow = _split_ints(args.ema_slows) or None
    grid_rsi = _split_floats(args.rsi_buffers) or None
    grid_rsi_long_reset = _split_floats(args.rsi_long_resets) or None
    grid_rsi_long_trigger = _split_floats(args.rsi_long_triggers) or None
    grid_rsi_short_reset = _split_floats(args.rsi_short_resets) or None
    grid_rsi_short_trigger = _split_floats(args.rsi_short_triggers) or None
    grid_tp = _split_floats(args.tp_mults) or None
    grid_entry = _split_floats(args.entry_pullbacks) or None
    grid_entry_timeout = _split_ints(args.entry_timeout_bars) or None
    grid_adx = _split_floats(args.adx_mins) or None
    grid_range = _split_floats(args.range_min_bps) or None
    grid_stop = _split_floats(args.stop_mults) or None
    grid_trail = _split_floats(args.trail_mults) or None
    grid_be = _split_floats(args.breakeven_trigger_atrs) or None
    grid_pos_mult = _split_floats(args.position_mults) or None
    grid_cd = _split_ints(args.cooldown_bars) or None
    grid_hold = _split_ints(args.max_hold_bars) or None

    field_order = [
        "ema_fast",
        "ema_slow",
        "rsi_trigger_buffer",
        "rsi_long_reset",
        "rsi_long_trigger",
        "rsi_short_reset",
        "rsi_short_trigger",
        "tp_atr_mult",
        "entry_pullback_bps",
        "entry_timeout_bars",
        "adx_min",
        "range_min_bps",
        "stop_atr_mult",
        "trail_atr_mult",
        "breakeven_trigger_atr",
        "position_mult",
        "cooldown_bars",
        "max_hold_bars",
    ]

    expanded: list[dict] = []
    seen: set[tuple] = set()
    for cand in candidates:
        grids = {
            "ema_fast": grid_ema_fast or [int(_candidate_value(cand, "ema_fast", getattr(base_cfg, "ema_fast", 18)))],
            "ema_slow": grid_ema_slow or [int(_candidate_value(cand, "ema_slow", getattr(base_cfg, "ema_slow", 55)))],
            "rsi_trigger_buffer": grid_rsi or [float(_candidate_value(cand, "rsi_trigger_buffer", getattr(base_cfg, "rsi_trigger_buffer", 0.0)))],
            "rsi_long_reset": grid_rsi_long_reset or [float(_candidate_value(cand, "rsi_long_reset", getattr(base_cfg, "rsi_long_reset", 48.0)))],
            "rsi_long_trigger": grid_rsi_long_trigger or [float(_candidate_value(cand, "rsi_long_trigger", getattr(base_cfg, "rsi_long_trigger", 50.0)))],
            "rsi_short_reset": grid_rsi_short_reset or [float(_candidate_value(cand, "rsi_short_reset", getattr(base_cfg, "rsi_short_reset", 52.0)))],
            "rsi_short_trigger": grid_rsi_short_trigger or [float(_candidate_value(cand, "rsi_short_trigger", getattr(base_cfg, "rsi_short_trigger", 50.0)))],
            "tp_atr_mult": grid_tp or [float(_candidate_value(cand, "tp_atr_mult", getattr(base_cfg, "tp_atr_mult", 0.45)))],
            "entry_pullback_bps": grid_entry or [float(_candidate_value(cand, "entry_pullback_bps", getattr(base_cfg, "entry_pullback_bps", 8.0)))],
            "entry_timeout_bars": grid_entry_timeout or [int(_candidate_value(cand, "entry_timeout_bars", getattr(base_cfg, "entry_timeout_bars", 2)))],
            "adx_min": grid_adx or [float(_candidate_value(cand, "adx_min", getattr(base_cfg, "adx_min", 11.0)))],
            "range_min_bps": grid_range or [float(_candidate_value(cand, "range_min_bps", getattr(base_cfg, "range_min_bps", 0.3)))],
            "stop_atr_mult": grid_stop or [float(_candidate_value(cand, "stop_atr_mult", getattr(base_cfg, "stop_atr_mult", 1.5)))],
            "trail_atr_mult": grid_trail or [float(_candidate_value(cand, "trail_atr_mult", getattr(base_cfg, "trail_atr_mult", 1.8)))],
            "breakeven_trigger_atr": grid_be or [float(_candidate_value(cand, "breakeven_trigger_atr", getattr(base_cfg, "breakeven_trigger_atr", 0.25)))],
            "position_mult": grid_pos_mult or [float(_candidate_value(cand, "position_mult", getattr(base_cfg, "position_mult", 1.0)))],
            "cooldown_bars": grid_cd or [int(_candidate_value(cand, "cooldown_bars", getattr(base_cfg, "cooldown_bars", 2)))],
            "max_hold_bars": grid_hold or [int(_candidate_value(cand, "max_hold_bars", getattr(base_cfg, "max_hold_bars", 30)))],
        }

        for combo in product(*(grids[name] for name in field_order)):
            item = {name: combo[idx] for idx, name in enumerate(field_order)}
            if int(item["ema_fast"]) >= int(item["ema_slow"]):
                continue
            if float(item["rsi_long_reset"]) >= float(item["rsi_long_trigger"]):
                continue
            if float(item["rsi_short_reset"]) <= float(item["rsi_short_trigger"]):
                continue
            if float(item["tp_atr_mult"]) <= 0 or float(item["stop_atr_mult"]) <= 0 or float(item["trail_atr_mult"]) <= 0:
                continue
            if int(item["entry_timeout_bars"]) <= 0 or int(item["max_hold_bars"]) <= 0:
                continue
            key = tuple(item[name] for name in field_order)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(item)

    if not expanded:
        raise RuntimeError("no expanded candidates generated")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results: list[dict] = []
    progress_path = out_dir / f"gp3_hc_search_progress_{stamp}.csv"

    for idx, cand in enumerate(expanded, start=1):
        cfg = _clone_cfg(base_cfg)
        for k, v in cand.items():
            setattr(cfg, k, v)

        feat = _apply_signal_rules(ind, cfg)
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
        strategy = IntervalAwareGp3HcStrategy(feat, cfg, rules, gp3, bar_ms=interval_ms)
        result = engine.run(bars, strategy, events_by_bar=None, external_events_by_ts=None)
        stats = summarize(result)
        round_trips = int(len(hcmod._extract_round_trips(result["fills"])))

        row = {
            **cand,
            "interval": str(args.interval),
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
            f"[{idx}/{len(expanded)}] "
            f"ret={row['net_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% "
            f"rt={row['round_trips']} tf={row['interval']} "
            f"ema={row['ema_fast']}/{row['ema_slow']} "
            f"rsi={row['rsi_trigger_buffer']:.2f} tp={row['tp_atr_mult']:.2f} "
            f"entry={row['entry_pullback_bps']:.1f}/{row['entry_timeout_bars']} "
            f"adx={row['adx_min']:.1f} range={row['range_min_bps']:.2f} "
            f"stop={row['stop_atr_mult']:.2f} trail={row['trail_atr_mult']:.2f} "
            f"be={row['breakeven_trigger_atr']:.2f} pm={row['position_mult']:.1f} "
            f"cd={row['cooldown_bars']} hold={row['max_hold_bars']}"
        )

    df = pd.DataFrame(results)
    if not df.empty:
        df["target_score"] = [
            _candidate_score(
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

    result_csv = out_dir / f"gp3_hc_search_results_{stamp}.csv"
    df.to_csv(result_csv, index=False)

    feasible = df[
        (df["net_return_pct"] >= 2000.0)
        & (df["max_drawdown_pct"] <= 30.0)
        & ((int(args.min_round_trips or 0) <= 0) | (df["round_trips"] >= int(args.min_round_trips or 0)))
        & ((int(args.max_round_trips or 0) <= 0) | (df["round_trips"] <= int(args.max_round_trips or 0)))
    ].copy()
    best = None if feasible.empty else feasible.iloc[0].to_dict()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_start_utc": start_dt.isoformat(),
        "window_end_utc": end_dt.isoformat(),
        "profile": str(Path(args.profile)),
        "symbol": "ETHUSDC",
        "interval": str(args.interval),
        "interval_minutes": int(interval_minutes),
        "source_bars_1m": int(len(raw)),
        "bars_after_resample": int(len(raw_tf)),
        "bars_after_features": int(len(ind)),
        "min_round_trips": int(args.min_round_trips or 0),
        "max_round_trips": int(args.max_round_trips or 0),
        "tested_candidates": int(len(df)),
        "feasible_count": int(len(feasible)),
        "result_csv": str(result_csv),
        "progress_csv": str(progress_path),
        "best_feasible": best,
        "best_overall": None if df.empty else df.iloc[0].to_dict(),
    }
    summary_path = out_dir / f"gp3_hc_search_summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_json: {summary_path}")
    print(f"results_csv: {result_csv}")
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
