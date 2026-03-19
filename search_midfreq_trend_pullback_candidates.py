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
from hc_engine import Bar, BacktestConfig, BacktestEngine, IntrabarPathModel, Side, StopTriggerSource, TimeInForce, summarize


SESSION_MASKS: dict[str, dict[str, set[int]]] = {
    "all": {"long": set(range(24)), "short": set(range(24))},
    "eu_us": {"long": set(range(7, 22)), "short": set(range(7, 22))},
    "asia_eu": {"long": set(range(0, 16)), "short": set(range(0, 16))},
    "ny_asia": {
        "long": {0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        "short": {0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
    },
    "cta_eth_side": {
        "long": {0, 8, 9, 13, 15, 16, 21, 23},
        "short": {1, 13, 14, 15, 17, 18, 19, 20},
    },
    "cta_eth_union": {
        "long": {0, 1, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23},
        "short": {0, 1, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23},
    },
}

CTA_CLOSE_LONG_RULES = {"none", "ema20_mid", "ema12_26_flip", "close_below_ema26"}
CTA_CLOSE_SHORT_RULES = {"none", "macd_hist_flip", "ema12_26_flip", "close_above_ema26"}
CTA_FLUSH_RULES = {"off", "ema12_26_flip"}


def _resolve_session_mask(name: str) -> dict[str, set[int]]:
    key = str(name or "").strip().lower()
    if key not in SESSION_MASKS:
        raise RuntimeError(f"unknown session mask: {name}")
    return SESSION_MASKS[key]


def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()
    avg_down = down.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / max(int(period), 1), adjust=False).mean()


def _calc_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=int(fast), adjust=False).mean()
    ema_slow = close.ewm(span=int(slow), adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=int(signal), adjust=False).mean()
    return macd_line - macd_signal


def _parse_utc(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty utc string")
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_window(args) -> tuple[int, int]:
    start_raw = str(getattr(args, "start_utc", "") or "").strip()
    end_raw = str(getattr(args, "end_utc", "") or "").strip()
    if bool(start_raw) != bool(end_raw):
        raise RuntimeError("start-utc and end-utc must be provided together")
    if start_raw and end_raw:
        start_dt = _parse_utc(start_raw)
        end_dt = _parse_utc(end_raw)
    else:
        end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=float(args.days))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    if end_ms <= start_ms:
        raise RuntimeError("invalid backtest window")
    return start_ms, end_ms


def _attach_confirmed_pivots(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_h = out["high"].shift(1)
    prev2_h = out["high"].shift(2)
    prev_l = out["low"].shift(1)
    prev2_l = out["low"].shift(2)
    out["pivot_high_confirm"] = prev_h.where((prev_h > prev2_h) & (prev_h >= out["high"]))
    out["pivot_low_confirm"] = prev_l.where((prev_l < prev2_l) & (prev_l <= out["low"]))

    last_hi = np.full(len(out), np.nan)
    prev_hi = np.full(len(out), np.nan)
    last_lo = np.full(len(out), np.nan)
    prev_lo = np.full(len(out), np.nan)

    hi_last = np.nan
    hi_prev = np.nan
    lo_last = np.nan
    lo_prev = np.nan
    for i in range(len(out)):
        ph = out.iloc[i]["pivot_high_confirm"]
        pl = out.iloc[i]["pivot_low_confirm"]
        if pd.notna(ph):
            hi_prev = hi_last
            hi_last = float(ph)
        if pd.notna(pl):
            lo_prev = lo_last
            lo_last = float(pl)
        last_hi[i] = hi_last
        prev_hi[i] = hi_prev
        last_lo[i] = lo_last
        prev_lo[i] = lo_prev

    out["last_pivot_high"] = last_hi
    out["prev_pivot_high"] = prev_hi
    out["last_pivot_low"] = last_lo
    out["prev_pivot_low"] = prev_lo
    return out


def _build_feature_frame(primary_raw: pd.DataFrame, filter_raw: pd.DataFrame, cand: dict, session_mask: dict[str, set[int]]) -> pd.DataFrame:
    out = primary_raw.copy()
    primary_ema_fast = int(cand["primary_ema_fast"])
    primary_ema_slow = int(cand["primary_ema_slow"])
    atr_period = int(cand["atr_period"])
    rsi_period = int(cand["rsi_period"])
    breakout_lookback = int(cand["breakout_lookback"])

    out["ema_fast"] = out["close"].ewm(span=primary_ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=primary_ema_slow, adjust=False).mean()
    out["ema12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["ema12_prev"] = out["ema12"].shift(1)
    out["ema20_prev"] = out["ema20"].shift(1)
    out["ema26_prev"] = out["ema26"].shift(1)
    out["atr"] = _calc_atr(out, atr_period)
    out["rsi"] = _calc_rsi(out["close"], rsi_period)
    out["close_prev"] = out["close"].shift(1)
    out["macd_hist"] = _calc_macd_hist(out["close"])
    out["macd_hist_prev"] = out["macd_hist"].shift(1)
    out["roll_high_prev"] = out["high"].shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).max()
    out["roll_low_prev"] = out["low"].shift(1).rolling(breakout_lookback, min_periods=breakout_lookback).min()
    out["pullback_long_atr"] = (out["ema_fast"] - out["low"]) / out["atr"].replace(0.0, np.nan)
    out["pullback_short_atr"] = (out["high"] - out["ema_fast"]) / out["atr"].replace(0.0, np.nan)
    out["reclaim_long_atr"] = (out["close"] - out["ema_fast"]) / out["atr"].replace(0.0, np.nan)
    out["reclaim_short_atr"] = (out["ema_fast"] - out["close"]) / out["atr"].replace(0.0, np.nan)
    out["dist_high_atr"] = (out["roll_high_prev"] - out["close"]) / out["atr"].replace(0.0, np.nan)
    out["dist_low_atr"] = (out["close"] - out["roll_low_prev"]) / out["atr"].replace(0.0, np.nan)
    out["body_frac"] = (out["close"] - out["open"]).abs() / (out["high"] - out["low"]).replace(0.0, np.nan)

    tf = filter_raw.copy()
    filter_ema_fast = int(cand["filter_ema_fast"])
    filter_ema_slow = int(cand["filter_ema_slow"])
    tf["filter_ema_fast"] = tf["close"].ewm(span=filter_ema_fast, adjust=False).mean()
    tf["filter_ema_slow"] = tf["close"].ewm(span=filter_ema_slow, adjust=False).mean()
    tf["filter_slope_bps"] = (
        (tf["filter_ema_fast"] - tf["filter_ema_fast"].shift(int(cand["filter_slope_lookback"])))
        / tf["close"].replace(0.0, np.nan)
    ) * 10_000.0
    tf["filter_spread_bps"] = (
        (tf["filter_ema_fast"] - tf["filter_ema_slow"]).abs() / tf["close"].replace(0.0, np.nan)
    ) * 10_000.0
    tf = _attach_confirmed_pivots(tf)
    tf["structure_up"] = (
        (tf["last_pivot_high"] > tf["prev_pivot_high"])
        & (tf["last_pivot_low"] > tf["prev_pivot_low"])
    )
    tf["structure_down"] = (
        (tf["last_pivot_high"] < tf["prev_pivot_high"])
        & (tf["last_pivot_low"] < tf["prev_pivot_low"])
    )
    tf["trend_up"] = (
        (tf["close"] > tf["filter_ema_fast"])
        & (tf["filter_ema_fast"] > tf["filter_ema_slow"])
        & (tf["filter_slope_bps"] >= float(cand["filter_slope_min_bps"]))
        & (tf["filter_spread_bps"] >= float(cand["filter_spread_min_bps"]))
        & tf["structure_up"]
    )
    tf["trend_down"] = (
        (tf["close"] < tf["filter_ema_fast"])
        & (tf["filter_ema_fast"] < tf["filter_ema_slow"])
        & (tf["filter_slope_bps"] <= -float(cand["filter_slope_min_bps"]))
        & (tf["filter_spread_bps"] >= float(cand["filter_spread_min_bps"]))
        & tf["structure_down"]
    )
    tf = tf[
        [
            "close_time",
            "trend_up",
            "trend_down",
            "filter_slope_bps",
            "filter_spread_bps",
            "last_pivot_high",
            "prev_pivot_high",
            "last_pivot_low",
            "prev_pivot_low",
        ]
    ].dropna()

    merged = pd.merge_asof(
        out.sort_values("close_time"),
        tf.sort_values("close_time"),
        on="close_time",
        direction="backward",
    )

    hour_utc = pd.to_datetime(merged["close_time"], unit="ms", utc=True).dt.hour.astype(int)
    allow_long = hour_utc.isin(sorted(session_mask["long"]))
    allow_short = hour_utc.isin(sorted(session_mask["short"]))

    merged["long_signal"] = (
        merged["trend_up"]
        & allow_long
        & (merged["ema_fast"] > merged["ema_slow"])
        & (merged["close"] > merged["ema_slow"])
        & (merged["rsi"] >= float(cand["rsi_long_min"]))
        & (merged["rsi"] <= float(cand["rsi_long_max"]))
        & (merged["pullback_long_atr"] >= float(cand["pullback_min_atr"]))
        & (merged["pullback_long_atr"] <= float(cand["pullback_max_atr"]))
        & (merged["reclaim_long_atr"] >= float(cand["reclaim_min_atr"]))
        & (merged["dist_high_atr"] <= float(cand["near_extreme_max_atr"]))
        & (merged["body_frac"] >= float(cand["body_frac_min"]))
        & (merged["close"] > merged["open"])
        & (merged["close"] >= merged["close_prev"])
    )
    merged["short_signal"] = (
        merged["trend_down"]
        & allow_short
        & (merged["ema_fast"] < merged["ema_slow"])
        & (merged["close"] < merged["ema_slow"])
        & (merged["rsi"] >= float(cand["rsi_short_min"]))
        & (merged["rsi"] <= float(cand["rsi_short_max"]))
        & (merged["pullback_short_atr"] >= float(cand["pullback_min_atr"]))
        & (merged["pullback_short_atr"] <= float(cand["pullback_max_atr"]))
        & (merged["reclaim_short_atr"] >= float(cand["reclaim_min_atr"]))
        & (merged["dist_low_atr"] <= float(cand["near_extreme_max_atr"]))
        & (merged["body_frac"] >= float(cand["body_frac_min"]))
        & (merged["close"] < merged["open"])
        & (merged["close"] <= merged["close_prev"])
    )

    need = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "close_prev",
        "volume",
        "close_time",
        "ema12",
        "ema12_prev",
        "ema20",
        "ema20_prev",
        "ema26",
        "ema26_prev",
        "ema_fast",
        "ema_slow",
        "atr",
        "rsi",
        "macd_hist",
        "macd_hist_prev",
        "trend_up",
        "trend_down",
        "pullback_long_atr",
        "pullback_short_atr",
        "reclaim_long_atr",
        "reclaim_short_atr",
        "dist_high_atr",
        "dist_low_atr",
        "body_frac",
        "long_signal",
        "short_signal",
    ]
    return merged[need].dropna().reset_index(drop=True)


def _cta_close_signal(row: dict | pd.Series, side: str, rule: str) -> bool:
    rule_name = str(rule or "none").strip().lower()
    if rule_name in {"", "none", "off"}:
        return False
    side_name = str(side or "").strip().upper()
    close_px = float(row.get("close", float("nan")))
    close_prev = float(row.get("close_prev", float("nan")))
    ema20 = float(row.get("ema20", float("nan")))
    ema20_prev = float(row.get("ema20_prev", float("nan")))
    ema12 = float(row.get("ema12", float("nan")))
    ema12_prev = float(row.get("ema12_prev", float("nan")))
    ema26 = float(row.get("ema26", float("nan")))
    ema26_prev = float(row.get("ema26_prev", float("nan")))
    macd_hist = float(row.get("macd_hist", float("nan")))
    macd_hist_prev = float(row.get("macd_hist_prev", float("nan")))

    if rule_name == "ema20_mid":
        if side_name == "LONG":
            return close_px <= ema20 and close_prev > ema20_prev
        return close_px >= ema20 and close_prev < ema20_prev
    if rule_name == "macd_hist_flip":
        if side_name == "LONG":
            return macd_hist <= 0.0 and macd_hist_prev > 0.0
        return macd_hist >= 0.0 and macd_hist_prev < 0.0
    if rule_name == "ema12_26_flip":
        if side_name == "LONG":
            return ema12 <= ema26 and ema12_prev > ema26_prev
        return ema12 >= ema26 and ema12_prev < ema26_prev
    if rule_name == "close_above_ema26":
        return side_name == "SHORT" and close_px >= ema26 and close_prev < ema26_prev
    if rule_name == "close_below_ema26":
        return side_name == "LONG" and close_px <= ema26 and close_prev > ema26_prev
    raise RuntimeError(f"unknown CTA close rule: {rule}")


class CtaAwareTrendPullbackStrategy(gp3scan.IntervalAwareGp3HcStrategy):
    def __init__(self, feat: pd.DataFrame, cfg, rules, gp3_mod, *, bar_ms: int):
        super().__init__(feat, cfg, rules, gp3_mod, bar_ms=bar_ms)

    def _cta_exit_trigger(self, row: pd.Series, qty: float) -> bool:
        side = "LONG" if float(qty) > 0 else "SHORT"
        primary_rule = str(getattr(self.cfg, "close_long_rule", "none") if side == "LONG" else getattr(self.cfg, "close_short_rule", "none"))
        flush_rule = str(getattr(self.cfg, "flush_rule", "off"))
        return _cta_close_signal(row, side, primary_rule) or _cta_close_signal(row, side, flush_rule)

    def on_bar_close(self, engine: BacktestEngine, bar: Bar) -> None:
        self._sync_fills(engine)
        if self.cur_idx < 0:
            return
        qty, entry = engine.get_position()
        if qty == 0.0:
            return
        if self.pending_exit is not None:
            return

        row = self.df.iloc[self.cur_idx]
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        atr = max(float(self.entry_atr or 0.0), float(self.rules.tick_size))

        self.bars_held += 1
        new_stop = None
        if qty > 0:
            if (not self.be_triggered) and (h - float(entry) >= float(self.cfg.breakeven_trigger_atr) * atr):
                be = float(entry) * (1.0 + float(self.cfg.breakeven_offset_bps) / 10_000.0)
                if be > float(self.stop_price):
                    new_stop = be
                    self.be_triggered = True
            if self.be_triggered:
                trail = h - float(self.cfg.trail_atr_mult) * atr
                if trail > float(self.stop_price):
                    new_stop = trail if new_stop is None else max(new_stop, trail)
        else:
            if (not self.be_triggered) and (float(entry) - l >= float(self.cfg.breakeven_trigger_atr) * atr):
                be = float(entry) * (1.0 - float(self.cfg.breakeven_offset_bps) / 10_000.0)
                if be < float(self.stop_price):
                    new_stop = be
                    self.be_triggered = True
            if self.be_triggered:
                trail = l + float(self.cfg.trail_atr_mult) * atr
                if trail < float(self.stop_price):
                    new_stop = trail if new_stop is None else min(new_stop, trail)

        if new_stop is not None:
            if qty > 0:
                stop = self.gp3.floor_to_step(float(new_stop), float(self.rules.tick_size))
                stop_side = Side.SELL
            else:
                stop = self.gp3.ceil_to_step(float(new_stop), float(self.rules.tick_size))
                stop_side = Side.BUY
            self.stop_price = float(stop)
            if self.sl_order_id:
                engine.cancel_order(self.sl_order_id)
            self.sl_order_id = engine.place_stop_market(
                side=stop_side,
                qty=abs(float(qty)),
                stop_price=float(stop),
                ts_ms=int(bar.close_time_ms),
                reduce_only=True,
                reason="gp3_sl_trailing",
            )

        trend_flip = (qty > 0 and ema_fast < ema_slow) or (qty < 0 and ema_fast > ema_slow)
        cta_exit = self._cta_exit_trigger(row, qty)
        exit_reason = ""
        if self.bars_held >= int(self.cfg.max_hold_bars):
            exit_reason = "gp3_max_hold"
        elif trend_flip:
            exit_reason = "gp3_trend_flip"
        elif cta_exit:
            exit_reason = "gp3_cta_close"

        if exit_reason:
            if qty > 0:
                px = self.gp3.ceil_to_step(float(c), float(self.rules.tick_size))
                side = Side.SELL
            else:
                px = self.gp3.floor_to_step(float(c), float(self.rules.tick_size))
                side = Side.BUY
            oid = engine.place_limit(
                side=side,
                qty=abs(float(qty)),
                limit_price=float(px),
                ts_ms=int(bar.close_time_ms),
                tif=TimeInForce.GTC,
                reduce_only=True,
                reason=exit_reason,
            )
            self.pending_exit = {
                "order_id": oid,
                "expire_idx": int(self.cur_idx + self.order_timeout_bars),
            }


def _build_grid(args) -> list[dict]:
    fields = [
        "session_mask",
        "primary_ema_fast",
        "primary_ema_slow",
        "filter_ema_fast",
        "filter_ema_slow",
        "filter_slope_lookback",
        "filter_slope_min_bps",
        "filter_spread_min_bps",
        "breakout_lookback",
        "pullback_min_atr",
        "pullback_max_atr",
        "reclaim_min_atr",
        "near_extreme_max_atr",
        "rsi_long_min",
        "rsi_long_max",
        "rsi_short_min",
        "rsi_short_max",
        "body_frac_min",
        "entry_pullback_bps",
        "entry_timeout_bars",
        "tp_atr_mult",
        "stop_atr_mult",
        "trail_atr_mult",
        "breakeven_trigger_atr",
        "breakeven_offset_bps",
        "position_mult",
        "cooldown_bars",
        "max_hold_bars",
        "close_long_rule",
        "close_short_rule",
        "flush_rule",
        "atr_period",
        "rsi_period",
    ]
    grids = {
        "session_mask": [x.strip() for x in str(args.session_masks).split(",") if x.strip()],
        "primary_ema_fast": gp3scan._split_ints(args.primary_ema_fasts),
        "primary_ema_slow": gp3scan._split_ints(args.primary_ema_slows),
        "filter_ema_fast": gp3scan._split_ints(args.filter_ema_fasts),
        "filter_ema_slow": gp3scan._split_ints(args.filter_ema_slows),
        "filter_slope_lookback": gp3scan._split_ints(args.filter_slope_lookbacks),
        "filter_slope_min_bps": gp3scan._split_floats(args.filter_slope_min_bps),
        "filter_spread_min_bps": gp3scan._split_floats(args.filter_spread_min_bps),
        "breakout_lookback": gp3scan._split_ints(args.breakout_lookbacks),
        "pullback_min_atr": gp3scan._split_floats(args.pullback_min_atrs),
        "pullback_max_atr": gp3scan._split_floats(args.pullback_max_atrs),
        "reclaim_min_atr": gp3scan._split_floats(args.reclaim_min_atrs),
        "near_extreme_max_atr": gp3scan._split_floats(args.near_extreme_max_atrs),
        "rsi_long_min": gp3scan._split_floats(args.rsi_long_mins),
        "rsi_long_max": gp3scan._split_floats(args.rsi_long_maxs),
        "rsi_short_min": gp3scan._split_floats(args.rsi_short_mins),
        "rsi_short_max": gp3scan._split_floats(args.rsi_short_maxs),
        "body_frac_min": gp3scan._split_floats(args.body_frac_mins),
        "entry_pullback_bps": gp3scan._split_floats(args.entry_pullbacks),
        "entry_timeout_bars": gp3scan._split_ints(args.entry_timeout_bars),
        "tp_atr_mult": gp3scan._split_floats(args.tp_mults),
        "stop_atr_mult": gp3scan._split_floats(args.stop_mults),
        "trail_atr_mult": gp3scan._split_floats(args.trail_mults),
        "breakeven_trigger_atr": gp3scan._split_floats(args.breakeven_trigger_atrs),
        "breakeven_offset_bps": gp3scan._split_floats(args.breakeven_offset_bps),
        "position_mult": gp3scan._split_floats(args.position_mults),
        "cooldown_bars": gp3scan._split_ints(args.cooldown_bars),
        "max_hold_bars": gp3scan._split_ints(args.max_hold_bars),
        "close_long_rule": [x.strip() for x in str(args.close_long_rules).split(",") if x.strip()],
        "close_short_rule": [x.strip() for x in str(args.close_short_rules).split(",") if x.strip()],
        "flush_rule": [x.strip() for x in str(args.flush_rules).split(",") if x.strip()],
        "atr_period": gp3scan._split_ints(args.atr_periods),
        "rsi_period": gp3scan._split_ints(args.rsi_periods),
    }
    for name, values in grids.items():
        if not values:
            raise RuntimeError(f"grid field empty: {name}")

    out: list[dict] = []
    for combo in product(*(grids[name] for name in fields)):
        cand = {fields[i]: combo[i] for i in range(len(fields))}
        if int(cand["primary_ema_fast"]) >= int(cand["primary_ema_slow"]):
            continue
        if int(cand["filter_ema_fast"]) >= int(cand["filter_ema_slow"]):
            continue
        if float(cand["pullback_min_atr"]) >= float(cand["pullback_max_atr"]):
            continue
        if float(cand["rsi_long_min"]) >= float(cand["rsi_long_max"]):
            continue
        if float(cand["rsi_short_min"]) >= float(cand["rsi_short_max"]):
            continue
        if str(cand["close_long_rule"]).lower() not in CTA_CLOSE_LONG_RULES:
            continue
        if str(cand["close_short_rule"]).lower() not in CTA_CLOSE_SHORT_RULES:
            continue
        if str(cand["flush_rule"]).lower() not in CTA_FLUSH_RULES:
            continue
        out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Search 5m/15m trend-pullback candidates on HC")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--klines-file", default="d:/project/hc/cache/klines_ETHUSDC_1741904460000_1773440460000_1m.pkl")
    ap.add_argument("--out-dir", default="d:/project/hc/output/midfreq_trend_pullback_search")
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--start-utc", default="")
    ap.add_argument("--end-utc", default="")
    ap.add_argument("--interval", default="5m", choices=["5m", "15m"])
    ap.add_argument("--filter-interval", default="15m", choices=["15m", "30m", "1h"])
    ap.add_argument("--stop-trigger-source", default="mark", choices=["last", "mark"])
    ap.add_argument("--tick-size", type=float, default=0.0)
    ap.add_argument("--step-size", type=float, default=0.0)
    ap.add_argument("--min-qty", type=float, default=0.0)
    ap.add_argument("--session-masks", default="all,cta_eth_side,eu_us")
    ap.add_argument("--primary-ema-fasts", default="8,13")
    ap.add_argument("--primary-ema-slows", default="21,34")
    ap.add_argument("--filter-ema-fasts", default="10,20")
    ap.add_argument("--filter-ema-slows", default="30,50")
    ap.add_argument("--filter-slope-lookbacks", default="1,2")
    ap.add_argument("--filter-slope-min-bps", default="4,8")
    ap.add_argument("--filter-spread-min-bps", default="6,12")
    ap.add_argument("--breakout-lookbacks", default="12,20")
    ap.add_argument("--pullback-min-atrs", default="0.05,0.15")
    ap.add_argument("--pullback-max-atrs", default="0.50,0.90")
    ap.add_argument("--reclaim-min-atrs", default="0.02,0.08")
    ap.add_argument("--near-extreme-max-atrs", default="0.30,0.80")
    ap.add_argument("--rsi-long-mins", default="48,52")
    ap.add_argument("--rsi-long-maxs", default="72")
    ap.add_argument("--rsi-short-mins", default="28")
    ap.add_argument("--rsi-short-maxs", default="52,48")
    ap.add_argument("--body-frac-mins", default="0.20,0.35")
    ap.add_argument("--entry-pullbacks", default="1,3")
    ap.add_argument("--entry-timeout-bars", default="1")
    ap.add_argument("--tp-mults", default="1.0,1.4")
    ap.add_argument("--stop-mults", default="0.8,1.2")
    ap.add_argument("--trail-mults", default="1.2,1.8")
    ap.add_argument("--breakeven-trigger-atrs", default="0.6,0.9")
    ap.add_argument("--breakeven-offset-bps", default="0")
    ap.add_argument("--position-mults", default="8,12")
    ap.add_argument("--cooldown-bars", default="0,1")
    ap.add_argument("--max-hold-bars", default="8,14")
    ap.add_argument("--close-long-rules", default="none,ema20_mid")
    ap.add_argument("--close-short-rules", default="none,macd_hist_flip,close_above_ema26")
    ap.add_argument("--flush-rules", default="off,ema12_26_flip")
    ap.add_argument("--atr-periods", default="14")
    ap.add_argument("--rsi-periods", default="7,9")
    ap.add_argument("--min-round-trips", type=int, default=240)
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

    start_ms, end_ms = _resolve_window(args)
    raw = hcmod._load_klines_local(Path(args.klines_file), start_ms=start_ms, end_ms=end_ms)
    primary_raw = gp3scan._resample_klines(raw, primary_minutes)
    filter_raw = gp3scan._resample_klines(raw, filter_minutes)
    candidates = _build_grid(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    progress_path = out_dir / f"midfreq_trend_pullback_progress_{stamp}.csv"
    result_csv = out_dir / f"midfreq_trend_pullback_results_{stamp}.csv"
    summary_path = out_dir / f"midfreq_trend_pullback_summary_{stamp}.json"

    results: list[dict] = []
    last_feat_rows = 0
    for idx, cand in enumerate(candidates, start=1):
        session_mask = _resolve_session_mask(str(cand["session_mask"]))
        feat = _build_feature_frame(primary_raw, filter_raw, cand, session_mask)
        last_feat_rows = int(len(feat))
        if feat.empty:
            continue
        bars = gp3scan._build_bars(feat)

        cfg = gp3scan._clone_cfg(base_cfg)
        cfg.entry_pullback_bps = float(cand["entry_pullback_bps"])
        cfg.entry_timeout_bars = int(cand["entry_timeout_bars"])
        cfg.tp_atr_mult = float(cand["tp_atr_mult"])
        cfg.stop_atr_mult = float(cand["stop_atr_mult"])
        cfg.trail_atr_mult = float(cand["trail_atr_mult"])
        cfg.breakeven_trigger_atr = float(cand["breakeven_trigger_atr"])
        cfg.breakeven_offset_bps = float(cand["breakeven_offset_bps"])
        cfg.position_mult = float(cand["position_mult"])
        cfg.cooldown_bars = int(cand["cooldown_bars"])
        cfg.max_hold_bars = int(cand["max_hold_bars"])
        cfg.close_long_rule = str(cand["close_long_rule"])
        cfg.close_short_rule = str(cand["close_short_rule"])
        cfg.flush_rule = str(cand["flush_rule"])
        cfg.order_timeout_sec = int(primary_minutes * 60)

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
            gtc_queue_delay_ms=300,
            maker_buffer_ticks=0,
            allow_same_bar_entry_exit=True,
            stop_trigger_source=StopTriggerSource(str(args.stop_trigger_source)),
            allow_marketable_gtc_as_taker=True,
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
        strategy = CtaAwareTrendPullbackStrategy(feat, cfg, rules, gp3, bar_ms=primary_minutes * 60_000)
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
            "round_trips": round_trips,
            "candidate_index": int(idx),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(progress_path, index=False)
        print(
            f"[{idx}/{len(candidates)}] ret={row['net_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% "
            f"rt={row['round_trips']} sess={row['session_mask']} pf={row['primary_ema_fast']}/{row['primary_ema_slow']} "
            f"ff={row['filter_ema_fast']}/{row['filter_ema_slow']} pb={row['pullback_min_atr']:.2f}-{row['pullback_max_atr']:.2f} "
            f"tp/sl={row['tp_atr_mult']:.2f}/{row['stop_atr_mult']:.2f}"
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
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": int(len(results)),
            "feature_rows": int(last_feat_rows),
            "window": {
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "interval": str(args.interval),
                "filter_interval": str(args.filter_interval),
            },
            "top10": df.head(10).to_dict(orient="records"),
        }
    else:
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": 0,
            "feature_rows": int(last_feat_rows),
            "window": {
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "interval": str(args.interval),
                "filter_interval": str(args.filter_interval),
            },
            "top10": [],
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] progress={progress_path}")
    print(f"[DONE] results={result_csv}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
