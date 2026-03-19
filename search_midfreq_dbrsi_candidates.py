from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import run_gp3_hc_backtest as hcmod
import search_gp3_hc_candidates as gp3scan
from hc_engine import BacktestConfig, BacktestEngine, IntrabarPathModel, OrderStatus, Side, StopTriggerSource, TimeInForce, summarize


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


def _roe_to_price(entry: float, side: str, roe_pct: float, leverage: float) -> float:
    if entry <= 0.0 or leverage <= 0.0:
        return 0.0
    delta = abs(float(roe_pct)) / 100.0 / float(leverage)
    if str(side).upper() == "LONG":
        return float(entry) * (1.0 + delta) if float(roe_pct) >= 0.0 else float(entry) * (1.0 - delta)
    return float(entry) * (1.0 - delta) if float(roe_pct) >= 0.0 else float(entry) * (1.0 + delta)


def _build_feature_frame(raw: pd.DataFrame, cand: dict, session_mask: dict[str, set[int]]) -> pd.DataFrame:
    out = raw.copy()
    basis_period = int(cand["basis_period"])
    fast_period = int(cand["rsi_fast_period"])
    slow_period = int(cand["rsi_slow_period"])

    out["basis"] = out["close"].rolling(basis_period).mean()
    std = out["close"].rolling(basis_period).std(ddof=0)
    out["bandwidth"] = (2.0 * std) / out["basis"].replace(0.0, np.nan)
    out["zscore"] = (out["close"] - out["basis"]) / std.replace(0.0, np.nan)
    out["rsi_fast"] = _calc_rsi(out["close"], fast_period)
    out["rsi_slow"] = _calc_rsi(out["close"], slow_period)
    out["rsi_fast_prev"] = out["rsi_fast"].shift(1)
    out["z_prev"] = out["zscore"].shift(1)

    hour_utc = pd.to_datetime(out["close_time"], unit="ms", utc=True).dt.hour.astype(int)
    allow_long = hour_utc.isin(sorted(session_mask["long"]))
    allow_short = hour_utc.isin(sorted(session_mask["short"]))

    signal_mode = int(cand["signal_mode"])
    fast_os = float(cand["fast_oversold"])
    fast_ob = float(cand["fast_overbought"])
    slow_long_max = float(cand["slow_long_max"])
    slow_short_min = float(cand["slow_short_min"])
    zscore_min = float(cand["zscore_min"])
    reclaim_z_max = float(cand["reclaim_z_max"])
    rsi_rebound_min = float(cand["rsi_rebound_min"])
    bandwidth_min = float(cand["bandwidth_min"])

    if signal_mode == 0:
        out["long_signal"] = (
            allow_long
            & (out["bandwidth"] >= bandwidth_min)
            & (out["rsi_fast_prev"] > fast_os)
            & (out["rsi_fast"] <= fast_os)
            & (out["rsi_slow"] <= slow_long_max)
            & (out["zscore"] <= -zscore_min)
        )
        out["short_signal"] = (
            allow_short
            & (out["bandwidth"] >= bandwidth_min)
            & (out["rsi_fast_prev"] < fast_ob)
            & (out["rsi_fast"] >= fast_ob)
            & (out["rsi_slow"] >= slow_short_min)
            & (out["zscore"] >= zscore_min)
        )
    else:
        out["long_signal"] = (
            allow_long
            & (out["bandwidth"] >= bandwidth_min)
            & (out["z_prev"] <= -zscore_min)
            & (out["zscore"] >= -reclaim_z_max)
            & (out["rsi_fast_prev"] <= fast_os)
            & (out["rsi_fast"] >= out["rsi_fast_prev"] + rsi_rebound_min)
            & (out["rsi_slow"] <= slow_long_max)
        )
        out["short_signal"] = (
            allow_short
            & (out["bandwidth"] >= bandwidth_min)
            & (out["z_prev"] >= zscore_min)
            & (out["zscore"] <= reclaim_z_max)
            & (out["rsi_fast_prev"] >= fast_ob)
            & (out["rsi_fast"] <= out["rsi_fast_prev"] - rsi_rebound_min)
            & (out["rsi_slow"] >= slow_short_min)
        )

    out["long_exit_flag"] = (out["close"] >= out["basis"]) | (out["rsi_fast"] >= float(cand["exit_long_rsi"]))
    out["short_exit_flag"] = (out["close"] <= out["basis"]) | (out["rsi_fast"] <= float(cand["exit_short_rsi"]))
    out["in_mid"] = (out["rsi_fast"] >= float(cand["mid_band_low"])) & (out["rsi_fast"] <= float(cand["mid_band_high"]))

    need = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "basis",
        "bandwidth",
        "zscore",
        "rsi_fast",
        "rsi_slow",
        "long_signal",
        "short_signal",
        "long_exit_flag",
        "short_exit_flag",
        "in_mid",
    ]
    return out[need].dropna().reset_index(drop=True)


class DbrsiHcStrategy:
    def __init__(self, feat: pd.DataFrame, cfg, rules, gp3_mod, *, bar_ms: int):
        self.df = feat.reset_index(drop=True).copy()
        self.cfg = cfg
        self.rules = rules
        self.gp3 = gp3_mod
        self.index_by_open = {int(v): int(i) for i, v in self.df["open_time"].items()}
        self.cur_idx = -1
        self.bar_settle_delay_ms = int(max(float(getattr(cfg, "bar_settle_delay_sec", 0.0) or 0.0), 0.0) * 1000.0)
        self.entry_bar_timeout = max(int(getattr(cfg, "entry_timeout_bars", 1) or 1), 1)

        self.pending_entry: Optional[dict] = None
        self.close_order_id: Optional[str] = None
        self.tp_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None
        self.stop_price: float = 0.0
        self.bars_held: int = 0
        self.be_triggered: bool = False
        self.cooldown_until: int = -1

        self.need_mid_long = False
        self.need_mid_short = False
        self.mid_ready_long = True
        self.mid_ready_short = True
        self.mid_long_streak = 0
        self.mid_short_streak = 0
        self._seen_fills = 0

    @staticmethod
    def _is_order_active(engine: BacktestEngine, order_id: Optional[str]) -> bool:
        if not order_id:
            return False
        order = engine.orders.get(str(order_id))
        if not order:
            return False
        return order.status in (OrderStatus.NEW, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED)

    def _clear_exit_orders(self, engine: BacktestEngine) -> None:
        for oid in (self.tp_order_id, self.sl_order_id):
            if oid:
                engine.cancel_order(str(oid))
        self.tp_order_id = None
        self.sl_order_id = None
        self.stop_price = 0.0

    def _close_cleanup(self, engine: BacktestEngine) -> None:
        self._clear_exit_orders(engine)
        if self.close_order_id:
            engine.cancel_order(str(self.close_order_id))
        self.close_order_id = None
        self.pending_entry = None
        self.bars_held = 0
        self.be_triggered = False
        if self.cur_idx >= 0:
            self.cooldown_until = max(self.cooldown_until, int(self.cur_idx + int(self.cfg.cooldown_bars)))

    def _place_exit_orders(self, engine: BacktestEngine, ts_ms: int) -> None:
        qty, entry = engine.get_position()
        if qty == 0.0 or float(entry) <= 0.0:
            return
        side = "LONG" if qty > 0.0 else "SHORT"
        tp_px = _roe_to_price(float(entry), side, float(self.cfg.tp_roe_pct), float(self.cfg.leverage))
        sl_px = _roe_to_price(float(entry), side, -abs(float(self.cfg.sl_roe_pct)), float(self.cfg.leverage))
        tick = float(self.rules.tick_size)
        if side == "LONG":
            tp_px = float(self.gp3.floor_to_step(tp_px, tick))
            self.stop_price = float(self.gp3.floor_to_step(sl_px, tick))
            exit_side = Side.SELL
        else:
            tp_px = float(self.gp3.ceil_to_step(tp_px, tick))
            self.stop_price = float(self.gp3.ceil_to_step(sl_px, tick))
            exit_side = Side.BUY

        self.tp_order_id = engine.place_limit(
            side=exit_side,
            qty=abs(float(qty)),
            limit_price=float(tp_px),
            ts_ms=int(ts_ms),
            tif=TimeInForce.GTC,
            reduce_only=True,
            reason="dbrsi_tp",
        )
        self.sl_order_id = engine.place_stop_market(
            side=exit_side,
            qty=abs(float(qty)),
            stop_price=float(self.stop_price),
            ts_ms=int(ts_ms),
            reduce_only=True,
            reason="dbrsi_sl",
        )

    def _mark_mid_reset(self, row: pd.Series) -> None:
        in_mid = bool(row.get("in_mid", False))
        reset_bars = max(int(self.cfg.mid_reset_bars), 1)
        if in_mid:
            if self.need_mid_long and not self.mid_ready_long:
                self.mid_long_streak += 1
                if self.mid_long_streak >= reset_bars:
                    self.mid_ready_long = True
                    self.mid_long_streak = 0
            if self.need_mid_short and not self.mid_ready_short:
                self.mid_short_streak += 1
                if self.mid_short_streak >= reset_bars:
                    self.mid_ready_short = True
                    self.mid_short_streak = 0
        else:
            self.mid_long_streak = 0
            self.mid_short_streak = 0

    def _sync_fills(self, engine: BacktestEngine) -> None:
        fills = engine.state.fills
        if self._seen_fills >= len(fills):
            return
        for fill in fills[self._seen_fills :]:
            oid = str(fill.order_id)
            if self.pending_entry and oid == str(self.pending_entry.get("order_id")):
                side = str(self.pending_entry.get("side", "")).upper()
                self.pending_entry = None
                self.close_order_id = None
                self.bars_held = 0
                self.be_triggered = False
                if side == "LONG":
                    self.need_mid_long = True
                    self.mid_ready_long = False
                    self.mid_long_streak = 0
                elif side == "SHORT":
                    self.need_mid_short = True
                    self.mid_ready_short = False
                    self.mid_short_streak = 0
                self._clear_exit_orders(engine)
                self._place_exit_orders(engine, int(fill.ts_ms))
                continue
            if self.close_order_id and oid == str(self.close_order_id):
                self._close_cleanup(engine)
                continue
            if oid == str(self.tp_order_id) or oid == str(self.sl_order_id):
                self._close_cleanup(engine)
                continue
        self._seen_fills = len(fills)

    def on_bar_open(self, engine: BacktestEngine, bar) -> None:
        self.cur_idx = self.index_by_open.get(int(bar.open_time_ms), -1)
        if self.cur_idx < 0:
            return
        self._sync_fills(engine)
        if self.pending_entry and self.cur_idx > int(self.pending_entry["expire_idx"]):
            engine.cancel_order(str(self.pending_entry["order_id"]))
            self.pending_entry = None

        signal_idx = int(self.cur_idx - 1)
        if signal_idx < 0:
            return
        row = self.df.iloc[signal_idx]
        self._mark_mid_reset(row)
        qty, _ = engine.get_position()
        if qty == 0.0:
            self._clear_exit_orders(engine)
        if qty == 0.0 and self.pending_entry is None and self.close_order_id is None and self.cur_idx >= self.cooldown_until:
            long_sig = bool(row.get("long_signal", False)) and self.mid_ready_long
            short_sig = bool(row.get("short_signal", False)) and self.mid_ready_short
            if long_sig != short_sig:
                side = "LONG" if long_sig else "SHORT"
                if side == "LONG":
                    px = self.gp3.floor_to_step(
                        float(bar.open) * (1.0 - float(self.cfg.entry_pullback_bps) / 10_000.0),
                        float(self.rules.tick_size),
                    )
                else:
                    px = self.gp3.ceil_to_step(
                        float(bar.open) * (1.0 + float(self.cfg.entry_pullback_bps) / 10_000.0),
                        float(self.rules.tick_size),
                    )
                cash = float(engine.state.cash)
                qty_v, _ = hcmod._calc_qty_compat(self.gp3, cash, float(px), self.rules, self.cfg)
                if float(qty_v) > 0.0:
                    order_ts = int(bar.open_time_ms) + int(self.bar_settle_delay_ms)
                    if order_ts >= int(bar.close_time_ms):
                        order_ts = int(bar.open_time_ms)
                    oid = engine.place_limit(
                        side=Side.BUY if side == "LONG" else Side.SELL,
                        qty=float(qty_v),
                        limit_price=float(px),
                        ts_ms=int(order_ts),
                        tif=TimeInForce.GTX,
                        reduce_only=False,
                        reason=f"dbrsi_entry_{side.lower()}",
                    )
                    self.pending_entry = {
                        "order_id": oid,
                        "side": side,
                        "expire_idx": int(signal_idx + self.entry_bar_timeout),
                    }
        if qty != 0.0 and self.close_order_id is None:
            exit_signal = (qty > 0.0 and bool(row.get("long_exit_flag", False))) or (
                qty < 0.0 and bool(row.get("short_exit_flag", False))
            )
            time_exit = self.bars_held >= int(self.cfg.max_hold_bars)
            if exit_signal or time_exit:
                self._clear_exit_orders(engine)
                self.close_order_id = engine.place_market(
                    side=Side.SELL if qty > 0.0 else Side.BUY,
                    qty=abs(float(qty)),
                    ts_ms=int(bar.open_time_ms),
                    reduce_only=True,
                    reason="dbrsi_exit_signal" if exit_signal else "dbrsi_exit_time",
                )

    def on_price_event(self, engine: BacktestEngine, event) -> None:
        self._sync_fills(engine)
        qty, _ = engine.get_position()
        if qty == 0.0 or self.close_order_id:
            return
        if (not self._is_order_active(engine, self.tp_order_id)) or (not self._is_order_active(engine, self.sl_order_id)):
            self._clear_exit_orders(engine)
            self._place_exit_orders(engine, int(event.ts_ms))

    def on_bar_close(self, engine: BacktestEngine, bar) -> None:
        self._sync_fills(engine)
        qty, entry = engine.get_position()
        if qty == 0.0 or self.cur_idx < 0 or self.close_order_id:
            return
        self.bars_held += 1
        trigger_roe = float(self.cfg.breakeven_trigger_roe_pct)
        if trigger_roe <= 0.0:
            return
        lev = max(float(self.cfg.leverage), 1.0)
        trigger_pct = trigger_roe / 100.0 / lev
        offset_pct = float(self.cfg.breakeven_offset_roe_pct) / 100.0 / lev
        row = self.df.iloc[self.cur_idx]
        new_stop = None
        if qty > 0.0:
            if (not self.be_triggered) and float(row["high"]) >= float(entry) * (1.0 + trigger_pct):
                new_stop = float(entry) * (1.0 + offset_pct)
                self.be_triggered = True
            if new_stop is not None and float(new_stop) > float(self.stop_price):
                self.stop_price = float(self.gp3.floor_to_step(float(new_stop), float(self.rules.tick_size)))
        else:
            if (not self.be_triggered) and float(row["low"]) <= float(entry) * (1.0 - trigger_pct):
                new_stop = float(entry) * (1.0 - offset_pct)
                self.be_triggered = True
            if new_stop is not None and (float(self.stop_price) <= 0.0 or float(new_stop) < float(self.stop_price)):
                self.stop_price = float(self.gp3.ceil_to_step(float(new_stop), float(self.rules.tick_size)))
        if new_stop is not None:
            if self._is_order_active(engine, self.sl_order_id):
                engine.cancel_order(str(self.sl_order_id))
            self.sl_order_id = engine.place_stop_market(
                side=Side.SELL if qty > 0.0 else Side.BUY,
                qty=abs(float(qty)),
                stop_price=float(self.stop_price),
                ts_ms=int(bar.close_time_ms),
                reduce_only=True,
                reason="dbrsi_be_sl",
            )


def _build_grid(args) -> list[dict]:
    fields = [
        "session_mask",
        "rsi_fast_period",
        "rsi_slow_period",
        "signal_mode",
        "fast_oversold",
        "fast_overbought",
        "slow_long_max",
        "slow_short_min",
        "basis_period",
        "bandwidth_min",
        "zscore_min",
        "reclaim_z_max",
        "rsi_rebound_min",
        "exit_long_rsi",
        "exit_short_rsi",
        "mid_band_low",
        "mid_band_high",
        "mid_reset_bars",
        "entry_pullback_bps",
        "entry_timeout_bars",
        "tp_roe_pct",
        "sl_roe_pct",
        "breakeven_trigger_roe_pct",
        "breakeven_offset_roe_pct",
        "position_mult",
        "cooldown_bars",
        "max_hold_bars",
    ]
    grids = {
        "session_mask": [x.strip() for x in str(args.session_masks).split(",") if x.strip()],
        "rsi_fast_period": gp3scan._split_ints(args.rsi_fast_periods),
        "rsi_slow_period": gp3scan._split_ints(args.rsi_slow_periods),
        "signal_mode": gp3scan._split_ints(args.signal_modes),
        "fast_oversold": gp3scan._split_floats(args.fast_oversolds),
        "fast_overbought": gp3scan._split_floats(args.fast_overboughts),
        "slow_long_max": gp3scan._split_floats(args.slow_long_maxs),
        "slow_short_min": gp3scan._split_floats(args.slow_short_mins),
        "basis_period": gp3scan._split_ints(args.basis_periods),
        "bandwidth_min": gp3scan._split_floats(args.bandwidth_mins),
        "zscore_min": gp3scan._split_floats(args.zscore_mins),
        "reclaim_z_max": gp3scan._split_floats(args.reclaim_z_maxs),
        "rsi_rebound_min": gp3scan._split_floats(args.rsi_rebound_mins),
        "exit_long_rsi": gp3scan._split_floats(args.exit_long_rsis),
        "exit_short_rsi": gp3scan._split_floats(args.exit_short_rsis),
        "mid_band_low": gp3scan._split_floats(args.mid_band_lows),
        "mid_band_high": gp3scan._split_floats(args.mid_band_highs),
        "mid_reset_bars": gp3scan._split_ints(args.mid_reset_bars),
        "entry_pullback_bps": gp3scan._split_floats(args.entry_pullbacks),
        "entry_timeout_bars": gp3scan._split_ints(args.entry_timeout_bars),
        "tp_roe_pct": gp3scan._split_floats(args.tp_roe_pcts),
        "sl_roe_pct": gp3scan._split_floats(args.sl_roe_pcts),
        "breakeven_trigger_roe_pct": gp3scan._split_floats(args.breakeven_trigger_roe_pcts),
        "breakeven_offset_roe_pct": gp3scan._split_floats(args.breakeven_offset_roe_pcts),
        "position_mult": gp3scan._split_floats(args.position_mults),
        "cooldown_bars": gp3scan._split_ints(args.cooldown_bars),
        "max_hold_bars": gp3scan._split_ints(args.max_hold_bars),
    }
    for name, values in grids.items():
        if not values:
            raise RuntimeError(f"grid field empty: {name}")
    out: list[dict] = []
    for combo in product(*(grids[name] for name in fields)):
        cand = {fields[i]: combo[i] for i in range(len(fields))}
        if int(cand["rsi_fast_period"]) >= int(cand["rsi_slow_period"]):
            continue
        if float(cand["fast_oversold"]) >= float(cand["fast_overbought"]):
            continue
        if float(cand["slow_long_max"]) >= float(cand["slow_short_min"]):
            continue
        if float(cand["mid_band_low"]) >= float(cand["mid_band_high"]):
            continue
        out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Search 5m/15m DBRSI-style mid-frequency candidates on HC")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--klines-file", default="d:/project/hc/cache/klines_ETHUSDC_1741904460000_1773440460000_1m.pkl")
    ap.add_argument("--out-dir", default="d:/project/hc/output/midfreq_dbrsi_search")
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--start-utc", default="")
    ap.add_argument("--end-utc", default="")
    ap.add_argument("--interval", default="5m", choices=["5m", "15m"])
    ap.add_argument("--stop-trigger-source", default="mark", choices=["last", "mark"])
    ap.add_argument("--tick-size", type=float, default=0.0)
    ap.add_argument("--step-size", type=float, default=0.0)
    ap.add_argument("--min-qty", type=float, default=0.0)
    ap.add_argument("--session-masks", default="all,cta_eth_side,eu_us")
    ap.add_argument("--rsi-fast-periods", default="6,8")
    ap.add_argument("--rsi-slow-periods", default="14,21")
    ap.add_argument("--signal-modes", default="0,1")
    ap.add_argument("--fast-oversolds", default="20,24")
    ap.add_argument("--fast-overboughts", default="76,80")
    ap.add_argument("--slow-long-maxs", default="45,50")
    ap.add_argument("--slow-short-mins", default="50,55")
    ap.add_argument("--basis-periods", default="18,24")
    ap.add_argument("--bandwidth-mins", default="0.0015,0.0025")
    ap.add_argument("--zscore-mins", default="1.0,1.4")
    ap.add_argument("--reclaim-z-maxs", default="0.25,0.50")
    ap.add_argument("--rsi-rebound-mins", default="2,4")
    ap.add_argument("--exit-long-rsis", default="52")
    ap.add_argument("--exit-short-rsis", default="48")
    ap.add_argument("--mid-band-lows", default="45")
    ap.add_argument("--mid-band-highs", default="55")
    ap.add_argument("--mid-reset-bars", default="1,2")
    ap.add_argument("--entry-pullbacks", default="2,4")
    ap.add_argument("--entry-timeout-bars", default="1,2")
    ap.add_argument("--tp-roe-pcts", default="4,6")
    ap.add_argument("--sl-roe-pcts", default="6,8")
    ap.add_argument("--breakeven-trigger-roe-pcts", default="3")
    ap.add_argument("--breakeven-offset-roe-pcts", default="0")
    ap.add_argument("--position-mults", default="4,6,8")
    ap.add_argument("--cooldown-bars", default="0,1")
    ap.add_argument("--max-hold-bars", default="6,10")
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
    interval_minutes = gp3scan._parse_interval_minutes(args.interval)
    start_ms, end_ms = _resolve_window(args)
    raw = hcmod._load_klines_local(Path(args.klines_file), start_ms=start_ms, end_ms=end_ms)
    primary_raw = gp3scan._resample_klines(raw, interval_minutes)
    candidates = _build_grid(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    progress_path = out_dir / f"midfreq_dbrsi_progress_{stamp}.csv"
    result_csv = out_dir / f"midfreq_dbrsi_results_{stamp}.csv"
    summary_path = out_dir / f"midfreq_dbrsi_summary_{stamp}.json"

    results: list[dict] = []
    last_feat_rows = 0
    for idx, cand in enumerate(candidates, start=1):
        session_mask = _resolve_session_mask(str(cand["session_mask"]))
        feat = _build_feature_frame(primary_raw, cand, session_mask)
        last_feat_rows = int(len(feat))
        if feat.empty:
            continue
        bars = gp3scan._build_bars(feat)

        cfg = gp3scan._clone_cfg(base_cfg)
        cfg.entry_pullback_bps = float(cand["entry_pullback_bps"])
        cfg.entry_timeout_bars = int(cand["entry_timeout_bars"])
        cfg.tp_roe_pct = float(cand["tp_roe_pct"])
        cfg.sl_roe_pct = float(cand["sl_roe_pct"])
        cfg.breakeven_trigger_roe_pct = float(cand["breakeven_trigger_roe_pct"])
        cfg.breakeven_offset_roe_pct = float(cand["breakeven_offset_roe_pct"])
        cfg.position_mult = float(cand["position_mult"])
        cfg.cooldown_bars = int(cand["cooldown_bars"])
        cfg.max_hold_bars = int(cand["max_hold_bars"])
        cfg.mid_reset_bars = int(cand["mid_reset_bars"])

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
        strategy = DbrsiHcStrategy(feat, cfg, rules, gp3, bar_ms=interval_minutes * 60_000)
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
            "round_trips": round_trips,
            "candidate_index": int(idx),
        }
        results.append(row)
        pd.DataFrame(results).to_csv(progress_path, index=False)
        print(
            f"[{idx}/{len(candidates)}] ret={row['net_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% "
            f"rt={row['round_trips']} mode={row['signal_mode']} session={row['session_mask']} "
            f"rsi={row['rsi_fast_period']}/{row['rsi_slow_period']} z={row['zscore_min']:.2f} "
            f"tp/sl={row['tp_roe_pct']:.1f}/{row['sl_roe_pct']:.1f} hold={row['max_hold_bars']}"
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
            "window": {"start_ms": int(start_ms), "end_ms": int(end_ms), "interval": str(args.interval)},
            "top10": df.head(10).to_dict(orient="records"),
        }
    else:
        summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": 0,
            "feature_rows": int(last_feat_rows),
            "window": {"start_ms": int(start_ms), "end_ms": int(end_ms), "interval": str(args.interval)},
            "top10": [],
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] progress={progress_path}")
    print(f"[DONE] results={result_csv}")
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
