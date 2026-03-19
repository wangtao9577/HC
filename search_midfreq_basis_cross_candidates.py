from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd

import run_gp3_hc_backtest as hcmod
import search_gp3_hc_candidates as gp3scan
from hc_engine import BacktestConfig, BacktestEngine, IntrabarPathModel, OrderStatus, Side, StopTriggerSource, TimeInForce, summarize


SESSION_MASKS: dict[str, dict[str, set[int]]] = {
    "all": {
        "long": set(range(24)),
        "short": set(range(24)),
    },
    "eu_us": {
        "long": set(range(7, 22)),
        "short": set(range(7, 22)),
    },
    "asia_eu": {
        "long": set(range(0, 16)),
        "short": set(range(0, 16)),
    },
    "ny_asia": {
        "long": {0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        "short": {0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
    },
    # Converted from CTA `cta_time_vol_best_hours.csv` ETH open_long/open_short BJ hours to UTC.
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


def _build_feature_frame(raw5m: pd.DataFrame, cand: dict, session_mask: dict[str, set[int]]) -> pd.DataFrame:
    out = raw5m.copy()

    boll_period = int(cand["boll_period"])
    boll_std = float(cand["boll_std"])
    macd_fast = int(cand["macd_fast"])
    macd_slow = int(cand["macd_slow"])
    macd_signal = int(cand["macd_signal"])
    tp_confirm_bars = int(cand["tp_confirm_bars"])
    slope_lookback = int(cand["slope_lookback"])

    out["basis"] = out["close"].rolling(boll_period).mean()
    std = out["close"].rolling(boll_period).std(ddof=0)
    out["upper"] = out["basis"] + boll_std * std
    out["lower"] = out["basis"] - boll_std * std
    out["bandwidth"] = (out["upper"] - out["lower"]) / out["basis"].replace(0.0, pd.NA)

    ema_fast = out["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = out["close"].ewm(span=macd_slow, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["signal"] = out["macd"].ewm(span=macd_signal, adjust=False).mean()
    out["hist"] = out["macd"] - out["signal"]
    out["macd_bps"] = (out["macd"] / out["close"].replace(0.0, pd.NA)) * 10_000.0
    out["hist_bps"] = (out["hist"] / out["close"].replace(0.0, pd.NA)) * 10_000.0
    out["slope_bps"] = ((out["basis"] - out["basis"].shift(slope_lookback)) / out["close"].replace(0.0, pd.NA)) * 10_000.0

    close_prev = out["close"].shift(1)
    basis_prev = out["basis"].shift(1)
    long_cross = (close_prev <= basis_prev) & (out["close"] > out["basis"])
    short_cross = (close_prev >= basis_prev) & (out["close"] < out["basis"])

    long_cond = (out["close"] < out["basis"]).astype(int)
    short_cond = (out["close"] > out["basis"]).astype(int)
    out["long_exit_flag"] = long_cond.rolling(tp_confirm_bars, min_periods=tp_confirm_bars).sum().eq(tp_confirm_bars)
    out["short_exit_flag"] = short_cond.rolling(tp_confirm_bars, min_periods=tp_confirm_bars).sum().eq(tp_confirm_bars)

    out["hour_utc"] = pd.to_datetime(out["close_time"], unit="ms", utc=True).dt.hour.astype(int)
    allowed_long = out["hour_utc"].isin(sorted(session_mask["long"]))
    allowed_short = out["hour_utc"].isin(sorted(session_mask["short"]))

    common = (
        (out["bandwidth"] >= float(cand["bandwidth_min"]))
        & (out["hist_bps"].abs() >= float(cand["hist_abs_min_bps"]))
        & (out["macd_bps"].abs() <= float(cand["macd_near_zero_max_bps"]))
    )
    out["long_signal"] = common & long_cross & (out["slope_bps"] >= float(cand["slope_long_min_bps"])) & allowed_long
    out["short_signal"] = common & short_cross & (out["slope_bps"] <= float(cand["slope_short_max_bps"])) & allowed_short

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
        "macd_bps",
        "hist_bps",
        "slope_bps",
        "long_signal",
        "short_signal",
        "long_exit_flag",
        "short_exit_flag",
    ]
    return out[need].dropna().reset_index(drop=True)


class BasisCrossHcStrategy:
    def __init__(self, feat: pd.DataFrame, cfg, rules, gp3_mod, *, bar_ms: int):
        self.df = feat.reset_index(drop=True).copy()
        self.cfg = cfg
        self.rules = rules
        self.gp3 = gp3_mod
        self.index_by_open = {int(v): int(i) for i, v in self.df["open_time"].items()}
        self.cur_idx = -1
        self.bar_ms = int(bar_ms)

        self.pending_entry: Optional[dict] = None
        self.pending_exit: Optional[dict] = None
        self.sl_order_id: Optional[str] = None

        self.entry_basis: float = 0.0
        self.stop_price: float = 0.0
        self.bars_held: int = 0
        self.be_triggered: bool = False
        self.cooldown_until: int = -1
        self.order_timeout_bars = max(1, math.ceil(float(cfg.order_timeout_sec) / max(float(bar_ms) / 1000.0, 1.0)))
        self.bar_settle_delay_ms = int(max(float(getattr(cfg, "bar_settle_delay_sec", 0.0) or 0.0), 0.0) * 1000.0)
        self._seen_fills = 0

    def _calc_initial_stop(self, side: str, entry_basis: float, entry_price: float) -> float:
        if side == "LONG":
            raw = float(entry_basis) * (1.0 - float(self.cfg.stop_offset_pct))
            cap = float(entry_price) * (1.0 - float(self.cfg.stop_cap_pct))
            return float(self.gp3.floor_to_step(max(raw, cap), float(self.rules.tick_size)))
        raw = float(entry_basis) * (1.0 + float(self.cfg.stop_offset_pct))
        cap = float(entry_price) * (1.0 + float(self.cfg.stop_cap_pct))
        return float(self.gp3.ceil_to_step(min(raw, cap), float(self.rules.tick_size)))

    def _place_stop(self, engine: BacktestEngine, ts_ms: int) -> None:
        qty, _ = engine.get_position()
        if qty == 0.0 or float(self.stop_price) <= 0.0:
            return
        side = Side.SELL if qty > 0 else Side.BUY
        self.sl_order_id = engine.place_stop_market(
            side=side,
            qty=abs(float(qty)),
            stop_price=float(self.stop_price),
            ts_ms=int(ts_ms),
            reduce_only=True,
            reason="basis_cross_sl",
        )

    def _clear_stop(self, engine: BacktestEngine) -> None:
        if self.sl_order_id:
            engine.cancel_order(str(self.sl_order_id))
        self.sl_order_id = None

    def _close_cleanup(self, engine: BacktestEngine) -> None:
        self._clear_stop(engine)
        if self.pending_exit:
            engine.cancel_order(str(self.pending_exit.get("order_id")))
        self.pending_entry = None
        self.pending_exit = None
        self.entry_basis = 0.0
        self.stop_price = 0.0
        self.be_triggered = False
        self.bars_held = 0
        if self.cur_idx >= 0:
            self.cooldown_until = max(self.cooldown_until, int(self.cur_idx + int(self.cfg.cooldown_bars)))

    @staticmethod
    def _is_order_active(engine: BacktestEngine, order_id: Optional[str]) -> bool:
        if not order_id:
            return False
        o = engine.orders.get(str(order_id))
        if not o:
            return False
        return o.status in (OrderStatus.NEW, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED)

    def _sync_fills(self, engine: BacktestEngine) -> None:
        fills = engine.state.fills
        if self._seen_fills >= len(fills):
            return
        for f in fills[self._seen_fills :]:
            oid = str(f.order_id)
            if self.pending_entry and oid == str(self.pending_entry.get("order_id")):
                o = engine.orders.get(oid)
                if o and o.status == OrderStatus.PARTIALLY_FILLED:
                    engine.cancel_order(oid)
                self.entry_basis = float(self.pending_entry.get("signal_basis", 0.0))
                self.stop_price = self._calc_initial_stop(
                    str(self.pending_entry.get("side")),
                    float(self.entry_basis),
                    float(f.price),
                )
                self.pending_entry = None
                self.pending_exit = None
                self.bars_held = 0
                self.be_triggered = False
                self._clear_stop(engine)
                self._place_stop(engine, int(f.ts_ms))
                continue

            if self.pending_exit and oid == str(self.pending_exit.get("order_id")):
                self._close_cleanup(engine)
                continue

            if oid == str(self.sl_order_id):
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
        if self.pending_exit and self.cur_idx > int(self.pending_exit["expire_idx"]):
            engine.cancel_order(str(self.pending_exit["order_id"]))
            self.pending_exit = None

        signal_idx = int(self.cur_idx - 1)
        if signal_idx < 0:
            return
        row = self.df.iloc[signal_idx]
        qty, _ = engine.get_position()

        if qty == 0.0 and self.pending_entry is None and self.pending_exit is None and self.cur_idx >= self.cooldown_until:
            long_sig = bool(row.get("long_signal", False))
            short_sig = bool(row.get("short_signal", False))
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
                if float(px) > 0:
                    cash = float(engine.state.cash)
                    qty_v, _ = hcmod._calc_qty_compat(self.gp3, cash, float(px), self.rules, self.cfg)
                    if float(qty_v) > 0:
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
                            reason=f"basis_cross_entry_{side.lower()}",
                        )
                        self.pending_entry = {
                            "order_id": oid,
                            "side": side,
                            "signal_basis": float(row["basis"]),
                            "expire_idx": int(signal_idx + int(self.cfg.entry_timeout_bars)),
                        }

        if qty != 0.0 and self.pending_exit is None:
            exit_signal = (qty > 0 and bool(row.get("long_exit_flag", False))) or (
                qty < 0 and bool(row.get("short_exit_flag", False))
            )
            time_exit = self.bars_held >= int(self.cfg.max_hold_bars)
            if exit_signal or time_exit:
                if qty > 0:
                    px = self.gp3.ceil_to_step(
                        float(bar.open) * (1.0 + float(getattr(self.cfg, "exit_push_bps", 0.0)) / 10_000.0),
                        float(self.rules.tick_size),
                    )
                    side = Side.SELL
                else:
                    px = self.gp3.floor_to_step(
                        float(bar.open) * (1.0 - float(getattr(self.cfg, "exit_push_bps", 0.0)) / 10_000.0),
                        float(self.rules.tick_size),
                    )
                    side = Side.BUY
                oid = engine.place_limit(
                    side=side,
                    qty=abs(float(qty)),
                    limit_price=float(px),
                    ts_ms=int(bar.open_time_ms),
                    tif=TimeInForce.GTC,
                    reduce_only=True,
                    reason="basis_cross_exit_time" if time_exit else "basis_cross_exit_signal",
                )
                self.pending_exit = {
                    "order_id": oid,
                    "expire_idx": int(self.cur_idx + self.order_timeout_bars),
                }

    def on_price_event(self, engine: BacktestEngine, event) -> None:
        self._sync_fills(engine)
        qty, _ = engine.get_position()
        if qty == 0.0:
            return
        if not self._is_order_active(engine, self.sl_order_id):
            self._clear_stop(engine)
            self._place_stop(engine, int(event.ts_ms))

    def on_bar_close(self, engine: BacktestEngine, bar) -> None:
        self._sync_fills(engine)
        qty, entry = engine.get_position()
        if qty == 0.0 or self.cur_idx < 0:
            return
        self.bars_held += 1

        row = self.df.iloc[self.cur_idx]
        h = float(row["high"])
        l = float(row["low"])
        new_stop = None
        if qty > 0:
            if (not self.be_triggered) and h >= float(entry) * (1.0 + float(self.cfg.breakeven_trigger_pct)):
                new_stop = float(entry) * (1.0 + float(self.cfg.breakeven_offset_pct))
                self.be_triggered = True
        else:
            if (not self.be_triggered) and l <= float(entry) * (1.0 - float(self.cfg.breakeven_trigger_pct)):
                new_stop = float(entry) * (1.0 - float(self.cfg.breakeven_offset_pct))
                self.be_triggered = True

        if new_stop is not None:
            if qty > 0 and float(new_stop) > float(self.stop_price):
                self.stop_price = float(self.gp3.floor_to_step(float(new_stop), float(self.rules.tick_size)))
            elif qty < 0 and float(new_stop) < float(self.stop_price):
                self.stop_price = float(self.gp3.ceil_to_step(float(new_stop), float(self.rules.tick_size)))
            if self._is_order_active(engine, self.sl_order_id):
                engine.cancel_order(str(self.sl_order_id))
            self._place_stop(engine, int(bar.close_time_ms))


def _build_grid(args) -> list[dict]:
    fields = [
        "session_mask",
        "boll_period",
        "boll_std",
        "macd_fast",
        "macd_slow",
        "macd_signal",
        "bandwidth_min",
        "hist_abs_min_bps",
        "macd_near_zero_max_bps",
        "slope_lookback",
        "slope_long_min_bps",
        "slope_short_max_bps",
        "tp_confirm_bars",
        "entry_pullback_bps",
        "entry_timeout_bars",
        "stop_offset_pct",
        "stop_cap_pct",
        "breakeven_trigger_pct",
        "breakeven_offset_pct",
        "position_mult",
        "cooldown_bars",
        "max_hold_bars",
        "order_timeout_sec",
        "exit_push_bps",
    ]
    grids = {
        "session_mask": [x.strip() for x in str(args.session_masks).split(",") if x.strip()],
        "boll_period": gp3scan._split_ints(args.boll_periods),
        "boll_std": gp3scan._split_floats(args.boll_stds),
        "macd_fast": gp3scan._split_ints(args.macd_fasts),
        "macd_slow": gp3scan._split_ints(args.macd_slows),
        "macd_signal": gp3scan._split_ints(args.macd_signals),
        "bandwidth_min": gp3scan._split_floats(args.bandwidth_mins),
        "hist_abs_min_bps": gp3scan._split_floats(args.hist_abs_min_bps),
        "macd_near_zero_max_bps": gp3scan._split_floats(args.macd_near_zero_max_bps),
        "slope_lookback": gp3scan._split_ints(args.slope_lookbacks),
        "slope_long_min_bps": gp3scan._split_floats(args.slope_long_min_bps),
        "slope_short_max_bps": gp3scan._split_floats(args.slope_short_max_bps),
        "tp_confirm_bars": gp3scan._split_ints(args.tp_confirm_bars),
        "entry_pullback_bps": gp3scan._split_floats(args.entry_pullbacks),
        "entry_timeout_bars": gp3scan._split_ints(args.entry_timeout_bars),
        "stop_offset_pct": gp3scan._split_floats(args.stop_offset_pcts),
        "stop_cap_pct": gp3scan._split_floats(args.stop_cap_pcts),
        "breakeven_trigger_pct": gp3scan._split_floats(args.breakeven_trigger_pcts),
        "breakeven_offset_pct": gp3scan._split_floats(args.breakeven_offset_pcts),
        "position_mult": gp3scan._split_floats(args.position_mults),
        "cooldown_bars": gp3scan._split_ints(args.cooldown_bars),
        "max_hold_bars": gp3scan._split_ints(args.max_hold_bars),
        "order_timeout_sec": gp3scan._split_ints(args.order_timeout_secs),
        "exit_push_bps": gp3scan._split_floats(args.exit_push_bps),
    }
    for name, values in grids.items():
        if not values:
            raise RuntimeError(f"grid field empty: {name}")

    out: list[dict] = []
    for combo in product(*(grids[name] for name in fields)):
        cand = {fields[i]: combo[i] for i in range(len(fields))}
        if int(cand["macd_fast"]) >= int(cand["macd_slow"]):
            continue
        out.append(cand)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Search 5m BOLL2-style basis-cross candidates on HC")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--klines-file", default="d:/project/hc/cache/klines_ETHUSDC_1741904460000_1773440460000_1m.pkl")
    ap.add_argument("--out-dir", default="d:/project/hc/output/midfreq_basis_cross_search")
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--interval", default="5m", choices=["5m"])
    ap.add_argument("--stop-trigger-source", default="mark", choices=["last", "mark"])
    ap.add_argument("--tick-size", type=float, default=0.0)
    ap.add_argument("--step-size", type=float, default=0.0)
    ap.add_argument("--min-qty", type=float, default=0.0)
    ap.add_argument("--session-masks", default="all,cta_eth_side,cta_eth_union,eu_us")
    ap.add_argument("--boll-periods", default="18,24")
    ap.add_argument("--boll-stds", default="2.0,2.4")
    ap.add_argument("--macd-fasts", default="12")
    ap.add_argument("--macd-slows", default="26")
    ap.add_argument("--macd-signals", default="9")
    ap.add_argument("--bandwidth-mins", default="0.002,0.003")
    ap.add_argument("--hist-abs-min-bps", default="1.0,2.0")
    ap.add_argument("--macd-near-zero-max-bps", default="8.0,12.0")
    ap.add_argument("--slope-lookbacks", default="1")
    ap.add_argument("--slope-long-min-bps", default="-2.0,0.0")
    ap.add_argument("--slope-short-max-bps", default="2.0,0.0")
    ap.add_argument("--tp-confirm-bars", default="1,2")
    ap.add_argument("--entry-pullbacks", default="1")
    ap.add_argument("--entry-timeout-bars", default="1")
    ap.add_argument("--stop-offset-pcts", default="0.008,0.012")
    ap.add_argument("--stop-cap-pcts", default="0.015,0.02")
    ap.add_argument("--breakeven-trigger-pcts", default="0.006,0.010")
    ap.add_argument("--breakeven-offset-pcts", default="0.0003")
    ap.add_argument("--position-mults", default="24")
    ap.add_argument("--cooldown-bars", default="0,1")
    ap.add_argument("--max-hold-bars", default="12,18")
    ap.add_argument("--order-timeout-secs", default="300")
    ap.add_argument("--exit-push-bps", default="0")
    ap.add_argument("--min-round-trips", type=int, default=250)
    ap.add_argument("--max-round-trips", type=int, default=400)
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
    start_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(days=float(args.days))
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    raw = hcmod._load_klines_local(Path(args.klines_file), start_ms=start_ms, end_ms=end_ms)
    raw5m = gp3scan._resample_klines(raw, interval_minutes)

    candidates = _build_grid(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    progress_path = out_dir / f"midfreq_basis_cross_progress_{stamp}.csv"
    result_csv = out_dir / f"midfreq_basis_cross_results_{stamp}.csv"
    summary_path = out_dir / f"midfreq_basis_cross_summary_{stamp}.json"

    results: list[dict] = []
    last_feat_rows = 0
    for idx, cand in enumerate(candidates, start=1):
        session_mask = _resolve_session_mask(str(cand["session_mask"]))
        feat = _build_feature_frame(raw5m, cand, session_mask)
        last_feat_rows = int(len(feat))
        if feat.empty:
            continue
        bars = gp3scan._build_bars(feat)

        cfg = gp3scan._clone_cfg(base_cfg)
        cfg.entry_pullback_bps = float(cand["entry_pullback_bps"])
        cfg.entry_timeout_bars = int(cand["entry_timeout_bars"])
        cfg.position_mult = float(cand["position_mult"])
        cfg.cooldown_bars = int(cand["cooldown_bars"])
        cfg.max_hold_bars = int(cand["max_hold_bars"])
        cfg.order_timeout_sec = int(cand["order_timeout_sec"])
        cfg.stop_offset_pct = float(cand["stop_offset_pct"])
        cfg.stop_cap_pct = float(cand["stop_cap_pct"])
        cfg.breakeven_trigger_pct = float(cand["breakeven_trigger_pct"])
        cfg.breakeven_offset_pct = float(cand["breakeven_offset_pct"])
        cfg.exit_push_bps = float(cand["exit_push_bps"])

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
        strategy = BasisCrossHcStrategy(feat, cfg, rules, gp3, bar_ms=interval_minutes * 60_000)
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
            f"[{idx}/{len(candidates)}] ret={row['net_return_pct']:.2f}% dd={row['max_drawdown_pct']:.2f}% "
            f"rt={row['round_trips']} session={row['session_mask']} bb={row['boll_period']}/{row['boll_std']:.2f} "
            f"bw={row['bandwidth_min']:.3f} hist={row['hist_abs_min_bps']:.1f} macd0={row['macd_near_zero_max_bps']:.1f} "
            f"hold={row['max_hold_bars']} confirm={row['tp_confirm_bars']}"
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
        "source_bars_1m": int(len(raw)),
        "bars_after_resample": int(len(raw5m)),
        "bars_after_last_features": int(last_feat_rows),
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
