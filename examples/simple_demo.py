from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc_engine import (
    BacktestConfig,
    BacktestEngine,
    Bar,
    IntrabarPathModel,
    Side,
    TimeInForce,
    summarize,
)


class DemoStrategy:
    def __init__(self):
        self.prev_bar: Bar | None = None
        self.tp_id: str | None = None
        self.sl_id: str | None = None

    def on_bar_open(self, engine: BacktestEngine, bar: Bar) -> None:
        qty, _ = engine.get_position()
        if qty != 0.0:
            return
        if self.prev_bar is None:
            return
        if self.prev_bar.close > self.prev_bar.open:
            engine.place_market(side=Side.BUY, qty=1.0, ts_ms=bar.open_time_ms, reason="demo_entry_long")
        elif self.prev_bar.close < self.prev_bar.open:
            engine.place_market(side=Side.SELL, qty=1.0, ts_ms=bar.open_time_ms, reason="demo_entry_short")

    def on_price_event(self, engine: BacktestEngine, event) -> None:
        qty, entry = engine.get_position()
        if qty == 0.0:
            self._clear_if_done(engine)
            return

        if self.tp_id or self.sl_id:
            return

        if qty > 0:
            tp = entry * 1.004
            sl = entry * 0.996
            self.tp_id = engine.place_limit(
                side=Side.SELL,
                qty=abs(qty),
                limit_price=tp,
                ts_ms=event.ts_ms,
                tif=TimeInForce.GTX,
                reduce_only=True,
                reason="demo_tp",
            )
            self.sl_id = engine.place_stop_market(
                side=Side.SELL,
                qty=abs(qty),
                stop_price=sl,
                ts_ms=event.ts_ms,
                reduce_only=True,
                reason="demo_sl",
            )
        else:
            tp = entry * 0.996
            sl = entry * 1.004
            self.tp_id = engine.place_limit(
                side=Side.BUY,
                qty=abs(qty),
                limit_price=tp,
                ts_ms=event.ts_ms,
                tif=TimeInForce.GTX,
                reduce_only=True,
                reason="demo_tp",
            )
            self.sl_id = engine.place_stop_market(
                side=Side.BUY,
                qty=abs(qty),
                stop_price=sl,
                ts_ms=event.ts_ms,
                reduce_only=True,
                reason="demo_sl",
            )

    def on_bar_close(self, engine: BacktestEngine, bar: Bar) -> None:
        self.prev_bar = bar
        self._clear_if_done(engine)

    def _clear_if_done(self, engine: BacktestEngine) -> None:
        # Reset local IDs when order no longer active.
        if self.tp_id:
            o = engine.orders.get(self.tp_id)
            if (o is None) or (getattr(o.status, "name", "") != "NEW"):
                self.tp_id = None
        if self.sl_id:
            o = engine.orders.get(self.sl_id)
            if (o is None) or (getattr(o.status, "name", "") != "NEW"):
                self.sl_id = None
        # When flat, cancel any lingering protective order.
        qty, _ = engine.get_position()
        if qty == 0.0:
            if self.tp_id:
                engine.cancel_order(self.tp_id)
                self.tp_id = None
            if self.sl_id:
                engine.cancel_order(self.sl_id)
                self.sl_id = None


def synthetic_bars(n: int = 1200, start_price: float = 2000.0) -> list[Bar]:
    out: list[Bar] = []
    ts = 0
    px = float(start_price)
    rng = random.Random(42)
    for i in range(n):
        t = i / 60.0
        drift = 0.12 * math.sin(t) + 0.06 * math.sin(t / 3.0)
        noise = rng.uniform(-0.35, 0.35)
        o = px
        c = max(1.0, o + drift + noise)
        wick = abs(drift) * 0.8 + rng.uniform(0.05, 0.6)
        h = max(o, c) + wick
        l = min(o, c) - wick
        out.append(
            Bar(
                open_time_ms=ts,
                close_time_ms=ts + 60_000,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=100.0 + rng.uniform(0.0, 40.0),
            )
        )
        px = c
        ts += 60_000
    return out


def main() -> None:
    bars = synthetic_bars()
    cfg = BacktestConfig(
        symbol="ETHUSDC",
        initial_cash=10_000.0,
        maker_fee_rate=0.0,
        taker_fee_rate=0.0004,
        market_slippage_bps=0.5,
        stop_slippage_bps=2.0,
        tick_size=0.01,
        maker_queue_delay_ms=1000,
        maker_buffer_ticks=0,
        allow_same_bar_entry_exit=True,
    )
    engine = BacktestEngine(config=cfg, path_model=IntrabarPathModel(mode="oc_aware"))
    result = engine.run(bars, DemoStrategy())
    summary = summarize(result)

    out_dir = Path("d:/project/hc/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["fills"]).to_csv(out_dir / "demo_fills.csv", index=False)
    pd.DataFrame(result["equity_curve"]).to_csv(out_dir / "demo_equity.csv", index=False)

    print("HC demo summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"fills_csv: {out_dir / 'demo_fills.csv'}")
    print(f"equity_csv: {out_dir / 'demo_equity.csv'}")


if __name__ == "__main__":
    main()
