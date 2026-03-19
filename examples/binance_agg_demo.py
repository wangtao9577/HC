from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc_engine import (  # noqa: E402
    AggTradeFetchSpec,
    BacktestConfig,
    BacktestEngine,
    IntrabarPathModel,
    Side,
    TimeInForce,
    aggtrades_to_bars_and_events,
    fetch_binance_futures_aggtrades,
    summarize,
)


class DemoStrategy:
    def __init__(self):
        self.prev_close = None
        self.tp_id = None
        self.sl_id = None

    def on_bar_open(self, engine: BacktestEngine, bar) -> None:
        qty, _ = engine.get_position()
        if qty != 0.0:
            return
        if self.prev_close is None:
            return
        if bar.open > self.prev_close:
            engine.place_market(side=Side.BUY, qty=1.0, ts_ms=bar.open_time_ms, reason="agg_demo_long")
        elif bar.open < self.prev_close:
            engine.place_market(side=Side.SELL, qty=1.0, ts_ms=bar.open_time_ms, reason="agg_demo_short")

    def on_price_event(self, engine: BacktestEngine, event) -> None:
        qty, entry = engine.get_position()
        if qty == 0.0:
            self._cleanup(engine)
            return
        if self.tp_id or self.sl_id:
            return

        if qty > 0:
            self.tp_id = engine.place_limit(
                side=Side.SELL,
                qty=abs(qty),
                limit_price=entry * 1.003,
                ts_ms=event.ts_ms,
                tif=TimeInForce.GTX,
                reduce_only=True,
                reason="agg_demo_tp",
            )
            self.sl_id = engine.place_stop_market(
                side=Side.SELL,
                qty=abs(qty),
                stop_price=entry * 0.997,
                ts_ms=event.ts_ms,
                reduce_only=True,
                reason="agg_demo_sl",
            )
        else:
            self.tp_id = engine.place_limit(
                side=Side.BUY,
                qty=abs(qty),
                limit_price=entry * 0.997,
                ts_ms=event.ts_ms,
                tif=TimeInForce.GTX,
                reduce_only=True,
                reason="agg_demo_tp",
            )
            self.sl_id = engine.place_stop_market(
                side=Side.BUY,
                qty=abs(qty),
                stop_price=entry * 1.003,
                ts_ms=event.ts_ms,
                reduce_only=True,
                reason="agg_demo_sl",
            )

    def on_bar_close(self, engine: BacktestEngine, bar) -> None:
        self.prev_close = float(bar.close)
        self._cleanup(engine)

    def _cleanup(self, engine: BacktestEngine) -> None:
        qty, _ = engine.get_position()
        if qty != 0.0:
            return
        if self.tp_id:
            engine.cancel_order(self.tp_id)
            self.tp_id = None
        if self.sl_id:
            engine.cancel_order(self.sl_id)
            self.sl_id = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDC")
    ap.add_argument("--minutes", type=int, default=60, help="recent window length")
    ap.add_argument("--bar-sec", type=int, default=60, help="bar size in seconds")
    args = ap.parse_args()

    # Keep default window short to avoid API throttling in high-volume symbols.
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(minutes=max(5, int(args.minutes)))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    spec = AggTradeFetchSpec(symbol=str(args.symbol).upper(), start_ms=start_ms, end_ms=end_ms)
    df = fetch_binance_futures_aggtrades(spec, use_cache=True, cache_dir=ROOT / "cache")
    bars, events_by_bar = aggtrades_to_bars_and_events(
        df,
        start_ms=start_ms,
        end_ms=end_ms,
        bar_ms=max(1, int(args.bar_sec)) * 1000,
        fill_empty_bars=True,
    )
    if not bars:
        raise RuntimeError("no bars generated from aggTrades")

    engine = BacktestEngine(
        BacktestConfig(
            symbol=str(args.symbol).upper(),
            initial_cash=10000.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0004,
            market_slippage_bps=0.5,
            stop_slippage_bps=2.0,
            tick_size=0.01,
            maker_queue_delay_ms=200,
            maker_buffer_ticks=0,
            allow_same_bar_entry_exit=True,
        ),
        path_model=IntrabarPathModel(mode="oc_aware"),
    )
    result = engine.run(bars, DemoStrategy(), events_by_bar=events_by_bar)
    stats = summarize(result)

    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result["fills"]).to_csv(out_dir / "agg_demo_fills.csv", index=False)
    pd.DataFrame(result["equity_curve"]).to_csv(out_dir / "agg_demo_equity.csv", index=False)
    pd.DataFrame(df).to_csv(out_dir / "agg_demo_raw_aggtrades.csv", index=False)

    print("HC aggTrades demo summary:")
    print(f"  window_utc: {start_dt.isoformat()} -> {end_dt.isoformat()}")
    print(f"  aggtrades: {len(df)}")
    print(f"  bars: {len(bars)}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"fills_csv: {out_dir / 'agg_demo_fills.csv'}")
    print(f"equity_csv: {out_dir / 'agg_demo_equity.csv'}")
    print(f"agg_raw_csv: {out_dir / 'agg_demo_raw_aggtrades.csv'}")


if __name__ == "__main__":
    main()
