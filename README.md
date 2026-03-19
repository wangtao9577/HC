# HC Backtest Core

`hc` is a standalone backtest model layer intended for reuse by multiple strategies.

## Design Goals

- Event-driven matching engine (not bar-close-only PnL math)
- No lookahead: orders only become active after submission + latency
- Same-bar open/close handled by intrabar path replay, not fixed `stop-first` or `tp-first`
- Progressive data fidelity:
  - Tier 1: tick/aggTrades replay (best)
  - Tier 2: second bars
  - Tier 3: 1m OHLC with O/C-aware path inference
- Exchange-like execution controls:
  - maker/taker fee split
  - post-only queue delay
  - slippage model for market/stop
  - reduce-only orders

## Folder Layout

- `hc_engine/types.py`: core dataclasses and enums
- `hc_engine/intrabar.py`: OHLC -> intrabar event path model
- `hc_engine/engine.py`: order book and position simulator
- `hc_engine/report.py`: basic performance metrics
- `hc_engine/binance_agg.py`: Binance futures aggTrades adapter + cache + event/bar conversion
- `examples/simple_demo.py`: runnable example
- `examples/binance_agg_demo.py`: live-data replay demo (aggTrades)

## Why This Is Closer To Live

The engine does not decide fills by using full bar high/low directly.
It replays a sequence of price events inside each bar and processes:

1. stop trigger
2. market/limit fill
3. reduce-only/position constraints
4. fee and slippage

in strict event order.

## Usage

```powershell
Set-Location d:\project\hc
python examples\simple_demo.py
```

AggTrades replay demo:

```powershell
Set-Location d:\project\hc
python examples\binance_agg_demo.py
```

Custom window example:

```powershell
Set-Location d:\project\hc
python examples\binance_agg_demo.py --symbol ETHUSDC --minutes 90 --bar-sec 60
```

This demo fetches recent Binance futures `aggTrades`, builds:

- `events_by_bar` from real trade timestamps/prices
- 1m bars from the same trade stream

and runs the engine with real intrabar event ordering.

## Integration With Future Strategies

Implement a strategy class with these hooks:

- `on_bar_open(engine, bar)`
- `on_price_event(engine, event)`
- `on_bar_close(engine, bar)`

The strategy can place/cancel orders through engine APIs.

## Data Fidelity Ladder

1. Use `events_by_bar` from real `aggTrades` (preferred)
2. Use second-level bars converted to events
3. Use 1m OHLC inferred path (`IntrabarPathModel`) as fallback

## GP3 Runner Notes

`run_gp3_hc_backtest.py` now supports:

- stop trigger source: `--stop-trigger-source last|mark`
- external exchange events: `--external-events-file <json/csv>`
- funding and fee tiers: `--funding-rate`, `--maker-fee-tiers`, `--taker-fee-tiers`
- partial fills:
  - `--partial-fill-enabled`
  - `--partial-fill-scope maker|all` (recommended: `maker`)
  - partial fills are only applied when event-level liquidity is known (`aggTrades`)
- resilient Binance kline fetch:
  - `--binance-fapi-base-urls`
  - `--binance-klines-max-conn-retries`
  - `--binance-klines-conn-backoff-sec`

Important live-like defaults in the current GP3 HC runner:

- entry signals are taken from the previous closed bar, not the current bar
- `BAR_SETTLE_DELAY_SEC` is respected for GP3 order submission timing
- GP3 feature construction follows the deployed `server_runtime` signal rules
- aggTrades replay keeps aggressor side (`is_buyer_maker`) so passive limit fills only happen on compatible tape direction
- when event-level liquidity is available, fills are always capped by traded quantity
- 1m historical mark-price klines are fetched and attached to HC events when available
- GTC limit orders can be modeled as taker when they become marketable against the tape
- initial margin rejection and maintenance-margin liquidation are now simulated in-engine

Recommended stable Binance command (no agg):

```powershell
Set-Location d:\project\hc
python run_gp3_hc_backtest.py --days 30 --no-agg --binance-klines-source hc --stop-trigger-source mark --partial-fill-enabled --partial-fill-scope maker --funding-rate 0.0001 --funding-interval-hours 8 --maker-fee 0 --taker-fee 0.0004
```

Recommended live-like command (use aggTrades replay):

```powershell
Set-Location d:\project\hc
python run_gp3_hc_backtest.py --days 30 --use-agg --stop-trigger-source mark --partial-fill-enabled --partial-fill-scope maker --funding-rate 0.0001 --funding-interval-hours 8 --maker-fee 0 --taker-fee 0.0004
```

Long-window use_agg verification can be run month-by-month with:

```powershell
Set-Location d:\project\hc
python run_gp3_useagg_monthly_batch.py --start-utc 2025-03-10T00:00:00Z --end-utc 2026-03-10T00:00:00Z
```

Upload notes for this repository:

- `run_gp3_hc_backtest.py --no-agg` is the stable low-dependency live-like fallback.
- `run_gp3_hc_backtest.py --use-agg` is the highest-fidelity live-like mode when Binance aggTrades are available.
- `cache/` and `output/` are runtime artifacts and are intentionally excluded from version control.

Live execution event import:

```powershell
Set-Location d:\project\hc
python build_external_events_from_live_trades.py --live-trades-file d:\path\to\live_trades.csv --db-path d:\path\to\dbrsi.db
```

This generates an `external-events.json` file that can be fed back into:

```powershell
python run_gp3_hc_backtest.py --external-events-file d:\project\hc\output\live_external_events.json
```
