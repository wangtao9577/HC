from __future__ import annotations

from typing import Protocol

from .types import Bar, PriceEvent


class StrategyProtocol(Protocol):
    def on_bar_open(self, engine: "BacktestEngine", bar: Bar) -> None:
        ...

    def on_price_event(self, engine: "BacktestEngine", event: PriceEvent) -> None:
        ...

    def on_bar_close(self, engine: "BacktestEngine", bar: Bar) -> None:
        ...
