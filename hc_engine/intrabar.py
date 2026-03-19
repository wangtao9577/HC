from __future__ import annotations

from dataclasses import dataclass

from .types import Bar, PriceEvent


@dataclass
class IntrabarPathModel:
    """
    Convert one OHLC bar into an ordered list of price events.

    Modes:
    - "oc_aware": infer path from O/C direction + wick shape
    - "long_worst": long-side pessimistic ordering
    - "short_worst": short-side pessimistic ordering
    """

    mode: str = "oc_aware"

    def events_from_bar(self, bar: Bar) -> list[PriceEvent]:
        o = float(bar.open)
        h = float(bar.high)
        l = float(bar.low)
        c = float(bar.close)
        t0 = int(bar.open_time_ms)
        t1 = int(bar.close_time_ms)
        dt = max(t1 - t0, 1)

        seq = self._price_sequence(o=o, h=h, l=l, c=c)
        # Spread events inside the bar, preserving order.
        offsets = [0.0, 0.25, 0.75, 0.999]
        out: list[PriceEvent] = []
        prev_price = None
        for i, px in enumerate(seq):
            if prev_price is not None and abs(px - prev_price) < 1e-12:
                continue
            off = offsets[min(i, len(offsets) - 1)]
            ts = t0 + int(dt * off)
            # Mark price proxy: linear O->C path, less sensitive to intrabar wick spikes.
            mark = o + (c - o) * float(off)
            out.append(
                PriceEvent(
                    ts_ms=ts,
                    price=float(px),
                    source=f"inferred:{self.mode}",
                    mark_price=float(mark),
                    trade_qty=0.0,
                )
            )
            prev_price = float(px)
        return out

    def _price_sequence(self, *, o: float, h: float, l: float, c: float) -> list[float]:
        if self.mode == "long_worst":
            return [o, l, h, c]
        if self.mode == "short_worst":
            return [o, h, l, c]

        # O/C-aware inference.
        if c >= o:
            # Bull bar.
            lower_wick = max(0.0, o - l)
            upper_wick = max(0.0, h - c)
            if lower_wick <= upper_wick:
                return [o, l, h, c]
            return [o, h, l, c]
        # Bear bar.
        upper_wick = max(0.0, h - o)
        lower_wick = max(0.0, c - l)
        if upper_wick <= lower_wick:
            return [o, h, l, c]
        return [o, l, h, c]
