from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable, Optional

from .intrabar import IntrabarPathModel
from .types import (
    BacktestConfig,
    Bar,
    EngineState,
    ExternalEvent,
    ExternalEventType,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    PriceEvent,
    Side,
    TimeInForce,
)


class BacktestEngine:
    def __init__(self, config: BacktestConfig, path_model: Optional[IntrabarPathModel] = None):
        self.config = config
        self.path_model = path_model or IntrabarPathModel(mode="oc_aware")
        self.state = EngineState(cash=float(config.initial_cash))

        self._order_seq = 0
        self._orders: dict[str, Order] = {}
        self._current_bar_open_ms: int = 0
        self._position_open_bar_ms: int = -1
        self._external_seq = 0
        self._next_funding_ts_ms: Optional[int] = None

    @property
    def orders(self) -> dict[str, Order]:
        return self._orders

    def get_position(self) -> tuple[float, float]:
        return float(self.state.position_qty), float(self.state.entry_price)

    def place_market(
        self,
        *,
        side: Side,
        qty: float,
        ts_ms: int,
        reduce_only: bool = False,
        reason: str = "",
    ) -> str:
        return self._register_order(
            side=side,
            order_type=OrderType.MARKET,
            qty=qty,
            ts_ms=ts_ms,
            reduce_only=reduce_only,
            reason=reason or "market",
        )

    def place_limit(
        self,
        *,
        side: Side,
        qty: float,
        limit_price: float,
        ts_ms: int,
        tif: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        reason: str = "",
    ) -> str:
        return self._register_order(
            side=side,
            order_type=OrderType.LIMIT,
            qty=qty,
            ts_ms=ts_ms,
            limit_price=float(limit_price),
            tif=tif,
            reduce_only=reduce_only,
            reason=reason or "limit",
        )

    def place_stop_market(
        self,
        *,
        side: Side,
        qty: float,
        stop_price: float,
        ts_ms: int,
        reduce_only: bool = False,
        reason: str = "",
    ) -> str:
        return self._register_order(
            side=side,
            order_type=OrderType.STOP_MARKET,
            qty=qty,
            ts_ms=ts_ms,
            stop_price=float(stop_price),
            reduce_only=reduce_only,
            reason=reason or "stop_market",
        )

    def cancel_order(self, order_id: str) -> bool:
        o = self._orders.get(str(order_id))
        if not o:
            return False
        if o.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            return False
        o.status = OrderStatus.CANCELED
        return True

    def cancel_all(self) -> None:
        for oid in list(self._orders.keys()):
            self.cancel_order(oid)

    def run(
        self,
        bars: Iterable[Bar],
        strategy,
        events_by_bar: Optional[dict[int, list[PriceEvent]]] = None,
        external_events_by_ts: Optional[dict[int, list[ExternalEvent]]] = None,
    ) -> dict:
        ext_items: list[tuple[int, ExternalEvent]] = []
        if external_events_by_ts:
            for ts_key, evs in external_events_by_ts.items():
                for ev in evs or []:
                    ext_items.append((int(getattr(ev, "ts_ms", ts_key)), ev))
            ext_items.sort(key=lambda x: int(x[0]))
        ext_idx = 0

        def drain_external(until_ts_ms: int, last_px: float) -> None:
            nonlocal ext_idx
            while ext_idx < len(ext_items) and int(ext_items[ext_idx][0]) <= int(until_ts_ms):
                _, ev = ext_items[ext_idx]
                self._apply_external_event(ev, ts_ms=int(until_ts_ms), last_price=float(last_px))
                ext_idx += 1

        for bar in bars:
            self._current_bar_open_ms = int(bar.open_time_ms)
            if self._next_funding_ts_ms is None:
                self._init_funding_schedule(int(bar.open_time_ms))
            strategy.on_bar_open(self, bar)
            drain_external(int(bar.open_time_ms), float(bar.open))

            if events_by_bar and int(bar.open_time_ms) in events_by_bar:
                events = sorted(list(events_by_bar[int(bar.open_time_ms)]), key=lambda x: int(x.ts_ms))
            else:
                events = self.path_model.events_from_bar(bar)

            for ev in events:
                self._apply_funding_until(int(ev.ts_ms), float(ev.mark_price if ev.mark_price else ev.price))
                strategy.on_price_event(self, ev)
                self._process_price_event(ev)
                drain_external(int(ev.ts_ms), float(ev.price))
                self._check_liquidation(int(ev.ts_ms), float(ev.mark_price if ev.mark_price else ev.price))
                self._mark_equity(int(ev.ts_ms), float(ev.price))

            strategy.on_bar_close(self, bar)
            close_ref = float(bar.mark_close) if getattr(bar, "mark_close", None) not in (None, 0, 0.0) else float(bar.close)
            self._apply_funding_until(int(bar.close_time_ms), close_ref)
            drain_external(int(bar.close_time_ms), float(bar.close))
            self._check_liquidation(int(bar.close_time_ms), close_ref)
            self._mark_equity(int(bar.close_time_ms), float(bar.close))

        return {
            "state": asdict(self.state),
            "fills": [asdict(f) for f in self.state.fills],
            "equity_curve": [{"ts_ms": t, "equity": e} for t, e in self.state.equity_curve],
            "open_orders": [asdict(o) for o in self._orders.values()],
        }

    def _register_order(
        self,
        *,
        side: Side,
        order_type: OrderType,
        qty: float,
        ts_ms: int,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        tif: TimeInForce = TimeInForce.GTC,
        reduce_only: bool = False,
        reason: str = "",
    ) -> str:
        q = max(0.0, float(qty))
        if q <= 0:
            raise ValueError("qty must be > 0")
        self._order_seq += 1
        oid = f"hc-{self._order_seq}"
        self._orders[oid] = Order(
            order_id=oid,
            side=Side(side),
            order_type=OrderType(order_type),
            qty=q,
            created_ts_ms=int(ts_ms),
            limit_price=limit_price,
            stop_price=stop_price,
            tif=TimeInForce(tif),
            reduce_only=bool(reduce_only),
            reason=str(reason or ""),
        )
        return oid

    @staticmethod
    def _remaining_qty(o: Order) -> float:
        return max(0.0, float(o.qty) - float(o.filled_qty))

    def _stop_trigger_price(self, ev: PriceEvent) -> float:
        src = str(getattr(self.config, "stop_trigger_source", "last"))
        if "." in src:
            src = src.split(".")[-1]
        src = src.lower().strip()
        if src == "mark":
            mp = float(ev.mark_price or 0.0)
            if mp > 0:
                return mp
        return float(ev.price)

    @staticmethod
    def _event_after_order_create(o: Order, ev_ts_ms: int) -> bool:
        # Orders created during the current event should only become matchable on later events.
        return int(ev_ts_ms) > int(o.created_ts_ms)

    def _process_price_event(self, ev: PriceEvent) -> None:
        stop_px = self._stop_trigger_price(ev)
        for o in list(self._orders.values()):
            if o.status not in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED) or o.order_type != OrderType.STOP_MARKET:
                continue
            if not self._event_after_order_create(o, int(ev.ts_ms)):
                continue
            if o.stop_price is None:
                o.status = OrderStatus.REJECTED
                continue
            if self._stop_triggered(o, stop_px):
                o.status = OrderStatus.TRIGGERED
                o.trigger_ts_ms = int(ev.ts_ms)

        remaining_liq = float(ev.trade_qty) if float(ev.trade_qty or 0.0) > 0 else None
        for o in list(self._orders.values()):
            if o.status not in (OrderStatus.NEW, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED):
                continue
            fill = self._build_fill_if_match(o, ev, remaining_liq=remaining_liq)
            if fill is None:
                continue

            applied = self._apply_fill(fill, reduce_only=bool(o.reduce_only))
            if not applied:
                if bool(o.reduce_only):
                    o.status = OrderStatus.REJECTED
                continue

            o.filled_qty = float(o.filled_qty) + float(fill.qty)
            if o.filled_qty + 1e-12 >= float(o.qty):
                o.status = OrderStatus.FILLED
            else:
                o.status = OrderStatus.PARTIALLY_FILLED

            if remaining_liq is not None:
                remaining_liq = max(0.0, float(remaining_liq) - float(fill.qty))
                if remaining_liq <= 1e-12:
                    remaining_liq = 0.0

    def _stop_triggered(self, o: Order, trigger_px: float) -> bool:
        if o.side == Side.BUY:
            return trigger_px >= float(o.stop_price or 0.0)
        return trigger_px <= float(o.stop_price or 0.0)

    def _compute_fill_qty(self, o: Order, ev: PriceEvent, remaining_liq: Optional[float]) -> float:
        rem = self._remaining_qty(o)
        if rem <= 0:
            return 0.0
        qty_cap = rem
        liq = remaining_liq
        if liq is not None:
            qty_cap = min(qty_cap, max(0.0, float(liq)))
        if not bool(self.config.partial_fill_enabled):
            return max(0.0, qty_cap)

        pf_scope = str(getattr(self.config, "partial_fill_scope", "maker") or "maker").lower().strip()
        # Partial fill is only meaningful when per-event liquidity is known (e.g., aggTrades replay).
        apply_partial = (
            bool(self.config.partial_fill_enabled)
            and remaining_liq is not None
            and (pf_scope == "all" or o.order_type == OrderType.LIMIT)
        )
        if apply_partial:
            min_part = max(0.0, float(self.config.min_partial_qty))
            part = max(min_part, rem * max(0.01, float(self.config.partial_fill_ratio)))
            qty_cap = min(qty_cap, part)
        return max(0.0, qty_cap)

    @staticmethod
    def _agg_passive_side_ok(o: Order, ev: PriceEvent) -> bool:
        if str(getattr(ev, "source", "")).lower() != "aggtrade":
            return True
        ibm = getattr(ev, "is_buyer_maker", None)
        if ibm is None:
            return True
        if o.side == Side.BUY:
            return bool(ibm) is True
        return bool(ibm) is False

    @staticmethod
    def _cap_taker_exec_price(limit_px: float, exec_px: float, side: Side) -> float:
        if side == Side.BUY:
            return min(float(limit_px), float(exec_px))
        return max(float(limit_px), float(exec_px))

    def _build_fill_if_match(self, o: Order, ev: PriceEvent, *, remaining_liq: Optional[float]) -> Optional[Fill]:
        px = float(ev.price)
        rem = self._remaining_qty(o)
        if rem <= 0:
            return None
        if not self._event_after_order_create(o, int(ev.ts_ms)):
            return None

        if o.order_type == OrderType.MARKET:
            exec_px = self._apply_slippage(px, o.side, self.config.market_slippage_bps)
            fill_qty = self._compute_fill_qty(o, ev, remaining_liq)
            if fill_qty <= 0:
                return None
            return self._make_fill(
                o,
                ev.ts_ms,
                exec_px,
                qty=fill_qty,
                is_maker=False,
                reason=o.reason or "market_fill",
            )

        if o.order_type == OrderType.STOP_MARKET and o.status == OrderStatus.TRIGGERED:
            exec_px = self._apply_slippage(px, o.side, self.config.stop_slippage_bps)
            fill_qty = self._compute_fill_qty(o, ev, remaining_liq)
            if fill_qty <= 0:
                return None
            return self._make_fill(
                o,
                ev.ts_ms,
                exec_px,
                qty=fill_qty,
                is_maker=False,
                reason=o.reason or "stop_fill",
            )

        if o.order_type != OrderType.LIMIT:
            return None
        if o.limit_price is None:
            return None

        limit_px = float(o.limit_price)
        matched = False
        if o.tif == TimeInForce.GTX:
            # Post-only approximation: wait queue delay and require buffer crossing.
            if int(ev.ts_ms) < int(o.created_ts_ms) + int(self.config.maker_queue_delay_ms):
                return None
            buf = float(self.config.maker_buffer_ticks) * float(self.config.tick_size)
            if o.side == Side.BUY and px <= (limit_px - buf):
                matched = True
            if o.side == Side.SELL and px >= (limit_px + buf):
                matched = True
            if not matched:
                return None
        else:
            crossed = False
            if o.side == Side.BUY and px <= limit_px:
                crossed = True
            if o.side == Side.SELL and px >= limit_px:
                crossed = True
            if not crossed:
                return None

            passive_ok = self._agg_passive_side_ok(o, ev)
            if bool(self.config.allow_marketable_gtc_as_taker) and not passive_ok:
                exec_px = self._cap_taker_exec_price(
                    limit_px,
                    self._apply_slippage(px, o.side, self.config.market_slippage_bps),
                    o.side,
                )
                fill_qty = self._compute_fill_qty(o, ev, remaining_liq)
                if fill_qty <= 0:
                    return None
                return self._make_fill(
                    o,
                    ev.ts_ms,
                    exec_px,
                    qty=fill_qty,
                    is_maker=False,
                    reason=o.reason or "limit_taker_fill",
                )

            if int(ev.ts_ms) < int(o.created_ts_ms) + int(self.config.gtc_queue_delay_ms):
                return None
            matched = True

        fill_qty = self._compute_fill_qty(o, ev, remaining_liq)
        if fill_qty <= 0:
            return None
        return self._make_fill(
            o,
            ev.ts_ms,
            limit_px,
            qty=fill_qty,
            is_maker=True,
            reason=o.reason or ("post_only_fill" if o.tif == TimeInForce.GTX else "limit_fill"),
        )

    @staticmethod
    def _normalize_tiers(tiers: list[tuple[float, float]]) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        for t in tiers or []:
            if not isinstance(t, (list, tuple)) or len(t) < 2:
                continue
            out.append((float(t[0]), float(t[1])))
        out.sort(key=lambda x: x[0])
        return out

    def _effective_fee_rate(self, *, is_maker: bool) -> float:
        base = float(self.config.maker_fee_rate if is_maker else self.config.taker_fee_rate)
        tiers = self._normalize_tiers(self.config.maker_fee_tiers if is_maker else self.config.taker_fee_tiers)
        if not tiers:
            return base
        vol = float(self.state.trade_notional_cum)
        rate = base
        for threshold, r in tiers:
            if vol >= float(threshold):
                rate = float(r)
            else:
                break
        return float(rate)

    @staticmethod
    def _infer_trigger_source(reason: str, order_type: OrderType) -> str:
        s = str(reason or "").lower()
        if "protection_invalid" in s:
            return "protection_invalid"
        if "external" in s:
            return "external"
        if "tp" in s:
            return "tp"
        if "sl" in s or order_type == OrderType.STOP_MARKET:
            return "sl"
        if "time_or_flip" in s:
            return "strategy_exit"
        if "entry" in s:
            return "entry"
        if "manual" in s:
            return "manual"
        if "system" in s:
            return "system"
        return "strategy"

    def _make_fill(
        self,
        o: Order,
        ts_ms: int,
        price: float,
        *,
        qty: float,
        is_maker: bool,
        reason: str,
        trigger_source: Optional[str] = None,
    ) -> Fill:
        q = max(0.0, float(qty))
        notional = abs(q * float(price))
        fee_rate = self._effective_fee_rate(is_maker=bool(is_maker))
        fee = notional * fee_rate
        return Fill(
            ts_ms=int(ts_ms),
            order_id=str(o.order_id),
            side=Side(o.side),
            qty=q,
            price=float(price),
            is_maker=bool(is_maker),
            fee=float(fee),
            reason=str(reason),
            fee_rate=float(fee_rate),
            notional=float(notional),
            trigger_source=str(trigger_source or self._infer_trigger_source(reason, o.order_type)),
        )

    def _apply_slippage(self, px: float, side: Side, bps: float) -> float:
        r = float(max(0.0, bps) / 10_000.0)
        if side == Side.BUY:
            return px * (1.0 + r)
        return px * (1.0 - r)

    def _equity_at_price(self, ref_price: float) -> float:
        qty = float(self.state.position_qty)
        entry = float(self.state.entry_price)
        unreal = qty * (float(ref_price) - entry) if qty != 0.0 and entry > 0.0 else 0.0
        return float(self.state.cash + unreal)

    def _initial_margin_requirement(self, qty: float, ref_price: float) -> float:
        lev = max(float(getattr(self.config, "leverage", 1.0) or 1.0), 1.0)
        return abs(float(qty)) * max(float(ref_price), 0.0) / lev

    def _maintenance_requirement(self, qty: float, ref_price: float) -> float:
        mmr = max(float(getattr(self.config, "maintenance_margin_rate", 0.0) or 0.0), 0.0)
        mma = max(float(getattr(self.config, "maintenance_amount", 0.0) or 0.0), 0.0)
        if abs(float(qty)) <= 1e-12 or float(ref_price) <= 0.0:
            return 0.0
        return abs(float(qty)) * float(ref_price) * mmr + mma

    def _apply_fill(self, f: Fill, *, reduce_only: bool) -> bool:
        signed = float(f.qty) if f.side == Side.BUY else -float(f.qty)
        cur = float(self.state.position_qty)

        # Optional lock to avoid same-bar entry+exit if desired.
        if (not self.config.allow_same_bar_entry_exit) and reduce_only and self._position_open_bar_ms == self._current_bar_open_ms:
            return False

        # Reduce-only guard.
        if reduce_only:
            if cur == 0.0:
                return False
            if (cur > 0 and signed > 0) or (cur < 0 and signed < 0):
                return False
            signed = math.copysign(min(abs(signed), abs(cur)), signed)

        realized = 0.0
        new_qty = cur
        new_entry = float(self.state.entry_price)

        if cur == 0.0:
            new_qty = signed
            new_entry = float(f.price)
            self._position_open_bar_ms = int(self._current_bar_open_ms)
        elif (cur > 0 and signed > 0) or (cur < 0 and signed < 0):
            # Add to existing side.
            tot = abs(cur) + abs(signed)
            if tot > 0:
                new_entry = (abs(cur) * new_entry + abs(signed) * float(f.price)) / tot
            new_qty = cur + signed
        else:
            # Close or reverse.
            close_qty = min(abs(cur), abs(signed))
            direction = 1.0 if cur > 0 else -1.0
            realized = close_qty * (float(f.price) - new_entry) * direction
            remainder = abs(signed) - close_qty
            if remainder <= 0:
                new_qty = cur + signed
                if abs(new_qty) < 1e-12:
                    new_qty = 0.0
                    new_entry = 0.0
            else:
                new_qty = math.copysign(remainder, signed)
                new_entry = float(f.price)
                self._position_open_bar_ms = int(self._current_bar_open_ms)

        new_cash = float(self.state.cash + float(realized) - float(f.fee))
        opens_or_adds_risk = abs(float(new_qty)) > abs(float(cur)) + 1e-12
        if bool(getattr(self.config, "reject_on_insufficient_margin", True)) and opens_or_adds_risk:
            unreal_after = (
                float(new_qty) * (float(f.price) - float(new_entry))
                if abs(float(new_qty)) > 1e-12 and float(new_entry) > 0.0
                else 0.0
            )
            equity_after = float(new_cash + unreal_after)
            req_margin = self._initial_margin_requirement(float(new_qty), float(f.price))
            if req_margin > equity_after + 1e-12:
                self.state.margin_reject_count += 1
                return False

        self.state.position_qty = float(new_qty)
        self.state.entry_price = float(new_entry)
        self.state.realized_pnl += float(realized)
        self.state.cash = float(new_cash)
        self.state.trade_notional_cum += float(f.notional)
        self.state.fee_paid_total += float(f.fee)
        if bool(f.is_maker):
            self.state.fee_paid_maker += float(f.fee)
        else:
            self.state.fee_paid_taker += float(f.fee)
        self.state.fills.append(f)
        return True

    def _mark_equity(self, ts_ms: int, last_price: float) -> None:
        qty = float(self.state.position_qty)
        entry = float(self.state.entry_price)
        unreal = qty * (float(last_price) - entry) if qty != 0.0 and entry > 0.0 else 0.0
        eq = float(self.state.cash + unreal)
        self.state.equity_curve.append((int(ts_ms), eq))

    def _check_liquidation(self, ts_ms: int, ref_price: float) -> None:
        if not bool(getattr(self.config, "liquidate_on_margin_breach", True)):
            return
        qty = float(self.state.position_qty)
        if abs(qty) <= 1e-12 or float(ref_price) <= 0.0:
            return
        equity = self._equity_at_price(float(ref_price))
        maint = self._maintenance_requirement(float(qty), float(ref_price))
        if equity > maint + 1e-12:
            return

        for oid in list(self._orders.keys()):
            self.cancel_order(oid)
        side = Side.SELL if qty > 0 else Side.BUY
        exec_px = self._apply_slippage(float(ref_price), side, self.config.market_slippage_bps)
        fee_rate = self._effective_fee_rate(is_maker=False)
        liq_fee_rate = max(float(getattr(self.config, "liquidation_fee_rate", 0.0) or 0.0), 0.0)
        notional = abs(float(qty) * exec_px)
        fill = Fill(
            ts_ms=int(ts_ms),
            order_id=self._next_external_order_id(),
            side=side,
            qty=abs(float(qty)),
            price=float(exec_px),
            is_maker=False,
            fee=float(notional * (fee_rate + liq_fee_rate)),
            reason="liquidation",
            fee_rate=float(fee_rate + liq_fee_rate),
            notional=float(notional),
            trigger_source="liquidation",
        )
        applied = self._apply_fill(fill, reduce_only=True)
        if applied:
            self.state.liquidation_count += 1
            self.state.liquidation_fee_paid += float(notional * liq_fee_rate)

    def _init_funding_schedule(self, first_ts_ms: int) -> None:
        interval = int(max(0, int(self.config.funding_interval_ms)))
        if interval <= 0:
            self._next_funding_ts_ms = None
            return
        self._next_funding_ts_ms = int(((int(first_ts_ms) // interval) + 1) * interval)

    def _apply_funding_until(self, ts_ms: int, ref_price: float) -> None:
        interval = int(max(0, int(self.config.funding_interval_ms)))
        if interval <= 0:
            return
        if self._next_funding_ts_ms is None:
            self._init_funding_schedule(int(ts_ms))
        if self._next_funding_ts_ms is None:
            return
        rate = float(self.config.funding_rate)
        if abs(rate) <= 0:
            while int(self._next_funding_ts_ms) <= int(ts_ms):
                self._next_funding_ts_ms = int(self._next_funding_ts_ms) + interval
            return

        while int(self._next_funding_ts_ms) <= int(ts_ms):
            qty = float(self.state.position_qty)
            px = float(ref_price)
            if qty != 0.0 and px > 0.0:
                notional = abs(qty) * px
                # Positive funding_rate: long pays, short receives.
                flow = (-notional * rate) if qty > 0 else (notional * rate)
                self.state.cash += float(flow)
                self.state.funding_pnl += float(flow)
            self._next_funding_ts_ms = int(self._next_funding_ts_ms) + interval

    def _next_external_order_id(self) -> str:
        self._external_seq += 1
        return f"hc-ext-{self._external_seq}"

    def _apply_external_event(self, ev: ExternalEvent, *, ts_ms: int, last_price: float) -> None:
        typ = ev.event_type
        if not isinstance(typ, ExternalEventType):
            typ = ExternalEventType(str(typ))

        if typ == ExternalEventType.CANCEL_ORDER:
            if ev.order_id:
                self.cancel_order(str(ev.order_id))
            return

        if typ == ExternalEventType.CANCEL_ALL_REDUCE:
            for o in self._orders.values():
                if bool(o.reduce_only) and o.status in (OrderStatus.NEW, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED):
                    o.status = OrderStatus.CANCELED
            return

        if typ != ExternalEventType.FORCE_FLAT:
            return

        pos_qty, _ = self.get_position()
        if abs(float(pos_qty)) <= 1e-12:
            return
        close_qty = abs(float(pos_qty))
        if ev.qty is not None:
            close_qty = min(close_qty, abs(float(ev.qty)))
        if close_qty <= 0:
            return

        side = Side.SELL if float(pos_qty) > 0 else Side.BUY
        px = float(ev.price) if (ev.price is not None and float(ev.price) > 0) else float(last_price)
        exec_px = self._apply_slippage(px, side, self.config.market_slippage_bps)
        fee_rate = self._effective_fee_rate(is_maker=False)
        notional = abs(close_qty * exec_px)
        fill = Fill(
            ts_ms=int(ts_ms),
            order_id=self._next_external_order_id(),
            side=side,
            qty=float(close_qty),
            price=float(exec_px),
            is_maker=False,
            fee=float(notional * fee_rate),
            reason=str(ev.reason or "external_force_flatten"),
            fee_rate=float(fee_rate),
            notional=float(notional),
            trigger_source=str(ev.trigger_source or "external"),
        )
        self._apply_fill(fill, reduce_only=True)
