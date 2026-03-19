from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"


class TimeInForce(str, Enum):
    GTC = "GTC"
    GTX = "GTX"  # post-only style


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    TRIGGERED = "TRIGGERED"


@dataclass
class Bar:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    mark_close: Optional[float] = None


@dataclass
class PriceEvent:
    ts_ms: int
    price: float
    source: str = "inferred"
    mark_price: Optional[float] = None
    trade_qty: float = 0.0
    is_buyer_maker: Optional[bool] = None


class StopTriggerSource(str, Enum):
    LAST = "last"
    MARK = "mark"


class ExternalEventType(str, Enum):
    FORCE_FLAT = "FORCE_FLAT"
    CANCEL_ORDER = "CANCEL_ORDER"
    CANCEL_ALL_REDUCE = "CANCEL_ALL_REDUCE"


@dataclass
class ExternalEvent:
    ts_ms: int
    event_type: ExternalEventType
    reason: str = ""
    order_id: str = ""
    price: Optional[float] = None
    qty: Optional[float] = None
    trigger_source: str = "external"


@dataclass
class Order:
    order_id: str
    side: Side
    order_type: OrderType
    qty: float
    created_ts_ms: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.NEW
    reason: str = ""
    trigger_ts_ms: int = 0
    filled_qty: float = 0.0


@dataclass
class Fill:
    ts_ms: int
    order_id: str
    side: Side
    qty: float
    price: float
    is_maker: bool
    fee: float
    reason: str = ""
    fee_rate: float = 0.0
    notional: float = 0.0
    trigger_source: str = "strategy"


@dataclass
class BacktestConfig:
    symbol: str = "ETHUSDC"
    initial_cash: float = 10000.0
    leverage: float = 1.0
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0004
    market_slippage_bps: float = 1.0
    stop_slippage_bps: float = 2.0
    tick_size: float = 0.01
    maker_queue_delay_ms: int = 1000
    gtc_queue_delay_ms: int = 300
    maker_buffer_ticks: int = 0
    allow_same_bar_entry_exit: bool = True
    stop_trigger_source: StopTriggerSource = StopTriggerSource.LAST
    allow_marketable_gtc_as_taker: bool = True

    partial_fill_enabled: bool = False
    partial_fill_ratio: float = 0.35
    min_partial_qty: float = 0.0
    # maker: only LIMIT/GTX orders, all: all order types
    partial_fill_scope: str = "maker"

    funding_interval_ms: int = 8 * 60 * 60 * 1000
    funding_rate: float = 0.0

    maker_fee_tiers: list[tuple[float, float]] = field(default_factory=list)
    taker_fee_tiers: list[tuple[float, float]] = field(default_factory=list)
    maintenance_margin_rate: float = 0.005
    maintenance_amount: float = 0.0
    liquidation_fee_rate: float = 0.0
    reject_on_insufficient_margin: bool = True
    liquidate_on_margin_breach: bool = True


@dataclass
class EngineState:
    cash: float
    realized_pnl: float = 0.0
    position_qty: float = 0.0  # >0 long, <0 short
    entry_price: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    fee_paid_total: float = 0.0
    fee_paid_maker: float = 0.0
    fee_paid_taker: float = 0.0
    funding_pnl: float = 0.0
    trade_notional_cum: float = 0.0
    margin_reject_count: int = 0
    liquidation_count: int = 0
    liquidation_fee_paid: float = 0.0
