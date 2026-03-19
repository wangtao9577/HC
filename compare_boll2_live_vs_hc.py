from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare BOLL2.1 live JSONL execution logs against backtest events/trades and HC outputs."
    )
    ap.add_argument(
        "--live-jsonl",
        action="append",
        required=True,
        help="Live JSONL file path or glob pattern. Repeatable.",
    )
    ap.add_argument("--backtest-events-file", default="", help="Optional backtest signal events CSV.")
    ap.add_argument("--backtest-trades-file", default="", help="Optional backtest trades CSV.")
    ap.add_argument("--hc-roundtrips-file", default="", help="Optional HC roundtrips CSV.")
    ap.add_argument("--hc-fills-file", default="", help="Optional HC fills CSV.")
    ap.add_argument("--symbol", default="ETHUSDC")
    ap.add_argument("--account", default="")
    ap.add_argument("--signal-match-window-sec", type=int, default=6 * 3600)
    ap.add_argument("--trade-match-window-sec", type=int, default=12 * 3600)
    ap.add_argument("--fill-match-window-sec", type=int, default=12 * 3600)
    ap.add_argument("--bj-tz", default="Asia/Shanghai")
    ap.add_argument("--out-dir", default="d:/project/hc/output/live_compare")
    return ap.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        if value == "":
            return None
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def side_to_ls(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in {"long", "buy", "b", "l"}:
        return "LONG"
    if s in {"short", "sell", "s"}:
        return "SHORT"
    return ""


def parse_ts_ms(raw: Any, *, assumed_tz: str = "UTC") -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.isdigit():
        v = int(s)
        if v > 10_000_000_000:
            return v
        return v * 1000
    try:
        dt = pd.to_datetime(s)
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.tz_localize(assumed_tz)
        dt = dt.tz_convert("UTC")
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def iso_utc(ts_ms: Any) -> str:
    v = to_int(ts_ms)
    if not v:
        return ""
    return pd.to_datetime(int(v), unit="ms", utc=True).isoformat()


def expand_live_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for raw in patterns:
        raw = str(raw or "").strip()
        if not raw:
            continue
        p = Path(raw)
        if any(ch in raw for ch in "*?[]"):
            base = p.parent if str(p.parent) not in {"", "."} else Path(".")
            for hit in sorted(base.glob(p.name)):
                key = str(hit.resolve())
                if key not in seen:
                    out.append(hit)
                    seen.add(key)
        elif p.exists():
            key = str(p.resolve())
            if key not in seen:
                out.append(p)
                seen.add(key)
    if not out:
        raise RuntimeError("no live JSONL files matched")
    return out


def load_live_records(paths: list[Path], *, symbol: str, account: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    symbol = str(symbol or "").upper()
    account = str(account or "").strip()
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except Exception as exc:
                    raise RuntimeError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
                if symbol and str(rec.get("symbol") or "").upper() != symbol:
                    continue
                if account and str(rec.get("account") or "").strip() != account:
                    continue
                rec["_source_file"] = str(path)
                rec["_line_no"] = int(line_no)
                rec["_seq"] = len(rows)
                rec["ts_ms"] = to_int(rec.get("ts_ms"))
                rows.append(rec)
    rows.sort(key=lambda x: (int(x.get("ts_ms") or 0), int(x.get("_seq") or 0)))
    if not rows:
        raise RuntimeError("no live JSONL rows found after symbol/account filtering")
    return rows


def signal_key_from_event(event: dict[str, Any]) -> tuple[Any, ...]:
    signal = side_to_ls(event.get("signal") or event.get("side"))
    basis = to_float(event.get("basis"))
    return (
        str(event.get("account") or ""),
        str(event.get("symbol") or "").upper(),
        signal,
        to_int(event.get("signal_ready_ts_ms")),
        to_int(event.get("bar_open_time_ms")),
        round(float(basis), 6) if basis is not None else None,
    )


def signal_row_from_event(event: dict[str, Any], *, synthetic_from: str = "") -> dict[str, Any]:
    return {
        "live_signal_id": None,
        "account": str(event.get("account") or ""),
        "symbol": str(event.get("symbol") or "").upper(),
        "session_id": str(event.get("session_id") or ""),
        "signal_side": side_to_ls(event.get("signal") or event.get("side")),
        "signal_basis": to_float(event.get("basis")),
        "signal_reason": str(event.get("signal_reason") or ""),
        "entry_signal_ts_ms": to_int(event.get("ts_ms")),
        "entry_signal_ts_utc": iso_utc(event.get("ts_ms")),
        "signal_bar_open_time_ms": to_int(event.get("bar_open_time_ms")),
        "signal_bar_open_time_utc": iso_utc(event.get("bar_open_time_ms")),
        "signal_ready_ts_ms": to_int(event.get("signal_ready_ts_ms")),
        "signal_ready_ts_utc": iso_utc(event.get("signal_ready_ts_ms")),
        "signal_px1m_open": to_float(event.get("px1m_open")),
        "signal_limit_price": to_float(event.get("price")),
        "signal_qty": to_float(event.get("qty")),
        "signal_order_notional": to_float(event.get("order_notional")),
        "entry_posted_ts_ms": None,
        "entry_posted_ts_utc": "",
        "entry_order_id": "",
        "entry_client_order_id": "",
        "entry_posted_price": None,
        "entry_posted_qty": None,
        "entry_post_status": "",
        "entry_post_error": "",
        "entry_filled_ts_ms": None,
        "entry_filled_ts_utc": "",
        "entry_fill_price": None,
        "entry_fill_qty": None,
        "entry_stop_price": None,
        "entry_timeout_ts_ms": None,
        "entry_timeout_ts_utc": "",
        "entry_pending_status": "",
        "synthetic_from": synthetic_from,
        "source_file": str(event.get("_source_file") or ""),
        "source_line_no": to_int(event.get("_line_no")),
    }


def choose_unmatched_signal(
    signals: list[dict[str, Any]],
    signal_queues: dict[tuple[Any, ...], deque[int]],
    event: dict[str, Any],
) -> Optional[int]:
    key = signal_key_from_event(event)
    q = signal_queues.get(key)
    if q:
        while q:
            idx = q[0]
            row = signals[idx]
            if row.get("entry_posted_ts_ms") is None and row.get("entry_post_status") != "POST_FAILED":
                return idx
            q.popleft()
    target_side = side_to_ls(event.get("signal") or event.get("side"))
    target_ts = to_int(event.get("ts_ms"))
    target_acct = str(event.get("account") or "")
    target_symbol = str(event.get("symbol") or "").upper()
    for idx in range(len(signals) - 1, -1, -1):
        row = signals[idx]
        if row.get("account") != target_acct or row.get("symbol") != target_symbol:
            continue
        if row.get("signal_side") != target_side:
            continue
        if row.get("entry_posted_ts_ms") is not None or row.get("entry_post_status") == "POST_FAILED":
            continue
        signal_ready_ts_ms = to_int(row.get("signal_ready_ts_ms"))
        if signal_ready_ts_ms is not None and target_ts is not None:
            if abs(int(target_ts) - int(signal_ready_ts_ms)) <= 120_000:
                return idx
    return None


def build_live_signals(records: list[dict[str, Any]]) -> pd.DataFrame:
    signals: list[dict[str, Any]] = []
    signal_queues: dict[tuple[Any, ...], deque[int]] = defaultdict(deque)
    by_order_id: dict[str, int] = {}
    by_client_order_id: dict[str, int] = {}

    for event in records:
        et = str(event.get("event_type") or "")
        if et == "entry_signal":
            row = signal_row_from_event(event)
            row["live_signal_id"] = len(signals) + 1
            idx = len(signals)
            signals.append(row)
            signal_queues[signal_key_from_event(event)].append(idx)
            continue

        if et not in {"entry_posted", "entry_post_failed", "entry_filled", "pending_timeout", "pending_finished"}:
            continue

        idx: Optional[int] = None
        order_id = str(event.get("order_id") or "")
        client_order_id = str(event.get("client_order_id") or "")
        pending_kind = str(event.get("kind") or "").lower()
        if client_order_id and client_order_id in by_client_order_id:
            idx = by_client_order_id[client_order_id]
        elif order_id and order_id in by_order_id:
            idx = by_order_id[order_id]
        elif et in {"entry_posted", "entry_post_failed"}:
            idx = choose_unmatched_signal(signals, signal_queues, event)

        if idx is None:
            if et in {"entry_posted", "entry_post_failed", "entry_filled"} or (et in {"pending_timeout", "pending_finished"} and pending_kind == "entry"):
                row = signal_row_from_event(event, synthetic_from=et)
                row["live_signal_id"] = len(signals) + 1
                idx = len(signals)
                signals.append(row)
            else:
                continue

        row = signals[idx]
        if et == "entry_posted":
            row["entry_posted_ts_ms"] = to_int(event.get("ts_ms"))
            row["entry_posted_ts_utc"] = iso_utc(event.get("ts_ms"))
            row["entry_order_id"] = order_id
            row["entry_client_order_id"] = client_order_id
            row["entry_posted_price"] = to_float(event.get("price"))
            row["entry_posted_qty"] = to_float(event.get("qty"))
            row["entry_post_status"] = "POSTED"
            if order_id:
                by_order_id[order_id] = idx
            if client_order_id:
                by_client_order_id[client_order_id] = idx
        elif et == "entry_post_failed":
            row["entry_posted_ts_ms"] = to_int(event.get("ts_ms"))
            row["entry_posted_ts_utc"] = iso_utc(event.get("ts_ms"))
            row["entry_post_status"] = "POST_FAILED"
            row["entry_post_error"] = str(event.get("error") or "")
        elif et == "entry_filled":
            row["entry_filled_ts_ms"] = to_int(event.get("ts_ms"))
            row["entry_filled_ts_utc"] = iso_utc(event.get("ts_ms"))
            row["entry_order_id"] = order_id or str(row.get("entry_order_id") or "")
            row["entry_client_order_id"] = client_order_id or str(row.get("entry_client_order_id") or "")
            row["entry_fill_price"] = to_float(event.get("avg_price"))
            row["entry_fill_qty"] = to_float(event.get("exec_qty"))
            row["entry_stop_price"] = to_float(event.get("stop_price"))
            row["entry_post_status"] = row.get("entry_post_status") or "POSTED"
            if row["entry_order_id"]:
                by_order_id[str(row["entry_order_id"])] = idx
            if row["entry_client_order_id"]:
                by_client_order_id[str(row["entry_client_order_id"])] = idx
        elif et == "pending_timeout" and pending_kind == "entry":
            row["entry_timeout_ts_ms"] = to_int(event.get("ts_ms"))
            row["entry_timeout_ts_utc"] = iso_utc(event.get("ts_ms"))
            row["entry_pending_status"] = "TIMEOUT"
        elif et == "pending_finished" and pending_kind == "entry":
            row["entry_pending_status"] = str(event.get("status") or "")

    for row in signals:
        if row.get("entry_filled_ts_ms"):
            row["signal_outcome"] = "FILLED"
        elif row.get("entry_post_status") == "POST_FAILED":
            row["signal_outcome"] = "POST_FAILED"
        elif row.get("entry_timeout_ts_ms"):
            row["signal_outcome"] = "TIMEOUT"
        elif row.get("entry_posted_ts_ms"):
            row["signal_outcome"] = "POSTED_OPEN"
        else:
            row["signal_outcome"] = "SIGNAL_ONLY"

    df = pd.DataFrame(signals)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "live_signal_id",
                "account",
                "symbol",
                "session_id",
                "signal_side",
                "signal_basis",
                "signal_ready_ts_ms",
                "entry_filled_ts_ms",
                "signal_outcome",
            ]
        )
    return df.sort_values(["signal_ready_ts_ms", "entry_signal_ts_ms", "live_signal_id"], na_position="last").reset_index(drop=True)


def infer_live_exit_category(trade: dict[str, Any]) -> str:
    source_event = str(trade.get("exit_source_event") or "").strip().lower()
    exit_event_type = str(trade.get("exit_event_type") or "").strip().lower()
    if source_event == "boll_exit_signal":
        return "boll_exit_signal"
    if exit_event_type == "position_reset_exchange_flat":
        if bool(trade.get("breakeven_triggered")):
            return "boll_stop_be"
        return "boll_stop"
    return source_event or "unknown"


def new_trade_from_signal(
    event: dict[str, Any],
    *,
    signal_row: Optional[dict[str, Any]] = None,
    synthetic_from: str = "",
) -> dict[str, Any]:
    base_signal = signal_row or {}
    side = side_to_ls(event.get("signal") or event.get("side") or base_signal.get("signal_side"))
    return {
        "live_trade_id": None,
        "account": str(event.get("account") or base_signal.get("account") or ""),
        "symbol": str(event.get("symbol") or base_signal.get("symbol") or "").upper(),
        "session_id": str(event.get("session_id") or base_signal.get("session_id") or ""),
        "is_takeover": False,
        "synthetic_from": synthetic_from,
        "side": side,
        "signal_basis": to_float(base_signal.get("signal_basis") or event.get("basis")),
        "signal_reason": str(base_signal.get("signal_reason") or event.get("signal_reason") or ""),
        "signal_bar_open_time_ms": to_int(base_signal.get("signal_bar_open_time_ms") or event.get("bar_open_time_ms")),
        "signal_bar_open_time_utc": iso_utc(base_signal.get("signal_bar_open_time_ms") or event.get("bar_open_time_ms")),
        "signal_ready_ts_ms": to_int(base_signal.get("signal_ready_ts_ms") or event.get("signal_ready_ts_ms")),
        "signal_ready_ts_utc": iso_utc(base_signal.get("signal_ready_ts_ms") or event.get("signal_ready_ts_ms")),
        "signal_ts_ms": to_int(base_signal.get("entry_signal_ts_ms") or event.get("ts_ms")),
        "signal_ts_utc": iso_utc(base_signal.get("entry_signal_ts_ms") or event.get("ts_ms")),
        "signal_px1m_open": to_float(base_signal.get("signal_px1m_open") or event.get("px1m_open")),
        "entry_posted_ts_ms": to_int(base_signal.get("entry_posted_ts_ms")),
        "entry_posted_ts_utc": iso_utc(base_signal.get("entry_posted_ts_ms")),
        "entry_limit_price": to_float(base_signal.get("entry_posted_price") or base_signal.get("signal_limit_price") or event.get("submitted_price")),
        "entry_posted_qty": to_float(base_signal.get("entry_posted_qty") or base_signal.get("signal_qty") or event.get("submitted_qty")),
        "entry_order_id": str(event.get("order_id") or base_signal.get("entry_order_id") or ""),
        "entry_client_order_id": str(event.get("client_order_id") or base_signal.get("entry_client_order_id") or ""),
        "entry_filled_ts_ms": to_int(event.get("ts_ms")),
        "entry_filled_ts_utc": iso_utc(event.get("ts_ms")),
        "entry_fill_price": to_float(event.get("avg_price")),
        "entry_fill_qty": to_float(event.get("exec_qty")),
        "initial_stop_price": to_float(event.get("stop_price")),
        "stop_posted_ts_ms": None,
        "stop_posted_ts_utc": "",
        "stop_order_id_initial": "",
        "stop_order_id_current": "",
        "stop_price_current": None,
        "breakeven_triggered": False,
        "breakeven_ts_ms": None,
        "breakeven_ts_utc": "",
        "breakeven_new_stop_price": None,
        "exit_posted_ts_ms": None,
        "exit_posted_ts_utc": "",
        "exit_order_id": "",
        "exit_client_order_id": "",
        "exit_source_event": "",
        "exit_limit_price": None,
        "exit_posted_qty": None,
        "exit_filled_ts_ms": None,
        "exit_filled_ts_utc": "",
        "exit_fill_price": None,
        "exit_fill_qty": None,
        "exit_event_type": "",
        "exit_category": "",
        "close_source": "",
        "close_position_entry_price": None,
        "close_position_qty": None,
    }


def build_live_trades(records: list[dict[str, Any]], live_signals: pd.DataFrame) -> pd.DataFrame:
    signal_rows = {str(r.get("entry_order_id") or ""): dict(r) for r in live_signals.to_dict("records") if str(r.get("entry_order_id") or "")}
    signal_rows_by_client = {
        str(r.get("entry_client_order_id") or ""): dict(r)
        for r in live_signals.to_dict("records")
        if str(r.get("entry_client_order_id") or "")
    }
    current_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []

    for event in records:
        et = str(event.get("event_type") or "")
        pair = (str(event.get("account") or ""), str(event.get("symbol") or "").upper())
        current = current_by_pair.get(pair)

        if et == "entry_filled":
            signal_row = None
            order_id = str(event.get("order_id") or "")
            client_order_id = str(event.get("client_order_id") or "")
            if client_order_id and client_order_id in signal_rows_by_client:
                signal_row = signal_rows_by_client[client_order_id]
            elif order_id and order_id in signal_rows:
                signal_row = signal_rows[order_id]
            trade = new_trade_from_signal(event, signal_row=signal_row, synthetic_from="" if signal_row else "entry_filled")
            trade["live_trade_id"] = len(trades) + 1
            current_by_pair[pair] = trade
            trades.append(trade)
            continue

        if et == "position_takeover":
            trade = {
                "live_trade_id": len(trades) + 1,
                "account": str(event.get("account") or ""),
                "symbol": str(event.get("symbol") or "").upper(),
                "session_id": str(event.get("session_id") or ""),
                "is_takeover": True,
                "synthetic_from": "position_takeover",
                "side": side_to_ls(event.get("side")),
                "signal_basis": to_float(event.get("basis")),
                "signal_reason": "",
                "signal_bar_open_time_ms": None,
                "signal_bar_open_time_utc": "",
                "signal_ready_ts_ms": None,
                "signal_ready_ts_utc": "",
                "signal_ts_ms": None,
                "signal_ts_utc": "",
                "signal_px1m_open": None,
                "entry_posted_ts_ms": None,
                "entry_posted_ts_utc": "",
                "entry_limit_price": None,
                "entry_posted_qty": None,
                "entry_order_id": str(event.get("order_id") or ""),
                "entry_client_order_id": str(event.get("client_order_id") or ""),
                "entry_filled_ts_ms": to_int(event.get("ts_ms")),
                "entry_filled_ts_utc": iso_utc(event.get("ts_ms")),
                "entry_fill_price": to_float(event.get("entry_price")),
                "entry_fill_qty": to_float(event.get("qty")),
                "initial_stop_price": to_float(event.get("stop_price")),
                "stop_posted_ts_ms": None,
                "stop_posted_ts_utc": "",
                "stop_order_id_initial": str(event.get("stop_order_id") or ""),
                "stop_order_id_current": str(event.get("stop_order_id") or ""),
                "stop_price_current": to_float(event.get("stop_price")),
                "breakeven_triggered": False,
                "breakeven_ts_ms": None,
                "breakeven_ts_utc": "",
                "breakeven_new_stop_price": None,
                "exit_posted_ts_ms": None,
                "exit_posted_ts_utc": "",
                "exit_order_id": "",
                "exit_client_order_id": "",
                "exit_source_event": "",
                "exit_limit_price": None,
                "exit_posted_qty": None,
                "exit_filled_ts_ms": None,
                "exit_filled_ts_utc": "",
                "exit_fill_price": None,
                "exit_fill_qty": None,
                "exit_event_type": "",
                "exit_category": "",
                "close_source": "",
                "close_position_entry_price": None,
                "close_position_qty": None,
            }
            current_by_pair[pair] = trade
            trades.append(trade)
            continue

        if current is None:
            continue

        if et == "stop_posted":
            stop_ts_ms = to_int(event.get("ts_ms"))
            if current.get("stop_posted_ts_ms") is None:
                current["stop_posted_ts_ms"] = stop_ts_ms
                current["stop_posted_ts_utc"] = iso_utc(stop_ts_ms)
                current["stop_order_id_initial"] = str(event.get("order_id") or "")
            current["stop_order_id_current"] = str(event.get("order_id") or current.get("stop_order_id_current") or "")
            current["stop_price_current"] = to_float(event.get("stop_price"))
            continue

        if et == "breakeven_activated":
            current["breakeven_triggered"] = True
            current["breakeven_ts_ms"] = to_int(event.get("ts_ms"))
            current["breakeven_ts_utc"] = iso_utc(event.get("ts_ms"))
            current["breakeven_new_stop_price"] = to_float(event.get("new_stop_price"))
            if current["breakeven_new_stop_price"] is not None:
                current["stop_price_current"] = current["breakeven_new_stop_price"]
            continue

        if et == "exit_posted":
            current["exit_posted_ts_ms"] = to_int(event.get("ts_ms"))
            current["exit_posted_ts_utc"] = iso_utc(event.get("ts_ms"))
            current["exit_order_id"] = str(event.get("order_id") or "")
            current["exit_client_order_id"] = str(event.get("client_order_id") or "")
            current["exit_source_event"] = str(event.get("source_event") or "")
            current["exit_limit_price"] = to_float(event.get("price"))
            current["exit_posted_qty"] = to_float(event.get("qty"))
            continue

        if et == "exit_filled":
            current["exit_filled_ts_ms"] = to_int(event.get("ts_ms"))
            current["exit_filled_ts_utc"] = iso_utc(event.get("ts_ms"))
            current["exit_fill_price"] = to_float(event.get("avg_price"))
            current["exit_fill_qty"] = to_float(event.get("exec_qty"))
            current["exit_event_type"] = "exit_filled"
            current["close_source"] = "strategy_exit_fill"
            closed_position = event.get("closed_position") or {}
            current["close_position_entry_price"] = to_float(closed_position.get("entry_price"))
            current["close_position_qty"] = to_float(closed_position.get("qty"))
            current["exit_category"] = infer_live_exit_category(current)
            del current_by_pair[pair]
            continue

        if et == "position_reset_exchange_flat":
            old_position = event.get("old_position") or event.get("position") or {}
            current["exit_filled_ts_ms"] = to_int(event.get("ts_ms"))
            current["exit_filled_ts_utc"] = iso_utc(event.get("ts_ms"))
            current["exit_fill_price"] = None
            current["exit_fill_qty"] = to_float(old_position.get("qty")) or current.get("entry_fill_qty")
            current["exit_event_type"] = "position_reset_exchange_flat"
            current["close_source"] = "exchange_flat_sync"
            current["close_position_entry_price"] = to_float(old_position.get("entry_price"))
            current["close_position_qty"] = to_float(old_position.get("qty"))
            current["exit_category"] = infer_live_exit_category(current)
            del current_by_pair[pair]
            continue

    for trade in trades:
        entry_ts_ms = to_int(trade.get("entry_filled_ts_ms"))
        exit_ts_ms = to_int(trade.get("exit_filled_ts_ms"))
        trade["entry_filled_ts_utc"] = iso_utc(entry_ts_ms)
        trade["exit_filled_ts_utc"] = iso_utc(exit_ts_ms)
        trade["duration_sec"] = (int(exit_ts_ms) - int(entry_ts_ms)) / 1000.0 if entry_ts_ms and exit_ts_ms else None
        trade["is_open"] = not bool(exit_ts_ms)
        if trade.get("exit_category") == "":
            trade["exit_category"] = infer_live_exit_category(trade)
        if trade.get("entry_fill_price") is not None and trade.get("exit_fill_price") is not None and trade.get("entry_fill_qty") is not None:
            qty = float(trade["entry_fill_qty"])
            if trade.get("side") == "LONG":
                trade["gross_pnl_quote"] = (float(trade["exit_fill_price"]) - float(trade["entry_fill_price"])) * qty
            elif trade.get("side") == "SHORT":
                trade["gross_pnl_quote"] = (float(trade["entry_fill_price"]) - float(trade["exit_fill_price"])) * qty
            else:
                trade["gross_pnl_quote"] = None
        else:
            trade["gross_pnl_quote"] = None
    df = pd.DataFrame(trades)
    if df.empty:
        return pd.DataFrame(columns=["live_trade_id", "account", "symbol", "side", "entry_filled_ts_ms", "exit_filled_ts_ms"])
    return df.sort_values(["entry_filled_ts_ms", "live_trade_id"], na_position="last").reset_index(drop=True)


def read_csv_any(path: Path) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"failed to read CSV: {path}; last_err={last_err}")


def load_backtest_events(path: str, *, bj_tz: str) -> pd.DataFrame:
    if not str(path).strip():
        return pd.DataFrame()
    df = read_csv_any(Path(path))
    if df.empty:
        return pd.DataFrame()
    time_col = "event_time_bj" if "event_time_bj" in df.columns else df.columns[0]
    side_col = "side" if "side" in df.columns else None
    basis_col = "basis" if "basis" in df.columns else None
    out = pd.DataFrame(
        {
            "ref_signal_id": range(1, len(df) + 1),
            "ref_event_ts_ms": df[time_col].map(lambda x: parse_ts_ms(x, assumed_tz=bj_tz)),
            "ref_event_ts_utc": df[time_col].map(lambda x: iso_utc(parse_ts_ms(x, assumed_tz=bj_tz))),
            "ref_side": df[side_col].map(side_to_ls) if side_col else "",
            "ref_basis": pd.to_numeric(df[basis_col], errors="coerce") if basis_col else None,
            "ref_source_file": str(path),
        }
    )
    return out.dropna(subset=["ref_event_ts_ms"]).reset_index(drop=True)


def load_backtest_trades(path: str) -> pd.DataFrame:
    if not str(path).strip():
        return pd.DataFrame()
    df = read_csv_any(Path(path))
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "ref_trade_id": range(1, len(df) + 1),
            "ref_entry_ts_ms": df["entry_time"].map(lambda x: parse_ts_ms(x, assumed_tz="UTC")) if "entry_time" in df.columns else None,
            "ref_exit_ts_ms": df["exit_time"].map(lambda x: parse_ts_ms(x, assumed_tz="UTC")) if "exit_time" in df.columns else None,
            "ref_side": df["side"].map(side_to_ls) if "side" in df.columns else "",
            "ref_entry_price": pd.to_numeric(df["entry_price"], errors="coerce") if "entry_price" in df.columns else None,
            "ref_exit_price": pd.to_numeric(df["exit_price"], errors="coerce") if "exit_price" in df.columns else None,
            "ref_qty": pd.to_numeric(df["qty"], errors="coerce") if "qty" in df.columns else None,
            "ref_pnl": pd.to_numeric(df["pnl"], errors="coerce") if "pnl" in df.columns else None,
            "ref_reason": df["reason"].astype(str) if "reason" in df.columns else "",
            "ref_be_triggered": df["be_triggered"].astype(str) if "be_triggered" in df.columns else "",
            "ref_source_file": str(path),
        }
    )
    out["ref_entry_ts_utc"] = out["ref_entry_ts_ms"].map(iso_utc)
    out["ref_exit_ts_utc"] = out["ref_exit_ts_ms"].map(iso_utc)
    return out.dropna(subset=["ref_entry_ts_ms"]).reset_index(drop=True)


def load_hc_roundtrips(path: str) -> pd.DataFrame:
    if not str(path).strip():
        return pd.DataFrame()
    df = read_csv_any(Path(path))
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["ref_trade_id"] = range(1, len(out) + 1)
    out["ref_entry_ts_ms"] = out["entry_ts_ms"].map(to_int)
    out["ref_exit_ts_ms"] = out["exit_ts_ms"].map(to_int)
    out["ref_side"] = out["side"].map(side_to_ls)
    out["ref_entry_price"] = pd.to_numeric(out["entry_price"], errors="coerce")
    out["ref_exit_price"] = pd.to_numeric(out["exit_price"], errors="coerce")
    out["ref_qty"] = pd.to_numeric(out["qty"], errors="coerce")
    out["ref_pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    out["ref_fee_total"] = pd.to_numeric(out["fee_total"], errors="coerce") if "fee_total" in out.columns else None
    out["ref_source_file"] = str(path)
    out["ref_entry_ts_utc"] = out["ref_entry_ts_ms"].map(iso_utc)
    out["ref_exit_ts_utc"] = out["ref_exit_ts_ms"].map(iso_utc)
    return out[
        [
            "ref_trade_id",
            "ref_entry_ts_ms",
            "ref_entry_ts_utc",
            "ref_exit_ts_ms",
            "ref_exit_ts_utc",
            "ref_side",
            "ref_entry_price",
            "ref_exit_price",
            "ref_qty",
            "ref_pnl",
            "ref_fee_total",
            "ref_source_file",
        ]
    ].reset_index(drop=True)


def classify_hc_fill_role(reason: Any) -> str:
    s = str(reason or "").strip().lower()
    if s.startswith("boll_entry_"):
        return "ENTRY"
    if s.startswith("boll_exit") or s.startswith("boll_stop"):
        return "EXIT"
    return ""


def infer_hc_fill_position_side(row: pd.Series) -> str:
    reason = str(row.get("reason") or "").strip().lower()
    side = str(row.get("side") or "").strip().upper()
    if reason == "boll_entry_long":
        return "LONG"
    if reason == "boll_entry_short":
        return "SHORT"
    if side == "SELL":
        return "LONG"
    if side == "BUY":
        return "SHORT"
    return ""


def load_hc_fills(path: str) -> pd.DataFrame:
    if not str(path).strip():
        return pd.DataFrame()
    df = read_csv_any(Path(path))
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["ref_fill_id"] = range(1, len(out) + 1)
    out["ref_fill_ts_ms"] = out["ts_ms"].map(to_int)
    out["ref_fill_ts_utc"] = out["ref_fill_ts_ms"].map(iso_utc)
    out["ref_fill_side"] = out["side"].astype(str).str.upper().str.strip()
    out["ref_fill_qty"] = pd.to_numeric(out["qty"], errors="coerce")
    out["ref_fill_price"] = pd.to_numeric(out["price"], errors="coerce")
    out["ref_fill_reason"] = out["reason"].astype(str)
    out["ref_order_id"] = out["order_id"].astype(str) if "order_id" in out.columns else ""
    out["ref_source_file"] = str(path)
    out["ref_position_side"] = out.apply(infer_hc_fill_position_side, axis=1)
    out["ref_fill_role"] = out["ref_fill_reason"].map(classify_hc_fill_role)
    return out[
        [
            "ref_fill_id",
            "ref_fill_ts_ms",
            "ref_fill_ts_utc",
            "ref_fill_side",
            "ref_position_side",
            "ref_fill_role",
            "ref_fill_qty",
            "ref_fill_price",
            "ref_fill_reason",
            "ref_order_id",
            "ref_source_file",
        ]
    ].reset_index(drop=True)


def greedy_match(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    left_ts_col: str,
    right_ts_col: str,
    left_side_col: str,
    right_side_col: str,
    max_delta_sec: int,
    left_filter: Optional[Callable[..., bool]] = None,
    right_filter: Optional[Callable[..., bool]] = None,
) -> list[tuple[int, int, float]]:
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    if left_df.empty or right_df.empty:
        return matches
    left_iter = left_df.reset_index()
    right_iter = right_df.reset_index()
    for left_row in left_iter.itertuples(index=False):
        li = int(left_row.index)
        lside = str(getattr(left_row, left_side_col))
        lts = to_int(getattr(left_row, left_ts_col))
        if not lside or lts is None:
            continue
        if left_filter and not left_filter(left_row):
            continue
        best: Optional[tuple[int, float]] = None
        for right_row in right_iter.itertuples(index=False):
            ri = int(right_row.index)
            if ri in used_right:
                continue
            rside = str(getattr(right_row, right_side_col))
            rts = to_int(getattr(right_row, right_ts_col))
            if not rside or rts is None:
                continue
            if rside != lside:
                continue
            if right_filter and not right_filter(right_row, left_row):
                continue
            delta_sec = abs(int(rts) - int(lts)) / 1000.0
            if delta_sec > float(max_delta_sec):
                continue
            if best is None or delta_sec < best[1]:
                best = (ri, delta_sec)
        if best is not None:
            used_right.add(best[0])
            matches.append((li, best[0], best[1]))
    return matches


def build_signal_compare(
    live_signals: pd.DataFrame,
    ref_events: pd.DataFrame,
    *,
    max_delta_sec: int,
) -> pd.DataFrame:
    if live_signals.empty:
        return pd.DataFrame()
    work = live_signals.copy()
    work["match_ts_ms"] = work["signal_ready_ts_ms"].fillna(work["entry_signal_ts_ms"])
    matches = greedy_match(
        work,
        ref_events,
        left_ts_col="match_ts_ms",
        right_ts_col="ref_event_ts_ms",
        left_side_col="signal_side",
        right_side_col="ref_side",
        max_delta_sec=max_delta_sec,
    )
    ref_by_idx = ref_events.to_dict("index")
    rows: list[dict[str, Any]] = []
    matched_left: set[int] = set()
    for li, ri, _ in matches:
        matched_left.add(li)
        lrow = work.loc[li].to_dict()
        rrow = ref_by_idx[ri]
        rows.append(
            {
                **lrow,
                **rrow,
                "matched": True,
                "delta_signal_ready_sec": (
                    (float(lrow["signal_ready_ts_ms"]) - float(rrow["ref_event_ts_ms"])) / 1000.0
                    if pd.notna(lrow.get("signal_ready_ts_ms")) and pd.notna(rrow.get("ref_event_ts_ms"))
                    else None
                ),
                "delta_entry_fill_sec": (
                    (float(lrow["entry_filled_ts_ms"]) - float(rrow["ref_event_ts_ms"])) / 1000.0
                    if pd.notna(lrow.get("entry_filled_ts_ms")) and pd.notna(rrow.get("ref_event_ts_ms"))
                    else None
                ),
                "delta_basis": (
                    float(lrow["signal_basis"]) - float(rrow["ref_basis"])
                    if pd.notna(lrow.get("signal_basis")) and pd.notna(rrow.get("ref_basis"))
                    else None
                ),
            }
        )
    for idx, row in work.iterrows():
        if int(idx) in matched_left:
            continue
        rows.append({**row.to_dict(), "matched": False})
    return pd.DataFrame(rows).sort_values(["matched", "signal_ready_ts_ms", "entry_signal_ts_ms"], ascending=[False, True, True], na_position="last").reset_index(drop=True)


def build_trade_compare(
    live_trades: pd.DataFrame,
    ref_trades: pd.DataFrame,
    *,
    max_delta_sec: int,
    left_match_ts_col: str,
    compare_label: str,
) -> pd.DataFrame:
    if live_trades.empty:
        return pd.DataFrame()
    matches = greedy_match(
        live_trades,
        ref_trades,
        left_ts_col=left_match_ts_col,
        right_ts_col="ref_entry_ts_ms",
        left_side_col="side",
        right_side_col="ref_side",
        max_delta_sec=max_delta_sec,
    )
    ref_by_idx = ref_trades.to_dict("index")
    rows: list[dict[str, Any]] = []
    matched_left: set[int] = set()
    for li, ri, _ in matches:
        matched_left.add(li)
        lrow = live_trades.loc[li].to_dict()
        rrow = ref_by_idx[ri]
        rows.append(
            {
                **lrow,
                **rrow,
                "compare_label": compare_label,
                "matched": True,
                "delta_match_entry_sec": (
                    (float(lrow[left_match_ts_col]) - float(rrow["ref_entry_ts_ms"])) / 1000.0
                    if pd.notna(lrow.get(left_match_ts_col)) and pd.notna(rrow.get("ref_entry_ts_ms"))
                    else None
                ),
                "delta_entry_fill_sec": (
                    (float(lrow["entry_filled_ts_ms"]) - float(rrow["ref_entry_ts_ms"])) / 1000.0
                    if pd.notna(lrow.get("entry_filled_ts_ms")) and pd.notna(rrow.get("ref_entry_ts_ms"))
                    else None
                ),
                "delta_exit_fill_sec": (
                    (float(lrow["exit_filled_ts_ms"]) - float(rrow["ref_exit_ts_ms"])) / 1000.0
                    if pd.notna(lrow.get("exit_filled_ts_ms")) and pd.notna(rrow.get("ref_exit_ts_ms"))
                    else None
                ),
                "delta_entry_price": (
                    float(lrow["entry_fill_price"]) - float(rrow["ref_entry_price"])
                    if pd.notna(lrow.get("entry_fill_price")) and pd.notna(rrow.get("ref_entry_price"))
                    else None
                ),
                "delta_exit_price": (
                    float(lrow["exit_fill_price"]) - float(rrow["ref_exit_price"])
                    if pd.notna(lrow.get("exit_fill_price")) and pd.notna(rrow.get("ref_exit_price"))
                    else None
                ),
            }
        )
    for idx, row in live_trades.iterrows():
        if int(idx) in matched_left:
            continue
        rows.append({**row.to_dict(), "compare_label": compare_label, "matched": False})
    return pd.DataFrame(rows).sort_values(["matched", left_match_ts_col, "entry_filled_ts_ms"], ascending=[False, True, True], na_position="last").reset_index(drop=True)


def live_entry_fill_table(live_trades: pd.DataFrame) -> pd.DataFrame:
    if live_trades.empty:
        return pd.DataFrame()
    out = live_trades.copy()
    out["live_fill_role"] = "ENTRY"
    out["live_fill_ts_ms"] = out["entry_filled_ts_ms"]
    out["live_fill_ts_utc"] = out["entry_filled_ts_utc"]
    out["live_fill_price"] = out["entry_fill_price"]
    out["live_fill_qty"] = out["entry_fill_qty"]
    out["live_fill_reason"] = out["side"].map(lambda x: "boll_entry_long" if x == "LONG" else ("boll_entry_short" if x == "SHORT" else ""))
    out["live_position_side"] = out["side"]
    return out


def live_exit_fill_table(live_trades: pd.DataFrame) -> pd.DataFrame:
    if live_trades.empty:
        return pd.DataFrame()
    out = live_trades.copy()
    out = out[out["exit_filled_ts_ms"].notna()].copy()
    out["live_fill_role"] = "EXIT"
    out["live_fill_ts_ms"] = out["exit_filled_ts_ms"]
    out["live_fill_ts_utc"] = out["exit_filled_ts_utc"]
    out["live_fill_price"] = out["exit_fill_price"]
    out["live_fill_qty"] = out["exit_fill_qty"].fillna(out["entry_fill_qty"])
    out["live_fill_reason"] = out["exit_category"]
    out["live_position_side"] = out["side"]
    return out


def build_fill_compare(
    live_fill_df: pd.DataFrame,
    hc_fill_df: pd.DataFrame,
    *,
    max_delta_sec: int,
    compare_label: str,
) -> pd.DataFrame:
    if live_fill_df.empty:
        return pd.DataFrame()
    matches = greedy_match(
        live_fill_df,
        hc_fill_df,
        left_ts_col="live_fill_ts_ms",
        right_ts_col="ref_fill_ts_ms",
        left_side_col="live_position_side",
        right_side_col="ref_position_side",
        max_delta_sec=max_delta_sec,
        right_filter=lambda right_row, left_row: str(getattr(right_row, "ref_fill_role")) == str(getattr(left_row, "live_fill_role")),
    )
    ref_by_idx = hc_fill_df.to_dict("index")
    rows: list[dict[str, Any]] = []
    matched_left: set[int] = set()
    for li, ri, _ in matches:
        matched_left.add(li)
        lrow = live_fill_df.loc[li].to_dict()
        rrow = ref_by_idx[ri]
        rows.append(
            {
                **lrow,
                **rrow,
                "compare_label": compare_label,
                "matched": True,
                "delta_fill_sec": (
                    (float(lrow["live_fill_ts_ms"]) - float(rrow["ref_fill_ts_ms"])) / 1000.0
                    if pd.notna(lrow.get("live_fill_ts_ms")) and pd.notna(rrow.get("ref_fill_ts_ms"))
                    else None
                ),
                "delta_fill_price": (
                    float(lrow["live_fill_price"]) - float(rrow["ref_fill_price"])
                    if pd.notna(lrow.get("live_fill_price")) and pd.notna(rrow.get("ref_fill_price"))
                    else None
                ),
                "delta_fill_qty": (
                    float(lrow["live_fill_qty"]) - float(rrow["ref_fill_qty"])
                    if pd.notna(lrow.get("live_fill_qty")) and pd.notna(rrow.get("ref_fill_qty"))
                    else None
                ),
                "reason_same": str(lrow.get("live_fill_reason") or "") == str(rrow.get("ref_fill_reason") or ""),
            }
        )
    for idx, row in live_fill_df.iterrows():
        if int(idx) in matched_left:
            continue
        rows.append({**row.to_dict(), "compare_label": compare_label, "matched": False})
    return pd.DataFrame(rows).sort_values(["matched", "live_fill_ts_ms"], ascending=[False, True], na_position="last").reset_index(drop=True)


def summarize_compare(df: pd.DataFrame, *, delta_cols: list[str]) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "matched": 0}
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "matched": int(pd.to_numeric(df.get("matched"), errors="coerce").fillna(False).astype(bool).sum()) if "matched" in df.columns else 0,
    }
    for col in delta_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary[col] = {
            "avg": float(series.mean()),
            "median": float(series.median()),
            "avg_abs": float(series.abs().mean()),
            "max_abs": float(series.abs().max()),
        }
    return summary


def write_df(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    live_paths = expand_live_paths(args.live_jsonl)
    records = load_live_records(live_paths, symbol=args.symbol, account=args.account)
    live_signals = build_live_signals(records)
    live_trades = build_live_trades(records, live_signals)

    backtest_events = load_backtest_events(args.backtest_events_file, bj_tz=args.bj_tz)
    backtest_trades = load_backtest_trades(args.backtest_trades_file)
    hc_roundtrips = load_hc_roundtrips(args.hc_roundtrips_file)
    hc_fills = load_hc_fills(args.hc_fills_file)

    compare_signals = build_signal_compare(live_signals, backtest_events, max_delta_sec=args.signal_match_window_sec)
    compare_backtest_trades = build_trade_compare(
        live_trades,
        backtest_trades,
        max_delta_sec=args.trade_match_window_sec,
        left_match_ts_col="signal_ready_ts_ms",
        compare_label="backtest_trades",
    )
    compare_hc_roundtrips = build_trade_compare(
        live_trades,
        hc_roundtrips,
        max_delta_sec=args.trade_match_window_sec,
        left_match_ts_col="entry_filled_ts_ms",
        compare_label="hc_roundtrips",
    )

    live_entry_fills = live_entry_fill_table(live_trades)
    live_exit_fills = live_exit_fill_table(live_trades)
    hc_entry_fills = hc_fills[hc_fills["ref_fill_role"] == "ENTRY"].copy() if not hc_fills.empty else pd.DataFrame()
    hc_exit_fills = hc_fills[hc_fills["ref_fill_role"] == "EXIT"].copy() if not hc_fills.empty else pd.DataFrame()
    compare_hc_entry_fills = build_fill_compare(
        live_entry_fills,
        hc_entry_fills,
        max_delta_sec=args.fill_match_window_sec,
        compare_label="hc_entry_fills",
    )
    compare_hc_exit_fills = build_fill_compare(
        live_exit_fills,
        hc_exit_fills,
        max_delta_sec=args.fill_match_window_sec,
        compare_label="hc_exit_fills",
    )

    files = {
        "live_signals_csv": out_dir / "live_signals.csv",
        "live_trades_csv": out_dir / "live_trades.csv",
        "compare_signals_csv": out_dir / "compare_live_vs_backtest_signals.csv",
        "compare_backtest_trades_csv": out_dir / "compare_live_vs_backtest_trades.csv",
        "compare_hc_roundtrips_csv": out_dir / "compare_live_vs_hc_roundtrips.csv",
        "compare_hc_entry_fills_csv": out_dir / "compare_live_vs_hc_entry_fills.csv",
        "compare_hc_exit_fills_csv": out_dir / "compare_live_vs_hc_exit_fills.csv",
        "summary_json": out_dir / "summary.json",
    }

    write_df(live_signals, files["live_signals_csv"])
    write_df(live_trades, files["live_trades_csv"])
    write_df(compare_signals, files["compare_signals_csv"])
    write_df(compare_backtest_trades, files["compare_backtest_trades_csv"])
    write_df(compare_hc_roundtrips, files["compare_hc_roundtrips_csv"])
    write_df(compare_hc_entry_fills, files["compare_hc_entry_fills_csv"])
    write_df(compare_hc_exit_fills, files["compare_hc_exit_fills_csv"])

    summary = {
        "inputs": {
            "live_jsonl_files": [str(p) for p in live_paths],
            "backtest_events_file": str(args.backtest_events_file or ""),
            "backtest_trades_file": str(args.backtest_trades_file or ""),
            "hc_roundtrips_file": str(args.hc_roundtrips_file or ""),
            "hc_fills_file": str(args.hc_fills_file or ""),
            "symbol": str(args.symbol),
            "account": str(args.account),
        },
        "counts": {
            "live_records": int(len(records)),
            "live_signals": int(len(live_signals)),
            "live_trades": int(len(live_trades)),
            "open_live_trades": int(live_trades["is_open"].fillna(False).astype(bool).sum()) if not live_trades.empty else 0,
            "backtest_events": int(len(backtest_events)),
            "backtest_trades": int(len(backtest_trades)),
            "hc_roundtrips": int(len(hc_roundtrips)),
            "hc_fills": int(len(hc_fills)),
        },
        "compare": {
            "signals": summarize_compare(compare_signals, delta_cols=["delta_signal_ready_sec", "delta_entry_fill_sec", "delta_basis"]),
            "backtest_trades": summarize_compare(
                compare_backtest_trades,
                delta_cols=["delta_match_entry_sec", "delta_entry_fill_sec", "delta_exit_fill_sec", "delta_entry_price", "delta_exit_price"],
            ),
            "hc_roundtrips": summarize_compare(
                compare_hc_roundtrips,
                delta_cols=["delta_match_entry_sec", "delta_entry_fill_sec", "delta_exit_fill_sec", "delta_entry_price", "delta_exit_price"],
            ),
            "hc_entry_fills": summarize_compare(compare_hc_entry_fills, delta_cols=["delta_fill_sec", "delta_fill_price", "delta_fill_qty"]),
            "hc_exit_fills": summarize_compare(compare_hc_exit_fills, delta_cols=["delta_fill_sec", "delta_fill_price", "delta_fill_qty"]),
        },
        "outputs": {k: str(v) for k, v in files.items()},
    }

    files["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
