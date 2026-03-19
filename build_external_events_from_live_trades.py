from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


TIME_COL_CANDIDATES = {
    "entry": [
        "entry_time",
        "open_time",
        "entry_time_bj",
        "开仓时间",
        "开仓时间(bj)",
    ],
    "exit": [
        "exit_time",
        "close_time",
        "last_close_time",
        "平仓时间",
        "最后平仓时间",
    ],
}

PRICE_COL_CANDIDATES = {
    "entry": ["entry_price", "open_price", "开仓价格"],
    "exit": ["exit_price", "avg_exit_price", "close_price", "平仓均价", "平仓价格"],
}

QTY_COL_CANDIDATES = [
    "qty",
    "size",
    "closed_qty",
    "已平仓量",
    "已平仓量(eth)",
    "已平仓仓位",
]

SIDE_COL_CANDIDATES = [
    "side",
    "position_side",
    "方向",
    "持仓方向",
]

SYMBOL_COL_CANDIDATES = ["symbol", "交易对", "品种"]


def _norm_col(s: str) -> str:
    return str(s or "").strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cmap = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        hit = cmap.get(_norm_col(cand))
        if hit:
            return hit
    return None


def _parse_side(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if any(x in s for x in ["long", "buy", "做多", "多"]):
        return "LONG"
    if any(x in s for x in ["short", "sell", "做空", "空"]):
        return "SHORT"
    return ""


def _parse_ts_ms(raw: Any, *, assumed_tz: str) -> int:
    if raw is None or str(raw).strip() == "":
        raise ValueError("empty timestamp")
    s = str(raw).strip()
    try:
        if s.isdigit():
            v = int(s)
            if v > 10_000_000_000_000:
                return v
            if v > 10_000_000_000:
                return v
            return v * 1000
        dt = pd.to_datetime(s)
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.tz_localize(assumed_tz)
        dt = dt.tz_convert("UTC")
        return int(dt.timestamp() * 1000)
    except Exception as e:
        raise ValueError(f"invalid timestamp '{raw}': {e}") from e


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if isinstance(raw.get("rows"), list):
                return [dict(x) for x in raw["rows"]]
            if isinstance(raw.get("trades"), list):
                return [dict(x) for x in raw["trades"]]
            if isinstance(raw.get("data"), list):
                return [dict(x) for x in raw["data"]]
        if isinstance(raw, list):
            return [dict(x) for x in raw]
        raise RuntimeError(f"unsupported json structure: {path}")

    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                return [dict(r) for r in csv.DictReader(f)]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"unable to read csv with supported encodings: {path}; last_err={last_err}")


def _load_trade_frame(
    path: Path,
    *,
    assumed_tz: str,
    symbol_filter: str,
    column_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    rows = _load_rows(path)
    if not rows:
        raise RuntimeError(f"no live trade rows found: {path}")
    df = pd.DataFrame(rows)

    column_map = dict(column_map or {})
    side_col = column_map.get("side") or _pick_col(df, SIDE_COL_CANDIDATES)
    entry_time_col = column_map.get("entry_time") or _pick_col(df, TIME_COL_CANDIDATES["entry"])
    exit_time_col = column_map.get("exit_time") or _pick_col(df, TIME_COL_CANDIDATES["exit"])
    entry_price_col = column_map.get("entry_price") or _pick_col(df, PRICE_COL_CANDIDATES["entry"])
    exit_price_col = column_map.get("exit_price") or _pick_col(df, PRICE_COL_CANDIDATES["exit"])
    qty_col = column_map.get("qty") or _pick_col(df, QTY_COL_CANDIDATES)
    symbol_col = column_map.get("symbol") or _pick_col(df, SYMBOL_COL_CANDIDATES)

    required = {
        "side": side_col,
        "entry_time": entry_time_col,
        "exit_time": exit_time_col,
        "entry_price": entry_price_col,
        "exit_price": exit_price_col,
        "qty": qty_col,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise RuntimeError(f"missing required live trade columns: {missing}; columns={list(df.columns)}")

    out = pd.DataFrame(
        {
            "side": df[side_col].map(_parse_side),
            "entry_ts_ms": df[entry_time_col].map(lambda x: _parse_ts_ms(x, assumed_tz=assumed_tz)),
            "exit_ts_ms": df[exit_time_col].map(lambda x: _parse_ts_ms(x, assumed_tz=assumed_tz)),
            "entry_price": pd.to_numeric(df[entry_price_col], errors="coerce"),
            "exit_price": pd.to_numeric(df[exit_price_col], errors="coerce"),
            "qty": pd.to_numeric(df[qty_col], errors="coerce"),
        }
    )
    if symbol_col:
        out["symbol"] = df[symbol_col].astype(str).str.upper().str.strip()
    else:
        out["symbol"] = str(symbol_filter or "").upper()

    out = out.dropna(subset=["entry_price", "exit_price", "qty"]).copy()
    out = out[out["side"].isin(["LONG", "SHORT"])].copy()
    if symbol_filter:
        out = out[out["symbol"] == str(symbol_filter).upper()].copy()
    out = out.sort_values(["entry_ts_ms", "exit_ts_ms"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("no usable live trade rows after filtering")
    return out


def _parse_jsonish(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _enrich_from_trade_events(
    df: pd.DataFrame,
    *,
    db_path: Path,
    symbol: str,
    strategy_type: str,
    account_name: str,
    match_window_sec: int,
) -> pd.DataFrame:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    rows = []
    for r in df.itertuples(index=False):
        start_utc = datetime.fromtimestamp((int(r.entry_ts_ms) - match_window_sec * 1000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        end_utc = datetime.fromtimestamp((int(r.exit_ts_ms) + match_window_sec * 1000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        q = """
            SELECT id, event_time, event_type, side, size, price, reason, raw
            FROM trade_events
            WHERE symbol = ?
              AND strategy_type = ?
              AND account_name = ?
              AND event_time >= ?
              AND event_time <= ?
            ORDER BY event_time, id
        """
        evs = [dict(x) for x in cur.execute(q, (symbol, strategy_type, account_name, start_utc, end_utc)).fetchall()]
        entry_event = None
        exit_event = None
        for ev in evs:
            if entry_event is None and str(ev.get("event_type") or "").lower() in {"entry_fill", "entry"}:
                entry_event = ev
            if str(ev.get("event_type") or "").lower() in {"exit_detected", "close", "exit"}:
                exit_event = ev
        exit_raw = _parse_jsonish(exit_event.get("raw")) if exit_event else {}
        entry_raw = _parse_jsonish(entry_event.get("raw")) if entry_event else {}
        rows.append(
            {
                **r._asdict(),
                "db_entry_event_id": entry_event.get("id") if entry_event else None,
                "db_entry_event_type": entry_event.get("event_type") if entry_event else "",
                "db_entry_reason": entry_event.get("reason") if entry_event else "",
                "db_exit_event_id": exit_event.get("id") if exit_event else None,
                "db_exit_event_type": exit_event.get("event_type") if exit_event else "",
                "db_exit_reason": exit_event.get("reason") if exit_event else "",
                "db_exit_source": str(exit_raw.get("source") or ""),
                "db_exit_detail": exit_raw.get("detail") if isinstance(exit_raw.get("detail"), dict) else {},
                "db_event_count": len(evs),
                "db_entry_raw": entry_raw,
                "db_exit_raw": exit_raw,
            }
        )
    con.close()
    return pd.DataFrame(rows)


def _classify_row(row: pd.Series) -> str:
    source = str(row.get("db_exit_source") or "").lower().strip()
    reason = str(row.get("db_exit_reason") or "").lower().strip()
    if source in {"external", "manual"}:
        return source
    if "external" in reason:
        return "external"
    if "manual" in reason:
        return "manual"
    if source in {"tp", "sl", "system"}:
        return source
    if "tp" in reason:
        return "tp"
    if "sl" in reason or "stop" in reason:
        return "sl"
    return "unknown"


def _build_external_events(
    df: pd.DataFrame,
    *,
    reason_prefix: str,
    trigger_source: str,
    only_classes: set[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for r in df.itertuples(index=False):
        cls = str(getattr(r, "classification", "") or "")
        if only_classes and cls not in only_classes:
            continue
        events.append(
            {
                "ts_ms": int(r.exit_ts_ms),
                "event_type": "FORCE_FLAT",
                "price": float(r.exit_price),
                "qty": float(r.qty),
                "reason": f"{reason_prefix}:{cls or 'live_close'}",
                "trigger_source": str(trigger_source or "live_external"),
            }
        )
    events.sort(key=lambda x: int(x["ts_ms"]))
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HC external-events from live closed trades")
    ap.add_argument("--live-trades-file", required=True, help="CSV/JSON export of live closed trades")
    ap.add_argument("--symbol", default="ETHUSDC")
    ap.add_argument("--assumed-timezone", default="Asia/Shanghai")
    ap.add_argument("--column-map-json", default="", help='Optional explicit column map, e.g. {"side":"方向","entry_time":"开仓时间"}')
    ap.add_argument("--db-path", default="", help="Optional DBRSI sqlite path for enrichment/classification")
    ap.add_argument("--strategy-type", default="gp_2")
    ap.add_argument("--account-name", default="GP")
    ap.add_argument("--match-window-sec", type=int, default=180)
    ap.add_argument("--include-classes", default="external,manual,unknown", help="comma separated classes to export as FORCE_FLAT events")
    ap.add_argument("--reason-prefix", default="live_close_imported")
    ap.add_argument("--trigger-source", default="live_external")
    ap.add_argument("--out-events-json", default="d:/project/hc/output/live_external_events.json")
    ap.add_argument("--out-summary-json", default="d:/project/hc/output/live_external_events_summary.json")
    args = ap.parse_args()

    live_path = Path(args.live_trades_file)
    column_map = json.loads(str(args.column_map_json)) if str(args.column_map_json).strip() else {}
    df = _load_trade_frame(
        live_path,
        assumed_tz=str(args.assumed_timezone),
        symbol_filter=str(args.symbol).upper(),
        column_map=column_map,
    )

    if str(args.db_path).strip():
        df = _enrich_from_trade_events(
            df,
            db_path=Path(args.db_path),
            symbol=str(args.symbol).upper(),
            strategy_type=str(args.strategy_type),
            account_name=str(args.account_name),
            match_window_sec=int(args.match_window_sec),
        )
    else:
        df = df.copy()

    df["classification"] = df.apply(_classify_row, axis=1)

    include_classes = {x.strip().lower() for x in str(args.include_classes or "").split(",") if x.strip()}
    events = _build_external_events(
        df,
        reason_prefix=str(args.reason_prefix),
        trigger_source=str(args.trigger_source),
        only_classes=include_classes,
    )

    out_events = Path(args.out_events_json)
    out_summary = Path(args.out_summary_json)
    out_events.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    events_payload = {"events": events}
    out_events.write_text(json.dumps(events_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_payload = {
        "symbol": str(args.symbol).upper(),
        "input_file": str(live_path),
        "db_path": str(args.db_path or ""),
        "rows_total": int(len(df)),
        "events_exported": int(len(events)),
        "class_counts": {
            str(k): int(v)
            for k, v in df["classification"].value_counts(dropna=False).to_dict().items()
        },
        "rows": df.to_dict(orient="records"),
        "external_events_file": str(out_events),
    }
    out_summary.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("live trade external-event build done:")
    print(f"  input_rows: {len(df)}")
    print(f"  events_exported: {len(events)}")
    print(f"  out_events_json: {out_events}")
    print(f"  out_summary_json: {out_summary}")


if __name__ == "__main__":
    main()
