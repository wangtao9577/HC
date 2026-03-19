from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare GP3 live trades against HC roundtrips, with optional fragment aggregation."
    )
    ap.add_argument("--live-trades-file", required=True, help="CSV/JSON/JSONL with live trade rows.")
    ap.add_argument("--hc-roundtrips-file", required=True, help="HC roundtrips CSV from run_gp3_hc_backtest.py")
    ap.add_argument("--match-window-sec", type=int, default=6 * 3600)
    ap.add_argument("--price-window-abs", type=float, default=10.0)
    ap.add_argument("--bj-tz", default="Asia/Shanghai")
    ap.add_argument("--group-hc-roundtrips", dest="group_hc_roundtrips", action="store_true")
    ap.add_argument("--raw-hc-roundtrips", dest="group_hc_roundtrips", action="store_false")
    ap.set_defaults(group_hc_roundtrips=True)
    ap.add_argument("--out-dir", default="d:/project/hc/output/gp3_live_compare")
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
        return v if v > 10_000_000_000 else v * 1000
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
    if v is None:
        return ""
    return pd.to_datetime(int(v), unit="ms", utc=True).isoformat()


def pick_first(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return pd.DataFrame(data["rows"])
        return pd.DataFrame([data])
    raise RuntimeError(f"unsupported live trades file: {path}")


def load_live_trades(path: Path, *, bj_tz: str) -> pd.DataFrame:
    raw = load_frame(path)
    ts_col = pick_first(
        raw,
        [
            "entry_ts_ms",
            "open_time_ms",
            "entry_time_ms",
            "open_ts_ms",
            "entry_time",
            "open_time",
            "opened_at",
            "ts",
        ],
    )
    side_col = pick_first(raw, ["side", "position_side", "direction", "dir", "signal"])
    entry_price_col = pick_first(raw, ["entry_price", "open_price", "price", "entry"])
    qty_col = pick_first(raw, ["qty", "size", "amount", "filled_qty"])
    pnl_col = pick_first(raw, ["pnl", "realized_pnl", "profit", "gross_pnl_quote"])
    exit_ts_col = pick_first(raw, ["exit_ts_ms", "close_time_ms", "exit_time", "close_time"])
    exit_price_col = pick_first(raw, ["exit_price", "close_price"])

    if ts_col is None or side_col is None or entry_price_col is None:
        raise RuntimeError(
            f"live trades file missing required columns; found={list(raw.columns)} "
            f"need one of ts/side/entry_price aliases"
        )

    out = pd.DataFrame(index=raw.index)
    out["live_trade_id"] = range(1, len(raw) + 1)
    out["entry_ts_ms"] = raw[ts_col].map(lambda x: parse_ts_ms(x, assumed_tz=bj_tz))
    out["side"] = raw[side_col].map(side_to_ls)
    out["entry_price"] = raw[entry_price_col].map(to_float)
    out["qty"] = raw[qty_col].map(to_float) if qty_col else pd.Series([None] * len(raw), index=raw.index)
    out["pnl"] = raw[pnl_col].map(to_float) if pnl_col else pd.Series([None] * len(raw), index=raw.index)
    out["exit_ts_ms"] = (
        raw[exit_ts_col].map(lambda x: parse_ts_ms(x, assumed_tz=bj_tz))
        if exit_ts_col
        else pd.Series([None] * len(raw), index=raw.index)
    )
    out["exit_price"] = (
        raw[exit_price_col].map(to_float)
        if exit_price_col
        else pd.Series([None] * len(raw), index=raw.index)
    )
    out["entry_ts_utc"] = out["entry_ts_ms"].map(iso_utc)
    out["entry_dt_local"] = pd.to_datetime(out["entry_ts_ms"], unit="ms", utc=True).dt.tz_convert(bj_tz)
    if "exit_ts_ms" in out.columns:
        out["exit_ts_utc"] = out["exit_ts_ms"].map(iso_utc)
    return out.dropna(subset=["entry_ts_ms", "entry_price"]).query("side != ''").reset_index(drop=True)


def _weighted_mean(df: pd.DataFrame, price_col: str, qty_col: str) -> Optional[float]:
    px = pd.to_numeric(df[price_col], errors="coerce")
    qty = pd.to_numeric(df[qty_col], errors="coerce").abs()
    valid = px.notna() & qty.notna()
    if not bool(valid.any()):
        return None
    qty_sum = float(qty[valid].sum())
    if qty_sum <= 0:
        return float(px[valid].mean())
    return float((px[valid] * qty[valid]).sum() / qty_sum)


def load_hc_roundtrips(path: Path, *, group_fragments: bool, bj_tz: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    need = {"entry_ts_ms", "exit_ts_ms", "side", "entry_price", "exit_price", "qty", "pnl", "fee_total"}
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise RuntimeError(f"hc roundtrips missing columns: {miss}")
    raw = raw.copy()
    for col in ["entry_ts_ms", "exit_ts_ms"]:
        raw[col] = raw[col].map(to_int)
    for col in ["entry_price", "exit_price", "qty", "pnl", "fee_total"]:
        raw[col] = raw[col].map(to_float)
    raw["side"] = raw["side"].map(side_to_ls)

    if not group_fragments:
        out = raw.copy()
        out["fragments"] = 1
        out["grouped_trade_id"] = range(1, len(out) + 1)
    else:
        rows: list[dict[str, Any]] = []
        grouped = raw.groupby(["entry_ts_ms", "side"], dropna=False, sort=True)
        for (entry_ts_ms, side), part in grouped:
            rows.append(
                {
                    "entry_ts_ms": to_int(entry_ts_ms),
                    "exit_ts_ms": max([to_int(x) or 0 for x in part["exit_ts_ms"]]) or None,
                    "side": side,
                    "entry_price": _weighted_mean(part, "entry_price", "qty"),
                    "exit_price": _weighted_mean(part, "exit_price", "qty"),
                    "qty": float(pd.to_numeric(part["qty"], errors="coerce").sum()),
                    "pnl": float(pd.to_numeric(part["pnl"], errors="coerce").sum()),
                    "fee_total": float(pd.to_numeric(part["fee_total"], errors="coerce").sum()),
                    "fragments": int(len(part)),
                }
            )
        out = pd.DataFrame(rows)
        out["grouped_trade_id"] = range(1, len(out) + 1)
    out["entry_ts_utc"] = out["entry_ts_ms"].map(iso_utc)
    out["exit_ts_utc"] = out["exit_ts_ms"].map(iso_utc)
    out["entry_dt_local"] = pd.to_datetime(out["entry_ts_ms"], unit="ms", utc=True).dt.tz_convert(bj_tz)
    out["exit_dt_local"] = pd.to_datetime(out["exit_ts_ms"], unit="ms", utc=True).dt.tz_convert(bj_tz)
    return out.sort_values(["entry_ts_ms", "grouped_trade_id"]).reset_index(drop=True)


def match_live_to_hc(
    live_df: pd.DataFrame,
    hc_df: pd.DataFrame,
    *,
    match_window_sec: int,
    price_window_abs: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    hc_reset = hc_df.reset_index(drop=True)

    for lrow in live_df.sort_values("entry_ts_ms").itertuples(index=False):
        best_idx = None
        best_key = None
        for ridx, rrow in hc_reset.iterrows():
            if ridx in used:
                continue
            if str(rrow.get("side") or "") != str(lrow.side or ""):
                continue
            ref_ts = to_int(rrow.get("entry_ts_ms"))
            live_ts = to_int(lrow.entry_ts_ms)
            if ref_ts is None or live_ts is None:
                continue
            delta_sec = abs(ref_ts - live_ts) / 1000.0
            if match_window_sec > 0 and delta_sec > float(match_window_sec):
                continue
            ref_price = to_float(rrow.get("entry_price"))
            live_price = to_float(lrow.entry_price)
            delta_price = abs(float(ref_price or 0.0) - float(live_price or 0.0))
            if price_window_abs > 0 and delta_price > float(price_window_abs):
                continue
            qty_live = to_float(getattr(lrow, "qty", None))
            qty_ref = to_float(rrow.get("qty"))
            delta_qty = (
                abs(float(qty_live or 0.0) - float(qty_ref or 0.0))
                if qty_live is not None and qty_ref is not None
                else float("inf")
            )
            key = (delta_sec, delta_price, delta_qty, ridx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = ridx

        base = {
            "live_trade_id": int(lrow.live_trade_id),
            "live_entry_ts_ms": to_int(lrow.entry_ts_ms),
            "live_entry_ts_utc": iso_utc(lrow.entry_ts_ms),
            "live_side": str(lrow.side or ""),
            "live_entry_price": to_float(lrow.entry_price),
            "live_qty": to_float(getattr(lrow, "qty", None)),
            "live_pnl": to_float(getattr(lrow, "pnl", None)),
            "matched": best_idx is not None,
        }
        if best_idx is None:
            rows.append(base)
            continue

        used.add(best_idx)
        rrow = hc_reset.iloc[int(best_idx)]
        ref_entry_ts_ms = to_int(rrow.get("entry_ts_ms"))
        ref_entry_price = to_float(rrow.get("entry_price"))
        ref_qty = to_float(rrow.get("qty"))
        rows.append(
            {
                **base,
                "ref_grouped_trade_id": to_int(rrow.get("grouped_trade_id")),
                "ref_entry_ts_ms": ref_entry_ts_ms,
                "ref_entry_ts_utc": iso_utc(ref_entry_ts_ms),
                "ref_exit_ts_ms": to_int(rrow.get("exit_ts_ms")),
                "ref_exit_ts_utc": iso_utc(rrow.get("exit_ts_ms")),
                "ref_side": str(rrow.get("side") or ""),
                "ref_entry_price": ref_entry_price,
                "ref_exit_price": to_float(rrow.get("exit_price")),
                "ref_qty": ref_qty,
                "ref_pnl": to_float(rrow.get("pnl")),
                "ref_fee_total": to_float(rrow.get("fee_total")),
                "ref_fragments": to_int(rrow.get("fragments")),
                "delta_entry_sec": (
                    (float(base["live_entry_ts_ms"]) - float(ref_entry_ts_ms)) / 1000.0
                    if base["live_entry_ts_ms"] is not None and ref_entry_ts_ms is not None
                    else None
                ),
                "delta_entry_price": (
                    float(base["live_entry_price"]) - float(ref_entry_price)
                    if base["live_entry_price"] is not None and ref_entry_price is not None
                    else None
                ),
                "delta_qty": (
                    float(base["live_qty"]) - float(ref_qty)
                    if base["live_qty"] is not None and ref_qty is not None
                    else None
                ),
                "delta_pnl": (
                    float(base["live_pnl"]) - float(rrow.get("pnl"))
                    if base["live_pnl"] is not None and to_float(rrow.get("pnl")) is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_compare(compare_df: pd.DataFrame, hc_df: pd.DataFrame) -> dict[str, Any]:
    matched = compare_df[compare_df["matched"] == True] if not compare_df.empty else compare_df
    return {
        "live_trades": int(len(compare_df)),
        "matched_live_trades": int(len(matched)),
        "unmatched_live_trades": int(len(compare_df) - len(matched)),
        "hc_grouped_trades": int(len(hc_df)),
        "unused_hc_grouped_trades": int(len(hc_df) - len(matched)),
        "delta_entry_sec_mean": float(matched["delta_entry_sec"].mean()) if "delta_entry_sec" in matched and not matched.empty else None,
        "delta_entry_price_mean": float(matched["delta_entry_price"].mean()) if "delta_entry_price" in matched and not matched.empty else None,
        "delta_qty_mean": float(matched["delta_qty"].mean()) if "delta_qty" in matched and not matched.empty else None,
        "live_pnl_sum": float(pd.to_numeric(compare_df.get("live_pnl"), errors="coerce").sum()) if "live_pnl" in compare_df else None,
        "matched_ref_pnl_sum": float(pd.to_numeric(matched.get("ref_pnl"), errors="coerce").sum()) if "ref_pnl" in matched else None,
    }


def write_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(Path(args.out_dir))

    live_df = load_live_trades(Path(args.live_trades_file), bj_tz=str(args.bj_tz))
    hc_df = load_hc_roundtrips(
        Path(args.hc_roundtrips_file),
        group_fragments=bool(args.group_hc_roundtrips),
        bj_tz=str(args.bj_tz),
    )
    compare_df = match_live_to_hc(
        live_df,
        hc_df,
        match_window_sec=int(args.match_window_sec),
        price_window_abs=float(args.price_window_abs),
    )

    files = {
        "live_trades_csv": out_dir / "live_trades_normalized.csv",
        "hc_roundtrips_csv": out_dir / "hc_roundtrips_normalized.csv",
        "compare_csv": out_dir / "compare_live_vs_hc_roundtrips.csv",
        "summary_json": out_dir / "summary.json",
    }
    write_df(live_df, files["live_trades_csv"])
    write_df(hc_df, files["hc_roundtrips_csv"])
    write_df(compare_df, files["compare_csv"])

    summary = {
        "inputs": {
            "live_trades_file": str(Path(args.live_trades_file)),
            "hc_roundtrips_file": str(Path(args.hc_roundtrips_file)),
            "group_hc_roundtrips": bool(args.group_hc_roundtrips),
            "match_window_sec": int(args.match_window_sec),
            "price_window_abs": float(args.price_window_abs),
        },
        "counts": {
            "live_trades": int(len(live_df)),
            "hc_roundtrips": int(len(hc_df)),
            "compare_rows": int(len(compare_df)),
        },
        "compare": summarize_compare(compare_df, hc_df),
        "files": {k: str(v) for k, v in files.items()},
    }
    files["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("GP3 live vs HC compare done:")
    print(f"  live_trades: {len(live_df)}")
    print(f"  hc_roundtrips: {len(hc_df)}")
    print(f"  matched_live_trades: {summary['compare']['matched_live_trades']}")
    print(f"  unmatched_live_trades: {summary['compare']['unmatched_live_trades']}")
    print(f"  compare_csv: {files['compare_csv']}")
    print(f"  summary_json: {files['summary_json']}")


if __name__ == "__main__":
    main()
