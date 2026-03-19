from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run BOLL2 use_agg backtests month-by-month with rolling capital and aggregate the yearly summary."
    )
    ap.add_argument("--profile", default="d:/project/BOLL2.1/profiles/gen2_plus_profile.env")
    ap.add_argument("--start-utc", required=True)
    ap.add_argument("--end-utc", required=True)
    ap.add_argument("--initial-capital", type=float, default=100000.0)
    ap.add_argument("--out-dir", default="d:/project/hc/output/boll2_useagg_monthly_batch")
    ap.add_argument("--agg-chunk-minutes", type=float, default=120.0)
    ap.add_argument("--agg-chunk-pause-sec", type=float, default=0.0)
    ap.add_argument("--agg-req-sleep-sec", type=float, default=0.02)
    ap.add_argument("--agg-timeout-sec", type=float, default=20.0)
    ap.add_argument("--agg-max-418-retries", type=int, default=40)
    ap.add_argument("--agg-ban-cooldown-sec", type=float, default=20.0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--resume", dest="resume", action="store_true")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=True)
    return ap.parse_args()


def parse_utc(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty UTC timestamp")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def next_month_boundary(cur: datetime) -> datetime:
    if cur.month == 12:
        return datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)


def build_month_slices(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime, str]]:
    if end_dt <= start_dt:
        raise RuntimeError("end-utc must be greater than start-utc")
    out: list[tuple[datetime, datetime, str]] = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(next_month_boundary(cur), end_dt)
        label = f"{cur.strftime('%Y%m%d')}_{nxt.strftime('%Y%m%d')}"
        out.append((cur, nxt, label))
        cur = nxt
    return out


def latest_stats_file(month_dir: Path) -> Path | None:
    files = sorted(month_dir.glob("*_boll2_hc_stats_*.json"))
    return files[-1] if files else None


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_progress(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    df = pd.DataFrame(rows)
    csv_path = out_dir / "monthly_progress.csv"
    json_path = out_dir / "monthly_progress.json"
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "label",
                "window_start_utc",
                "window_end_utc",
                "status",
                "initial_capital",
                "ending_equity",
                "net_return_pct",
                "max_drawdown_pct",
                "round_trips",
                "wins",
                "losses",
                "final_position_qty",
                "stats_json",
                "error",
            ]
        )
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def calc_max_drawdown_pct(equity_df: pd.DataFrame) -> float:
    if equity_df.empty:
        return 0.0
    eq = pd.to_numeric(equity_df["equity"], errors="coerce").dropna()
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, pd.NA)
    return float(dd.fillna(0.0).max() * 100.0)


def build_aggregate_summary(rows: list[dict], out_dir: Path) -> dict:
    ok_rows = [r for r in rows if r.get("status") in {"ok", "cached"} and str(r.get("stats_json") or "").strip()]
    if not ok_rows:
        return {
            "status": "no_successful_slices",
            "completed_slices": 0,
        }

    month_stats = [load_stats(Path(r["stats_json"])) for r in ok_rows]
    first_stats = month_stats[0]
    last_stats = month_stats[-1]

    equity_parts: list[pd.DataFrame] = []
    rt_parts: list[pd.DataFrame] = []
    fills_parts: list[pd.DataFrame] = []
    open_boundary_rows: list[dict] = []

    for stats in month_stats:
        outputs = stats.get("outputs", {}) or {}
        eq_path = Path(str(outputs.get("equity_csv") or ""))
        rt_path = Path(str(outputs.get("roundtrips_csv") or ""))
        fills_path = Path(str(outputs.get("fills_csv") or ""))
        if eq_path.exists():
            eq_df = pd.read_csv(eq_path)
            if not eq_df.empty:
                equity_parts.append(eq_df[["ts_ms", "equity"]].copy())
        if rt_path.exists():
            rt_df = pd.read_csv(rt_path)
            if not rt_df.empty:
                rt_parts.append(rt_df.copy())
        if fills_path.exists():
            fills_df = pd.read_csv(fills_path)
            if not fills_df.empty:
                fills_parts.append(fills_df.copy())

        final_position = stats.get("final_position", {}) or {}
        qty = float(final_position.get("qty") or 0.0)
        if abs(qty) > 1e-12:
            open_boundary_rows.append(
                {
                    "window_start_utc": stats.get("window_start_utc"),
                    "window_end_utc": stats.get("window_end_utc"),
                    "qty": qty,
                    "entry_price": float(final_position.get("entry_price") or 0.0),
                    "stats_json": stats.get("outputs", {}).get("stats_json"),
                }
            )

    equity_all = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame(columns=["ts_ms", "equity"])
    if not equity_all.empty:
        equity_all["ts_ms"] = pd.to_numeric(equity_all["ts_ms"], errors="coerce")
        equity_all["equity"] = pd.to_numeric(equity_all["equity"], errors="coerce")
        equity_all = equity_all.dropna(subset=["ts_ms", "equity"]).sort_values("ts_ms").drop_duplicates(subset=["ts_ms"], keep="last")
        equity_all.to_csv(out_dir / "aggregate_equity.csv", index=False)

    rt_all = pd.concat(rt_parts, ignore_index=True) if rt_parts else pd.DataFrame()
    if not rt_all.empty:
        rt_all.to_csv(out_dir / "aggregate_roundtrips.csv", index=False)

    fills_all = pd.concat(fills_parts, ignore_index=True) if fills_parts else pd.DataFrame()
    if not fills_all.empty:
        fills_all.to_csv(out_dir / "aggregate_fills.csv", index=False)

    starting_equity = float((first_stats.get("engine_summary", {}) or {}).get("starting_equity") or 0.0)
    ending_equity = float((last_stats.get("engine_summary", {}) or {}).get("ending_equity") or 0.0)
    net_return_pct = ((ending_equity / starting_equity) - 1.0) * 100.0 if starting_equity > 0 else 0.0
    wins = int(sum(int(s.get("wins") or 0) for s in month_stats))
    losses = int(sum(int(s.get("losses") or 0) for s in month_stats))
    round_trips = int(sum(int(s.get("round_trips") or 0) for s in month_stats))
    win_rate_pct = (wins / round_trips * 100.0) if round_trips else 0.0
    fee_paid_total = float(sum(float((s.get("engine_summary", {}) or {}).get("fee_paid_total") or 0.0) for s in month_stats))
    funding_pnl = float(sum(float((s.get("engine_summary", {}) or {}).get("funding_pnl") or 0.0) for s in month_stats))
    margin_reject_count = float(sum(float((s.get("engine_summary", {}) or {}).get("margin_reject_count") or 0.0) for s in month_stats))
    liquidation_count = float(sum(float((s.get("engine_summary", {}) or {}).get("liquidation_count") or 0.0) for s in month_stats))

    summary = {
        "status": "ok",
        "profile": first_stats.get("profile"),
        "symbol": first_stats.get("symbol"),
        "use_agg": True,
        "window_start_utc": first_stats.get("window_start_utc"),
        "window_end_utc": last_stats.get("window_end_utc"),
        "completed_slices": len(month_stats),
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "net_return_pct": net_return_pct,
        "max_drawdown_pct": calc_max_drawdown_pct(equity_all),
        "round_trips": round_trips,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate_pct,
        "fee_paid_total": fee_paid_total,
        "funding_pnl": funding_pnl,
        "margin_reject_count": margin_reject_count,
        "liquidation_count": liquidation_count,
        "open_boundary_months": open_boundary_rows,
        "aggregate_outputs": {
            "equity_csv": str(out_dir / "aggregate_equity.csv") if not equity_all.empty else "",
            "roundtrips_csv": str(out_dir / "aggregate_roundtrips.csv") if not rt_all.empty else "",
            "fills_csv": str(out_dir / "aggregate_fills.csv") if not fills_all.empty else "",
        },
        "monthly_stats_jsons": [str(r["stats_json"]) for r in ok_rows],
    }
    return summary


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    runner = root / "run_boll2_hc_backtest.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = parse_utc(args.start_utc)
    end_dt = parse_utc(args.end_utc)
    slices = build_month_slices(start_dt, end_dt)
    rows: list[dict] = []
    current_initial = float(args.initial_capital)

    for start_cur, end_cur, label in slices:
        month_dir = out_dir / label
        month_dir.mkdir(parents=True, exist_ok=True)

        if bool(args.resume):
            cached = latest_stats_file(month_dir)
            if cached is not None:
                stats = load_stats(cached)
                es = stats.get("engine_summary", {}) or {}
                current_initial = float(es.get("ending_equity") or current_initial)
                final_position = stats.get("final_position", {}) or {}
                rows.append(
                    {
                        "label": label,
                        "window_start_utc": stats.get("window_start_utc"),
                        "window_end_utc": stats.get("window_end_utc"),
                        "status": "cached",
                        "initial_capital": stats.get("initial_capital_override") or es.get("starting_equity"),
                        "ending_equity": es.get("ending_equity"),
                        "net_return_pct": es.get("net_return_pct"),
                        "max_drawdown_pct": es.get("max_drawdown_pct"),
                        "round_trips": stats.get("round_trips"),
                        "wins": stats.get("wins"),
                        "losses": stats.get("losses"),
                        "final_position_qty": final_position.get("qty"),
                        "stats_json": str(cached),
                        "error": "",
                    }
                )
                write_progress(rows, out_dir)
                continue

        cmd = [
            sys.executable,
            str(runner),
            "--profile",
            str(args.profile),
            "--start-utc",
            start_cur.isoformat().replace("+00:00", "Z"),
            "--end-utc",
            end_cur.isoformat().replace("+00:00", "Z"),
            "--initial-capital-override",
            str(current_initial),
            "--use-agg",
            "--stop-trigger-source",
            "mark",
            "--agg-chunk-minutes",
            str(args.agg_chunk_minutes),
            "--agg-chunk-pause-sec",
            str(args.agg_chunk_pause_sec),
            "--agg-req-sleep-sec",
            str(args.agg_req_sleep_sec),
            "--agg-timeout-sec",
            str(args.agg_timeout_sec),
            "--agg-max-418-retries",
            str(args.agg_max_418_retries),
            "--agg-ban-cooldown-sec",
            str(args.agg_ban_cooldown_sec),
            "--out-dir",
            str(month_dir),
        ]
        proc = None
        stats_path = None
        last_error = ""
        for attempt in range(1, max(int(args.max_attempts), 1) + 1):
            proc = subprocess.run(
                cmd,
                cwd=str(root.parent),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stats_path = latest_stats_file(month_dir)
            if proc.returncode == 0 and stats_path is not None:
                break
            last_error = (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip()

        if proc is not None and proc.returncode == 0 and stats_path is not None:
            stats = load_stats(stats_path)
            es = stats.get("engine_summary", {}) or {}
            final_position = stats.get("final_position", {}) or {}
            current_initial = float(es.get("ending_equity") or current_initial)
            rows.append(
                {
                    "label": label,
                    "window_start_utc": stats.get("window_start_utc"),
                    "window_end_utc": stats.get("window_end_utc"),
                    "status": "ok",
                    "initial_capital": stats.get("initial_capital_override") or es.get("starting_equity"),
                    "ending_equity": es.get("ending_equity"),
                    "net_return_pct": es.get("net_return_pct"),
                    "max_drawdown_pct": es.get("max_drawdown_pct"),
                    "round_trips": stats.get("round_trips"),
                    "wins": stats.get("wins"),
                    "losses": stats.get("losses"),
                    "final_position_qty": final_position.get("qty"),
                    "attempts": attempt,
                    "stats_json": str(stats_path),
                    "error": "",
                }
            )
            write_progress(rows, out_dir)
            continue

        rows.append(
            {
                "label": label,
                "window_start_utc": start_cur.isoformat(),
                "window_end_utc": end_cur.isoformat(),
                "status": "error",
                "initial_capital": current_initial,
                "ending_equity": None,
                "net_return_pct": None,
                "max_drawdown_pct": None,
                "round_trips": None,
                "wins": None,
                "losses": None,
                "final_position_qty": None,
                "attempts": max(int(args.max_attempts), 1),
                "stats_json": str(stats_path) if stats_path is not None else "",
                "error": last_error,
            }
        )
        write_progress(rows, out_dir)
        break

    csv_path, json_path = write_progress(rows, out_dir)
    summary = build_aggregate_summary(rows, out_dir)
    summary_path = out_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BOLL2 use_agg monthly batch done:")
    print(f"  slices_recorded: {len(rows)}")
    print(f"  progress_csv: {csv_path}")
    print(f"  progress_json: {json_path}")
    print(f"  aggregate_summary_json: {summary_path}")
    if summary.get("status") == "ok":
        print(f"  ending_equity: {summary.get('ending_equity')}")
        print(f"  net_return_pct: {summary.get('net_return_pct')}")
        print(f"  max_drawdown_pct: {summary.get('max_drawdown_pct')}")


if __name__ == "__main__":
    main()
