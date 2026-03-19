from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run GP3 use_agg backtests month-by-month and persist progress for long-running verification."
    )
    ap.add_argument("--symbol", default="ETHUSDC")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--start-utc", required=True)
    ap.add_argument("--end-utc", required=True)
    ap.add_argument("--klines-file", default="d:/project/gp3.0/results/live_like_scan/ETHUSDC_1m_20250310_20260310.pkl")
    ap.add_argument("--tick-size", type=float, default=0.01)
    ap.add_argument("--step-size", type=float, default=0.001)
    ap.add_argument("--min-qty", type=float, default=0.001)
    ap.add_argument("--out-dir", default="d:/project/hc/output/gp3_monthly_useagg_batch")
    ap.add_argument("--agg-chunk-minutes", type=float, default=120.0)
    ap.add_argument("--agg-chunk-pause-sec", type=float, default=0.0)
    ap.add_argument("--agg-req-sleep-sec", type=float, default=0.0)
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
        label = f"{cur.year:04d}-{cur.month:02d}"
        out.append((cur, nxt, label))
        cur = nxt
    return out


def latest_stats_file(month_dir: Path) -> Optional[Path]:
    files = sorted(month_dir.glob("*_gp3_hc_stats_*.json"))
    return files[-1] if files else None


def load_stats(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_progress(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "monthly_progress.csv"
    json_path = out_dir / "monthly_progress.json"
    headers = [
        "label",
        "window_start_utc",
        "window_end_utc",
        "status",
        "use_agg",
        "net_return_pct",
        "max_drawdown_pct",
        "round_trips",
        "wins",
        "losses",
        "stats_json",
        "error",
    ]
    lines = [",".join(headers)]
    for row in rows:
        vals = []
        for h in headers:
            val = row.get(h, "")
            text = "" if val is None else str(val)
            text = text.replace("\"", "\"\"")
            if any(ch in text for ch in [",", "\"", "\n"]):
                text = f"\"{text}\""
            vals.append(text)
        lines.append(",".join(vals))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    runner = root / "run_gp3_hc_backtest.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_dt = parse_utc(args.start_utc)
    end_dt = parse_utc(args.end_utc)
    slices = build_month_slices(start_dt, end_dt)
    rows: list[dict] = []

    for start_cur, end_cur, label in slices:
        month_dir = out_dir / label
        month_dir.mkdir(parents=True, exist_ok=True)

        if bool(args.resume):
            cached = latest_stats_file(month_dir)
            if cached is not None:
                stats = load_stats(cached)
                es = stats.get("engine_summary", {}) or {}
                rows.append(
                    {
                        "label": label,
                        "window_start_utc": stats.get("window_start_utc"),
                        "window_end_utc": stats.get("window_end_utc"),
                        "status": "cached",
                        "use_agg": stats.get("use_agg"),
                        "net_return_pct": es.get("net_return_pct"),
                        "max_drawdown_pct": es.get("max_drawdown_pct"),
                        "round_trips": stats.get("round_trips"),
                        "wins": stats.get("wins"),
                        "losses": stats.get("losses"),
                        "stats_json": str(cached),
                        "error": "",
                    }
                )
                write_progress(rows, out_dir)
                continue

        cmd = [
            sys.executable,
            str(runner),
            "--symbol",
            str(args.symbol).upper(),
            "--profile",
            str(args.profile),
            "--start-utc",
            start_cur.isoformat().replace("+00:00", "Z"),
            "--end-utc",
            end_cur.isoformat().replace("+00:00", "Z"),
            "--use-agg",
            "--stop-trigger-source",
            "mark",
            "--funding-rate",
            "0.0001",
            "--funding-interval-hours",
            "8",
            "--maker-fee-tiers",
            "0:0",
            "--taker-fee-tiers",
            "0:0.0004,50000000:0.00035",
            "--klines-file",
            str(args.klines_file),
            "--tick-size",
            str(args.tick_size),
            "--step-size",
            str(args.step_size),
            "--min-qty",
            str(args.min_qty),
            "--agg-chunk-minutes",
            str(args.agg_chunk_minutes),
            "--agg-chunk-pause-sec",
            str(args.agg_chunk_pause_sec),
            "--agg-req-sleep-sec",
            str(args.agg_req_sleep_sec),
            "--out-dir",
            str(month_dir),
        ]
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
            stats = load_stats(stats_path)
            es = stats.get("engine_summary", {}) or {}
            rows.append(
                {
                    "label": label,
                    "window_start_utc": stats.get("window_start_utc"),
                    "window_end_utc": stats.get("window_end_utc"),
                    "status": "ok",
                    "use_agg": stats.get("use_agg"),
                    "net_return_pct": es.get("net_return_pct"),
                    "max_drawdown_pct": es.get("max_drawdown_pct"),
                    "round_trips": stats.get("round_trips"),
                    "wins": stats.get("wins"),
                    "losses": stats.get("losses"),
                    "stats_json": str(stats_path),
                    "error": "",
                }
            )
        else:
            rows.append(
                {
                    "label": label,
                    "window_start_utc": start_cur.isoformat(),
                    "window_end_utc": end_cur.isoformat(),
                    "status": "error",
                    "use_agg": True,
                    "net_return_pct": None,
                    "max_drawdown_pct": None,
                    "round_trips": None,
                    "wins": None,
                    "losses": None,
                    "stats_json": str(stats_path) if stats_path is not None else "",
                    "error": (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip(),
                }
            )
        write_progress(rows, out_dir)

    csv_path, json_path = write_progress(rows, out_dir)
    print("GP3 use_agg monthly batch done:")
    print(f"  slices: {len(rows)}")
    print(f"  progress_csv: {csv_path}")
    print(f"  progress_json: {json_path}")


if __name__ == "__main__":
    main()
