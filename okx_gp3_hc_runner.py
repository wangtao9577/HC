from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "hc-okx-gp3-runner/1.0",
            "Accept": "application/json",
        }
    )
    return s


def _proxy_dict(proxy_url: str) -> dict | None:
    p = str(proxy_url or "").strip()
    if not p:
        return None
    return {"http": p, "https": p}


def fetch_okx_rules(inst_id: str, proxy_url: str, timeout_sec: float = 20.0) -> dict:
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "SWAP", "instId": inst_id}
    r = _session().get(url, params=params, proxies=_proxy_dict(proxy_url), timeout=timeout_sec)
    r.raise_for_status()
    j = r.json()
    if str(j.get("code")) != "0":
        raise RuntimeError(f"okx instruments error: {j}")
    data = j.get("data") or []
    if not data:
        raise RuntimeError(f"no instruments data for {inst_id}")
    row = data[0]
    tick = float(row.get("tickSz") or 0.0)
    lot = float(row.get("lotSz") or 0.0)
    min_sz = float(row.get("minSz") or 0.0)
    if tick <= 0 or lot <= 0 or min_sz <= 0:
        raise RuntimeError(f"invalid rules from okx: {row}")
    return {
        "instId": inst_id,
        "tick_size": tick,
        "step_size": lot,
        "min_qty": min_sz,
        "raw": row,
    }


def fetch_okx_1m_candles(
    inst_id: str,
    start_ms: int,
    end_ms: int,
    *,
    proxy_url: str,
    limit: int = 100,
    req_sleep_sec: float = 0.03,
    timeout_sec: float = 15.0,
    max_retries: int = 5,
) -> pd.DataFrame:
    url = "https://www.okx.com/api/v5/market/history-candles"
    proxies = _proxy_dict(proxy_url)
    s = _session()
    rows: list[dict] = []
    seen = set()

    after = None
    while True:
        params = {"instId": inst_id, "bar": "1m", "limit": str(max(1, min(int(limit), 300)))}
        if after is not None:
            params["after"] = str(after)

        resp = None
        last_err = None
        for i in range(1, max_retries + 1):
            try:
                resp = s.get(url, params=params, proxies=proxies, timeout=timeout_sec)
                resp.raise_for_status()
                break
            except Exception as e:
                last_err = e
                time.sleep(min(1.5 * i, 8.0))
        if resp is None:
            raise RuntimeError(f"okx history-candles failed: {last_err}")

        j = resp.json()
        if str(j.get("code")) != "0":
            raise RuntimeError(f"okx history-candles error: {j}")

        data = j.get("data") or []
        if not data:
            break

        min_ts = None
        max_ts = None
        for d in data:
            ts = int(d[0])
            if ts in seen:
                continue
            seen.add(ts)
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts
            rows.append(
                {
                    "open_time": ts,
                    "open": float(d[1]),
                    "high": float(d[2]),
                    "low": float(d[3]),
                    "close": float(d[4]),
                    "volume": float(d[5]),
                    "close_time": ts + 59_999,
                }
            )

        if min_ts is None:
            break
        if min_ts <= int(start_ms):
            break
        if after is not None and min_ts >= int(after):
            break
        after = int(min_ts)
        if req_sleep_sec > 0:
            time.sleep(req_sleep_sec)

    if not rows:
        raise RuntimeError("okx candles empty")

    out = pd.DataFrame(rows)
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    out = out[(out["open_time"] >= int(start_ms)) & (out["open_time"] < int(end_ms))].copy()
    if out.empty:
        raise RuntimeError("okx candles empty in requested window")
    return out


def run(args) -> dict:
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=float(args.days))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    rules = fetch_okx_rules(args.inst_id, args.proxy_url, timeout_sec=float(args.timeout_sec))
    klines = fetch_okx_1m_candles(
        args.inst_id,
        start_ms,
        end_ms,
        proxy_url=args.proxy_url,
        limit=int(args.page_limit),
        req_sleep_sec=float(args.req_sleep_sec),
        timeout_sec=float(args.timeout_sec),
        max_retries=int(args.max_retries),
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    kline_path = cache_dir / f"okx_{args.inst_id}_1m_{stamp}.pkl"
    rules_path = cache_dir / f"okx_{args.inst_id}_rules_{stamp}.json"
    klines.to_pickle(kline_path)
    rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [
        "python",
        str(Path(args.hc_runner)),
        "--symbol",
        str(args.symbol_for_gp3),
        "--profile",
        str(args.profile),
        "--days",
        str(args.days),
        "--no-agg",
        "--path-mode",
        str(args.path_mode),
        "--klines-file",
        str(kline_path),
        "--tick-size",
        str(rules["tick_size"]),
        "--step-size",
        str(rules["step_size"]),
        "--min-qty",
        str(rules["min_qty"]),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result = {
        "window_start_utc": start_dt.isoformat(),
        "window_end_utc": end_dt.isoformat(),
        "inst_id": args.inst_id,
        "symbol_for_gp3": args.symbol_for_gp3,
        "rows": int(len(klines)),
        "kline_file": str(kline_path),
        "rules_file": str(rules_path),
        "rules": rules,
        "runner_cmd": cmd,
        "runner_exit_code": int(proc.returncode),
        "runner_stdout": proc.stdout,
        "runner_stderr": proc.stderr,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch OKX 1m candles then run GP3.0 on HC model")
    ap.add_argument("--inst-id", default="ETH-USDT-SWAP")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--proxy-url", default="http://127.0.0.1:7890")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--symbol-for-gp3", default="ETHUSDC")
    ap.add_argument("--path-mode", default="oc_aware", choices=["oc_aware", "long_worst", "short_worst"])
    ap.add_argument("--page-limit", type=int, default=100)
    ap.add_argument("--req-sleep-sec", type=float, default=0.03)
    ap.add_argument("--timeout-sec", type=float, default=15.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--cache-dir", default="d:/project/hc/cache")
    ap.add_argument("--hc-runner", default="d:/project/hc/run_gp3_hc_backtest.py")
    ap.add_argument("--out-json", default="d:/project/hc/output/okx_gp3_hc_last_run.json")
    args = ap.parse_args()

    out = run(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"okx fetch + gp3 hc done, exit={out['runner_exit_code']}")
    print(f"inst_id={out['inst_id']} rows={out['rows']}")
    print(f"kline_file={out['kline_file']}")
    print(f"rules_file={out['rules_file']}")
    print(f"runner_cmd={' '.join(out['runner_cmd'])}")
    print("runner_stdout:")
    print(out["runner_stdout"].strip())
    if out["runner_stderr"].strip():
        print("runner_stderr:")
        print(out["runner_stderr"].strip())
    print(f"report_json={out_path}")


if __name__ == "__main__":
    main()
