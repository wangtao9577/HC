from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent
BOLL_ROOT = Path("d:/project/BOLL2.1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc_engine import (  # noqa: E402
    AggTradeFetchSpec,
    BacktestConfig,
    BacktestEngine,
    Bar,
    IntrabarPathModel,
    OrderStatus,
    Side,
    StopTriggerSource,
    TimeInForce,
    aggtrades_to_bars_and_events,
    fetch_binance_futures_aggtrades,
    summarize,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    return int(float(str(raw).strip()))


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    return float(str(raw).strip())


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            os.environ[k] = v


def _parse_utc_timestamp(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = pd.to_datetime(text).to_pydatetime()
        except Exception as exc:
            raise ValueError(f"invalid UTC timestamp: {raw}") from exc
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_time_window(args) -> tuple[datetime, datetime]:
    if str(getattr(args, "start_utc", "") or "").strip() or str(getattr(args, "end_utc", "") or "").strip():
        end_dt = (
            _parse_utc_timestamp(args.end_utc)
            if str(getattr(args, "end_utc", "") or "").strip()
            else datetime.now(timezone.utc).replace(second=0, microsecond=0)
        )
        start_dt = (
            _parse_utc_timestamp(args.start_utc)
            if str(getattr(args, "start_utc", "") or "").strip()
            else end_dt - timedelta(days=float(args.days))
        )
    else:
        end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=float(args.days))
    if end_dt <= start_dt:
        raise RuntimeError(f"invalid backtest window: start={start_dt.isoformat()} end={end_dt.isoformat()}")
    return start_dt, end_dt


def _load_boll_module():
    mod_path = BOLL_ROOT / "backtest_ethusdc.py"
    spec = importlib.util.spec_from_file_location("boll2_backtest_ethusdc", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _rules_cache_path() -> Path:
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "symbol_rules_cache.json"


def _load_symbol_rules_from_cache(boll, symbol: str):
    cpath = _rules_cache_path()
    if not cpath.exists():
        return None
    try:
        data = json.loads(cpath.read_text(encoding="utf-8"))
        rec = data.get(str(symbol).upper(), {}) or {}
        tick = float(rec.get("tick_size") or 0.0)
        step = float(rec.get("step_size") or 0.0)
        min_qty = float(rec.get("min_qty") or 0.0)
        if tick > 0 and step > 0 and min_qty > 0:
            return boll.SymbolRules(tick_size=tick, step_size=step, min_qty=min_qty)
    except Exception:
        return None
    return None


def _save_symbol_rules_to_cache(symbol: str, rules) -> None:
    cpath = _rules_cache_path()
    payload = {}
    if cpath.exists():
        try:
            payload = json.loads(cpath.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload[str(symbol).upper()] = {
        "tick_size": float(getattr(rules, "tick_size")),
        "step_size": float(getattr(rules, "step_size")),
        "min_qty": float(getattr(rules, "min_qty")),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    cpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_symbol_rules_with_cache(boll, symbol: str):
    cached = _load_symbol_rules_from_cache(boll, symbol)
    last_error: Exception | None = None
    for _ in range(3):
        try:
            rules = boll.fetch_exchange_info(symbol)
            _save_symbol_rules_to_cache(symbol, rules)
            return rules
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    if cached is not None:
        return cached
    raise RuntimeError(f"failed to fetch symbol rules for {symbol}: {last_error}") from last_error


def _apply_profile_to_boll_module(boll) -> None:
    boll.SYMBOL = os.getenv("SYMBOL", getattr(boll, "SYMBOL", "ETHUSDC"))
    boll.INITIAL_CAPITAL = _env_float("INITIAL_CAPITAL", float(getattr(boll, "INITIAL_CAPITAL", 100000.0)))

    boll.BOLL_PERIOD = _env_int("BOLL_PERIOD", int(getattr(boll, "BOLL_PERIOD", 28)))
    boll.BOLL_STD = _env_float("BOLL_STD", float(getattr(boll, "BOLL_STD", 2.8)))
    boll.MACD_FAST = _env_int("MACD_FAST", int(getattr(boll, "MACD_FAST", 16)))
    boll.MACD_SLOW = _env_int("MACD_SLOW", int(getattr(boll, "MACD_SLOW", 34)))
    boll.MACD_SIGNAL = _env_int("MACD_SIGNAL", int(getattr(boll, "MACD_SIGNAL", 9)))

    boll.BANDWIDTH_MIN = _env_float("BANDWIDTH_MIN", float(getattr(boll, "BANDWIDTH_MIN", 0.003)))
    boll.HIST_ABS_MIN = _env_float("HIST_ABS_MIN", float(getattr(boll, "HIST_ABS_MIN", 0.05)))
    boll.MACD_NEAR_ZERO_MAX = _env_float("MACD_NEAR_ZERO_MAX", float(getattr(boll, "MACD_NEAR_ZERO_MAX", 8.0)))
    boll.SLOPE_LOOKBACK = _env_int("SLOPE_LOOKBACK", int(getattr(boll, "SLOPE_LOOKBACK", 1)))
    boll.SLOPE_LONG_MIN = _env_float("SLOPE_LONG_MIN", float(getattr(boll, "SLOPE_LONG_MIN", -1.2)))
    boll.SLOPE_SHORT_MAX = _env_float("SLOPE_SHORT_MAX", float(getattr(boll, "SLOPE_SHORT_MAX", 0.6)))

    boll.ENTRY_LIMIT_MULT = _env_float("ENTRY_LIMIT_MULT", float(getattr(boll, "ENTRY_LIMIT_MULT", 1.0)))
    order_timeout_sec = _env_float("ORDER_TIMEOUT_SEC", float(getattr(boll, "ORDER_TIMEOUT_MS", 900_000) / 1000.0))
    boll.ORDER_TIMEOUT_MS = int(order_timeout_sec * 1000)
    boll.STOP_OFFSET_PCT = _env_float("STOP_OFFSET_PCT", float(getattr(boll, "STOP_OFFSET_PCT", 0.02)))
    boll.STOP_CAP_PCT = _env_float("STOP_CAP_PCT", float(getattr(boll, "STOP_CAP_PCT", 0.05)))
    boll.BREAKEVEN_TRIGGER_PCT = _env_float("BREAKEVEN_TRIGGER_PCT", float(getattr(boll, "BREAKEVEN_TRIGGER_PCT", 0.014)))
    boll.BREAKEVEN_OFFSET_PCT = _env_float("BREAKEVEN_OFFSET_PCT", float(getattr(boll, "BREAKEVEN_OFFSET_PCT", 0.0003)))
    boll.TP_CONFIRM_BARS = _env_int("TP_CONFIRM_BARS", int(getattr(boll, "TP_CONFIRM_BARS", 1)))
    boll.POSITION_MULT = _env_float("POSITION_MULT", float(getattr(boll, "POSITION_MULT", 2.45)))
    boll.SIGNAL_EXEC_DELAY_SEC = _env_float("SIGNAL_EXEC_DELAY_SEC", float(getattr(boll, "SIGNAL_EXEC_DELAY_SEC", 5.0)))
    boll.MAKER_FILL_BUFFER_TICKS = _env_int("MAKER_FILL_BUFFER_TICKS", int(getattr(boll, "MAKER_FILL_BUFFER_TICKS", 1)))
    boll.STOP_SLIPPAGE_BPS = _env_float("STOP_SLIPPAGE_BPS", float(getattr(boll, "STOP_SLIPPAGE_BPS", 2.0)))
    boll.MAKER_FEE = _env_float("MAKER_FEE", float(getattr(boll, "MAKER_FEE", 0.0)))
    boll.TAKER_FEE = _env_float("TAKER_FEE", float(getattr(boll, "TAKER_FEE", 0.0004)))


def _build_http_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _klines_cache_path(cache_dir: Path, symbol: str, interval: str, start_ms: int, end_ms: int) -> Path:
    return cache_dir / f"klines_{str(symbol).upper()}_{interval}_{int(start_ms)}_{int(end_ms)}.pkl"


def _parse_ban_until_ms(text: str) -> Optional[int]:
    raw = str(text or "")
    m = re.search(r"banned until\s+(\d{10,14})", raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _fetch_binance_klines_chunked(symbol: str, interval: str, start_ms: int, end_ms: int, args) -> pd.DataFrame:
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = _klines_cache_path(cache_dir, symbol, interval, start_ms, end_ms)
    if bool(args.binance_klines_use_cache) and cpath.exists():
        return pd.read_pickle(cpath)

    base_urls = [u.strip().rstrip("/") for u in str(getattr(args, "binance_fapi_base_urls", "")).split(",") if u.strip()]
    if not base_urls:
        base_urls = ["https://fapi.binance.com"]
    endpoint = "/fapi/v1/klines"
    limit = max(1, min(int(args.binance_klines_limit), 1500))
    timeout_sec = float(args.binance_klines_timeout_sec)
    page_sleep_sec = max(0.0, float(args.binance_klines_page_sleep_sec))
    max_418_retries = max(1, int(args.binance_klines_max_418_retries))
    ban_cooldown_sec = max(1.0, float(args.binance_klines_ban_cooldown_sec))
    max_conn_retries = max(0, int(args.binance_klines_max_conn_retries))
    conn_backoff_sec = max(0.2, float(args.binance_klines_conn_backoff_sec))

    sess = _build_http_session()
    rows: list = []
    cursor = int(start_ms)
    step_ms = 60_000 if interval == "1m" else 3_600_000
    consecutive_418 = 0
    while cursor < int(end_ms):
        params = {
            "symbol": str(symbol).upper(),
            "interval": interval,
            "limit": limit,
            "startTime": int(cursor),
            "endTime": int(end_ms),
        }
        resp = None
        last_err: Optional[Exception] = None
        for conn_try in range(max_conn_retries + 1):
            base = base_urls[conn_try % len(base_urls)]
            req_url = f"{base}{endpoint}"
            try:
                resp = sess.get(req_url, params=params, timeout=timeout_sec)
                break
            except requests.RequestException as e:
                last_err = e
                if conn_try >= max_conn_retries:
                    raise RuntimeError(
                        f"binance klines connection failed after {max_conn_retries + 1} tries, last_url={req_url}: {e}"
                    ) from e
                time.sleep(min(30.0, conn_backoff_sec * (1.8 ** conn_try)))
        if resp is None:
            raise RuntimeError(f"binance klines request returned no response: {last_err}")

        if resp.status_code in {418, 429}:
            consecutive_418 += 1
            if consecutive_418 > max_418_retries:
                raise RuntimeError(
                    f"binance klines throttle exceeded: status={resp.status_code} retries>{max_418_retries}"
                )
            retry_after = float(resp.headers.get("Retry-After", "0") or 0)
            ban_until_ms = _parse_ban_until_ms(resp.text)
            wait_sec = max(retry_after, ban_cooldown_sec)
            if ban_until_ms:
                now_ms = int(time.time() * 1000)
                wait_sec = max(wait_sec, (ban_until_ms - now_ms) / 1000.0 + 1.0)
            time.sleep(max(1.0, wait_sec))
            continue

        consecutive_418 = 0
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)

        last_open = int(batch[-1][0])
        nxt = int(last_open + step_ms)
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < limit:
            break
        if page_sleep_sec > 0:
            time.sleep(page_sleep_sec)

    if not rows:
        raise RuntimeError(f"binance {interval} klines fetch returned empty rows")

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    for c in ["open_time", "close_time"]:
        df[c] = df[c].astype("int64")
    df = df[(df["open_time"] >= int(start_ms)) & (df["close_time"] < int(end_ms))]
    out = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"binance {interval} klines window is empty after filtering")
    if bool(args.binance_klines_use_cache):
        out.to_pickle(cpath)
    return out


def _fetch_binance_mark_klines_chunked(symbol: str, start_ms: int, end_ms: int, args) -> pd.DataFrame:
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"mark_klines_{str(symbol).upper()}_{int(start_ms)}_{int(end_ms)}_1m.pkl"
    if bool(args.binance_klines_use_cache) and cpath.exists():
        return pd.read_pickle(cpath)

    base_urls = [u.strip().rstrip("/") for u in str(getattr(args, "binance_fapi_base_urls", "")).split(",") if u.strip()]
    if not base_urls:
        base_urls = ["https://fapi.binance.com"]
    endpoints = ["/fapi/v1/markPriceKlines", "/fapi/v1/premiumIndexKlines"]
    limit = max(1, min(int(args.binance_klines_limit), 1500))
    timeout_sec = float(args.binance_klines_timeout_sec)
    page_sleep_sec = max(0.0, float(args.binance_klines_page_sleep_sec))
    max_418_retries = max(1, int(args.binance_klines_max_418_retries))
    ban_cooldown_sec = max(1.0, float(args.binance_klines_ban_cooldown_sec))
    max_conn_retries = max(0, int(args.binance_klines_max_conn_retries))
    conn_backoff_sec = max(0.2, float(args.binance_klines_conn_backoff_sec))

    sess = _build_http_session()
    last_error: Optional[Exception] = None
    for endpoint in endpoints:
        rows: list = []
        cursor = int(start_ms)
        consecutive_418 = 0
        try:
            while cursor < int(end_ms):
                params = {
                    "symbol": str(symbol).upper(),
                    "interval": "1m",
                    "limit": limit,
                    "startTime": int(cursor),
                    "endTime": int(end_ms),
                }
                resp = None
                req_err: Optional[Exception] = None
                for conn_try in range(max_conn_retries + 1):
                    base = base_urls[conn_try % len(base_urls)]
                    req_url = f"{base}{endpoint}"
                    try:
                        resp = sess.get(req_url, params=params, timeout=timeout_sec)
                        break
                    except requests.RequestException as e:
                        req_err = e
                        if conn_try >= max_conn_retries:
                            raise RuntimeError(
                                f"binance mark klines connection failed after {max_conn_retries + 1} tries, last_url={req_url}: {e}"
                            ) from e
                        time.sleep(min(30.0, conn_backoff_sec * (1.8 ** conn_try)))
                if resp is None:
                    raise RuntimeError(f"binance mark klines request returned no response: {req_err}")
                if resp.status_code in {418, 429}:
                    consecutive_418 += 1
                    if consecutive_418 > max_418_retries:
                        raise RuntimeError(
                            f"binance mark klines throttle exceeded: status={resp.status_code} retries>{max_418_retries}"
                        )
                    retry_after = float(resp.headers.get("Retry-After", "0") or 0)
                    ban_until_ms = _parse_ban_until_ms(resp.text)
                    wait_sec = max(retry_after, ban_cooldown_sec)
                    if ban_until_ms:
                        now_ms = int(time.time() * 1000)
                        wait_sec = max(wait_sec, (ban_until_ms - now_ms) / 1000.0 + 1.0)
                    time.sleep(max(1.0, wait_sec))
                    continue

                consecutive_418 = 0
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                rows.extend(batch)
                last_open = int(batch[-1][0])
                nxt = int(last_open + 60_000)
                if nxt <= cursor:
                    break
                cursor = nxt
                if len(batch) < limit:
                    break
                if page_sleep_sec > 0:
                    time.sleep(page_sleep_sec)

            if not rows:
                continue
            cols = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "ignore1",
                "close_time",
                "ignore2",
                "ignore3",
                "ignore4",
                "ignore5",
                "ignore6",
            ]
            df = pd.DataFrame(rows, columns=cols)
            df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
            for c in ["open", "high", "low", "close"]:
                df[c] = df[c].astype(float)
            for c in ["open_time", "close_time"]:
                df[c] = df[c].astype("int64")
            out = df[["open_time", "open", "high", "low", "close", "close_time"]].copy()
            out["volume"] = 0.0
            out = out[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
            out = out[(out["open_time"] >= int(start_ms)) & (out["close_time"] < int(end_ms))].reset_index(drop=True)
            if out.empty:
                continue
            if bool(args.binance_klines_use_cache):
                out.to_pickle(cpath)
            return out
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"unable to fetch mark klines for {symbol}: {last_error}")


def _path_interp_price(path_mode: str, o: float, h: float, l: float, c: float, frac: float) -> float:
    model = IntrabarPathModel(mode=str(path_mode))
    seq = [float(x) for x in model._price_sequence(o=float(o), h=float(h), l=float(l), c=float(c))]
    offsets = [0.0, 0.25, 0.75, 0.999]
    f = min(max(float(frac), 0.0), 0.999)
    if f <= offsets[0]:
        return seq[0]
    for i in range(1, len(seq)):
        left_t = offsets[i - 1]
        right_t = offsets[min(i, len(offsets) - 1)]
        if f <= right_t:
            left_px = seq[i - 1]
            right_px = seq[i]
            span = max(right_t - left_t, 1e-9)
            ratio = (f - left_t) / span
            return float(left_px + (right_px - left_px) * ratio)
    return float(seq[-1])


def _apply_mark_prices_to_bars_and_events(
    bars: list[Bar],
    events_by_bar: dict[int, list],
    mark_df: Optional[pd.DataFrame],
    *,
    path_mode: str,
) -> None:
    if mark_df is None or mark_df.empty:
        return
    mark_map = {int(r.open_time): r for r in mark_df.itertuples(index=False)}
    for bar in bars:
        mr = mark_map.get(int(bar.open_time_ms))
        if mr is None:
            continue
        bar.mark_close = float(mr.close)
        evs = events_by_bar.get(int(bar.open_time_ms), [])
        dt = max(int(bar.close_time_ms) - int(bar.open_time_ms), 1)
        for ev in evs:
            frac = (int(ev.ts_ms) - int(bar.open_time_ms)) / dt
            ev.mark_price = _path_interp_price(
                path_mode,
                float(mr.open),
                float(mr.high),
                float(mr.low),
                float(mr.close),
                frac,
            )


def _build_inferred_events_with_mark(bars: list[Bar], mark_df: Optional[pd.DataFrame], *, path_mode: str) -> dict[int, list]:
    model = IntrabarPathModel(mode=str(path_mode))
    events_by_bar = {int(bar.open_time_ms): model.events_from_bar(bar) for bar in bars}
    _apply_mark_prices_to_bars_and_events(bars, events_by_bar, mark_df, path_mode=path_mode)
    return events_by_bar


def _fetch_agg_chunked(symbol: str, start_ms: int, end_ms: int, args) -> pd.DataFrame:
    s_ms = int(start_ms)
    e_ms = int(end_ms)
    if e_ms <= s_ms:
        return pd.DataFrame(columns=["agg_id", "price", "qty", "ts_ms", "is_buyer_maker"])

    chunk_min = max(1.0, float(args.agg_chunk_minutes))
    chunk_ms = int(chunk_min * 60_000)
    pause_sec = max(0.0, float(args.agg_chunk_pause_sec))
    out_parts: list[pd.DataFrame] = []

    cur = s_ms
    while cur < e_ms:
        nxt = min(cur + chunk_ms, e_ms)
        part = fetch_binance_futures_aggtrades(
            AggTradeFetchSpec(
                symbol=str(symbol).upper(),
                start_ms=int(cur),
                end_ms=int(nxt),
                limit=1000,
                sleep_sec=float(args.agg_req_sleep_sec),
                timeout_sec=float(args.agg_timeout_sec),
                max_418_retries=int(args.agg_max_418_retries),
                ban_cooldown_sec=float(args.agg_ban_cooldown_sec),
            ),
            use_cache=True,
            cache_dir=ROOT / "cache",
        )
        if part is not None and len(part):
            out_parts.append(part)
        cur = nxt
        if pause_sec > 0 and cur < e_ms:
            time.sleep(pause_sec)

    if not out_parts:
        return pd.DataFrame(columns=["agg_id", "price", "qty", "ts_ms", "is_buyer_maker"])
    out = pd.concat(out_parts, ignore_index=True)
    return out.sort_values("ts_ms").drop_duplicates(subset=["agg_id"]).reset_index(drop=True)


def _extract_round_trips(fills: list[dict]) -> pd.DataFrame:
    rows = []
    pos_qty = 0.0
    entry_px = 0.0
    entry_ts = 0
    cum_fee_open = 0.0
    for f in fills:
        side_raw = f.get("side")
        if hasattr(side_raw, "value"):
            side = str(getattr(side_raw, "value")).upper()
        else:
            side = str(side_raw).upper()
            if "." in side:
                side = side.split(".")[-1]
        qty = float(f.get("qty") or 0.0)
        px = float(f.get("price") or 0.0)
        ts = int(f.get("ts_ms") or 0)
        fee = float(f.get("fee") or 0.0)
        signed = qty if side == "BUY" else -qty
        cur = pos_qty
        if abs(cur) < 1e-12:
            pos_qty = signed
            entry_px = px
            entry_ts = ts
            cum_fee_open = fee
            continue
        if (cur > 0 and signed > 0) or (cur < 0 and signed < 0):
            tot = abs(cur) + abs(signed)
            entry_px = (abs(cur) * entry_px + abs(signed) * px) / max(tot, 1e-12)
            pos_qty = cur + signed
            cum_fee_open += fee
            continue
        close_qty = min(abs(cur), abs(signed))
        if close_qty > 0:
            if cur > 0:
                pnl = (px - entry_px) * close_qty - (cum_fee_open * (close_qty / max(abs(cur), 1e-12)) + fee)
                s = "LONG"
            else:
                pnl = (entry_px - px) * close_qty - (cum_fee_open * (close_qty / max(abs(cur), 1e-12)) + fee)
                s = "SHORT"
            rows.append(
                {
                    "entry_ts_ms": int(entry_ts),
                    "exit_ts_ms": int(ts),
                    "side": s,
                    "entry_price": float(entry_px),
                    "exit_price": float(px),
                    "qty": float(close_qty),
                    "pnl": float(pnl),
                    "fee_total": float(cum_fee_open * (close_qty / max(abs(cur), 1e-12)) + fee),
                }
            )
        remain = abs(signed) - close_qty
        if remain <= 1e-12:
            pos_qty = cur + signed
            if abs(pos_qty) < 1e-12:
                pos_qty = 0.0
                entry_px = 0.0
                entry_ts = 0
                cum_fee_open = 0.0
            else:
                cum_fee_open = 0.0
        else:
            pos_qty = math.copysign(remain, signed)
            entry_px = px
            entry_ts = ts
            cum_fee_open = fee
    return pd.DataFrame(rows)


class Boll2HcStrategy:
    def __init__(self, bars_df: pd.DataFrame, entry_events: dict[int, dict], exit_events: dict[int, set], rules, boll_mod):
        self.df = bars_df.reset_index(drop=True).copy()
        self.idx_by_open = {int(v): int(i) for i, v in self.df["open_time"].items()}
        self.cur_idx = -1
        self.entry_events = entry_events
        self.exit_events = exit_events
        self.rules = rules
        self.b = boll_mod

        self.pending_entry: Optional[dict] = None
        self.pending_exit: Optional[dict] = None
        self.stop_order_id: Optional[str] = None
        self.be_triggered = False
        self._seen_fills = 0
        self.entry_exec_mode = str(getattr(self.b, "ENTRY_EXEC_MODE", "baseline") or "baseline").strip().lower()
        self.rejected_entries = 0
        self.repriced_entries = 0
        self.market_fallback_entries = 0
        self.direct_taker_entries = 0
        self._probe_cache: dict[int, Optional[list[dict]]] = {}

    def on_bar_open(self, engine: BacktestEngine, bar: Bar) -> None:
        self.cur_idx = self.idx_by_open.get(int(bar.open_time_ms), -1)
        self._sync_fills(engine)

        if self.pending_entry and int(bar.open_time_ms) >= int(self.pending_entry["expire_ts_ms"]):
            engine.cancel_order(str(self.pending_entry["order_id"]))
            self.pending_entry = None

        qty, _ = engine.get_position()
        if self.pending_exit and int(bar.open_time_ms) >= int(self.pending_exit["expire_ts_ms"]):
            engine.cancel_order(str(self.pending_exit["order_id"]))
            self.pending_exit = None
            if qty != 0.0:
                self._place_exit_limit(engine, ts_ms=int(bar.open_time_ms), px_open=float(bar.open), qty=abs(float(qty)))

        if qty == 0.0 and self.pending_entry is None and self.pending_exit is None:
            ev = self.entry_events.get(int(bar.open_time_ms))
            if ev:
                signal_side = str(ev["side"]).upper()
                if signal_side == "LONG":
                    px = self.b.floor_to_step(float(bar.open) / float(self.b.ENTRY_LIMIT_MULT), float(self.rules.tick_size))
                    side = Side.BUY
                else:
                    px = self.b.ceil_to_step(float(bar.open) * float(self.b.ENTRY_LIMIT_MULT), float(self.rules.tick_size))
                    side = Side.SELL
                cash = float(engine.state.cash)
                qty_v = self.b.floor_to_step((cash * float(self.b.POSITION_MULT)) / max(px, 1e-12), float(self.rules.step_size))
                if float(qty_v) >= float(self.rules.min_qty):
                    self._submit_entry(
                        engine,
                        ts_ms=int(bar.open_time_ms),
                        signal_side=signal_side,
                        signal_basis=float(ev["basis"]),
                        side=side,
                        qty=float(qty_v),
                        base_px=float(px),
                    )

        qty, _ = engine.get_position()
        if qty != 0.0 and self.pending_exit is None and self.pending_entry is None:
            side_name = "LONG" if qty > 0 else "SHORT"
            ev = self.exit_events.get(int(bar.open_time_ms), set())
            if side_name in ev:
                self._place_exit_limit(engine, ts_ms=int(bar.open_time_ms), px_open=float(bar.open), qty=abs(float(qty)))

    def _place_exit_limit(self, engine: BacktestEngine, *, ts_ms: int, px_open: float, qty: float) -> None:
        pos_qty, _ = engine.get_position()
        if pos_qty > 0:
            side = Side.SELL
            px = self.b.ceil_to_step(float(px_open) * float(self.b.ENTRY_LIMIT_MULT), float(self.rules.tick_size))
        else:
            side = Side.BUY
            px = self.b.floor_to_step(float(px_open) / float(self.b.ENTRY_LIMIT_MULT), float(self.rules.tick_size))
        oid = engine.place_limit(
            side=side,
            qty=float(qty),
            limit_price=float(px),
            ts_ms=int(ts_ms),
            tif=TimeInForce.GTC,
            reduce_only=True,
            reason="boll_exit_signal",
        )
        self.pending_exit = {"order_id": oid, "expire_ts_ms": int(ts_ms) + int(self.b.ORDER_TIMEOUT_MS)}

    def _submit_entry(
        self,
        engine: BacktestEngine,
        *,
        ts_ms: int,
        signal_side: str,
        signal_basis: float,
        side: Side,
        qty: float,
        base_px: float,
    ) -> None:
        mode = self.entry_exec_mode
        if mode == "always_taker":
            self._place_entry_market(
                engine,
                ts_ms=ts_ms,
                signal_side=signal_side,
                signal_basis=signal_basis,
                side=side,
                qty=qty,
                reason=f"boll_entry_{signal_side.lower()}_direct_taker",
            )
            self.direct_taker_entries += 1
            return

        probe_rows = self._load_probe_rows(ts_ms) if mode != "baseline" else None
        reject_row = None
        order_side = str(side.value)
        if probe_rows:
            reject_row = self.b.detect_post_only_reject(probe_rows, order_side, float(base_px), float(self.rules.tick_size))

        if reject_row is None:
            self._place_entry_limit(
                engine,
                ts_ms=ts_ms,
                signal_side=signal_side,
                signal_basis=signal_basis,
                side=side,
                qty=qty,
                price=float(base_px),
                tif=TimeInForce.GTX,
            )
            return

        self.rejected_entries += 1
        if mode == "reject_skip":
            return

        if mode == "reject_to_taker":
            self._place_entry_market(
                engine,
                ts_ms=ts_ms,
                signal_side=signal_side,
                signal_basis=signal_basis,
                side=side,
                qty=qty,
                reason=f"boll_entry_{signal_side.lower()}_fallback_taker",
            )
            self.market_fallback_entries += 1
            return

        if mode != "maker_reprice_2ticks_2x_then_taker":
            raise RuntimeError(f"unsupported ENTRY_EXEC_MODE={mode}")

        max_attempts = max(int(getattr(self.b, "POST_ONLY_MAX_REPRICE_ATTEMPTS", 2) or 2), 0)
        for attempt in range(1, max_attempts + 1):
            reprice_px = float(
                self.b.repriced_entry_price(order_side, float(base_px), float(self.rules.tick_size), int(attempt))
            )
            reprice_match = None
            if probe_rows:
                reprice_match = self.b.detect_post_only_reject(
                    probe_rows,
                    order_side,
                    float(reprice_px),
                    float(self.rules.tick_size),
                )
            if reprice_match is None:
                self._place_entry_limit(
                    engine,
                    ts_ms=ts_ms,
                    signal_side=signal_side,
                    signal_basis=signal_basis,
                    side=side,
                    qty=qty,
                    price=float(reprice_px),
                    tif=TimeInForce.GTX,
                    fallback_kind="reprice",
                    reprice_attempt=int(attempt),
                )
                self.repriced_entries += 1
                return

        self._place_entry_market(
            engine,
            ts_ms=ts_ms,
            signal_side=signal_side,
            signal_basis=signal_basis,
            side=side,
            qty=qty,
            reason=f"boll_entry_{signal_side.lower()}_fallback_taker",
            fallback_kind="market",
            reprice_attempt=max(max_attempts, 1),
        )
        self.market_fallback_entries += 1

    def _place_entry_limit(
        self,
        engine: BacktestEngine,
        *,
        ts_ms: int,
        signal_side: str,
        signal_basis: float,
        side: Side,
        qty: float,
        price: float,
        tif: TimeInForce,
        fallback_kind: str = "",
        reprice_attempt: int = 0,
    ) -> None:
        oid = engine.place_limit(
            side=side,
            qty=float(qty),
            limit_price=float(price),
            ts_ms=int(ts_ms),
            tif=tif,
            reduce_only=False,
            reason=f"boll_entry_{signal_side.lower()}",
        )
        self.pending_entry = {
            "order_id": oid,
            "signal_side": signal_side,
            "signal_basis": float(signal_basis),
            "expire_ts_ms": int(ts_ms) + int(self.b.ORDER_TIMEOUT_MS),
            "fallback_kind": str(fallback_kind or ""),
            "reprice_attempt": int(reprice_attempt),
        }

    def _place_entry_market(
        self,
        engine: BacktestEngine,
        *,
        ts_ms: int,
        signal_side: str,
        signal_basis: float,
        side: Side,
        qty: float,
        reason: str,
        fallback_kind: str = "",
        reprice_attempt: int = 0,
    ) -> None:
        # Submit 1ms before the bar-open event so the first intrabar event can execute as taker.
        submit_ts = max(int(ts_ms) - 1, 0)
        oid = engine.place_market(
            side=side,
            qty=float(qty),
            ts_ms=submit_ts,
            reduce_only=False,
            reason=str(reason),
        )
        self.pending_entry = {
            "order_id": oid,
            "signal_side": signal_side,
            "signal_basis": float(signal_basis),
            "expire_ts_ms": int(ts_ms) + int(self.b.ORDER_TIMEOUT_MS),
            "fallback_kind": str(fallback_kind or "direct_taker"),
            "reprice_attempt": int(reprice_attempt),
        }

    def _load_probe_rows(self, ts_ms: int) -> Optional[list[dict]]:
        cache_key = int(ts_ms)
        if cache_key in self._probe_cache:
            return self._probe_cache[cache_key]
        try:
            rows = self.b.load_probe_rows(str(getattr(self.b, "SYMBOL", "ETHUSDC")).upper(), int(ts_ms))
            self._probe_cache[cache_key] = rows
            return rows
        except Exception:
            self._probe_cache[cache_key] = None
            return None

    def on_price_event(self, engine: BacktestEngine, event) -> None:
        self._sync_fills(engine)
        qty, _ = engine.get_position()
        if qty == 0.0:
            return
        if not self._is_order_active(engine, self.stop_order_id):
            close_side = Side.SELL if qty > 0 else Side.BUY
            engine.place_market(
                side=close_side,
                qty=abs(float(qty)),
                ts_ms=int(event.ts_ms),
                reduce_only=True,
                reason="protection_invalid",
            )
            self._close_cleanup(engine)

    def on_bar_close(self, engine: BacktestEngine, bar: Bar) -> None:
        self._sync_fills(engine)
        qty, entry = engine.get_position()
        if qty == 0.0 or not self._is_order_active(engine, self.stop_order_id):
            return

        new_stop = None
        if qty > 0:
            if (not self.be_triggered) and float(bar.high) >= float(entry) * (1.0 + float(self.b.BREAKEVEN_TRIGGER_PCT)):
                be = float(entry) * (1.0 + float(self.b.BREAKEVEN_OFFSET_PCT))
                o = engine.orders.get(str(self.stop_order_id))
                old = float(o.stop_price) if o and o.stop_price is not None else -1e18
                if be > old:
                    new_stop = be
                self.be_triggered = True
        else:
            if (not self.be_triggered) and float(bar.low) <= float(entry) * (1.0 - float(self.b.BREAKEVEN_TRIGGER_PCT)):
                be = float(entry) * (1.0 - float(self.b.BREAKEVEN_OFFSET_PCT))
                o = engine.orders.get(str(self.stop_order_id))
                old = float(o.stop_price) if o and o.stop_price is not None else 1e18
                if be < old:
                    new_stop = be
                self.be_triggered = True

        if new_stop is None:
            return
        engine.cancel_order(str(self.stop_order_id))
        if qty > 0:
            stop_px = self.b.floor_to_step(float(new_stop), float(self.rules.tick_size))
            stop_side = Side.SELL
        else:
            stop_px = self.b.ceil_to_step(float(new_stop), float(self.rules.tick_size))
            stop_side = Side.BUY
        self.stop_order_id = engine.place_stop_market(
            side=stop_side,
            qty=abs(float(qty)),
            stop_price=float(stop_px),
            ts_ms=int(bar.close_time_ms),
            reduce_only=True,
            reason="boll_stop_be",
        )

    def _sync_fills(self, engine: BacktestEngine) -> None:
        fills = engine.state.fills
        if self._seen_fills >= len(fills):
            return
        for f in fills[self._seen_fills :]:
            oid = str(f.order_id)

            if self.pending_entry and oid == str(self.pending_entry.get("order_id")):
                o = engine.orders.get(oid)
                if o and o.status == OrderStatus.PARTIALLY_FILLED:
                    engine.cancel_order(oid)
                qty, entry = engine.get_position()
                if qty != 0.0:
                    side = str(self.pending_entry.get("signal_side", "LONG")).upper()
                    basis = float(self.pending_entry.get("signal_basis") or 0.0)
                    stop = float(self.b.calc_initial_stop(side, basis, float(entry)))
                    if qty > 0:
                        stop_px = self.b.floor_to_step(stop, float(self.rules.tick_size))
                        stop_side = Side.SELL
                    else:
                        stop_px = self.b.ceil_to_step(stop, float(self.rules.tick_size))
                        stop_side = Side.BUY
                    self.stop_order_id = engine.place_stop_market(
                        side=stop_side,
                        qty=abs(float(qty)),
                        stop_price=float(stop_px),
                        ts_ms=int(f.ts_ms),
                        reduce_only=True,
                        reason="boll_stop",
                    )
                    self.be_triggered = False
                self.pending_entry = None
                continue

            if self.pending_exit and oid == str(self.pending_exit.get("order_id")):
                self._close_cleanup(engine)
                continue

            if oid == str(self.stop_order_id):
                self._close_cleanup(engine)
                continue

        self._seen_fills = len(fills)

    def _close_cleanup(self, engine: BacktestEngine) -> None:
        if self.stop_order_id:
            engine.cancel_order(str(self.stop_order_id))
        if self.pending_exit:
            engine.cancel_order(str(self.pending_exit.get("order_id")))
        self.pending_entry = None
        self.pending_exit = None
        self.stop_order_id = None
        self.be_triggered = False

    @staticmethod
    def _is_order_active(engine: BacktestEngine, order_id: Optional[str]) -> bool:
        if not order_id:
            return False
        o = engine.orders.get(str(order_id))
        if not o:
            return False
        return o.status in (OrderStatus.NEW, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED)


def run(args) -> dict:
    _load_env_file(Path("d:/project/.env"))
    _load_env_file(Path(args.profile))
    initial_capital_override = getattr(args, "initial_capital_override", None)
    prev_initial_capital = os.environ.get("INITIAL_CAPITAL")
    if initial_capital_override is not None:
        os.environ["INITIAL_CAPITAL"] = str(float(initial_capital_override))
    boll = _load_boll_module()
    _apply_profile_to_boll_module(boll)

    start_dt, end_dt = _resolve_time_window(args)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    try:
        symbol = str(getattr(boll, "SYMBOL", "ETHUSDC")).upper()
        df1h = _fetch_binance_klines_chunked(symbol, "1h", start_ms, end_ms, args)
        df1m = _fetch_binance_klines_chunked(symbol, "1m", start_ms, end_ms, args)
        mark_df = _fetch_binance_mark_klines_chunked(symbol, start_ms, end_ms, args) if bool(args.use_real_mark) else None

        rules = _fetch_symbol_rules_with_cache(boll, symbol)
        df1h_ind = boll.add_indicators_1h(df1h)
        entry_events, exit_events = boll.build_events(df1h_ind)

        bars = [
            Bar(
                open_time_ms=int(r.open_time),
                close_time_ms=int(r.close_time),
                open=float(r.open),
                high=float(r.high),
                low=float(r.low),
                close=float(r.close),
                volume=float(r.volume),
            )
            for r in df1m.itertuples(index=False)
        ]

        events_by_bar = None
        if bool(args.use_agg):
            s_ms = int(df1m["open_time"].iloc[0])
            e_ms = int(df1m["close_time"].iloc[-1]) + 1
            agg_df = _fetch_agg_chunked(symbol, s_ms, e_ms, args)
            _, events_by_bar = aggtrades_to_bars_and_events(
                agg_df,
                start_ms=s_ms,
                end_ms=e_ms,
                bar_ms=60_000,
                fill_empty_bars=False,
            )
            _apply_mark_prices_to_bars_and_events(bars, events_by_bar, mark_df, path_mode=str(args.path_mode))
        else:
            events_by_bar = _build_inferred_events_with_mark(bars, mark_df, path_mode=str(args.path_mode))

        engine_cfg = BacktestConfig(
            symbol=symbol,
            initial_cash=float(getattr(boll, "INITIAL_CAPITAL", 1000.0)),
            leverage=float(_env_float("LEVERAGE", 1.0)),
            maker_fee_rate=float(getattr(boll, "MAKER_FEE", 0.0)),
            taker_fee_rate=float(getattr(boll, "TAKER_FEE", 0.0004)),
            market_slippage_bps=0.5,
            stop_slippage_bps=float(getattr(boll, "STOP_SLIPPAGE_BPS", 2.0)),
            tick_size=float(rules.tick_size),
            maker_queue_delay_ms=int(args.maker_delay_ms),
            gtc_queue_delay_ms=int(args.gtc_queue_delay_ms),
            maker_buffer_ticks=int(getattr(boll, "MAKER_FILL_BUFFER_TICKS", 1)),
            allow_same_bar_entry_exit=True,
            stop_trigger_source=StopTriggerSource(str(args.stop_trigger_source)),
            allow_marketable_gtc_as_taker=bool(args.allow_marketable_gtc_as_taker),
            partial_fill_enabled=bool(args.partial_fill_enabled),
            partial_fill_ratio=float(args.partial_fill_ratio),
            partial_fill_scope=str(args.partial_fill_scope),
            funding_interval_ms=int(float(args.funding_interval_hours) * 3600 * 1000),
            funding_rate=float(args.funding_rate),
            maintenance_margin_rate=float(args.maintenance_margin_rate),
            maintenance_amount=float(args.maintenance_amount),
            liquidation_fee_rate=float(args.liquidation_fee_rate),
            reject_on_insufficient_margin=bool(args.reject_on_insufficient_margin),
            liquidate_on_margin_breach=bool(args.liquidate_on_margin_breach),
        )
        engine = BacktestEngine(engine_cfg, path_model=IntrabarPathModel(mode=str(args.path_mode)))
        strategy = Boll2HcStrategy(df1m, entry_events, exit_events, rules, boll)

        result = engine.run(bars, strategy, events_by_bar=events_by_bar)
        stats = summarize(result)
        fills_df = pd.DataFrame(result["fills"])
        eq_df = pd.DataFrame(result["equity_curve"])
        rt_df = _extract_round_trips(result["fills"])
        final_pos_qty, final_pos_entry = engine.get_position()

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fills_path = out_dir / f"{symbol}_boll2_hc_fills_{stamp}.csv"
        eq_path = out_dir / f"{symbol}_boll2_hc_equity_{stamp}.csv"
        rt_path = out_dir / f"{symbol}_boll2_hc_roundtrips_{stamp}.csv"
        stats_path = out_dir / f"{symbol}_boll2_hc_stats_{stamp}.json"
        fills_df.to_csv(fills_path, index=False)
        eq_df.to_csv(eq_path, index=False)
        rt_df.to_csv(rt_path, index=False)

        wins = int((rt_df["pnl"] > 0).sum()) if not rt_df.empty else 0
        losses = int((rt_df["pnl"] <= 0).sum()) if not rt_df.empty else 0
        win_rate = (wins / len(rt_df) * 100.0) if len(rt_df) else 0.0
        out_stats = {
            "window_start_utc": start_dt.isoformat(),
            "window_end_utc": end_dt.isoformat(),
            "profile": str(Path(args.profile)),
            "symbol": symbol,
            "use_agg": bool(args.use_agg),
            "path_mode": str(args.path_mode),
            "initial_capital_override": float(initial_capital_override) if initial_capital_override is not None else None,
            "bars_1m": int(len(df1m)),
            "bars_1h": int(len(df1h)),
            "entry_events": int(len(entry_events)),
            "exit_events": int(sum(len(v) for v in exit_events.values())),
            "fills": int(len(fills_df)),
            "round_trips": int(len(rt_df)),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": win_rate,
            "entry_exec_mode": str(getattr(strategy, "entry_exec_mode", "baseline")),
            "rejected_entries": int(getattr(strategy, "rejected_entries", 0)),
            "repriced_entries": int(getattr(strategy, "repriced_entries", 0)),
            "market_fallback_entries": int(getattr(strategy, "market_fallback_entries", 0)),
            "direct_taker_entries": int(getattr(strategy, "direct_taker_entries", 0)),
            "post_only_reprice_ticks": int(max(int(getattr(boll, "POST_ONLY_REPRICE_TICKS", 2) or 2), 1)),
            "post_only_max_reprice_attempts": int(
                max(int(getattr(boll, "POST_ONLY_MAX_REPRICE_ATTEMPTS", 2) or 2), 0)
            ),
            "engine_summary": stats,
            "final_position": {
                "qty": float(final_pos_qty),
                "entry_price": float(final_pos_entry),
            },
            "engine_config": asdict(engine_cfg),
            "outputs": {
                "fills_csv": str(fills_path),
                "equity_csv": str(eq_path),
                "roundtrips_csv": str(rt_path),
                "stats_json": str(stats_path),
            },
        }
        stats_path.write_text(json.dumps(out_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_stats
    finally:
        if initial_capital_override is not None:
            if prev_initial_capital is None:
                os.environ.pop("INITIAL_CAPITAL", None)
            else:
                os.environ["INITIAL_CAPITAL"] = prev_initial_capital


def main() -> None:
    ap = argparse.ArgumentParser(description="Run BOLL2.0plus strategy on HC engine")
    ap.add_argument("--profile", default="d:/project/hc/profiles/boll3_live_latest.env")
    ap.add_argument("--days", type=float, default=365.0)
    ap.add_argument("--start-utc", default="", help="Backtest window start in UTC, e.g. 2025-03-17T10:11:00+00:00")
    ap.add_argument("--end-utc", default="", help="Backtest window end in UTC, e.g. 2025-04-01T00:00:00+00:00")
    ap.add_argument("--initial-capital-override", type=float, default=None, help="Override INITIAL_CAPITAL for segmented compounding runs")
    ap.add_argument("--use-agg", dest="use_agg", action="store_true")
    ap.add_argument("--no-agg", dest="use_agg", action="store_false")
    ap.set_defaults(use_agg=False)

    ap.add_argument("--path-mode", default="oc_aware", choices=["oc_aware", "long_worst", "short_worst"])
    ap.add_argument("--maker-delay-ms", type=int, default=800)
    ap.add_argument("--gtc-queue-delay-ms", type=int, default=300)
    ap.add_argument("--out-dir", default="d:/project/hc/output")

    ap.add_argument("--stop-trigger-source", default="mark", choices=["last", "mark"])
    ap.add_argument("--use-real-mark", dest="use_real_mark", action="store_true")
    ap.add_argument("--no-real-mark", dest="use_real_mark", action="store_false")
    ap.add_argument("--allow-marketable-gtc-as-taker", dest="allow_marketable_gtc_as_taker", action="store_true")
    ap.add_argument("--disallow-marketable-gtc-as-taker", dest="allow_marketable_gtc_as_taker", action="store_false")
    ap.add_argument("--partial-fill-enabled", dest="partial_fill_enabled", action="store_true")
    ap.add_argument("--no-partial-fill", dest="partial_fill_enabled", action="store_false")
    ap.add_argument("--partial-fill-ratio", type=float, default=0.35)
    ap.add_argument("--partial-fill-scope", default="maker", choices=["maker", "all"])
    ap.set_defaults(partial_fill_enabled=True)

    ap.add_argument("--funding-interval-hours", type=float, default=8.0)
    ap.add_argument("--funding-rate", type=float, default=0.0)
    ap.add_argument("--maintenance-margin-rate", type=float, default=0.005)
    ap.add_argument("--maintenance-amount", type=float, default=0.0)
    ap.add_argument("--liquidation-fee-rate", type=float, default=0.0)
    ap.add_argument("--reject-on-insufficient-margin", dest="reject_on_insufficient_margin", action="store_true")
    ap.add_argument("--no-reject-on-insufficient-margin", dest="reject_on_insufficient_margin", action="store_false")
    ap.add_argument("--liquidate-on-margin-breach", dest="liquidate_on_margin_breach", action="store_true")
    ap.add_argument("--no-liquidate-on-margin-breach", dest="liquidate_on_margin_breach", action="store_false")

    ap.add_argument("--agg-chunk-minutes", type=float, default=20.0)
    ap.add_argument("--agg-chunk-pause-sec", type=float, default=1.0)
    ap.add_argument("--agg-req-sleep-sec", type=float, default=0.08)
    ap.add_argument("--agg-timeout-sec", type=float, default=20.0)
    ap.add_argument("--agg-max-418-retries", type=int, default=40)
    ap.add_argument("--agg-ban-cooldown-sec", type=float, default=20.0)

    ap.add_argument("--binance-fapi-base-urls", default="https://fapi.binance.com,https://fstream.binance.com")
    ap.add_argument("--binance-klines-limit", type=int, default=1000)
    ap.add_argument("--binance-klines-page-sleep-sec", type=float, default=0.12)
    ap.add_argument("--binance-klines-timeout-sec", type=float, default=20.0)
    ap.add_argument("--binance-klines-max-conn-retries", type=int, default=8)
    ap.add_argument("--binance-klines-conn-backoff-sec", type=float, default=1.5)
    ap.add_argument("--binance-klines-max-418-retries", type=int, default=30)
    ap.add_argument("--binance-klines-ban-cooldown-sec", type=float, default=20.0)
    ap.add_argument("--binance-klines-use-cache", dest="binance_klines_use_cache", action="store_true")
    ap.add_argument("--no-binance-klines-cache", dest="binance_klines_use_cache", action="store_false")
    ap.set_defaults(
        binance_klines_use_cache=True,
        use_real_mark=True,
        allow_marketable_gtc_as_taker=True,
        reject_on_insufficient_margin=True,
        liquidate_on_margin_breach=True,
    )

    args = ap.parse_args()
    stats = run(args)
    es = stats["engine_summary"]
    print("BOLL2.0plus on HC backtest done:")
    print(f"  symbol: {stats['symbol']}")
    print(f"  period: {stats['window_start_utc']} -> {stats['window_end_utc']}")
    print(f"  use_agg: {stats['use_agg']}, path_mode: {stats['path_mode']}")
    print(f"  entry_exec_mode: {stats['entry_exec_mode']}")
    print(f"  bars_1m: {stats['bars_1m']}, entry_events: {stats['entry_events']}, round_trips: {stats['round_trips']}")
    print(f"  ending_equity: {es.get('ending_equity')}")
    print(f"  net_return_pct: {es.get('net_return_pct')}")
    print(f"  max_drawdown_pct: {es.get('max_drawdown_pct')}")
    print(f"  sharpe_like: {es.get('sharpe_like')}")
    print(f"  calmar_like: {es.get('calmar_like')}")
    print(f"  fee_paid_total: {es.get('fee_paid_total')}")
    print(f"  funding_pnl: {es.get('funding_pnl')}")
    print(f"  win_rate_pct: {stats.get('win_rate_pct')}")
    print(f"  stats_json: {stats['outputs']['stats_json']}")


if __name__ == "__main__":
    main()
