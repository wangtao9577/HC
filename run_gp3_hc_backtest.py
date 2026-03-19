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
GP3_ROOT = Path("d:/project/gp3.0")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hc_engine import (  # noqa: E402
    AggTradeFetchSpec,
    BacktestConfig,
    BacktestEngine,
    Bar,
    ExternalEvent,
    ExternalEventType,
    IntrabarPathModel,
    OrderStatus,
    Side,
    StopTriggerSource,
    TimeInForce,
    aggtrades_to_bars_and_events,
    fetch_binance_futures_aggtrades,
    summarize,
)


def _load_gp3_module():
    candidates = [
        ("gp3_live_bot", GP3_ROOT / "live_gp_bot.py"),
        ("gp3_backtest_hf_ethusdc", GP3_ROOT / "backtest_hf_ethusdc.py"),
    ]
    for name, mod_path in candidates:
        if not mod_path.exists():
            continue
        spec = importlib.util.spec_from_file_location(name, str(mod_path))
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        # dataclasses may inspect sys.modules during class decoration
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    raise RuntimeError(f"cannot load GP3 module from: {candidates}")


def _load_env_compat(gp3, path: Path) -> None:
    p = Path(path)
    if hasattr(gp3, "load_env"):
        gp3.load_env(p)
        return
    if hasattr(gp3, "load_env_file"):
        gp3.load_env_file(p)
        return
    raise RuntimeError("gp3 module missing env loader (load_env/load_env_file)")


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
        except Exception as exc:  # pragma: no cover - defensive fallback
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


def _fetch_symbol_rules_public(gp3, symbol: str):
    resp = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20)
    resp.raise_for_status()
    data = resp.json()
    sym = str(symbol).upper()
    for s in data.get("symbols", []):
        if str(s.get("symbol", "")).upper() != sym:
            continue
        tick_size = step_size = min_qty = None
        for f in s.get("filters", []):
            ftyp = str(f.get("filterType", ""))
            if ftyp == "PRICE_FILTER":
                tick_size = float(f.get("tickSize", 0) or 0)
            elif ftyp == "LOT_SIZE":
                step_size = float(f.get("stepSize", 0) or 0)
                min_qty = float(f.get("minQty", 0) or 0)
        if tick_size and step_size and min_qty:
            return gp3.SymbolRules(tick_size=tick_size, step_size=step_size, min_qty=min_qty)
    raise RuntimeError(f"symbol not found in exchangeInfo: {sym}")


def _calc_qty_compat(gp3, cash: float, price: float, rules, cfg) -> tuple[float, float]:
    if price <= 0:
        return 0.0, 0.0

    # Legacy backtest signature.
    if hasattr(gp3, "calc_qty"):
        try:
            out = gp3.calc_qty(float(cash), float(price), rules, cfg)
            if isinstance(out, (tuple, list)):
                return float(out[0] or 0.0), float(out[1] or 0.0)
        except TypeError:
            pass

    # Reproduce GP sizing from config (works for live_gp_bot module).
    buffer_ratio = min(max(float(getattr(cfg, "margin_buffer_pct", 0.0) or 0.0) / 100.0, 0.0), 0.5)
    effective_cash = float(cash) * (1.0 - buffer_ratio)
    desired_notional = effective_cash * float(getattr(cfg, "position_mult", 1.0) or 1.0)
    leverage_capped_notional = effective_cash * max(float(getattr(cfg, "leverage", 1.0) or 1.0), 1.0)
    notional = min(desired_notional, leverage_capped_notional)
    max_notional = float(getattr(cfg, "max_notional", 0.0) or 0.0)
    if max_notional > 0:
        notional = min(notional, max_notional)
    min_notional = float(getattr(cfg, "min_notional", 0.0) or 0.0)
    if notional < min_notional:
        return 0.0, 0.0

    if hasattr(gp3, "floor_to_step"):
        qty = float(gp3.floor_to_step(notional / price, float(rules.step_size)))
    else:
        step = float(rules.step_size)
        qty = math.floor((notional / price) / step) * step
    if qty < float(rules.min_qty):
        return 0.0, float(notional)
    return float(qty), float(notional)


def _build_gp3_features_server_like(gp3, raw: pd.DataFrame, cfg) -> pd.DataFrame:
    out = raw.copy()
    out["ema_fast"] = out["close"].ewm(span=int(cfg.ema_fast), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=int(cfg.ema_slow), adjust=False).mean()

    prev_close = out["close"].shift(1)
    tr1 = (out["high"] - out["low"]).abs()
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"] = out["tr"].ewm(alpha=1 / int(cfg.atr_period), adjust=False).mean()
    out["adx"] = gp3.calc_adx(out, int(cfg.adx_period))

    delta = out["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / int(cfg.rsi_period), adjust=False).mean()
    avg_down = down.ewm(alpha=1 / int(cfg.rsi_period), adjust=False).mean()
    rs = avg_up / avg_down.replace(0, pd.NA)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["rsi_prev"] = out["rsi"].shift(1)

    out["vol_ma"] = out["volume"].rolling(int(cfg.volume_ma)).mean()
    out["vol_ok"] = out["volume"] >= out["vol_ma"] * float(cfg.volume_mult)
    out["ema_fast_slope"] = out["ema_fast"] - out["ema_fast"].shift(3)
    out["bar_range_bps"] = ((out["high"] - out["low"]) / out["close"]) * 10_000.0
    spread_bps = ((out["ema_fast"] - out["ema_slow"]).abs() / out["close"]) * 10_000.0
    out["spread_ok"] = spread_bps >= float(cfg.spread_min_bps)

    rsi_buf = float(getattr(cfg, "rsi_trigger_buffer", 0.0) or 0.0)
    long_rsi_rebound = (out["rsi_prev"] <= (float(cfg.rsi_long_reset) - rsi_buf)) & (
        out["rsi"] >= (float(cfg.rsi_long_trigger) + rsi_buf)
    )
    short_rsi_rebound = (out["rsi_prev"] >= (float(cfg.rsi_short_reset) + rsi_buf)) & (
        out["rsi"] <= (float(cfg.rsi_short_trigger) - rsi_buf)
    )
    range_ok = out["bar_range_bps"] >= float(cfg.range_min_bps)

    out["long_signal"] = (
        (out["ema_fast"] > out["ema_slow"])
        & (out["ema_fast_slope"] > 0)
        & (out["close"] >= out["ema_fast"])
        & long_rsi_rebound
        & range_ok
        & (out["adx"] >= float(cfg.adx_min))
        & out["spread_ok"]
    )
    out["short_signal"] = (
        (out["ema_fast"] < out["ema_slow"])
        & (out["ema_fast_slope"] < 0)
        & (out["close"] <= out["ema_fast"])
        & short_rsi_rebound
        & range_ok
        & (out["adx"] >= float(cfg.adx_min))
        & out["spread_ok"]
    )
    out["sig_long_trend"] = (out["ema_fast"] > out["ema_slow"]) & (out["ema_fast_slope"] > 0)
    out["sig_long_rsi"] = long_rsi_rebound
    out["sig_short_trend"] = (out["ema_fast"] < out["ema_slow"]) & (out["ema_fast_slope"] < 0)
    out["sig_short_rsi"] = short_rsi_rebound
    out["sig_adx"] = out["adx"] >= float(cfg.adx_min)
    out["sig_spread"] = out["spread_ok"]
    out["sig_range"] = range_ok
    return out.dropna().copy()


def _load_klines_local(path: Path, start_ms: int, end_ms: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"klines file not found: {path}")
    if path.suffix.lower() == ".pkl":
        raw = pd.read_pickle(path)
    elif path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    need = {"open_time", "open", "high", "low", "close", "volume", "close_time"}
    miss = [c for c in need if c not in raw.columns]
    if miss:
        raise RuntimeError(f"klines file missing columns: {miss}")
    out = raw.copy()
    out["open_time"] = out["open_time"].astype("int64")
    out["close_time"] = out["close_time"].astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = out[c].astype(float)
    out = out[(out["open_time"] >= int(start_ms)) & (out["close_time"] < int(end_ms))].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f"klines window empty in local file: {path}")
    return out


def _build_rules(gp3, cfg, args):
    if args.tick_size and args.step_size and args.min_qty:
        return gp3.SymbolRules(
            tick_size=float(args.tick_size),
            step_size=float(args.step_size),
            min_qty=float(args.min_qty),
        )
    cache_file = ROOT / "cache" / "symbol_rules_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    sym = str(cfg.symbol).upper()
    try:
        if hasattr(gp3, "fetch_symbol_rules"):
            rules = gp3.fetch_symbol_rules(sym)
        else:
            rules = _fetch_symbol_rules_public(gp3, sym)
        cache[sym] = {
            "tick_size": float(rules.tick_size),
            "step_size": float(rules.step_size),
            "min_qty": float(rules.min_qty),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return rules
    except Exception:
        data = cache.get(sym)
        if data:
            return gp3.SymbolRules(
                tick_size=float(data["tick_size"]),
                step_size=float(data["step_size"]),
                min_qty=float(data["min_qty"]),
            )
        # Last fallback for ETHUSDC to keep offline/blocked runs executable.
        if sym == "ETHUSDC":
            return gp3.SymbolRules(tick_size=0.01, step_size=0.001, min_qty=0.001)
        raise


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


def _klines_cache_path(cache_dir: Path, symbol: str, start_ms: int, end_ms: int) -> Path:
    return cache_dir / f"klines_{str(symbol).upper()}_{int(start_ms)}_{int(end_ms)}_1m.pkl"


def _parse_ban_until_ms(text: str) -> Optional[int]:
    raw = str(text or "")
    m = re.search(r"banned until\s+(\d{10,14})", raw)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _fetch_binance_klines_chunked(symbol: str, start_ms: int, end_ms: int, args) -> pd.DataFrame:
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = _klines_cache_path(cache_dir, symbol, start_ms, end_ms)
    if bool(args.binance_klines_use_cache) and cpath.exists():
        return _load_klines_local(cpath, start_ms=start_ms, end_ms=end_ms)

    base_urls = [u.strip().rstrip("/") for u in str(getattr(args, "binance_fapi_base_urls", "")).split(",") if u.strip()]
    if not base_urls:
        base_urls = ["https://fapi.binance.com"]
    endpoint = "/fapi/v1/klines"
    limit = max(1, min(int(args.binance_klines_limit), 1500))
    timeout_sec = float(args.binance_klines_timeout_sec)
    page_sleep_sec = max(0.0, float(args.binance_klines_page_sleep_sec))
    max_418_retries = max(1, int(args.binance_klines_max_418_retries))
    ban_cooldown_sec = max(1.0, float(args.binance_klines_ban_cooldown_sec))
    max_conn_retries = max(0, int(getattr(args, "binance_klines_max_conn_retries", 6)))
    conn_backoff_sec = max(0.2, float(getattr(args, "binance_klines_conn_backoff_sec", 1.5)))

    sess = _build_http_session()
    rows: list = []
    cursor = int(start_ms)
    consecutive_418 = 0
    while cursor < int(end_ms):
        params = {
            "symbol": str(symbol).upper(),
            "interval": "1m",
            "limit": limit,
            "startTime": int(cursor),
            "endTime": int(end_ms),
        }
        last_err: Optional[Exception] = None
        resp = None
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
                    f"binance klines throttle exceeded: status={resp.status_code} "
                    f"retries>{max_418_retries}"
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
        raise RuntimeError("binance klines fetch returned empty rows")

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
        raise RuntimeError("binance klines window is empty after filtering")
    if bool(args.binance_klines_use_cache):
        out.to_pickle(cpath)
    return out


def _fetch_binance_mark_klines_chunked(symbol: str, start_ms: int, end_ms: int, args) -> pd.DataFrame:
    cache_dir = ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"mark_klines_{str(symbol).upper()}_{int(start_ms)}_{int(end_ms)}_1m.pkl"
    if bool(args.binance_klines_use_cache) and cpath.exists():
        return _load_klines_local(cpath, start_ms=start_ms, end_ms=end_ms)

    base_urls = [u.strip().rstrip("/") for u in str(getattr(args, "binance_fapi_base_urls", "")).split(",") if u.strip()]
    if not base_urls:
        base_urls = ["https://fapi.binance.com"]
    endpoints = ["/fapi/v1/markPriceKlines", "/fapi/v1/premiumIndexKlines"]
    limit = max(1, min(int(args.binance_klines_limit), 1500))
    timeout_sec = float(args.binance_klines_timeout_sec)
    page_sleep_sec = max(0.0, float(args.binance_klines_page_sleep_sec))
    max_418_retries = max(1, int(args.binance_klines_max_418_retries))
    ban_cooldown_sec = max(1.0, float(args.binance_klines_ban_cooldown_sec))
    max_conn_retries = max(0, int(getattr(args, "binance_klines_max_conn_retries", 6)))
    conn_backoff_sec = max(0.2, float(getattr(args, "binance_klines_conn_backoff_sec", 1.5)))

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
    events_by_bar: dict[int, list[PriceEvent]],
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
        if part is not None and not part.empty:
            out_parts.append(part)
        cur = nxt
        if cur < e_ms and pause_sec > 0:
            time.sleep(pause_sec)

    if not out_parts:
        return pd.DataFrame(columns=["agg_id", "price", "qty", "ts_ms", "is_buyer_maker"])
    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["ts_ms", "agg_id"]).drop_duplicates(subset=["agg_id"], keep="first").reset_index(drop=True)
    return out


def _parse_fee_tiers(raw: str) -> list[tuple[float, float]]:
    s = str(raw or "").strip()
    if not s:
        return []
    out: list[tuple[float, float]] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        if ":" not in p:
            raise ValueError(f"invalid fee tier '{p}', expected threshold:rate")
        a, b = p.split(":", 1)
        out.append((float(a.strip()), float(b.strip())))
    out.sort(key=lambda x: x[0])
    return out


def _load_external_events(path: str) -> dict[int, list[ExternalEvent]]:
    raw = str(path or "").strip()
    if not raw:
        return {}
    p = Path(raw)
    if not p.exists():
        raise FileNotFoundError(f"external events file not found: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        rows = df.to_dict(orient="records")
    else:
        rows = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("events", [])
    out: dict[int, list[ExternalEvent]] = {}
    for r in rows or []:
        ts = int(r.get("ts_ms"))
        ev_type = str(r.get("event_type", "FORCE_FLAT")).strip().upper()
        ev = ExternalEvent(
            ts_ms=ts,
            event_type=ExternalEventType(ev_type),
            reason=str(r.get("reason", "") or ""),
            order_id=str(r.get("order_id", "") or ""),
            price=(float(r.get("price")) if r.get("price") not in (None, "", "nan") else None),
            qty=(float(r.get("qty")) if r.get("qty") not in (None, "", "nan") else None),
            trigger_source=str(r.get("trigger_source", "external") or "external"),
        )
        out.setdefault(int(ts), []).append(ev)
    return out


class Gp3HcStrategy:
    def __init__(self, feat: pd.DataFrame, cfg, rules, gp3_mod):
        self.df = feat.reset_index(drop=True).copy()
        self.cfg = cfg
        self.rules = rules
        self.gp3 = gp3_mod
        self.index_by_open = {int(v): int(i) for i, v in self.df["open_time"].items()}
        self.cur_idx = -1

        self.pending_entry: Optional[dict] = None
        self.pending_exit: Optional[dict] = None
        self.tp_order_id: Optional[str] = None
        self.sl_order_id: Optional[str] = None

        self.entry_atr: float = 0.0
        self.stop_price: float = 0.0
        self.bars_held: int = 0
        self.be_triggered: bool = False
        self.cooldown_until: int = -1
        self.order_timeout_bars = max(1, math.ceil(float(cfg.order_timeout_sec) / 60.0))
        self.bar_settle_delay_ms = int(
            max(float(getattr(cfg, "bar_settle_delay_sec", 0.0) or 0.0), 0.0) * 1000.0
        )
        self._seen_fills = 0

    def on_bar_open(self, engine: BacktestEngine, bar: Bar) -> None:
        self.cur_idx = self.index_by_open.get(int(bar.open_time_ms), -1)
        if self.cur_idx < 0:
            return
        self._sync_fills(engine)

        if self.pending_entry and self.cur_idx > int(self.pending_entry["expire_idx"]):
            engine.cancel_order(str(self.pending_entry["order_id"]))
            self.pending_entry = None
        if self.pending_exit and self.cur_idx > int(self.pending_exit["expire_idx"]):
            engine.cancel_order(str(self.pending_exit["order_id"]))
            self.pending_exit = None

        signal_idx = int(self.cur_idx - 1)
        if signal_idx < 0:
            return

        qty, _ = engine.get_position()
        if qty != 0.0 or self.pending_entry or self.pending_exit or self.cur_idx < self.cooldown_until:
            return

        row = self.df.iloc[signal_idx]
        long_sig = bool(row.get("long_signal", False))
        short_sig = bool(row.get("short_signal", False))
        if long_sig == short_sig:
            return

        close_px = float(row["close"])
        ema_fast = float(row["ema_fast"])
        pullback = float(self.cfg.entry_pullback_bps) / 10_000.0
        if long_sig:
            side = "LONG"
            px = min(close_px * (1.0 - pullback), ema_fast)
            px = self.gp3.floor_to_step(px, float(self.rules.tick_size))
        else:
            side = "SHORT"
            px = max(close_px * (1.0 + pullback), ema_fast)
            px = self.gp3.ceil_to_step(px, float(self.rules.tick_size))
        if px <= 0:
            return

        cash = float(engine.state.cash)
        qty_v, _ = _calc_qty_compat(self.gp3, cash, float(px), self.rules, self.cfg)
        if float(qty_v) <= 0:
            return

        order_ts = int(bar.open_time_ms) + int(self.bar_settle_delay_ms)
        if order_ts >= int(bar.close_time_ms):
            order_ts = int(bar.open_time_ms)
        oid = engine.place_limit(
            side=Side.BUY if side == "LONG" else Side.SELL,
            qty=float(qty_v),
            limit_price=float(px),
            ts_ms=int(order_ts),
            tif=TimeInForce.GTX,
            reduce_only=False,
            reason=f"gp3_entry_{side.lower()}",
        )
        self.pending_entry = {
            "order_id": oid,
            "side": side,
            "signal_atr": float(max(float(row.get("atr", 0.0)), float(self.rules.tick_size))),
            "expire_idx": int(signal_idx + int(self.cfg.entry_timeout_bars)),
        }

    def _clear_protection_orders(self, engine: BacktestEngine) -> None:
        if self.tp_order_id:
            engine.cancel_order(self.tp_order_id)
        if self.sl_order_id:
            engine.cancel_order(self.sl_order_id)
        self.tp_order_id = None
        self.sl_order_id = None

    def _place_protection_orders(self, engine: BacktestEngine, ts_ms: int) -> None:
        qty, entry = engine.get_position()
        if qty == 0.0:
            return

        side = "LONG" if qty > 0 else "SHORT"
        atr = max(float(self.entry_atr or 0.0), float(self.rules.tick_size))
        if atr <= 0:
            idx = max(int(self.cur_idx - 1), 0) if self.cur_idx >= 0 else -1
            if idx >= 0:
                atr = max(float(self.df.iloc[idx].get("atr", 0.0)), float(self.rules.tick_size))
            else:
                atr = float(self.rules.tick_size)
            self.entry_atr = float(atr)

        if side == "LONG":
            tp = self.gp3.floor_to_step(float(entry) + float(self.cfg.tp_atr_mult) * atr, float(self.rules.tick_size))
            stop = self.gp3.floor_to_step(float(entry) - float(self.cfg.stop_atr_mult) * atr, float(self.rules.tick_size))
            tp_side = Side.SELL
            sl_side = Side.SELL
        else:
            tp = self.gp3.ceil_to_step(float(entry) - float(self.cfg.tp_atr_mult) * atr, float(self.rules.tick_size))
            stop = self.gp3.ceil_to_step(float(entry) + float(self.cfg.stop_atr_mult) * atr, float(self.rules.tick_size))
            tp_side = Side.BUY
            sl_side = Side.BUY

        self.stop_price = float(stop)
        self.tp_order_id = engine.place_limit(
            side=tp_side,
            qty=abs(float(qty)),
            limit_price=float(tp),
            ts_ms=int(ts_ms),
            tif=TimeInForce.GTX,
            reduce_only=True,
            reason="gp3_tp",
        )
        self.sl_order_id = engine.place_stop_market(
            side=sl_side,
            qty=abs(float(qty)),
            stop_price=float(stop),
            ts_ms=int(ts_ms),
            reduce_only=True,
            reason="gp3_sl",
        )

    def on_price_event(self, engine: BacktestEngine, event) -> None:
        self._sync_fills(engine)
        qty, entry = engine.get_position()
        if qty == 0.0:
            return
        if (self.tp_order_id or self.sl_order_id) and (not self._protection_orders_valid(engine)):
            self._clear_protection_orders(engine)
        if self.tp_order_id and self.sl_order_id:
            return
        self._place_protection_orders(engine, int(event.ts_ms))

    def on_bar_close(self, engine: BacktestEngine, bar: Bar) -> None:
        self._sync_fills(engine)
        if self.cur_idx < 0:
            return
        qty, entry = engine.get_position()
        if qty == 0.0:
            return
        if self.pending_exit is not None:
            return

        row = self.df.iloc[self.cur_idx]
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        ema_fast = float(row["ema_fast"])
        ema_slow = float(row["ema_slow"])
        atr = max(float(self.entry_atr or 0.0), float(self.rules.tick_size))

        self.bars_held += 1
        new_stop = None
        if qty > 0:
            if (not self.be_triggered) and (h - float(entry) >= float(self.cfg.breakeven_trigger_atr) * atr):
                be = float(entry) * (1.0 + float(self.cfg.breakeven_offset_bps) / 10_000.0)
                if be > float(self.stop_price):
                    new_stop = be
                    self.be_triggered = True
            if self.be_triggered:
                trail = h - float(self.cfg.trail_atr_mult) * atr
                if trail > float(self.stop_price):
                    new_stop = trail if new_stop is None else max(new_stop, trail)
        else:
            if (not self.be_triggered) and (float(entry) - l >= float(self.cfg.breakeven_trigger_atr) * atr):
                be = float(entry) * (1.0 - float(self.cfg.breakeven_offset_bps) / 10_000.0)
                if be < float(self.stop_price):
                    new_stop = be
                    self.be_triggered = True
            if self.be_triggered:
                trail = l + float(self.cfg.trail_atr_mult) * atr
                if trail < float(self.stop_price):
                    new_stop = trail if new_stop is None else min(new_stop, trail)

        if new_stop is not None:
            if qty > 0:
                stop = self.gp3.floor_to_step(float(new_stop), float(self.rules.tick_size))
                stop_side = Side.SELL
            else:
                stop = self.gp3.ceil_to_step(float(new_stop), float(self.rules.tick_size))
                stop_side = Side.BUY
            self.stop_price = float(stop)
            if self.sl_order_id:
                engine.cancel_order(self.sl_order_id)
            self.sl_order_id = engine.place_stop_market(
                side=stop_side,
                qty=abs(float(qty)),
                stop_price=float(stop),
                ts_ms=int(bar.close_time_ms),
                reduce_only=True,
                reason="gp3_sl_trailing",
            )

        trend_flip = (qty > 0 and ema_fast < ema_slow) or (qty < 0 and ema_fast > ema_slow)
        if self.bars_held >= int(self.cfg.max_hold_bars) or trend_flip:
            if qty > 0:
                px = self.gp3.ceil_to_step(float(c), float(self.rules.tick_size))
                side = Side.SELL
            else:
                px = self.gp3.floor_to_step(float(c), float(self.rules.tick_size))
                side = Side.BUY
            oid = engine.place_limit(
                side=side,
                qty=abs(float(qty)),
                limit_price=float(px),
                ts_ms=int(bar.close_time_ms),
                tif=TimeInForce.GTC,
                reduce_only=True,
                reason="gp3_time_or_flip",
            )
            self.pending_exit = {
                "order_id": oid,
                "expire_idx": int(self.cur_idx + self.order_timeout_bars),
            }

    def _sync_fills(self, engine: BacktestEngine) -> None:
        fills = engine.state.fills
        if self._seen_fills >= len(fills):
            return
        for f in fills[self._seen_fills :]:
            oid = str(f.order_id)
            if self.pending_entry and oid == str(self.pending_entry.get("order_id")):
                # If entry is partially filled, stop adding residual fragments that distort position lifecycle.
                o = engine.orders.get(oid)
                if o and o.status == OrderStatus.PARTIALLY_FILLED:
                    engine.cancel_order(oid)
                self.entry_atr = float(self.pending_entry.get("signal_atr", 0.0))
                self.bars_held = 0
                self.be_triggered = False
                self.pending_entry = None
                self._clear_protection_orders(engine)
                self._place_protection_orders(engine, int(f.ts_ms))
                continue

            if self.pending_exit and oid == str(self.pending_exit.get("order_id")):
                self._close_cleanup(engine)
                continue

            if oid == str(self.tp_order_id) or oid == str(self.sl_order_id):
                self._close_cleanup(engine)
                continue
        self._seen_fills = len(fills)

    def _close_cleanup(self, engine: BacktestEngine) -> None:
        if self.tp_order_id:
            engine.cancel_order(self.tp_order_id)
        if self.sl_order_id:
            engine.cancel_order(self.sl_order_id)
        if self.pending_exit:
            engine.cancel_order(str(self.pending_exit.get("order_id")))
        self.tp_order_id = None
        self.sl_order_id = None
        self.pending_exit = None
        self.pending_entry = None
        self.entry_atr = 0.0
        self.stop_price = 0.0
        self.be_triggered = False
        self.bars_held = 0
        if self.cur_idx >= 0:
            self.cooldown_until = max(self.cooldown_until, int(self.cur_idx + int(self.cfg.cooldown_bars)))

    @staticmethod
    def _is_order_active(engine: BacktestEngine, order_id: Optional[str]) -> bool:
        if not order_id:
            return False
        o = engine.orders.get(str(order_id))
        if not o:
            return False
        return o.status in (OrderStatus.NEW, OrderStatus.TRIGGERED, OrderStatus.PARTIALLY_FILLED)

    def _protection_orders_valid(self, engine: BacktestEngine) -> bool:
        return self._is_order_active(engine, self.tp_order_id) and self._is_order_active(engine, self.sl_order_id)


def _extract_round_trips(fills: list[dict]) -> pd.DataFrame:
    rows = []
    pos_qty = 0.0
    entry_px = 0.0
    entry_ts = 0
    cum_fee_open = 0.0
    entry_reason = ""
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
            entry_reason = str(f.get("reason") or "")
            continue
        if (cur > 0 and signed > 0) or (cur < 0 and signed < 0):
            tot = abs(cur) + abs(signed)
            entry_px = (abs(cur) * entry_px + abs(signed) * px) / max(tot, 1e-12)
            pos_qty = cur + signed
            cum_fee_open += fee
            continue

        close_qty = min(abs(cur), abs(signed))
        direction = 1.0 if cur > 0 else -1.0
        pnl = close_qty * (px - entry_px) * direction - cum_fee_open - fee
        rows.append(
            {
                "entry_ts_ms": entry_ts,
                "exit_ts_ms": ts,
                "side": "LONG" if cur > 0 else "SHORT",
                "entry_price": entry_px,
                "exit_price": px,
                "qty": close_qty,
                "pnl": pnl,
                "fee_total": cum_fee_open + fee,
                "entry_reason": entry_reason,
                "exit_reason": str(f.get("reason") or ""),
                "exit_trigger_source": str(f.get("trigger_source") or ""),
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
                entry_reason = ""
            else:
                cum_fee_open = 0.0
        else:
            pos_qty = math.copysign(remain, signed)
            entry_px = px
            entry_ts = ts
            cum_fee_open = fee
            entry_reason = str(f.get("reason") or "")
    return pd.DataFrame(rows)


def run(args) -> dict:
    gp3 = _load_gp3_module()
    _load_env_compat(gp3, Path("d:/project/.env"))
    _load_env_compat(gp3, Path(args.profile))
    cfg = gp3.build_config_from_env()
    if not hasattr(cfg, "initial_capital"):
        cfg.initial_capital = float(getattr(args, "initial_capital", 100.0) or 100.0)
    if not hasattr(cfg, "bar_settle_delay_sec"):
        cfg.bar_settle_delay_sec = float(os.getenv("BAR_SETTLE_DELAY_SEC", "0") or 0.0)
    if not hasattr(cfg, "maker_fee"):
        cfg.maker_fee = 0.0
    if not hasattr(cfg, "taker_fee"):
        cfg.taker_fee = 0.0004
    if not hasattr(cfg, "stop_slippage_bps"):
        cfg.stop_slippage_bps = 2.0
    cfg.symbol = str(args.symbol).upper()
    cfg.skip_entry_bar_exit_checks = False

    start_dt, end_dt = _resolve_time_window(args)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    rules = _build_rules(gp3, cfg, args)
    if args.klines_file:
        raw = _load_klines_local(Path(args.klines_file), start_ms=start_ms, end_ms=end_ms)
    else:
        try:
            if str(args.binance_klines_source).lower() == "gp3" and hasattr(gp3, "fetch_klines_1m"):
                raw = gp3.fetch_klines_1m(cfg.symbol, start_ms, end_ms)
            else:
                raw = _fetch_binance_klines_chunked(cfg.symbol, start_ms, end_ms, args)
        except Exception:
            fallback = Path("d:/project/gp3.0/results/live_like_scan/ETHUSDC_1m_20250310_20260310.pkl")
            if str(cfg.symbol).upper() == "ETHUSDC" and fallback.exists():
                raw = _load_klines_local(fallback, start_ms=start_ms, end_ms=end_ms)
            else:
                raise
    mark_raw = _fetch_binance_mark_klines_chunked(cfg.symbol, start_ms, end_ms, args) if bool(args.use_real_mark) else None
    feat = _build_gp3_features_server_like(gp3, raw, cfg).dropna().reset_index(drop=True)
    if feat.empty:
        raise RuntimeError("feature frame is empty")

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
        for r in feat.itertuples(index=False)
    ]

    events_by_bar = None
    external_events = _load_external_events(args.external_events_file) if str(args.external_events_file).strip() else None
    if bool(args.use_agg):
        s_ms = int(feat["open_time"].iloc[0])
        e_ms = int(feat["close_time"].iloc[-1]) + 1
        agg_df = _fetch_agg_chunked(cfg.symbol, s_ms, e_ms, args)
        _, events_by_bar = aggtrades_to_bars_and_events(
            agg_df,
            start_ms=s_ms,
            end_ms=e_ms,
            bar_ms=60_000,
            fill_empty_bars=False,
        )
        _apply_mark_prices_to_bars_and_events(bars, events_by_bar, mark_raw, path_mode=str(args.path_mode))
    else:
        events_by_bar = _build_inferred_events_with_mark(bars, mark_raw, path_mode=str(args.path_mode))

    engine_cfg = BacktestConfig(
        symbol=cfg.symbol,
        initial_cash=float(cfg.initial_capital),
        leverage=float(getattr(cfg, "leverage", 1.0) or 1.0),
        maker_fee_rate=float(cfg.maker_fee),
        taker_fee_rate=float(cfg.taker_fee),
        market_slippage_bps=0.5,
        stop_slippage_bps=float(cfg.stop_slippage_bps),
        tick_size=float(rules.tick_size),
        maker_queue_delay_ms=int(args.maker_delay_ms),
        gtc_queue_delay_ms=int(args.gtc_queue_delay_ms),
        maker_buffer_ticks=int(args.maker_buffer_ticks),
        allow_same_bar_entry_exit=True,
        stop_trigger_source=StopTriggerSource(str(args.stop_trigger_source)),
        allow_marketable_gtc_as_taker=bool(args.allow_marketable_gtc_as_taker),
        partial_fill_enabled=bool(args.partial_fill_enabled),
        partial_fill_ratio=float(args.partial_fill_ratio),
        min_partial_qty=float(args.min_partial_qty),
        partial_fill_scope=str(args.partial_fill_scope),
        funding_interval_ms=int(float(args.funding_interval_hours) * 3600 * 1000),
        funding_rate=float(args.funding_rate),
        maker_fee_tiers=_parse_fee_tiers(args.maker_fee_tiers),
        taker_fee_tiers=_parse_fee_tiers(args.taker_fee_tiers),
        maintenance_margin_rate=float(args.maintenance_margin_rate),
        maintenance_amount=float(args.maintenance_amount),
        liquidation_fee_rate=float(args.liquidation_fee_rate),
        reject_on_insufficient_margin=bool(args.reject_on_insufficient_margin),
        liquidate_on_margin_breach=bool(args.liquidate_on_margin_breach),
    )
    engine = BacktestEngine(engine_cfg, path_model=IntrabarPathModel(mode=str(args.path_mode)))
    strategy = Gp3HcStrategy(feat, cfg, rules, gp3)
    result = engine.run(bars, strategy, events_by_bar=events_by_bar, external_events_by_ts=external_events)
    stats = summarize(result)

    fills_df = pd.DataFrame(result["fills"])
    eq_df = pd.DataFrame(result["equity_curve"])
    rt_df = _extract_round_trips(result["fills"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fills_path = out_dir / f"{cfg.symbol}_gp3_hc_fills_{stamp}.csv"
    eq_path = out_dir / f"{cfg.symbol}_gp3_hc_equity_{stamp}.csv"
    rt_path = out_dir / f"{cfg.symbol}_gp3_hc_roundtrips_{stamp}.csv"
    stats_path = out_dir / f"{cfg.symbol}_gp3_hc_stats_{stamp}.json"
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
        "symbol": cfg.symbol,
        "use_agg": bool(args.use_agg),
        "path_mode": str(args.path_mode),
        "bars": int(len(bars)),
        "fills": int(len(fills_df)),
        "round_trips": int(len(rt_df)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "engine_summary": stats,
        "external_events_count": int(sum(len(v) for v in (external_events or {}).values())),
        "config": asdict(cfg),
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GP3.0 strategy on HC engine")
    ap.add_argument("--symbol", default="ETHUSDC")
    ap.add_argument("--profile", default="d:/project/gp3.0/profiles/server_live_current.env")
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--start-utc", default="", help="Explicit UTC window start, e.g. 2026-03-11T00:00:00Z")
    ap.add_argument("--end-utc", default="", help="Explicit UTC window end, e.g. 2026-03-13T00:00:00Z")
    ap.add_argument("--use-agg", dest="use_agg", action="store_true")
    ap.add_argument("--no-agg", dest="use_agg", action="store_false")
    ap.set_defaults(use_agg=True)
    ap.add_argument("--path-mode", default="oc_aware", choices=["oc_aware", "long_worst", "short_worst"])
    ap.add_argument("--maker-delay-ms", type=int, default=800)
    ap.add_argument("--gtc-queue-delay-ms", type=int, default=300)
    ap.add_argument("--maker-buffer-ticks", type=int, default=0)
    ap.add_argument("--out-dir", default="d:/project/hc/output")
    ap.add_argument("--klines-file", default="", help="Local 1m klines file (.pkl/.parquet/.csv)")
    ap.add_argument("--tick-size", type=float, default=0.0, help="Symbol tick size override for offline mode")
    ap.add_argument("--step-size", type=float, default=0.0, help="Symbol step size override for offline mode")
    ap.add_argument("--min-qty", type=float, default=0.0, help="Symbol min qty override for offline mode")
    ap.add_argument("--agg-chunk-minutes", type=float, default=120.0, help="aggTrades fetch chunk size in minutes")
    ap.add_argument("--agg-chunk-pause-sec", type=float, default=2.0, help="pause between aggTrades chunks")
    ap.add_argument("--agg-req-sleep-sec", type=float, default=0.05, help="sleep between aggTrades requests")
    ap.add_argument("--agg-timeout-sec", type=float, default=20.0, help="aggTrades request timeout seconds")
    ap.add_argument("--agg-max-418-retries", type=int, default=40, help="max retries on HTTP 418 ban response")
    ap.add_argument("--agg-ban-cooldown-sec", type=float, default=20.0, help="cooldown seconds before retry on 418")

    ap.add_argument("--stop-trigger-source", default="last", choices=["last", "mark"], help="stop trigger source")
    ap.add_argument("--use-real-mark", dest="use_real_mark", action="store_true")
    ap.add_argument("--no-real-mark", dest="use_real_mark", action="store_false")
    ap.add_argument("--allow-marketable-gtc-as-taker", dest="allow_marketable_gtc_as_taker", action="store_true")
    ap.add_argument("--disallow-marketable-gtc-as-taker", dest="allow_marketable_gtc_as_taker", action="store_false")
    ap.add_argument("--partial-fill-enabled", dest="partial_fill_enabled", action="store_true")
    ap.add_argument("--no-partial-fill", dest="partial_fill_enabled", action="store_false")
    ap.add_argument("--partial-fill-ratio", type=float, default=0.35, help="max ratio filled per event when partial fill enabled")
    ap.add_argument("--min-partial-qty", type=float, default=0.0, help="minimum partial fill qty")
    ap.add_argument("--partial-fill-scope", default="maker", choices=["maker", "all"], help="apply partial fills to maker orders only or all orders")
    ap.add_argument("--funding-interval-hours", type=float, default=8.0, help="funding interval in hours")
    ap.add_argument("--funding-rate", type=float, default=0.0, help="funding rate per interval, positive means long pays")
    ap.add_argument("--maintenance-margin-rate", type=float, default=0.005, help="maintenance margin ratio used for liquidation check")
    ap.add_argument("--maintenance-amount", type=float, default=0.0, help="fixed maintenance margin add-on")
    ap.add_argument("--liquidation-fee-rate", type=float, default=0.0, help="extra liquidation fee rate on forced liquidation")
    ap.add_argument("--reject-on-insufficient-margin", dest="reject_on_insufficient_margin", action="store_true")
    ap.add_argument("--no-reject-on-insufficient-margin", dest="reject_on_insufficient_margin", action="store_false")
    ap.add_argument("--liquidate-on-margin-breach", dest="liquidate_on_margin_breach", action="store_true")
    ap.add_argument("--no-liquidate-on-margin-breach", dest="liquidate_on_margin_breach", action="store_false")
    ap.add_argument("--maker-fee-tiers", default="", help="tiered maker fee, e.g. 0:0,50000000:-0.00005")
    ap.add_argument("--taker-fee-tiers", default="", help="tiered taker fee, e.g. 0:0.0004,50000000:0.00035")
    ap.add_argument("--external-events-file", default="", help="JSON/CSV external events for force-flat/cancel simulation")

    ap.add_argument("--binance-klines-source", default="hc", choices=["hc", "gp3"], help="1m klines fetch backend")
    ap.add_argument("--binance-fapi-base-urls", default="https://fapi.binance.com", help="comma-separated Binance futures base urls for failover")
    ap.add_argument("--binance-klines-limit", type=int, default=1000, help="per-request kline limit")
    ap.add_argument("--binance-klines-page-sleep-sec", type=float, default=0.12, help="sleep between kline pages")
    ap.add_argument("--binance-klines-timeout-sec", type=float, default=20.0, help="kline request timeout seconds")
    ap.add_argument("--binance-klines-max-conn-retries", type=int, default=6, help="max retries for network/connection errors")
    ap.add_argument("--binance-klines-conn-backoff-sec", type=float, default=1.5, help="base backoff seconds for connection retries")
    ap.add_argument("--binance-klines-max-418-retries", type=int, default=30, help="max retries on 418/429")
    ap.add_argument("--binance-klines-ban-cooldown-sec", type=float, default=20.0, help="cooldown on ban/rate limit")
    ap.add_argument("--binance-klines-use-cache", dest="binance_klines_use_cache", action="store_true")
    ap.add_argument("--no-binance-klines-cache", dest="binance_klines_use_cache", action="store_false")
    ap.set_defaults(
        binance_klines_use_cache=True,
        partial_fill_enabled=True,
        use_real_mark=True,
        allow_marketable_gtc_as_taker=True,
        reject_on_insufficient_margin=True,
        liquidate_on_margin_breach=True,
    )
    args = ap.parse_args()

    stats = run(args)
    print("GP3 on HC backtest done:")
    print(f"  symbol: {stats['symbol']}")
    print(f"  period: {stats['window_start_utc']} -> {stats['window_end_utc']}")
    print(f"  use_agg: {stats['use_agg']}, path_mode: {stats['path_mode']}")
    print(f"  bars: {stats['bars']}, fills: {stats['fills']}, round_trips: {stats['round_trips']}")
    es = stats["engine_summary"]
    print(f"  ending_equity: {es.get('ending_equity')}")
    print(f"  net_return_pct: {es.get('net_return_pct')}")
    print(f"  max_drawdown_pct: {es.get('max_drawdown_pct')}")
    print(f"  fee_paid_total: {es.get('fee_paid_total')}")
    print(f"  funding_pnl: {es.get('funding_pnl')}")
    print(f"  win_rate_pct: {stats.get('win_rate_pct')}")
    print(f"  stats_json: {stats['outputs']['stats_json']}")


if __name__ == "__main__":
    main()
