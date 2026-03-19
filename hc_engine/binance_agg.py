from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .types import Bar, PriceEvent

BASE_URL = "https://fapi.binance.com"


@dataclass
class AggTradeFetchSpec:
    symbol: str
    start_ms: int
    end_ms: int
    limit: int = 1000
    sleep_sec: float = 0.02
    timeout_sec: float = 20.0
    max_418_retries: int = 20
    ban_cooldown_sec: float = 15.0
    max_conn_retries: int = 12
    conn_backoff_sec: float = 1.2


def _cache_path(cache_dir: Path, spec: AggTradeFetchSpec) -> Path:
    return cache_dir / (
        f"agg_{spec.symbol.upper()}_{int(spec.start_ms)}_{int(spec.end_ms)}_"
        f"l{int(spec.limit)}.csv"
    )


def fetch_binance_futures_aggtrades(
    spec: AggTradeFetchSpec,
    *,
    use_cache: bool = True,
    cache_dir: str | Path = "cache",
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    sym = str(spec.symbol or "").upper().strip()
    if not sym:
        raise ValueError("symbol is required")
    if int(spec.end_ms) <= int(spec.start_ms):
        raise ValueError("end_ms must be > start_ms")

    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    cpath = _cache_path(cdir, spec)
    if use_cache and cpath.exists():
        df = pd.read_csv(cpath)
        return _normalize_agg_df(df)

    sess = session or _build_session()
    rows: list[dict] = []
    cur = int(spec.start_ms)
    end_ms = int(spec.end_ms)
    limit = max(1, min(int(spec.limit), 1000))
    last_agg_id = None
    consecutive_418 = 0
    consecutive_conn_err = 0

    while cur <= end_ms:
        params = {
            "symbol": sym,
            "startTime": int(cur),
            "endTime": int(end_ms),
            "limit": limit,
        }
        try:
            resp = sess.get(f"{BASE_URL}/fapi/v1/aggTrades", params=params, timeout=float(spec.timeout_sec))
        except requests.RequestException:
            consecutive_conn_err += 1
            if consecutive_conn_err > int(max(1, spec.max_conn_retries)):
                raise
            time.sleep(max(0.5, float(spec.conn_backoff_sec) * consecutive_conn_err))
            continue
        consecutive_conn_err = 0
        # Handle exchange throttling explicitly for stability.
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "1") or 1)
            time.sleep(max(1.0, retry_after))
            continue
        if resp.status_code == 418:
            consecutive_418 += 1
            if consecutive_418 > int(max(1, spec.max_418_retries)):
                raise RuntimeError(
                    f"binance returned 418 too many times (>{spec.max_418_retries}) "
                    f"for {sym} {spec.start_ms}-{spec.end_ms}"
                )
            retry_after = float(resp.headers.get("Retry-After", "0") or 0)
            cool = max(float(spec.ban_cooldown_sec), retry_after, 5.0)
            time.sleep(cool)
            continue
        consecutive_418 = 0
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break

        new_count = 0
        for x in batch:
            agg_id = int(x.get("a"))
            if last_agg_id is not None and agg_id == last_agg_id:
                continue
            last_agg_id = agg_id
            t = int(x.get("T"))
            if t < int(spec.start_ms) or t > end_ms:
                continue
            rows.append(
                {
                    "agg_id": agg_id,
                    "price": float(x.get("p")),
                    "qty": float(x.get("q")),
                    "ts_ms": t,
                    "is_buyer_maker": bool(x.get("m")),
                }
            )
            new_count += 1

        last_ts = int(batch[-1].get("T"))
        if new_count <= 0 and last_ts <= cur:
            break
        cur = max(cur + 1, last_ts + 1)
        if float(spec.sleep_sec) > 0:
            time.sleep(float(spec.sleep_sec))

    df = pd.DataFrame(rows)
    df = _normalize_agg_df(df)
    if use_cache:
        df.to_csv(cpath, index=False)
    return df


def _build_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def _normalize_agg_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["agg_id", "price", "qty", "ts_ms", "is_buyer_maker"])
    out = df.copy()
    out["agg_id"] = out["agg_id"].astype("int64")
    out["price"] = out["price"].astype(float)
    out["qty"] = out["qty"].astype(float)
    out["ts_ms"] = out["ts_ms"].astype("int64")
    out["is_buyer_maker"] = out["is_buyer_maker"].astype(bool)
    out = out.sort_values(["ts_ms", "agg_id"]).drop_duplicates(subset=["agg_id"], keep="first")
    return out.reset_index(drop=True)


def aggtrades_to_price_events(df: pd.DataFrame) -> list[PriceEvent]:
    if df is None or df.empty:
        return []
    out: list[PriceEvent] = []
    # Running VWAP as mark-price proxy when only trade tape is available.
    cum_notional = 0.0
    cum_qty = 0.0
    for r in df.itertuples(index=False):
        px = float(getattr(r, "price"))
        q = float(getattr(r, "qty"))
        cum_notional += px * q
        cum_qty += q
        mark = (cum_notional / cum_qty) if cum_qty > 0 else px
        out.append(
            PriceEvent(
                ts_ms=int(getattr(r, "ts_ms")),
                price=px,
                source="aggTrade",
                mark_price=float(mark),
                trade_qty=float(q),
                is_buyer_maker=bool(getattr(r, "is_buyer_maker")),
            )
        )
    return out


def aggtrades_to_bars_and_events(
    df: pd.DataFrame,
    *,
    start_ms: int,
    end_ms: int,
    bar_ms: int = 60_000,
    fill_empty_bars: bool = True,
) -> tuple[list[Bar], dict[int, list[PriceEvent]]]:
    if int(end_ms) <= int(start_ms):
        raise ValueError("end_ms must be > start_ms")
    bms = int(bar_ms)
    if bms <= 0:
        raise ValueError("bar_ms must be > 0")

    events_by_bar: dict[int, list[PriceEvent]] = {}
    if df is not None and not df.empty:
        tmp: dict[int, list[tuple[int, float, float, bool]]] = {}
        for r in df.itertuples(index=False):
            ts = int(getattr(r, "ts_ms"))
            if ts < int(start_ms) or ts >= int(end_ms):
                continue
            p = float(getattr(r, "price"))
            q = float(getattr(r, "qty"))
            m = bool(getattr(r, "is_buyer_maker"))
            bopen = (ts // bms) * bms
            tmp.setdefault(bopen, []).append((ts, p, q, m))

        for bopen, vals in tmp.items():
            vals = sorted(vals, key=lambda x: int(x[0]))
            cum_notional = 0.0
            cum_qty = 0.0
            evs: list[PriceEvent] = []
            for ts, p, q, m in vals:
                cum_notional += p * q
                cum_qty += q
                mark = (cum_notional / cum_qty) if cum_qty > 0 else p
                evs.append(
                    PriceEvent(
                        ts_ms=int(ts),
                        price=float(p),
                        source="aggTrade",
                        mark_price=float(mark),
                        trade_qty=float(q),
                        is_buyer_maker=bool(m),
                    )
                )
            events_by_bar[bopen] = evs

    for k in list(events_by_bar.keys()):
        events_by_bar[k] = sorted(events_by_bar[k], key=lambda x: int(x.ts_ms))

    bars: list[Bar] = []
    cur = (int(start_ms) // bms) * bms
    end_floor = ((int(end_ms) - 1) // bms) * bms
    last_close = None
    while cur <= end_floor:
        evs = events_by_bar.get(cur, [])
        if evs:
            o = float(evs[0].price)
            c = float(evs[-1].price)
            prices = [float(e.price) for e in evs]
            h = max(prices)
            l = min(prices)
            v = float(len(evs))
            last_close = c
        else:
            if not fill_empty_bars:
                cur += bms
                continue
            if last_close is None:
                # Cannot infer empty bars before first trade; skip until first non-empty bar.
                cur += bms
                continue
            o = h = l = c = float(last_close)
            v = 0.0

        bars.append(
            Bar(
                open_time_ms=int(cur),
                close_time_ms=int(cur + bms),
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
            )
        )
        cur += bms

    return bars, events_by_bar
