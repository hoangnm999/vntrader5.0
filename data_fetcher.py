# ============================================================
# VN TRADER BOT V5 — data_fetcher.py
# Primary:  TCBS API  (apipubaws.tcbs.com.vn)
# Fallback: TCBS v2   (different endpoint format)
# Features: retry logic, full browser headers, symbol cache
# ============================================================

import time
import logging
import requests
import pandas as pd
from datetime import datetime

from config import DEFAULT_RESOLUTION, DEFAULT_LOOKBACK

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
_TCBS_BASE     = "https://apipubaws.tcbs.com.vn/stock-insight/v1"
_TCBS_BASE_V2  = "https://apipubaws.tcbs.com.vn/stock-insight/v2"
_CACHE_TTL     = 3600   # 1 hour for VNI cache
_RETRY_COUNT   = 3
_RETRY_DELAY   = 1.5    # seconds between retries

# ── Headers — full browser profile to avoid 403/block ────────
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://tcinvest.tcbs.com.vn/",
    "Origin":          "https://tcinvest.tcbs.com.vn",
    "DNT":             "1",
    "Connection":      "keep-alive",
}

# ── Per-symbol & VNI cache ────────────────────────────────────
_symbol_cache: dict[str, dict] = {}
_vni_cache: dict = {"data": None, "fetched_at": None}


# ── Column normaliser ─────────────────────────────────────────
_COL_MAP = {
    # TCBS v1
    "tradingDate": "date",
    # TCBS v2 / alternate
    "TradingDate": "date",
    "trading_date": "date",
    "o": "open",  "h": "high",  "l": "low",  "c": "close",  "v": "volume",
    "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    # Already correct
    "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume",
    "date": "date",
}

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns})
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=list(required))
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


# ── HTTP GET with retry ───────────────────────────────────────
def _get(url: str, params: dict = None) -> dict | None:
    for attempt in range(1, _RETRY_COUNT + 1):
        try:
            resp = requests.get(
                url, params=params, headers=_HEADERS,
                timeout=15, allow_redirects=True,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"HTTP {resp.status_code} on attempt {attempt}: {url}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout attempt {attempt}: {url}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error attempt {attempt}: {e}")
        except Exception as e:
            logger.warning(f"Request error attempt {attempt}: {e}")
        if attempt < _RETRY_COUNT:
            time.sleep(_RETRY_DELAY * attempt)
    return None


# ── TCBS v1 endpoint ─────────────────────────────────────────
def _fetch_v1(symbol: str, resolution: str, count: int) -> pd.DataFrame:
    to_ts = int(time.time())
    url   = f"{_TCBS_BASE}/stock-bardata"
    data  = _get(url, params={
        "ticker": symbol, "type": resolution,
        "to": to_ts, "count": count,
    })
    if not data:
        return pd.DataFrame()

    bars = data.get("data") or data.get("bars") or []
    if not bars:
        return pd.DataFrame()

    return _normalise(pd.DataFrame(bars))


# ── TCBS v2 endpoint (fallback) ───────────────────────────────
def _fetch_v2(symbol: str, resolution: str, count: int) -> pd.DataFrame:
    to_ts   = int(time.time())
    from_ts = to_ts - count * 86400 * 2   # rough estimate, more than enough
    url     = f"{_TCBS_BASE_V2}/stock-bardata"
    data    = _get(url, params={
        "ticker": symbol, "type": resolution,
        "from": from_ts, "to": to_ts,
    })
    if not data:
        return pd.DataFrame()

    bars = data.get("data") or data.get("bars") or []
    if not bars:
        return pd.DataFrame()

    df = _normalise(pd.DataFrame(bars))
    # Trim to requested count
    if len(df) > count:
        df = df.iloc[-count:].reset_index(drop=True)
    return df


# ── Public fetch (v1 → v2 fallback) ──────────────────────────
def fetch_ohlcv(
    symbol: str,
    resolution: str = DEFAULT_RESOLUTION,
    count: int = DEFAULT_LOOKBACK,
    use_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV for a symbol.
    Tries TCBS v1 first, falls back to v2 automatically.
    use_cache=True skips network if fresh data exists in memory.
    """
    cache_key = f"{symbol}_{resolution}_{count}"
    if use_cache and cache_key in _symbol_cache:
        entry = _symbol_cache[cache_key]
        if (datetime.utcnow() - entry["fetched_at"]).total_seconds() < _CACHE_TTL:
            return entry["data"]

    df = _fetch_v1(symbol, resolution, count)
    if df.empty:
        logger.info(f"[{symbol}] v1 empty — trying v2 fallback")
        df = _fetch_v2(symbol, resolution, count)

    if df.empty:
        logger.error(f"[{symbol}] Both endpoints returned no data")
    else:
        logger.info(f"[{symbol}] Fetched {len(df)} bars")
        if use_cache:
            _symbol_cache[cache_key] = {"data": df, "fetched_at": datetime.utcnow()}

    return df


def fetch_vni(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch VNINDEX daily data. Cached for 1 hour.
    Used for Relative Strength pre-filter (needs 63+ bars).
    """
    now     = datetime.utcnow()
    cached  = _vni_cache["data"]
    fetched = _vni_cache["fetched_at"]

    if (
        not force_refresh and cached is not None and fetched is not None
        and (now - fetched).total_seconds() < _CACHE_TTL
    ):
        return cached

    df = fetch_ohlcv("VNINDEX", count=200)
    if not df.empty:
        _vni_cache["data"]       = df
        _vni_cache["fetched_at"] = now
        logger.info(f"VNI cache refreshed ({len(df)} bars)")
    else:
        logger.warning("VNI fetch failed — using stale cache")
        if cached is not None:
            return cached

    return df


def fetch_all_symbols(
    symbols: list,
    resolution: str = DEFAULT_RESOLUTION,
    count: int = DEFAULT_LOOKBACK,
    delay_sec: float = 0.4,
) -> dict[str, pd.DataFrame]:
    """
    Fetch all symbols. Returns {symbol: DataFrame}.
    Slightly longer delay to avoid rate limiting.
    """
    results = {}
    for i, sym in enumerate(symbols):
        df = fetch_ohlcv(sym, resolution, count)
        if not df.empty:
            results[sym] = df
        else:
            logger.warning(f"[{sym}] Skipped — no data after retries")
        if i < len(symbols) - 1:
            time.sleep(delay_sec)

    logger.info(f"Fetched {len(results)}/{len(symbols)} symbols")
    return results
