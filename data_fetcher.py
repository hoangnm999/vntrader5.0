# ============================================================
# VN TRADER BOT V5 — data_fetcher.py
# 
# Nguồn dữ liệu (theo thứ tự fallback):
#   1. TCBS v1  — /stock-insight/v1/stock/bars-long-term
#   2. TCBS v2  — /stock-insight/v2/stock/bars-long-term
#   3. SSI      — iboard-query.ssi.com.vn
#
# Features: retry 3x, full browser headers, in-memory cache
# ============================================================

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

from config import DEFAULT_RESOLUTION, DEFAULT_LOOKBACK

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
_TCBS_HOST  = "https://apipubaws.tcbs.com.vn"
_SSI_HOST   = "https://iboard-query.ssi.com.vn"

_CACHE_TTL  = 3600   # 1 hour
_RETRY      = 3
_RETRY_WAIT = 1.5    # seconds (multiplied by attempt number)

# ── Full browser headers ──────────────────────────────────────
_HEADERS_TCBS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":           "application/json, text/plain, */*",
    "Accept-Language":  "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Accept-Encoding":  "gzip, deflate, br",
    "Referer":          "https://tcinvest.tcbs.com.vn/",
    "Origin":           "https://tcinvest.tcbs.com.vn",
    "sec-ch-ua":        '"Chromium";v="122", "Not(A:Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-fetch-dest":   "empty",
    "sec-fetch-mode":   "cors",
    "sec-fetch-site":   "cross-site",
    "DNT":              "1",
    "Connection":       "keep-alive",
}

_HEADERS_SSI = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer":         "https://iboard.ssi.com.vn/",
    "Origin":          "https://iboard.ssi.com.vn",
    "DNT":             "1",
}

# ── Caches ────────────────────────────────────────────────────
_vni_cache: dict = {"data": None, "fetched_at": None}
_sym_cache:  dict[str, dict] = {}


# ── HTTP helper ───────────────────────────────────────────────
def _get(url: str, params: dict, headers: dict) -> dict | None:
    """GET with retry. Returns parsed JSON or None."""
    for attempt in range(1, _RETRY + 1):
        try:
            r = requests.get(
                url, params=params, headers=headers,
                timeout=15, allow_redirects=True,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning(f"HTTP {r.status_code} attempt {attempt}: {url}")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout attempt {attempt}: {url}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"ConnError attempt {attempt}: {e}")
        except Exception as e:
            logger.warning(f"Error attempt {attempt}: {e}")
        if attempt < _RETRY:
            time.sleep(_RETRY_WAIT * attempt)
    return None


# ── Column normaliser ─────────────────────────────────────────
_COL = {
    # TCBS bars-long-term response keys
    "tradingDate": "date",  "TradingDate": "date",
    "open":  "open",  "Open":  "open",  "o": "open",
    "high":  "high",  "High":  "high",  "h": "high",
    "low":   "low",   "Low":   "low",   "l": "low",
    "close": "close", "Close": "close", "c": "close",
    "volume":"volume","Volume":"volume", "v": "volume",
    # SSI keys
    "time": "date", "Time": "date",
}

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in _COL.items() if k in df.columns})
    need = {"date","open","high","low","close","volume"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        logger.debug(f"Missing cols after rename: {missing} | got: {list(df.columns)}")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(need))
    return df.sort_values("date").reset_index(drop=True)[list(need)]


# ── Source 1: TCBS v1 ─────────────────────────────────────────
def _tcbs_v1(symbol: str, count: int) -> pd.DataFrame:
    """
    GET /stock-insight/v1/stock/bars-long-term
    params: ticker, type, from, to  (unix timestamps)
    """
    to_ts   = int(time.time())
    from_ts = to_ts - count * 86400 * 2   # 2× buffer for weekends/holidays
    url     = f"{_TCBS_HOST}/stock-insight/v1/stock/bars-long-term"
    data    = _get(url, {
        "ticker": symbol,
        "type":   "D",
        "from":   from_ts,
        "to":     to_ts,
    }, _HEADERS_TCBS)

    if not data:
        return pd.DataFrame()

    # Response có thể là list trực tiếp hoặc dict có key data/bars
    if isinstance(data, list):
        bars = data
    else:
        bars = (data.get("data") or data.get("bars") or
                data.get("listStock") or data.get("ohlcList") or [])

    if not bars:
        logger.debug(f"[{symbol}] TCBS v1: empty bars, keys={list(data.keys()) if isinstance(data,dict) else 'list'}")
        return pd.DataFrame()

    df = _normalise(pd.DataFrame(bars))
    if not df.empty and len(df) > count:
        df = df.iloc[-count:].reset_index(drop=True)
    return df


# ── Source 2: TCBS v2 ─────────────────────────────────────────
def _tcbs_v2(symbol: str, count: int) -> pd.DataFrame:
    """
    GET /stock-insight/v2/stock/bars-long-term
    Same params as v1 but different response structure.
    """
    to_ts   = int(time.time())
    from_ts = to_ts - count * 86400 * 2
    url     = f"{_TCBS_HOST}/stock-insight/v2/stock/bars-long-term"
    data    = _get(url, {
        "ticker": symbol,
        "type":   "D",
        "from":   from_ts,
        "to":     to_ts,
    }, _HEADERS_TCBS)

    if not data:
        return pd.DataFrame()

    if isinstance(data, list):
        bars = data
    else:
        bars = (data.get("data") or data.get("bars") or
                data.get("listStock") or data.get("ohlcList") or [])

    if not bars:
        return pd.DataFrame()

    df = _normalise(pd.DataFrame(bars))
    if not df.empty and len(df) > count:
        df = df.iloc[-count:].reset_index(drop=True)
    return df


# ── Source 3: SSI iBoard (final fallback) ────────────────────
def _ssi(symbol: str, count: int) -> pd.DataFrame:
    """
    GET https://iboard-query.ssi.com.vn/v2/stock/bars-long-term
    SSI uses symbol without exchange suffix.
    """
    to_ts   = int(time.time())
    from_ts = to_ts - count * 86400 * 2
    url     = f"{_SSI_HOST}/v2/stock/bars-long-term"
    data    = _get(url, {
        "symbol":     symbol,
        "resolution": "D",
        "from":       from_ts,
        "to":         to_ts,
    }, _HEADERS_SSI)

    if not data:
        return pd.DataFrame()

    # SSI response: {"data": {"t":[...], "o":[...], "h":[...], "l":[...], "c":[...], "v":[...]}}
    inner = data.get("data", data)
    if isinstance(inner, dict) and "t" in inner:
        try:
            df = pd.DataFrame({
                "date":   pd.to_datetime(inner["t"], unit="s"),
                "open":   inner["o"],
                "high":   inner["h"],
                "low":    inner["l"],
                "close":  inner["c"],
                "volume": inner["v"],
            })
            df = df.sort_values("date").reset_index(drop=True)
            for c in ("open","high","low","close","volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna()
            if len(df) > count:
                df = df.iloc[-count:].reset_index(drop=True)
            return df
        except Exception as e:
            logger.warning(f"[{symbol}] SSI parse error: {e}")
            return pd.DataFrame()

    # Fallback: try as list of dicts
    bars = inner if isinstance(inner, list) else []
    if not bars:
        return pd.DataFrame()
    return _normalise(pd.DataFrame(bars))


# ── Public fetch (cascading fallback) ────────────────────────
def fetch_ohlcv(
    symbol: str,
    resolution: str = DEFAULT_RESOLUTION,
    count: int = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """
    Fetch OHLCV with 3-source fallback cascade:
      TCBS v1 → TCBS v2 → SSI
    Returns empty DataFrame only if all 3 fail.
    """
    cache_key = f"{symbol}_{count}"
    if cache_key in _sym_cache:
        e = _sym_cache[cache_key]
        if (datetime.utcnow() - e["at"]).total_seconds() < _CACHE_TTL:
            return e["df"]

    # Try each source in order
    sources = [
        ("TCBS-v1", lambda: _tcbs_v1(symbol, count)),
        ("TCBS-v2", lambda: _tcbs_v2(symbol, count)),
        ("SSI",     lambda: _ssi(symbol, count)),
    ]

    for name, fn in sources:
        try:
            df = fn()
            if not df.empty:
                logger.info(f"[{symbol}] ✓ {name} → {len(df)} bars")
                _sym_cache[cache_key] = {"df": df, "at": datetime.utcnow()}
                return df
            logger.info(f"[{symbol}] {name} → empty, trying next")
        except Exception as e:
            logger.warning(f"[{symbol}] {name} exception: {e}")

    logger.error(f"[{symbol}] ❌ All 3 sources failed")
    return pd.DataFrame()


def fetch_vni(force_refresh: bool = False) -> pd.DataFrame:
    """VNINDEX data, cached 1h. Needs 200+ bars for RS filter."""
    now    = datetime.utcnow()
    cached = _vni_cache["data"]
    at     = _vni_cache["fetched_at"]

    if (not force_refresh and cached is not None and at is not None
            and (now - at).total_seconds() < _CACHE_TTL):
        return cached

    df = fetch_ohlcv("VNINDEX", count=250)
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
    delay_sec: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Fetch all symbols sequentially. Skip symbols with no data."""
    results = {}
    for i, sym in enumerate(symbols):
        df = fetch_ohlcv(sym, resolution, count)
        if not df.empty:
            results[sym] = df
        else:
            logger.warning(f"[{sym}] Skipped — no data from any source")
        if i < len(symbols) - 1:
            time.sleep(delay_sec)
    logger.info(f"Fetch complete: {len(results)}/{len(symbols)} symbols OK")
    return results
