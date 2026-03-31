# ============================================================
# VN TRADER BOT V5 — data_fetcher.py
# Fetches OHLCV from TCBS. VNI loaded once and cached.
# ============================================================

import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from config import TCBS_BASE_URL, DEFAULT_RESOLUTION, DEFAULT_LOOKBACK

logger = logging.getLogger(__name__)

# ── Internal cache ────────────────────────────────────────────
_vni_cache: dict = {"data": None, "fetched_at": None}
_CACHE_TTL_SECONDS = 3600  # 1 hour


# ── Low-level TCBS fetch ──────────────────────────────────────

def _fetch_tcbs_ohlcv(symbol: str, resolution: str, count: int) -> pd.DataFrame:
    """
    Fetch up to `count` candles for `symbol` from TCBS.
    Returns DataFrame with columns: date, open, high, low, close, volume.
    Returns empty DataFrame on failure.
    """
    to_ts = int(time.time())
    url = (
        f"{TCBS_BASE_URL}/stock-bardata"
        f"?ticker={symbol}&type={resolution}"
        f"&to={to_ts}&count={count}"
    )
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "DNT": "1",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        bars = raw.get("data", [])
        if not bars:
            logger.warning(f"[{symbol}] TCBS returned empty data")
            return pd.DataFrame()

        df = pd.DataFrame(bars)

        # Normalise column names
        rename_map = {
            "tradingDate": "date",
            "open":        "open",
            "high":        "high",
            "low":         "low",
            "close":       "close",
            "volume":      "volume",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.warning(f"[{symbol}] Missing columns: {required - set(df.columns)}")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        return df[["date", "open", "high", "low", "close", "volume"]]

    except requests.exceptions.Timeout:
        logger.error(f"[{symbol}] TCBS request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"[{symbol}] TCBS request failed: {e}")
    except Exception as e:
        logger.error(f"[{symbol}] Unexpected error: {e}")

    return pd.DataFrame()


# ── Public API ────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    resolution: str = DEFAULT_RESOLUTION,
    count: int = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """Fetch OHLCV for a single symbol."""
    df = _fetch_tcbs_ohlcv(symbol, resolution, count)
    if df.empty:
        logger.warning(f"[{symbol}] No data returned")
    return df


def fetch_vni(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch VN-Index (VNINDEX) data. Cached for 1 hour.
    Pass force_refresh=True to bypass cache.
    """
    now = datetime.utcnow()
    cached = _vni_cache["data"]
    fetched_at = _vni_cache["fetched_at"]

    if (
        not force_refresh
        and cached is not None
        and fetched_at is not None
        and (now - fetched_at).total_seconds() < _CACHE_TTL_SECONDS
    ):
        return cached

    df = _fetch_tcbs_ohlcv("VNINDEX", DEFAULT_RESOLUTION, 200)
    if not df.empty:
        _vni_cache["data"] = df
        _vni_cache["fetched_at"] = now
        logger.info("VNI cache refreshed")
    else:
        logger.warning("VNI fetch failed — returning stale cache if available")
        if cached is not None:
            return cached

    return df


def fetch_all_symbols(
    symbols: list,
    resolution: str = DEFAULT_RESOLUTION,
    count: int = DEFAULT_LOOKBACK,
    delay_sec: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for all symbols.
    Returns dict {symbol: DataFrame}.
    Skips symbols with empty data.
    """
    results = {}
    for i, sym in enumerate(symbols):
        df = fetch_ohlcv(sym, resolution, count)
        if not df.empty:
            results[sym] = df
        else:
            logger.warning(f"[{sym}] Skipped — empty data")
        if i < len(symbols) - 1:
            time.sleep(delay_sec)   # rate-limit courtesy

    logger.info(f"Fetched {len(results)}/{len(symbols)} symbols successfully")
    return results
