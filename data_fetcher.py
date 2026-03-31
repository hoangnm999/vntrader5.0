# ============================================================
# VN TRADER BOT V5 — data_fetcher.py
# 
# Dùng thư viện vnstock (luôn cập nhật endpoint mới nhất)
# thay vì tự gọi raw API → tránh bị break khi TCBS đổi URL.
#
# Fallback cascade:
#   1. vnstock  source="VCI"  (ổn định nhất)
#   2. vnstock  source="TCBS"
#   3. vnstock  source="MSN"  (Microsoft/Bing Finance)
# ============================================================

import logging
from datetime import datetime, timedelta
import pandas as pd

from config import DEFAULT_LOOKBACK

logger = logging.getLogger(__name__)

_CACHE_TTL  = 3600
_vni_cache: dict = {"data": None, "at": None}
_sym_cache:  dict[str, dict] = {}


# ── vnstock wrapper ───────────────────────────────────────────

def _vnstock_fetch(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    """Fetch via vnstock library for a given source."""
    try:
        from vnstock import Vnstock  # lazy import — only load when needed
        stock = Vnstock().stock(symbol=symbol, source=source)
        df = stock.quote.history(start=start, end=end, interval="1D")
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        logger.debug(f"[{symbol}] vnstock {source} error: {e}")
        return pd.DataFrame()


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise vnstock output to standard OHLCV columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    # vnstock returns: time/date, open, high, low, close, volume
    col_map = {
        "time":  "date", "tradingDate": "date",
        "open":  "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    need = {"date", "open", "high", "low", "close", "volume"}
    if not need.issubset(df.columns):
        logger.debug(f"Missing cols: {need - set(df.columns)}, got: {list(df.columns)}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=list(need))
    return df.sort_values("date").reset_index(drop=True)[list(need)]


# ── Public API ────────────────────────────────────────────────

def fetch_ohlcv(
    symbol: str,
    resolution: str = "D",
    count: int = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """
    Fetch OHLCV with 3-source fallback: VCI → TCBS → MSN.
    Uses in-memory cache (1 hour TTL).
    """
    cache_key = f"{symbol}_{count}"
    cached = _sym_cache.get(cache_key)
    if cached and (datetime.utcnow() - cached["at"]).total_seconds() < _CACHE_TTL:
        return cached["df"]

    # Date range: count bars back (add 50% buffer for weekends/holidays)
    end_date   = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=int(count * 1.6))).strftime("%Y-%m-%d")

    sources = ["VCI", "TCBS", "MSN"]
    for src in sources:
        try:
            df_raw = _vnstock_fetch(symbol, start_date, end_date, src)
            df     = _normalise(df_raw)
            if not df.empty:
                # Trim to requested count
                if len(df) > count:
                    df = df.iloc[-count:].reset_index(drop=True)
                logger.info(f"[{symbol}] ✓ {src} → {len(df)} bars")
                _sym_cache[cache_key] = {"df": df, "at": datetime.utcnow()}
                return df
            logger.info(f"[{symbol}] {src} → empty, trying next")
        except Exception as e:
            logger.warning(f"[{symbol}] {src} exception: {e}")

    logger.error(f"[{symbol}] ❌ All sources failed (VCI/TCBS/MSN)")
    return pd.DataFrame()


def fetch_vni(force_refresh: bool = False) -> pd.DataFrame:
    """VNINDEX daily data, cached 1h. Needs 200+ bars for RS filter."""
    now    = datetime.utcnow()
    cached = _vni_cache["data"]
    at     = _vni_cache["at"]

    if (not force_refresh and cached is not None and at is not None
            and (now - at).total_seconds() < _CACHE_TTL):
        return cached

    df = fetch_ohlcv("VNINDEX", count=250)
    if not df.empty:
        _vni_cache["data"] = df
        _vni_cache["at"]   = now
        logger.info(f"VNI cache refreshed ({len(df)} bars)")
    else:
        logger.warning("VNI fetch failed — using stale cache")
        if cached is not None:
            return cached
    return df


def fetch_all_symbols(
    symbols: list,
    resolution: str = "D",
    count: int = DEFAULT_LOOKBACK,
    delay_sec: float = 0.3,
) -> dict[str, pd.DataFrame]:
    """Fetch all symbols. Skip those with no data."""
    import time
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
