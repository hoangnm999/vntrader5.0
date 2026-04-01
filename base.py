# ============================================================
# signals/base.py — Shared dataclasses & math helpers
# ============================================================

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class SignalResult:
    """Output của mỗi strategy cho 1 symbol tại 1 thời điểm."""
    symbol:        str
    strategy:      str          # "breakout" | "momentum" | "mean_reversion"
    date:          pd.Timestamp
    close:         float
    signal:        int          # +1 BUY / -1 SELL / 0 NEUTRAL
    confidence:    float        # 0.0–1.0
    entry_price:   float        # suggested entry (thường = close)
    sl_price:      float        # stop loss
    tp_price:      float        # take profit
    rr:            float        # risk-reward ratio
    reason:        str          # plain text explanation
    values:        dict = field(default_factory=dict)  # raw indicator values


@dataclass
class AggregatedSignal:
    """Tổng hợp tất cả strategies cho 1 symbol."""
    symbol:    str
    date:      pd.Timestamp
    close:     float
    signals:   list[SignalResult]    # 1 entry per active strategy
    # Convenience
    buy_count:  int = 0
    sell_count: int = 0

    def best_buy(self) -> Optional[SignalResult]:
        buys = [s for s in self.signals if s.signal == 1]
        return max(buys, key=lambda x: x.confidence) if buys else None

    def best_sell(self) -> Optional[SignalResult]:
        sells = [s for s in self.signals if s.signal == -1]
        return max(sells, key=lambda x: x.confidence) if sells else None


# ── Shared math helpers ───────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX only — no DI."""
    tr_s  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up    = high - high.shift(1)
    down  = low.shift(1) - low
    pdm   = np.where((up > down) & (up > 0),   up,   0.0)
    mdm   = np.where((down > up) & (down > 0), down, 0.0)
    atr_s = pd.Series(tr_s).ewm(span=period, adjust=False).mean()
    pdi   = 100 * pd.Series(pdm).ewm(span=period, adjust=False).mean() / atr_s
    mdi   = 100 * pd.Series(mdm).ewm(span=period, adjust=False).mean() / atr_s
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def vol_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume / rolling mean."""
    ma = volume.rolling(period, min_periods=1).mean()
    return volume / ma.replace(0, np.nan)
