# ============================================================
# indicators.py — Compatibility shim cho backtest.py
#
# backtest.py dùng _precompute_signals() nội bộ (vectorized).
# File này giữ compute_atr() để backtest không bị lỗi import.
# Signals thực tế giờ nằm trong signals/ package.
# ============================================================

import pandas as pd
import numpy as np


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """ATR tại bar cuối cùng."""
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
