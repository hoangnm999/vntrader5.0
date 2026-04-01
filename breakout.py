# ============================================================
# signals/breakout.py — Breakout Strategy
#
# Logic: Giá phá đỉnh N ngày + Volume xác nhận mạnh + ADX tăng
#
# Phù hợp VN vì:
#   - VN có nhiều đợt pump rõ ràng sau tích lũy
#   - Breakout khỏi range thường kèm volume đột biến lớn
#   - False breakout lọc được qua ADX + volume double-confirm
#
# Entry: Close phá highest(N) kèm vol surge
# Exit:  TP = entry + 1.5×ATR×RR  |  SL = entry - 1.5×ATR
# ============================================================

import pandas as pd
import numpy as np
from .base import SignalResult, ema, atr, adx, vol_ratio, rsi


# ── Default params ────────────────────────────────────────────
DEFAULT_PARAMS = {
    "breakout_window":  20,     # highest/lowest N bars
    "vol_confirm":      2.0,    # volume phải > MA × X để xác nhận
    "adx_min":          20,     # trend đang hình thành
    "rsi_max_entry":    75,     # không vào khi overbought
    "rsi_min_entry":    25,     # không vào sell khi oversold
    "atr_period":       14,
    "atr_sl_mult":      1.5,    # SL = entry - mult × ATR
    "atr_tp_mult":      2.5,    # TP = entry + mult × ATR  (RR ~1.67)
    "min_bars":         30,     # bars tối thiểu để tính được indicator
}


def compute(
    symbol: str,
    df: pd.DataFrame,
    params: dict = None,
) -> SignalResult:
    """
    Breakout signal cho 1 symbol.

    Bullish breakout:
      Close >= highest(close, N) × 0.998   ← phá hoặc test đỉnh N ngày
      Volume > MA20 × vol_confirm           ← dòng tiền xác nhận
      ADX > adx_min                         ← trend đang có lực
      RSI < rsi_max_entry                   ← chưa overbought cực

    Bearish breakdown:
      Close <= lowest(close, N) × 1.002
      Volume > MA20 × vol_confirm
      ADX > adx_min
      RSI > rsi_min_entry
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if len(df) < p["min_bars"]:
        return _neutral(symbol, df, p, "Not enough bars")

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # Indicators
    atr_s   = atr(high, low, close, p["atr_period"])
    adx_s   = adx(high, low, close, 14)
    rsi_s   = rsi(close, 14)
    vr      = vol_ratio(volume, 20)

    # Latest values
    cur        = float(close.iloc[-1])
    cur_atr    = float(atr_s.iloc[-1])
    cur_adx    = float(adx_s.iloc[-1])
    cur_rsi    = float(rsi_s.iloc[-1])
    cur_vr     = float(vr.iloc[-1])
    cur_date   = df["date"].iloc[-1]

    # Breakout levels — exclude current bar to avoid look-ahead
    n   = p["breakout_window"]
    h20 = float(close.iloc[-(n+1):-1].max()) if len(close) > n else float(close.max())
    l20 = float(close.iloc[-(n+1):-1].min()) if len(close) > n else float(close.min())

    # Conditions
    is_breakout  = cur >= h20 * 0.998
    is_breakdown = cur <= l20 * 1.002
    vol_ok       = cur_vr >= p["vol_confirm"]
    adx_ok       = cur_adx >= p["adx_min"]
    not_ob       = cur_rsi < p["rsi_max_entry"]
    not_os       = cur_rsi > p["rsi_min_entry"]

    values = {
        "high20": round(h20, 2), "low20": round(l20, 2),
        "atr": round(cur_atr, 2), "adx": round(cur_adx, 1),
        "rsi": round(cur_rsi, 1), "vol_ratio": round(cur_vr, 2),
    }

    # ── BUY signal ────────────────────────────────────────────
    if is_breakout and vol_ok and adx_ok and not_ob:
        sl    = cur - cur_atr * p["atr_sl_mult"]
        tp    = cur + cur_atr * p["atr_tp_mult"]
        risk  = cur - sl
        rr    = round((tp - cur) / risk, 2) if risk > 0 else 0
        conf  = _confidence(cur_vr, p["vol_confirm"], cur_adx, p["adx_min"], cur_rsi, "bull")
        reason = (
            f"Breakout > {h20:.0f} | "
            f"Vol={cur_vr:.1f}×MA | "
            f"ADX={cur_adx:.1f} | RSI={cur_rsi:.1f}"
        )
        return SignalResult(
            symbol=symbol, strategy="breakout", date=cur_date,
            close=cur, signal=1, confidence=conf,
            entry_price=cur, sl_price=round(sl,2), tp_price=round(tp,2), rr=rr,
            reason=reason, values=values,
        )

    # ── SELL signal ───────────────────────────────────────────
    if is_breakdown and vol_ok and adx_ok and not_os:
        sl    = cur + cur_atr * p["atr_sl_mult"]
        tp    = cur - cur_atr * p["atr_tp_mult"]
        risk  = sl - cur
        rr    = round((cur - tp) / risk, 2) if risk > 0 else 0
        conf  = _confidence(cur_vr, p["vol_confirm"], cur_adx, p["adx_min"], cur_rsi, "bear")
        reason = (
            f"Breakdown < {l20:.0f} | "
            f"Vol={cur_vr:.1f}×MA | "
            f"ADX={cur_adx:.1f} | RSI={cur_rsi:.1f}"
        )
        return SignalResult(
            symbol=symbol, strategy="breakout", date=cur_date,
            close=cur, signal=-1, confidence=conf,
            entry_price=cur, sl_price=round(sl,2), tp_price=round(tp,2), rr=rr,
            reason=reason, values=values,
        )

    # ── Neutral ───────────────────────────────────────────────
    missing = []
    if not is_breakout and not is_breakdown:
        missing.append(f"price in range [{l20:.0f},{h20:.0f}]")
    if not vol_ok:
        missing.append(f"vol={cur_vr:.1f}× (need {p['vol_confirm']}×)")
    if not adx_ok:
        missing.append(f"ADX={cur_adx:.1f} (need {p['adx_min']})")

    return _neutral(symbol, df, p, " | ".join(missing), values)


def _confidence(vr, vr_min, adx_val, adx_min, rsi_val, direction):
    """Score 0–1 dựa trên độ mạnh của các điều kiện."""
    vol_score = min(1.0, (vr - vr_min) / vr_min)
    adx_score = min(1.0, (adx_val - adx_min) / 20)
    if direction == "bull":
        rsi_score = max(0, (75 - rsi_val) / 25)
    else:
        rsi_score = max(0, (rsi_val - 25) / 25)
    return round(0.4 + (vol_score * 0.3 + adx_score * 0.2 + rsi_score * 0.1), 3)


def _neutral(symbol, df, p, reason, values=None):
    cur  = float(df["close"].iloc[-1])
    date = df["date"].iloc[-1]
    atr_val = float(atr(df["high"], df["low"], df["close"], p["atr_period"]).iloc[-1])
    return SignalResult(
        symbol=symbol, strategy="breakout", date=date,
        close=cur, signal=0, confidence=0.0,
        entry_price=cur,
        sl_price=round(cur - atr_val * p["atr_sl_mult"], 2),
        tp_price=round(cur + atr_val * p["atr_tp_mult"], 2),
        rr=p["atr_tp_mult"] / p["atr_sl_mult"],
        reason=f"Breakout NEUTRAL — {reason}",
        values=values or {},
    )
