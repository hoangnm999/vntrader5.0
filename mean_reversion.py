# ============================================================
# signals/mean_reversion.py — Mean Reversion Strategy
#
# Logic: Mua ở đáy vùng tích lũy khi giá pullback về hỗ trợ
#
# Phù hợp VN vì:
#   - 60-70% thời gian VN market sideways → mean rev thắng
#   - Cổ phiếu VN có xu hướng bounce từ MA20 trong sideways
#   - Volume thấp khi pullback = không có distribution = safe entry
#
# Entry: RSI oversold + price về Bollinger Band dưới + vol thấp
# Exit:  TP = MA20 (mean) | SL = dưới BB lower thêm 1 ATR
# ============================================================

import pandas as pd
import numpy as np
from .base import SignalResult, ema, rsi, atr, vol_ratio


DEFAULT_PARAMS = {
    # Bollinger Bands
    "bb_period":        20,
    "bb_std":           2.0,
    # RSI thresholds
    "rsi_os":           35,     # oversold threshold (nới lỏng hơn 30)
    "rsi_ob":           65,     # overbought threshold
    # Range filter: EMA20/50 không cách nhau quá xa (sideways)
    "ema_gap_max_pct":  0.05,   # EMA20 vs EMA50 < 5% → sideways
    # Volume: pullback phải có vol thấp (không phải distribution)
    "vol_max":          1.2,    # volume < MA20 × 1.2 khi pullback
    # Exit
    "atr_period":       14,
    "atr_sl_mult":      1.0,    # SL = BB lower - 1× ATR
    "min_bars":         30,
}


def compute(
    symbol: str,
    df: pd.DataFrame,
    params: dict = None,
) -> SignalResult:
    """
    Mean reversion signal.

    Bullish (mua ở đáy):
      Close <= BB_lower × 1.005    ← chạm hoặc dưới band dưới
      RSI(14) < rsi_os (35)        ← oversold
      Volume < MA20 × vol_max      ← pullback nhẹ, không phải sell-off
      |EMA20 - EMA50| / EMA50 < 5% ← market đang sideways

    Bearish (short ở đỉnh):
      Close >= BB_upper × 0.995
      RSI(14) > rsi_ob (65)
      Volume < MA20 × vol_max
      Sideways condition

    Exit logic:
      TP = BB middle (MA20) — quay về mean
      SL = BB lower - 1×ATR (dưới hỗ trợ)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if len(df) < p["min_bars"]:
        return _neutral(symbol, df, p, "Not enough bars")

    close  = df["close"]
    volume = df["volume"]

    # Bollinger Bands
    bb_mid = close.rolling(p["bb_period"]).mean()
    bb_std = close.rolling(p["bb_period"]).std()
    bb_upper = bb_mid + p["bb_std"] * bb_std
    bb_lower = bb_mid - p["bb_std"] * bb_std

    rsi_s  = rsi(close, 14)
    atr_s  = atr(df["high"], df["low"], close, p["atr_period"])
    vr     = vol_ratio(volume, 20)
    ema20  = ema(close, 20)
    ema50  = ema(close, 50)

    cur      = float(close.iloc[-1])
    cur_rsi  = float(rsi_s.iloc[-1])
    cur_atr  = float(atr_s.iloc[-1])
    cur_vr   = float(vr.iloc[-1])
    cur_date = df["date"].iloc[-1]

    cur_bb_mid   = float(bb_mid.iloc[-1])
    cur_bb_upper = float(bb_upper.iloc[-1])
    cur_bb_lower = float(bb_lower.iloc[-1])
    cur_ema20    = float(ema20.iloc[-1])
    cur_ema50    = float(ema50.iloc[-1])

    # Sideways filter
    ema_gap_pct = abs(cur_ema20 - cur_ema50) / cur_ema50 if cur_ema50 != 0 else 1.0
    is_sideways = ema_gap_pct < p["ema_gap_max_pct"]

    # BB width % (đo độ rộng của range)
    bb_width_pct = (cur_bb_upper - cur_bb_lower) / cur_bb_mid if cur_bb_mid != 0 else 0

    values = {
        "bb_upper": round(cur_bb_upper, 2),
        "bb_mid":   round(cur_bb_mid, 2),
        "bb_lower": round(cur_bb_lower, 2),
        "bb_width_pct": round(bb_width_pct * 100, 1),
        "rsi":      round(cur_rsi, 1),
        "vol_ratio": round(cur_vr, 2),
        "ema_gap_pct": round(ema_gap_pct * 100, 2),
        "atr":      round(cur_atr, 2),
    }

    # ── BUY: Chạm đáy Bollinger ───────────────────────────────
    at_lower = cur <= cur_bb_lower * 1.005
    bull_rsi = cur_rsi <= p["rsi_os"]
    low_vol  = cur_vr  <= p["vol_max"]

    if at_lower and bull_rsi and low_vol and is_sideways:
        sl   = round(cur_bb_lower - cur_atr * p["atr_sl_mult"], 2)
        tp   = round(cur_bb_mid, 2)   # TP = MA20 (mean)
        risk = cur - sl
        rr   = round((tp - cur) / risk, 2) if risk > 0 else 0
        conf = _confidence(cur_rsi, p["rsi_os"], cur_vr, ema_gap_pct, "bull")
        return SignalResult(
            symbol=symbol, strategy="mean_reversion", date=cur_date,
            close=cur, signal=1, confidence=conf,
            entry_price=cur, sl_price=sl, tp_price=tp, rr=rr,
            reason=(
                f"BB lower touch ({cur_bb_lower:.0f}) | "
                f"RSI={cur_rsi:.1f} (OS) | "
                f"Vol={cur_vr:.1f}× (quiet) | "
                f"Sideways (EMA gap={ema_gap_pct*100:.1f}%)"
            ),
            values=values,
        )

    # ── SELL: Chạm đỉnh Bollinger ─────────────────────────────
    at_upper = cur >= cur_bb_upper * 0.995
    bear_rsi = cur_rsi >= p["rsi_ob"]

    if at_upper and bear_rsi and low_vol and is_sideways:
        sl   = round(cur_bb_upper + cur_atr * p["atr_sl_mult"], 2)
        tp   = round(cur_bb_mid, 2)   # TP = MA20
        risk = sl - cur
        rr   = round((cur - tp) / risk, 2) if risk > 0 else 0
        conf = _confidence(cur_rsi, p["rsi_ob"], cur_vr, ema_gap_pct, "bear")
        return SignalResult(
            symbol=symbol, strategy="mean_reversion", date=cur_date,
            close=cur, signal=-1, confidence=conf,
            entry_price=cur, sl_price=sl, tp_price=tp, rr=rr,
            reason=(
                f"BB upper touch ({cur_bb_upper:.0f}) | "
                f"RSI={cur_rsi:.1f} (OB) | "
                f"Vol={cur_vr:.1f}× (quiet) | "
                f"Sideways (EMA gap={ema_gap_pct*100:.1f}%)"
            ),
            values=values,
        )

    # ── Neutral ───────────────────────────────────────────────
    missing = []
    if not is_sideways:
        missing.append(f"trending (EMA gap={ema_gap_pct*100:.1f}%>{p['ema_gap_max_pct']*100:.0f}%)")
    elif not at_lower and not at_upper:
        pct_pos = (cur - cur_bb_lower) / (cur_bb_upper - cur_bb_lower) * 100 if (cur_bb_upper - cur_bb_lower) > 0 else 50
        missing.append(f"price mid-range ({pct_pos:.0f}% of BB)")
    if not bull_rsi and not bear_rsi:
        missing.append(f"RSI={cur_rsi:.1f} not extreme")

    return _neutral(symbol, df, p, " | ".join(missing), values)


def _confidence(rsi_val, rsi_threshold, vr_val, ema_gap, direction):
    if direction == "bull":
        rsi_score = max(0, (rsi_threshold - rsi_val) / rsi_threshold)
    else:
        rsi_score = max(0, (rsi_val - rsi_threshold) / (100 - rsi_threshold))
    vol_score  = max(0, 1 - vr_val)                      # thấp hơn = tốt hơn
    side_score = max(0, 1 - ema_gap / 0.05)              # gap nhỏ hơn = tốt hơn
    return round(0.4 + rsi_score * 0.3 + vol_score * 0.2 + side_score * 0.1, 3)


def _neutral(symbol, df, p, reason, values=None):
    cur  = float(df["close"].iloc[-1])
    date = df["date"].iloc[-1]
    bb_mid = float(df["close"].rolling(p["bb_period"]).mean().iloc[-1])
    atr_v  = float(atr(df["high"], df["low"], df["close"], p["atr_period"]).iloc[-1])
    return SignalResult(
        symbol=symbol, strategy="mean_reversion", date=date,
        close=cur, signal=0, confidence=0.0,
        entry_price=cur,
        sl_price=round(cur - atr_v, 2),
        tp_price=round(bb_mid, 2),
        rr=0.0,
        reason=f"MeanRev NEUTRAL — {reason}",
        values=values or {},
    )
