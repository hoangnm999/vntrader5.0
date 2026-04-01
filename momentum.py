# ============================================================
# signals/momentum.py — Short-term Momentum Strategy
#
# Logic: Bắt sóng ngắn 5-10 ngày đặc trưng VN
#
# Phù hợp VN vì:
#   - VN có nhiều đợt pump 1-3 tuần theo tin tức / dòng tiền
#   - Momentum có xu hướng tiếp tục ngắn hạn (2-5 ngày)
#   - Exit nhanh tránh bị kẹp khi sóng kết thúc đột ngột
#
# Entry: Giá tăng 3%+ trong 5 ngày + RSI vùng momentum + Vol tăng
# Exit:  TP cố định 5% | SL 3% | Max hold 8 ngày
# ============================================================

import pandas as pd
import numpy as np
from .base import SignalResult, rsi, atr, vol_ratio


DEFAULT_PARAMS = {
    # Entry conditions
    "mom_window":       5,      # lookback cho price momentum
    "mom_bull_thresh":  0.03,   # +3% trong 5 ngày → bull momentum
    "mom_bear_thresh": -0.03,   # -3% trong 5 ngày → bear momentum
    "rsi_bull_low":    52,      # RSI tối thiểu để confirm bull (không quá yếu)
    "rsi_bull_high":   72,      # RSI tối đa — không vào khi overbought
    "rsi_bear_low":    28,      # RSI tối thiểu bear — không vào khi oversold
    "rsi_bear_high":   48,      # RSI tối đa bear confirm
    "vol_min":          1.2,    # volume tăng nhẹ xác nhận dòng tiền theo
    # Continuation filter: hôm nay không đảo chiều
    "day_change_bull":  0.0,    # close[0] > close[-1]
    "day_change_bear":  0.0,
    # Exit
    "tp_pct":           0.05,   # chốt lời 5%
    "sl_pct":           0.03,   # cắt lỗ 3%
    "atr_period":       14,
    "min_bars":         20,
}


def compute(
    symbol: str,
    df: pd.DataFrame,
    params: dict = None,
) -> SignalResult:
    """
    Short-term momentum signal.

    Bullish:
      Price change(5d) > +3%       ← momentum đang chạy
      Close > Close[-1]            ← hôm nay tiếp tục tăng
      RSI(14) trong [52, 72]       ← momentum zone, chưa extreme
      Volume > MA10 × 1.2          ← dòng tiền theo

    Bearish: ngược lại.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if len(df) < p["min_bars"]:
        return _neutral(symbol, df, p, "Not enough bars")

    close  = df["close"]
    volume = df["volume"]

    rsi_s  = rsi(close, 14)
    atr_s  = atr(df["high"], df["low"], close, p["atr_period"])
    vr     = vol_ratio(volume, 10)   # 10-bar MA cho momentum (nhanh hơn)

    cur      = float(close.iloc[-1])
    prev     = float(close.iloc[-2])
    past_n   = float(close.iloc[-(p["mom_window"] + 1)])
    cur_rsi  = float(rsi_s.iloc[-1])
    cur_atr  = float(atr_s.iloc[-1])
    cur_vr   = float(vr.iloc[-1])
    cur_date = df["date"].iloc[-1]

    # Price momentum: % change over N days
    mom = (cur - past_n) / past_n if past_n != 0 else 0.0
    # Day change: hôm nay so với hôm qua
    day_chg = (cur - prev) / prev if prev != 0 else 0.0

    values = {
        "mom_5d_pct":  round(mom * 100, 2),
        "day_chg_pct": round(day_chg * 100, 2),
        "rsi":         round(cur_rsi, 1),
        "vol_ratio":   round(cur_vr, 2),
        "atr":         round(cur_atr, 2),
    }

    # ── BUY conditions ────────────────────────────────────────
    bull_mom  = mom >= p["mom_bull_thresh"]
    bull_cont = day_chg >= p["day_change_bull"]   # không đảo chiều
    bull_rsi  = p["rsi_bull_low"] <= cur_rsi <= p["rsi_bull_high"]
    bull_vol  = cur_vr >= p["vol_min"]

    if bull_mom and bull_cont and bull_rsi and bull_vol:
        sl   = cur * (1 - p["sl_pct"])
        tp   = cur * (1 + p["tp_pct"])
        rr   = round(p["tp_pct"] / p["sl_pct"], 2)
        conf = _confidence(mom, p["mom_bull_thresh"], cur_rsi, cur_vr, "bull")
        return SignalResult(
            symbol=symbol, strategy="momentum", date=cur_date,
            close=cur, signal=1, confidence=conf,
            entry_price=cur, sl_price=round(sl, 2), tp_price=round(tp, 2), rr=rr,
            reason=(
                f"Momentum +{mom*100:.1f}%/{p['mom_window']}d | "
                f"Day={day_chg*100:+.1f}% | "
                f"RSI={cur_rsi:.1f} | Vol={cur_vr:.1f}×"
            ),
            values=values,
        )

    # ── SELL conditions ───────────────────────────────────────
    bear_mom  = mom <= p["mom_bear_thresh"]
    bear_cont = day_chg <= -p["day_change_bear"]
    bear_rsi  = p["rsi_bear_low"] <= cur_rsi <= p["rsi_bear_high"]
    bear_vol  = cur_vr >= p["vol_min"]

    if bear_mom and bear_cont and bear_rsi and bear_vol:
        sl   = cur * (1 + p["sl_pct"])
        tp   = cur * (1 - p["tp_pct"])
        rr   = round(p["tp_pct"] / p["sl_pct"], 2)
        conf = _confidence(abs(mom), p["mom_bull_thresh"], cur_rsi, cur_vr, "bear")
        return SignalResult(
            symbol=symbol, strategy="momentum", date=cur_date,
            close=cur, signal=-1, confidence=conf,
            entry_price=cur, sl_price=round(sl, 2), tp_price=round(tp, 2), rr=rr,
            reason=(
                f"Momentum {mom*100:.1f}%/{p['mom_window']}d | "
                f"Day={day_chg*100:+.1f}% | "
                f"RSI={cur_rsi:.1f} | Vol={cur_vr:.1f}×"
            ),
            values=values,
        )

    # ── Neutral ───────────────────────────────────────────────
    missing = []
    if not bull_mom and not bear_mom:
        missing.append(f"mom={mom*100:.1f}% (need ±{p['mom_bull_thresh']*100:.0f}%)")
    if not bull_rsi and not bear_rsi:
        missing.append(f"RSI={cur_rsi:.1f} out of momentum zones")
    if not bull_vol:
        missing.append(f"vol={cur_vr:.1f}× (need {p['vol_min']}×)")
    return _neutral(symbol, df, p, " | ".join(missing), values)


def _confidence(mom_val, mom_thr, rsi_val, vr_val, direction):
    mom_score = min(1.0, (mom_val - mom_thr) / mom_thr)
    vol_score = min(1.0, (vr_val - 1.0) / 1.0)
    if direction == "bull":
        rsi_score = max(0, (72 - rsi_val) / 20)
    else:
        rsi_score = max(0, (rsi_val - 28) / 20)
    return round(0.4 + mom_score * 0.35 + vol_score * 0.15 + rsi_score * 0.1, 3)


def _neutral(symbol, df, p, reason, values=None):
    cur  = float(df["close"].iloc[-1])
    date = df["date"].iloc[-1]
    return SignalResult(
        symbol=symbol, strategy="momentum", date=date,
        close=cur, signal=0, confidence=0.0,
        entry_price=cur,
        sl_price=round(cur * (1 - p["sl_pct"]), 2),
        tp_price=round(cur * (1 + p["tp_pct"]), 2),
        rr=round(p["tp_pct"] / p["sl_pct"], 2),
        reason=f"Momentum NEUTRAL — {reason}",
        values=values or {},
    )
