# ============================================================
# VN TRADER BOT V5 — indicators.py  (revised)
# Pre-filter : Relative Strength vs VNINDEX (63-day)
# Layer 1    : Trend    — EMA20/50 + ADX (no DI redundancy)
# Layer 2    : Momentum — RSI slope + MACD histogram
# Layer 3    : Volume   — Vol surge + VWAP position
# Signal gate: ALL-3 agree (default) | 2-of-3 (alt mode)
# ============================================================

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


# ── Data classes ──────────────────────────────────────────────

@dataclass
class LayerResult:
    """Result from a single indicator layer."""
    name: str
    signal: int          # +1 bullish, -1 bearish, 0 neutral
    score: float         # 0.0 – 1.0 confidence within the layer
    reason: str          # human-readable explanation
    values: dict = field(default_factory=dict)  # raw indicator values


@dataclass
class SignalResult:
    """Aggregated result for one symbol on one bar."""
    symbol: str
    date: pd.Timestamp
    close: float
    trend:        LayerResult
    momentum:     LayerResult
    volume:       LayerResult
    rs_filter:    bool          # passed Relative Strength pre-filter?
    rs_value:     float         # stock 63d return - VNI 63d return (pct pts)
    layers_agree: int           # how many layers are bullish/bearish
    final_signal: int           # +1 / -1 / 0
    confidence:   float         # average layer score when signal fires
    reason:       str


# ── Helpers ───────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_f = _ema(close, fast)
    ema_s = _ema(close, slow)
    macd_line = ema_f - ema_s
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    """Returns ADX, +DI, -DI as Series."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr      = pd.Series(tr).ewm(span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _vwap_anchored(df: pd.DataFrame, anchor_bars: int = 20) -> pd.Series:
    """
    Simple rolling VWAP anchored to N bars ago.
    Uses typical price × volume / cumulative volume within window.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"]
    cum_tpv = (tp * vol).rolling(anchor_bars).sum()
    cum_vol = vol.rolling(anchor_bars).sum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def _adx_only(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Returns only ADX (no DI). Cleaner — avoids DI redundancy with EMA direction.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr      = pd.Series(tr).ewm(span=period, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


# ── Pre-filter: Relative Strength ────────────────────────────

RS_WINDOW = 63  # ~3 months of trading days

def relative_strength_filter(
    df_stock: pd.DataFrame,
    df_vni: pd.DataFrame,
    window: int = RS_WINDOW,
) -> tuple[bool, float]:
    """
    Compare stock 63-day return vs VNINDEX 63-day return.

    Returns:
        (passed: bool, rs_value: float)
        rs_value > 0  → stock outperforming index  (bullish pre-filter)
        rs_value < 0  → stock underperforming index (bearish / skip)

    Requires at least window+1 bars in both DataFrames.
    Falls back to (True, 0.0) if insufficient data to avoid blocking signals.
    """
    if len(df_stock) < window + 1 or len(df_vni) < window + 1:
        return True, 0.0   # not enough history → don't block

    stock_now  = float(df_stock["close"].iloc[-1])
    stock_past = float(df_stock["close"].iloc[-(window + 1)])
    vni_now    = float(df_vni["close"].iloc[-1])
    vni_past   = float(df_vni["close"].iloc[-(window + 1)])

    if stock_past == 0 or vni_past == 0:
        return True, 0.0

    stock_ret = (stock_now - stock_past) / stock_past * 100
    vni_ret   = (vni_now   - vni_past)   / vni_past   * 100
    rs_value  = round(stock_ret - vni_ret, 2)

    return rs_value > 0, rs_value

# ── Layer 1: Trend ────────────────────────────────────────────

def trend_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """
    Bullish:  EMA_fast > EMA_slow  AND  ADX > adx_min
    Bearish:  EMA_fast < EMA_slow  AND  ADX > adx_min
    Neutral:  ADX < adx_min  (choppy / ranging market)

    DI removed — EMA cross already captures direction cleanly.
    ADX confirms trend strength only (direction-agnostic).
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    ema_f   = _ema(close, cfg["ema_fast"])
    ema_s   = _ema(close, cfg["ema_slow"])
    adx     = _adx_only(high, low, close, cfg["adx_period"])

    ef      = float(ema_f.iloc[-1])
    es      = float(ema_s.iloc[-1])
    adx_val = float(adx.iloc[-1])

    adx_ok   = adx_val >= cfg["adx_min"]
    ema_bull = ef > es
    ema_bear = ef < es

    if adx_ok and ema_bull:
        signal = 1
        score  = min(1.0, (adx_val - cfg["adx_min"]) / 20 + 0.5)
        reason = f"EMA bullish ({ef:.1f}>{es:.1f}), ADX={adx_val:.1f}"
    elif adx_ok and ema_bear:
        signal = -1
        score  = min(1.0, (adx_val - cfg["adx_min"]) / 20 + 0.5)
        reason = f"EMA bearish ({ef:.1f}<{es:.1f}), ADX={adx_val:.1f}"
    else:
        signal = 0
        score  = 0.0
        reason = f"No trend — ADX={adx_val:.1f} (min {cfg['adx_min']}), EMA gap={ef-es:.1f}"

    return LayerResult(
        name="Trend", signal=signal, score=round(score, 3), reason=reason,
        values={"ema_fast": round(ef, 2), "ema_slow": round(es, 2), "adx": round(adx_val, 2)},
    )


# ── Layer 2: Momentum ─────────────────────────────────────────

def momentum_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """
    Bullish:  RSI > 50  AND  RSI rising (slope > 0)  AND  MACD hist > 0 & rising
    Bearish:  RSI < 50  AND  RSI falling              AND  MACD hist < 0 & falling
    Neutral:  RSI extreme (OB/OS) or mixed signals

    RSI slope replaces zone logic — captures momentum acceleration,
    not just position. Predictive power is higher.
    """
    close = df["close"]

    rsi_s = _rsi(close, cfg["rsi_period"])
    _, _, hist = _macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])

    rsi_now  = float(rsi_s.iloc[-1])
    rsi_prev = float(rsi_s.iloc[-2]) if len(rsi_s) > 1 else rsi_now
    hist_now  = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) > 1 else 0.0

    rsi_rising  = rsi_now > rsi_prev
    rsi_falling = rsi_now < rsi_prev
    hist_rising  = hist_now > hist_prev
    hist_falling = hist_now < hist_prev

    rsi_ob = cfg["rsi_ob"]
    rsi_os = cfg["rsi_os"]
    extreme = rsi_now >= rsi_ob or rsi_now <= rsi_os

    bull = (not extreme) and rsi_now > 50 and rsi_rising  and hist_now > 0 and hist_rising
    bear = (not extreme) and rsi_now < 50 and rsi_falling and hist_now < 0 and hist_falling

    if bull:
        signal = 1
        score  = min(1.0, 0.4 + (rsi_now - 50) / 50)
        reason = f"RSI={rsi_now:.1f} >50 ↑, MACD hist rising ({hist_now:.4f})"
    elif bear:
        signal = -1
        score  = min(1.0, 0.4 + (50 - rsi_now) / 50)
        reason = f"RSI={rsi_now:.1f} <50 ↓, MACD hist falling ({hist_now:.4f})"
    else:
        signal = 0
        score  = 0.0
        if extreme:
            tag = "overbought" if rsi_now >= rsi_ob else "oversold"
            reason = f"RSI {tag} ({rsi_now:.1f}) — no entry"
        else:
            slope_str = "↑" if rsi_rising else "↓"
            reason = f"Momentum mixed — RSI={rsi_now:.1f}{slope_str}, hist={hist_now:.4f}"

    return LayerResult(
        name="Momentum", signal=signal, score=round(score, 3), reason=reason,
        values={
            "rsi": round(rsi_now, 2),
            "rsi_rising": rsi_rising,
            "macd_hist": round(hist_now, 4),
            "hist_rising": hist_rising,
        },
    )


# ── Layer 3: Volume ───────────────────────────────────────────

def volume_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """
    Confirms signal with volume surge + optional VWAP position.
    Bullish:  volume > vol_MA * multiplier  AND  (if use_vwap) close > VWAP
    Bearish:  volume > vol_MA * multiplier  AND  (if use_vwap) close < VWAP
    Neutral:  volume below threshold
    """
    vol   = df["volume"]
    close = df["close"]

    vol_ma  = vol.rolling(cfg["vol_ma_period"]).mean()
    cur_vol = float(vol.iloc[-1])
    ma_val  = float(vol_ma.iloc[-1])
    threshold = ma_val * cfg["vol_multiplier"]
    vol_ok = cur_vol >= threshold

    cur_close = float(close.iloc[-1])
    vwap_ok_bull = True
    vwap_ok_bear = True
    vwap_val = None

    if cfg.get("use_vwap"):
        vwap = _vwap_anchored(df)
        vwap_val = float(vwap.iloc[-1])
        vwap_ok_bull = cur_close > vwap_val
        vwap_ok_bear = cur_close < vwap_val

    vol_ratio = round(cur_vol / ma_val, 2) if ma_val > 0 else 0.0

    if vol_ok and vwap_ok_bull:
        signal = 1
        score  = min(1.0, 0.4 + (vol_ratio - 1) * 0.3)
        vwap_str = f", above VWAP ({vwap_val:.2f})" if vwap_val else ""
        reason = f"Volume surge {vol_ratio}× MA{cfg['vol_ma_period']}{vwap_str}"
    elif vol_ok and vwap_ok_bear:
        signal = -1
        score  = min(1.0, 0.4 + (vol_ratio - 1) * 0.3)
        vwap_str = f", below VWAP ({vwap_val:.2f})" if vwap_val else ""
        reason = f"Volume surge {vol_ratio}× MA{cfg['vol_ma_period']}{vwap_str}"
    else:
        signal = 0
        score  = 0.0
        reason = f"Volume weak — {vol_ratio}× MA (need {cfg['vol_multiplier']}×)"

    return LayerResult(
        name="Volume", signal=signal, score=round(score, 3), reason=reason,
        values={"vol_ratio": vol_ratio, "vwap": round(vwap_val, 2) if vwap_val else None},
    )


# ── Aggregator ────────────────────────────────────────────────

def compute_signal(
    symbol: str,
    df: pd.DataFrame,
    cfg: dict,
    df_vni: pd.DataFrame = None,
    mode: str = "all3",          # "all3" | "2of3"
) -> Optional[SignalResult]:
    """
    Run RS pre-filter + all 3 layers. Return SignalResult.

    mode="all3"  → signal only when Trend + Momentum + Volume all agree (default)
    mode="2of3"  → signal when any 2-of-3 layers agree (higher frequency)

    RS pre-filter: stock must outperform VNINDEX over 63 days.
    If df_vni is None, RS filter is skipped (always passes).

    Requires at least cfg['ema_slow'] + 10 bars.
    """
    min_bars = cfg["ema_slow"] + 10
    if len(df) < min_bars:
        return None

    # ── RS pre-filter ──────────────────────────────────────────
    if df_vni is not None and not df_vni.empty:
        rs_passed, rs_value = relative_strength_filter(df, df_vni)
    else:
        rs_passed, rs_value = True, 0.0

    # ── 3 layers ───────────────────────────────────────────────
    trend    = trend_layer(df, cfg)
    momentum = momentum_layer(df, cfg)
    volume   = volume_layer(df, cfg)

    layers = [trend, momentum, volume]
    bull_count = sum(1 for l in layers if l.signal == 1)
    bear_count = sum(1 for l in layers if l.signal == -1)

    # ── Signal gate ────────────────────────────────────────────
    if mode == "all3":
        fire_bull = rs_passed and bull_count == 3
        fire_bear = bear_count == 3          # RS filter only gates longs
    else:  # 2of3
        fire_bull = rs_passed and bull_count >= 2
        fire_bear = bear_count >= 2

    active_layers = bull_count if fire_bull else (bear_count if fire_bear else 0)
    avg_score = round(sum(l.score for l in layers) / 3, 3)

    if fire_bull:
        final   = 1
        summary = f"🟢 BUY ({mode}) — RS+{rs_value:+.1f}%, {bull_count}/3 layers bullish"
    elif fire_bear:
        final   = -1
        summary = f"🔴 SELL ({mode}) — {bear_count}/3 layers bearish"
    else:
        final   = 0
        avg_score = 0.0
        rs_str  = f"RS {rs_value:+.1f}%" if df_vni is not None else "RS N/A"
        disagree = [l.name for l in layers if l.signal not in (
            ([1] if bull_count >= bear_count else [-1])
        )]
        summary = f"⚪ NEUTRAL — {rs_str}, bull={bull_count} bear={bear_count}"

    return SignalResult(
        symbol=symbol,
        date=df["date"].iloc[-1],
        close=float(df["close"].iloc[-1]),
        trend=trend,
        momentum=momentum,
        volume=volume,
        rs_filter=rs_passed,
        rs_value=rs_value,
        layers_agree=active_layers,
        final_signal=final,
        confidence=avg_score,
        reason=summary,
    )


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Convenience: latest ATR value for position sizing / SL calc."""
    atr = _atr(df["high"], df["low"], df["close"], period)
    return float(atr.iloc[-1])
