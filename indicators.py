# ============================================================
# VN TRADER BOT V5 — indicators.py  (v3)
#
# PRE-FILTER  Stock 3M > VNI 3M  AND  Stock 1M > VNI 1M
# LAYER 1     Trend    — EMA20/50 + ADX > 18
# LAYER 2     Momentum — RSI > rsi_min (50|55) + slope + MACD hist
# LAYER 3     Volume   — Vol/MA20 > 1.5 + Price > VWAP  (REQUIRED)
#
# SIGNAL GATE  Volume=1  AND  (Trend=1 OR Momentum=1)
#              i.e. Volume is mandatory in every signal
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
    rs_filter:    bool          # passed BOTH RS pre-filters?
    rs_3m:        float         # stock 63d return - VNI 63d return (pct pts)
    rs_1m:        float         # stock 21d return - VNI 21d return (pct pts)
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


# ── Pre-filter: Dual Relative Strength ───────────────────────

RS_WINDOW_3M = 63   # ~3 months trading days
RS_WINDOW_1M = 21   # ~1 month  trading days


def _rs_return(df_stock: pd.DataFrame, df_vni: pd.DataFrame, window: int) -> float:
    """
    Compute (stock_return - vni_return) over `window` bars.
    Returns 0.0 if insufficient data.
    """
    if len(df_stock) < window + 1 or len(df_vni) < window + 1:
        return 0.0
    s_now  = float(df_stock["close"].iloc[-1])
    s_past = float(df_stock["close"].iloc[-(window + 1)])
    v_now  = float(df_vni["close"].iloc[-1])
    v_past = float(df_vni["close"].iloc[-(window + 1)])
    if s_past == 0 or v_past == 0:
        return 0.0
    return round((s_now - s_past) / s_past * 100 - (v_now - v_past) / v_past * 100, 2)


def relative_strength_filter(
    df_stock: pd.DataFrame,
    df_vni: pd.DataFrame,
) -> tuple[bool, float, float]:
    """
    Dual RS pre-filter: stock must outperform VNINDEX on BOTH timeframes.

    Returns:
        (passed: bool, rs_3m: float, rs_1m: float)
        passed = True only when rs_3m > 0 AND rs_1m > 0
        Falls back gracefully when data is insufficient.
    """
    rs_3m = _rs_return(df_stock, df_vni, RS_WINDOW_3M)
    rs_1m = _rs_return(df_stock, df_vni, RS_WINDOW_1M)

    # If we don't have enough history for a window, don't penalise
    has_3m = len(df_stock) >= RS_WINDOW_3M + 1 and len(df_vni) >= RS_WINDOW_3M + 1
    has_1m = len(df_stock) >= RS_WINDOW_1M + 1 and len(df_vni) >= RS_WINDOW_1M + 1

    pass_3m = (rs_3m > 0) if has_3m else True
    pass_1m = (rs_1m > 0) if has_1m else True

    return (pass_3m and pass_1m), rs_3m, rs_1m

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
    Bullish:  RSI > rsi_min  AND  RSI rising  AND  MACD hist > 0 & rising
    Bearish:  RSI < (100 - rsi_min)  AND  RSI falling  AND  MACD hist < 0 & falling

    rsi_min = 50 (default) or 55 (strict mode) — set in config per symbol.
    RSI slope captures acceleration, not just level.
    """
    close = df["close"]

    rsi_s = _rsi(close, cfg["rsi_period"])
    _, _, hist = _macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])

    rsi_now   = float(rsi_s.iloc[-1])
    rsi_prev  = float(rsi_s.iloc[-2]) if len(rsi_s) > 1 else rsi_now
    hist_now  = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) > 1 else 0.0

    rsi_rising  = rsi_now > rsi_prev
    rsi_falling = rsi_now < rsi_prev
    hist_rising  = hist_now > hist_prev
    hist_falling = hist_now < hist_prev

    rsi_ob  = cfg["rsi_ob"]
    rsi_os  = cfg["rsi_os"]
    rsi_min = cfg.get("rsi_min", 50)          # NEW: configurable floor
    rsi_max = 100 - rsi_min                   # symmetric bear threshold

    extreme = rsi_now >= rsi_ob or rsi_now <= rsi_os

    bull = (not extreme) and rsi_now > rsi_min  and rsi_rising  and hist_now > 0 and hist_rising
    bear = (not extreme) and rsi_now < rsi_max  and rsi_falling and hist_now < 0 and hist_falling

    if bull:
        signal = 1
        score  = min(1.0, 0.4 + (rsi_now - rsi_min) / (rsi_ob - rsi_min))
        reason = f"RSI={rsi_now:.1f} >{rsi_min} ↑, MACD hist rising ({hist_now:.4f})"
    elif bear:
        signal = -1
        score  = min(1.0, 0.4 + (rsi_max - rsi_now) / (rsi_max - rsi_os))
        reason = f"RSI={rsi_now:.1f} <{rsi_max} ↓, MACD hist falling ({hist_now:.4f})"
    else:
        signal = 0
        score  = 0.0
        if extreme:
            tag = "overbought" if rsi_now >= rsi_ob else "oversold"
            reason = f"RSI {tag} ({rsi_now:.1f}) — no entry"
        else:
            slope_str = "↑" if rsi_rising else "↓"
            reason = f"Momentum weak — RSI={rsi_now:.1f}{slope_str} (need >{rsi_min}), hist={hist_now:.4f}"

    return LayerResult(
        name="Momentum", signal=signal, score=round(score, 3), reason=reason,
        values={
            "rsi": round(rsi_now, 2),
            "rsi_min": rsi_min,
            "rsi_rising": rsi_rising,
            "macd_hist": round(hist_now, 4),
            "hist_rising": hist_rising,
        },
    )


# ── Layer 3: Volume ───────────────────────────────────────────

def volume_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """
    REQUIRED layer — every signal must have Volume = +1 or -1.

    Bullish:  Vol/MA20 > vol_multiplier (1.5)  AND  Price > VWAP
    Bearish:  Vol/MA20 > vol_multiplier         AND  Price < VWAP
    Neutral:  volume below threshold OR VWAP conflict

    VWAP is always computed (not opt-in) — it's the dòng tiền anchor.
    vol_multiplier default raised to 1.5 to require clear surge.
    """
    vol   = df["volume"]
    close = df["close"]

    vol_ma    = vol.rolling(cfg["vol_ma_period"]).mean()
    cur_vol   = float(vol.iloc[-1])
    ma_val    = float(vol_ma.iloc[-1])
    vol_ratio = round(cur_vol / ma_val, 2) if ma_val > 0 else 0.0
    vol_ok    = vol_ratio >= cfg["vol_multiplier"]

    # VWAP always on
    vwap     = _vwap_anchored(df)
    vwap_val = float(vwap.iloc[-1])
    cur_close = float(close.iloc[-1])
    above_vwap = cur_close > vwap_val
    below_vwap = cur_close < vwap_val

    if vol_ok and above_vwap:
        signal = 1
        score  = min(1.0, 0.4 + (vol_ratio - 1) * 0.2)
        reason = f"Vol {vol_ratio}×MA20 ≥{cfg['vol_multiplier']}×, price > VWAP ({vwap_val:.1f})"
    elif vol_ok and below_vwap:
        signal = -1
        score  = min(1.0, 0.4 + (vol_ratio - 1) * 0.2)
        reason = f"Vol {vol_ratio}×MA20 ≥{cfg['vol_multiplier']}×, price < VWAP ({vwap_val:.1f})"
    else:
        signal = 0
        score  = 0.0
        if not vol_ok:
            reason = f"Vol weak — {vol_ratio}×MA20 (need ≥{cfg['vol_multiplier']}×)"
        else:
            side = "above" if above_vwap else "below"
            reason = f"Vol ok ({vol_ratio}×) but VWAP conflict — price {side} {vwap_val:.1f}"

    return LayerResult(
        name="Volume", signal=signal, score=round(score, 3), reason=reason,
        values={
            "vol_ratio": vol_ratio,
            "vol_threshold": cfg["vol_multiplier"],
            "vwap": round(vwap_val, 2),
            "above_vwap": above_vwap,
        },
    )


# ── Aggregator ────────────────────────────────────────────────

def compute_signal(
    symbol: str,
    df: pd.DataFrame,
    cfg: dict,
    df_vni: pd.DataFrame = None,
    mode: str = "vol_required",   # "vol_required" | "all3" | "2of3"
) -> Optional[SignalResult]:
    """
    Full pipeline: dual RS pre-filter → 3 layers → signal gate.

    mode="vol_required" (default)
        BUY:  RS pass + Volume=+1 + (Trend=+1 OR Momentum=+1)
        SELL: Volume=-1 + (Trend=-1 OR Momentum=-1)   [RS not required for sells]

    mode="all3"
        All 3 layers must agree. RS required for BUY.

    mode="2of3"
        Any 2-of-3 layers agree. RS required for BUY.
        Note: less selective, use only for backtest comparison.

    Requires at least cfg['ema_slow'] + 10 bars.
    """
    min_bars = cfg["ema_slow"] + 10
    if len(df) < min_bars:
        return None

    # ── Dual RS pre-filter ─────────────────────────────────────
    if df_vni is not None and not df_vni.empty:
        rs_passed, rs_3m, rs_1m = relative_strength_filter(df, df_vni)
    else:
        rs_passed, rs_3m, rs_1m = True, 0.0, 0.0

    # ── 3 layers ───────────────────────────────────────────────
    trend    = trend_layer(df, cfg)
    momentum = momentum_layer(df, cfg)
    volume   = volume_layer(df, cfg)

    bull_count = sum(1 for l in [trend, momentum, volume] if l.signal == 1)
    bear_count = sum(1 for l in [trend, momentum, volume] if l.signal == -1)

    vol_bull = volume.signal == 1
    vol_bear = volume.signal == -1
    non_vol_bull = sum(1 for l in [trend, momentum] if l.signal == 1)
    non_vol_bear = sum(1 for l in [trend, momentum] if l.signal == -1)

    # ── Signal gate ────────────────────────────────────────────
    if mode == "vol_required":
        fire_bull = rs_passed and vol_bull and non_vol_bull >= 1
        fire_bear = vol_bear and non_vol_bear >= 1   # RS not required for sells
    elif mode == "all3":
        fire_bull = rs_passed and bull_count == 3
        fire_bear = bear_count == 3
    else:  # 2of3
        fire_bull = rs_passed and bull_count >= 2
        fire_bear = bear_count >= 2

    active_layers = bull_count if fire_bull else (bear_count if fire_bear else 0)
    avg_score = round(sum(l.score for l in [trend, momentum, volume]) / 3, 3)

    if fire_bull:
        final   = 1
        rs_tag  = f"RS 3M={rs_3m:+.1f}% 1M={rs_1m:+.1f}%"
        summary = f"🟢 BUY [{mode}] — {rs_tag} | {bull_count}/3 layers"
    elif fire_bear:
        final   = -1
        summary = f"🔴 SELL [{mode}] — {bear_count}/3 layers bearish"
    else:
        final     = 0
        avg_score = 0.0
        rs_tag    = f"RS 3M={rs_3m:+.1f}% 1M={rs_1m:+.1f}%" if df_vni is not None else "RS N/A"
        vol_tag   = f"vol={'✅' if vol_bull else '❌'}"
        summary   = f"⚪ NEUTRAL — {rs_tag} | {vol_tag} | bull={bull_count} bear={bear_count}"

    return SignalResult(
        symbol=symbol,
        date=df["date"].iloc[-1],
        close=float(df["close"].iloc[-1]),
        trend=trend,
        momentum=momentum,
        volume=volume,
        rs_filter=rs_passed,
        rs_3m=rs_3m,
        rs_1m=rs_1m,
        layers_agree=active_layers,
        final_signal=final,
        confidence=avg_score,
        reason=summary,
    )


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Convenience: latest ATR value for position sizing / SL calc."""
    atr = _atr(df["high"], df["low"], df["close"], period)
    return float(atr.iloc[-1])
