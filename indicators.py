# ============================================================
# VN TRADER BOT V5.1 — indicators.py
#
# THAY ĐỔI SO VỚI v3:
#
# 1. RS PRE-FILTER: Dual (3M+1M) → Single 3M only
#    Lý do: 1M quá noise trên VN market, loại bỏ nhiều signal tốt
#
# 2. RS BENCHMARK: VNINDEX → Sector peers
#    Lý do: VNI bị chi phối bởi 5-6 blue chip → sai lệch cho mid-cap
#    Logic: so sánh stock vs trung bình các mã cùng ngành
#
# 3. LAYER 2 MOMENTUM: Bỏ RSI slope requirement
#    Lý do: VN market pump nhanh, RSI tăng 1 bar là đủ tín hiệu
#    Mới:   RSI > rsi_min + MACD hist rising (bỏ rsi_rising)
#
# 4. LAYER 3 VOLUME: Giảm vol_multiplier 1.5x → 1.2x (default)
#    VWAP: chỉ là tham khảo, không bắt buộc trong gate
#
# 5. LAYER 4 (MỚI): Price Momentum / Breakout — đặc trưng VN
#    Breakout:  Close > highest(20)  ← pump signal VN
#    Momentum:  Close/Close[10] > 1.03  ← 3% tăng trong 10 ngày
#    → Layer này boost score khi có, không bắt buộc
#
# 6. SIGNAL GATE: 2-of-3 thay vì Volume bắt buộc
#    Volume vẫn quan trọng nhưng breakout không luôn có vol surge ngay
# ============================================================

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


# ── Sector map VN ─────────────────────────────────────────────
# Dùng để tính RS vs sector peers thay vì vs VNINDEX
SECTOR_MAP = {
    # Banking
    "VCB": "bank", "BID": "bank", "CTG": "bank",
    "ACB": "bank", "MBB": "bank", "TCB": "bank",
    "VPB": "bank", "HDB": "bank", "STB": "bank",
    "SHB": "bank", "SSB": "bank", "TPB": "bank",
    # Industrial / Steel
    "HPG": "industrial", "GVR": "industrial",
    # Tech
    "FPT": "tech",
    # Real Estate
    "VHM": "realestate", "VIC": "realestate", "BCM": "realestate",
    # Consumer / Retail
    "MWG": "consumer", "VNM": "consumer", "SAB": "consumer", "MSN": "consumer",
    # Energy
    "GAS": "energy", "POW": "energy", "PLX": "energy",
    # Finance / Securities
    "SSI": "finance", "BVH": "finance",
}

# Sector peers lookup — mỗi symbol → danh sách peers cùng ngành
def _get_sector_peers(symbol: str) -> list:
    sector = SECTOR_MAP.get(symbol, "other")
    return [s for s, sec in SECTOR_MAP.items() if sec == sector and s != symbol]


# ── Data classes ──────────────────────────────────────────────

@dataclass
class LayerResult:
    name: str
    signal: int      # +1 bull / -1 bear / 0 neutral
    score: float     # 0.0–1.0
    reason: str
    values: dict = field(default_factory=dict)


@dataclass
class SignalResult:
    symbol: str
    date: pd.Timestamp
    close: float
    trend:        LayerResult
    momentum:     LayerResult
    volume:       LayerResult
    breakout:     LayerResult      # NEW: price momentum/breakout layer
    rs_filter:    bool
    rs_3m:        float            # vs sector (not VNI)
    rs_1m:        float            # kept for display, not used in gate
    layers_agree: int
    final_signal: int
    confidence:   float
    reason:       str


# ── Math helpers ──────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    macd_line   = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line


def _adx_only(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """ADX only — no DI (direction captured by EMA cross)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    up   = high - high.shift(1)
    down = low.shift(1) - low
    pdm  = np.where((up > down) & (up > 0),   up,   0.0)
    mdm  = np.where((down > up) & (down > 0), down, 0.0)
    atr  = pd.Series(tr).ewm(span=period, adjust=False).mean()
    pdi  = 100 * pd.Series(pdm).ewm(span=period, adjust=False).mean() / atr
    mdi  = 100 * pd.Series(mdm).ewm(span=period, adjust=False).mean() / atr
    dx   = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _vwap_rolling(df: pd.DataFrame, window: int = 20) -> pd.Series:
    tp      = (df["high"] + df["low"] + df["close"]) / 3
    cum_tpv = (tp * df["volume"]).rolling(window).sum()
    cum_vol = df["volume"].rolling(window).sum()
    return cum_tpv / cum_vol.replace(0, np.nan)


# ── RS Pre-filter (v5.1: sector-based, 3M only) ───────────────

RS_WINDOW_3M = 63
RS_WINDOW_1M = 21   # kept for display in SignalResult


def _period_return(df: pd.DataFrame, window: int) -> float:
    """Return % over last `window` bars. 0.0 if insufficient data."""
    if len(df) < window + 1:
        return 0.0
    now  = float(df["close"].iloc[-1])
    past = float(df["close"].iloc[-(window + 1)])
    return (now - past) / past * 100 if past != 0 else 0.0


def relative_strength_filter(
    symbol: str,
    df_stock: pd.DataFrame,
    df_benchmark: pd.DataFrame,      # VNI or sector proxy
    df_sector_peers: dict = None,    # {sym: df} for sector RS
) -> tuple[bool, float, float]:
    """
    Sector-aware RS filter (v5.1).

    Nếu có df_sector_peers → so sánh stock vs trung bình sector peers (3M).
    Fallback về VNINDEX nếu không có peer data.

    Returns: (passed, rs_3m, rs_1m)
    passed = rs_3m > 0  (chỉ cần 3M, bỏ 1M requirement)
    """
    # 3M return của stock
    stock_3m = _period_return(df_stock, RS_WINDOW_3M)
    stock_1m = _period_return(df_stock, RS_WINDOW_1M)

    # Benchmark: sector peers nếu có, fallback VNI
    if df_sector_peers and len(df_sector_peers) >= 2:
        peer_returns_3m = []
        for peer_df in df_sector_peers.values():
            r = _period_return(peer_df, RS_WINDOW_3M)
            if r != 0.0:
                peer_returns_3m.append(r)
        bench_3m = float(np.mean(peer_returns_3m)) if peer_returns_3m else _period_return(df_benchmark, RS_WINDOW_3M)
    else:
        bench_3m = _period_return(df_benchmark, RS_WINDOW_3M)

    bench_1m = _period_return(df_benchmark, RS_WINDOW_1M)

    rs_3m = round(stock_3m - bench_3m, 2)
    rs_1m = round(stock_1m - bench_1m, 2)

    # Gate: chỉ cần 3M (bỏ AND 1M)
    has_enough = len(df_stock) >= RS_WINDOW_3M + 1
    passed     = (rs_3m > 0) if has_enough else True

    return passed, rs_3m, rs_1m


# ── Layer 1: Trend ────────────────────────────────────────────

def trend_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """EMA20/50 cross + ADX > adx_min. Không đổi."""
    close = df["close"]
    ema_f = _ema(close, cfg["ema_fast"])
    ema_s = _ema(close, cfg["ema_slow"])
    adx   = _adx_only(df["high"], df["low"], close, cfg["adx_period"])

    ef  = float(ema_f.iloc[-1])
    es  = float(ema_s.iloc[-1])
    adx_val = float(adx.iloc[-1])
    adx_ok  = adx_val >= cfg["adx_min"]

    if adx_ok and ef > es:
        signal = 1
        score  = min(1.0, (adx_val - cfg["adx_min"]) / 20 + 0.5)
        reason = f"EMA bull ({ef:.1f}>{es:.1f}), ADX={adx_val:.1f}"
    elif adx_ok and ef < es:
        signal = -1
        score  = min(1.0, (adx_val - cfg["adx_min"]) / 20 + 0.5)
        reason = f"EMA bear ({ef:.1f}<{es:.1f}), ADX={adx_val:.1f}"
    else:
        signal, score = 0, 0.0
        reason = f"No trend — ADX={adx_val:.1f} (need {cfg['adx_min']}), gap={ef-es:.1f}"

    return LayerResult("Trend", signal, round(score, 3), reason,
                       {"ema_fast": round(ef,2), "ema_slow": round(es,2), "adx": round(adx_val,2)})


# ── Layer 2: Momentum (v5.1: bỏ RSI slope) ───────────────────

def momentum_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """
    v5.1: Bỏ điều kiện RSI rising.
    Bull: RSI > rsi_min  AND  MACD hist > 0 & rising
    Bear: RSI < rsi_max  AND  MACD hist < 0 & falling

    Lý do bỏ slope: VN market pump nhanh 1-2 bars, yêu cầu slope
    làm miss entry điểm đầu của sóng tăng.
    """
    close    = df["close"]
    rsi_s    = _rsi(close, cfg["rsi_period"])
    _, _, h  = _macd(close, cfg["macd_fast"], cfg["macd_slow"], cfg["macd_signal"])

    rsi_now  = float(rsi_s.iloc[-1])
    hist_now = float(h.iloc[-1])
    hist_prev= float(h.iloc[-2]) if len(h) > 1 else 0.0
    hist_rising  = hist_now > hist_prev
    hist_falling = hist_now < hist_prev

    rsi_min = cfg.get("rsi_min", 50)
    rsi_max = 100 - rsi_min
    rsi_ob  = cfg["rsi_ob"]
    rsi_os  = cfg["rsi_os"]
    extreme = rsi_now >= rsi_ob or rsi_now <= rsi_os

    # v5.1: chỉ cần RSI > threshold + MACD direction (bỏ RSI slope)
    bull = (not extreme) and rsi_now > rsi_min and hist_now > 0 and hist_rising
    bear = (not extreme) and rsi_now < rsi_max and hist_now < 0 and hist_falling

    if bull:
        signal = 1
        score  = min(1.0, 0.4 + (rsi_now - rsi_min) / (rsi_ob - rsi_min))
        reason = f"RSI={rsi_now:.1f}>{rsi_min}, MACD hist↑ ({hist_now:.4f})"
    elif bear:
        signal = -1
        score  = min(1.0, 0.4 + (rsi_max - rsi_now) / (rsi_max - rsi_os))
        reason = f"RSI={rsi_now:.1f}<{rsi_max}, MACD hist↓ ({hist_now:.4f})"
    else:
        signal, score = 0, 0.0
        if extreme:
            tag = "OB" if rsi_now >= rsi_ob else "OS"
            reason = f"RSI {tag}={rsi_now:.1f} — no entry"
        else:
            reason = f"Momentum neutral — RSI={rsi_now:.1f} (need>{rsi_min}), hist={hist_now:.4f}"

    return LayerResult("Momentum", signal, round(score, 3), reason,
                       {"rsi": round(rsi_now,2), "macd_hist": round(hist_now,4),
                        "hist_rising": hist_rising})


# ── Layer 3: Volume (v5.1: vol_mult 1.2x, VWAP tham khảo) ───

def volume_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """
    v5.1: vol_multiplier giảm về 1.2x (default mới trong config).
    VWAP tính nhưng không bắt buộc trong gate — chỉ hiển thị tham khảo.
    Gate: Vol/MA20 >= vol_multiplier là đủ để pass.
    """
    vol    = df["volume"]
    close  = df["close"]
    vol_ma = vol.rolling(cfg["vol_ma_period"]).mean()

    cur_vol   = float(vol.iloc[-1])
    ma_val    = float(vol_ma.iloc[-1])
    vol_ratio = round(cur_vol / ma_val, 2) if ma_val > 0 else 0.0
    vol_ok    = vol_ratio >= cfg["vol_multiplier"]

    # VWAP: tính nhưng chỉ dùng để hiển thị + tiebreak
    vwap     = _vwap_rolling(df)
    vwap_val = float(vwap.iloc[-1])
    cur_close = float(close.iloc[-1])
    above_vwap = cur_close > vwap_val

    if vol_ok:
        signal = 1 if above_vwap else -1
        score  = min(1.0, 0.4 + (vol_ratio - 1) * 0.3)
        vwap_tag = f"price>VWAP({vwap_val:.0f})" if above_vwap else f"price<VWAP({vwap_val:.0f})"
        reason = f"Vol {vol_ratio}×MA20, {vwap_tag}"
    else:
        signal, score = 0, 0.0
        reason = f"Vol weak — {vol_ratio}×MA20 (need ≥{cfg['vol_multiplier']}×)"

    return LayerResult("Volume", signal, round(score, 3), reason,
                       {"vol_ratio": vol_ratio, "vwap": round(vwap_val,2),
                        "above_vwap": above_vwap})


# ── Layer 4 (NEW): Price Momentum / Breakout ──────────────────

def breakout_layer(df: pd.DataFrame, cfg: dict) -> LayerResult:
    """
    Layer mới đặc trưng VN market.

    Bullish signal khi có 1 trong 2:
      Breakout:  Close > highest(close, 20 bars)  ← phá đỉnh
      Momentum:  Close / Close[10] > 1.03         ← tăng 3%+ trong 10 ngày

    Bearish signal khi:
      Breakdown: Close < lowest(close, 20 bars)
      Downmom:   Close / Close[10] < 0.97

    Layer này không bắt buộc trong signal gate (boost score).
    Nếu có breakout → confidence tăng đáng kể.
    """
    close = df["close"]
    n_break = cfg.get("breakout_window", 20)
    n_mom   = cfg.get("momentum_window", 10)
    mom_thr = cfg.get("momentum_threshold", 0.03)  # 3%

    cur   = float(close.iloc[-1])
    high20 = float(close.rolling(n_break).max().iloc[-1])
    low20  = float(close.rolling(n_break).min().iloc[-1])

    # Price momentum: so sánh với n_mom bars trước
    if len(close) > n_mom:
        past  = float(close.iloc[-(n_mom + 1)])
        pmom  = (cur - past) / past if past != 0 else 0.0
    else:
        pmom  = 0.0

    is_breakout  = cur >= high20 * 0.995   # 0.5% tolerance
    is_breakdown = cur <= low20  * 1.005
    is_bull_mom  = pmom >= mom_thr
    is_bear_mom  = pmom <= -mom_thr

    if is_breakout or is_bull_mom:
        signal = 1
        score  = 0.8 if is_breakout else 0.5
        tags   = []
        if is_breakout: tags.append(f"breakout>{high20:.0f}")
        if is_bull_mom: tags.append(f"pmom=+{pmom*100:.1f}%")
        reason = " + ".join(tags)
    elif is_breakdown or is_bear_mom:
        signal = -1
        score  = 0.8 if is_breakdown else 0.5
        tags   = []
        if is_breakdown: tags.append(f"breakdown<{low20:.0f}")
        if is_bear_mom:  tags.append(f"pmom={pmom*100:.1f}%")
        reason = " + ".join(tags)
    else:
        signal, score = 0, 0.0
        reason = f"No breakout — price={cur:.0f}, range=[{low20:.0f},{high20:.0f}], pmom={pmom*100:.1f}%"

    return LayerResult("Breakout", signal, round(score, 3), reason,
                       {"high20": round(high20,2), "low20": round(low20,2),
                        "pmom_pct": round(pmom*100,2)})


# ── Aggregator (v5.1: 2-of-3 core layers + breakout boost) ───

def compute_signal(
    symbol: str,
    df: pd.DataFrame,
    cfg: dict,
    df_vni: pd.DataFrame = None,
    df_sector_peers: dict = None,
    mode: str = "2of3",       # v5.1 default: 2of3 (nới lỏng từ vol_required)
) -> Optional[SignalResult]:
    """
    v5.1 pipeline:

    PRE-FILTER: RS 3M > sector peers (hoặc VNINDEX nếu không có peer data)
    LAYERS:     Trend + Momentum + Volume (core 3)
                Breakout (boost layer — không bắt buộc)
    GATE (2of3): bất kỳ 2/3 core layers đồng thuận + RS pass

    Breakout layer nếu đồng thuận → tăng confidence score.
    Nếu breakout trái chiều → cảnh báo nhưng không block signal.

    mode="2of3"       → 2/3 core layers (default v5.1)
    mode="vol_required" → Volume bắt buộc (v3 strict)
    mode="all3"       → cả 3 bắt buộc (v3 strictest)
    """
    min_bars = cfg["ema_slow"] + 10
    if len(df) < min_bars:
        return None

    # ── RS pre-filter (sector-aware) ───────────────────────────
    if df_vni is not None and not df_vni.empty:
        rs_passed, rs_3m, rs_1m = relative_strength_filter(
            symbol, df, df_vni, df_sector_peers
        )
    else:
        rs_passed, rs_3m, rs_1m = True, 0.0, 0.0

    # ── Core layers ────────────────────────────────────────────
    trend    = trend_layer(df, cfg)
    momentum = momentum_layer(df, cfg)
    volume   = volume_layer(df, cfg)
    breakout = breakout_layer(df, cfg)

    core_layers = [trend, momentum, volume]
    bull_count  = sum(1 for l in core_layers if l.signal == 1)
    bear_count  = sum(1 for l in core_layers if l.signal == -1)

    # ── Signal gate ────────────────────────────────────────────
    if mode == "2of3":
        fire_bull = rs_passed and bull_count >= 2
        fire_bear = bear_count >= 2   # RS not required for sells
    elif mode == "vol_required":
        vol_bull     = volume.signal == 1
        vol_bear     = volume.signal == -1
        non_vol_bull = sum(1 for l in [trend, momentum] if l.signal == 1)
        non_vol_bear = sum(1 for l in [trend, momentum] if l.signal == -1)
        fire_bull = rs_passed and vol_bull and non_vol_bull >= 1
        fire_bear = vol_bear and non_vol_bear >= 1
    else:  # all3
        fire_bull = rs_passed and bull_count == 3
        fire_bear = bear_count == 3

    # ── Breakout boost ─────────────────────────────────────────
    # Tăng confidence nếu breakout layer đồng thuận
    base_score = round(sum(l.score for l in core_layers) / 3, 3)
    if fire_bull and breakout.signal == 1:
        confidence = min(1.0, round(base_score * 1.2 + 0.05, 3))
        bo_tag = " + BREAKOUT"
    elif fire_bear and breakout.signal == -1:
        confidence = min(1.0, round(base_score * 1.2 + 0.05, 3))
        bo_tag = " + BREAKDOWN"
    else:
        confidence = base_score if (fire_bull or fire_bear) else 0.0
        bo_tag = ""

    active_layers = bull_count if fire_bull else (bear_count if fire_bear else 0)

    if fire_bull:
        final   = 1
        rs_tag  = f"RS={rs_3m:+.1f}%(sector)"
        summary = f"🟢 BUY [{mode}] — {rs_tag} | {bull_count}/3 layers{bo_tag}"
    elif fire_bear:
        final   = -1
        summary = f"🔴 SELL [{mode}] — {bear_count}/3 layers{bo_tag}"
    else:
        final      = 0
        confidence = 0.0
        rs_tag     = f"RS={rs_3m:+.1f}%" if df_vni is not None else "RS N/A"
        summary    = f"⚪ NEUTRAL — {rs_tag} | bull={bull_count} bear={bear_count}"

    return SignalResult(
        symbol=symbol,
        date=df["date"].iloc[-1],
        close=float(df["close"].iloc[-1]),
        trend=trend, momentum=momentum,
        volume=volume, breakout=breakout,
        rs_filter=rs_passed,
        rs_3m=rs_3m, rs_1m=rs_1m,
        layers_agree=active_layers,
        final_signal=final,
        confidence=confidence,
        reason=summary,
    )


def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    return float(_atr(df["high"], df["low"], df["close"], period).iloc[-1])
