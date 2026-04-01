# ============================================================
# signals.py — 3 strategies trong 1 file duy nhất
# Breakout | Momentum | Mean Reversion
# Flat file tránh vấn đề package/folder trên Railway
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ── Shared dataclasses ────────────────────────────────────────

@dataclass
class SignalResult:
    symbol:      str
    strategy:    str       # "breakout" | "momentum" | "mean_reversion"
    date:        pd.Timestamp
    close:       float
    signal:      int       # +1 BUY / -1 SELL / 0 NEUTRAL
    confidence:  float     # 0.0–1.0
    entry_price: float
    sl_price:    float
    tp_price:    float
    rr:          float
    reason:      str
    values:      dict = field(default_factory=dict)


@dataclass
class AggregatedSignal:
    symbol:     str
    date:       pd.Timestamp
    close:      float
    signals:    list
    buy_count:  int = 0
    sell_count: int = 0

    def best_buy(self):
        buys = [s for s in self.signals if s.signal == 1]
        return max(buys, key=lambda x: x.confidence) if buys else None


# ── Shared math ───────────────────────────────────────────────

def _ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()

def _rsi(close: pd.Series, p: int = 14) -> pd.Series:
    d    = close.diff()
    gain = d.clip(lower=0).ewm(span=p, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=p, adjust=False).mean()
    return 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

def _atr(high, low, close, p: int = 14) -> pd.Series:
    tr = pd.concat([high-low, (high-close.shift(1)).abs(),
                    (low-close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def _adx(high, low, close, p: int = 14) -> pd.Series:
    tr  = pd.concat([high-low, (high-close.shift(1)).abs(),
                     (low-close.shift(1)).abs()], axis=1).max(axis=1)
    up  = high - high.shift(1)
    dn  = low.shift(1) - low
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a   = pd.Series(tr).ewm(span=p, adjust=False).mean()
    pdi = 100 * pd.Series(pdm).ewm(span=p, adjust=False).mean() / a
    mdi = 100 * pd.Series(mdm).ewm(span=p, adjust=False).mean() / a
    dx  = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean()

def _volr(volume: pd.Series, p: int = 20) -> pd.Series:
    return volume / volume.rolling(p, min_periods=1).mean().replace(0, np.nan)


# ── Strategy 1: Breakout ──────────────────────────────────────

BREAKOUT_PARAMS = {
    "window": 20, "vol_confirm": 2.0, "adx_min": 20,
    "rsi_max": 75, "rsi_min": 25,
    "sl_mult": 1.5, "tp_mult": 2.5, "atr_p": 14, "min_bars": 30,
}

def _breakout(symbol: str, df: pd.DataFrame, p: dict = None) -> SignalResult:
    p = {**BREAKOUT_PARAMS, **(p or {})}
    if len(df) < p["min_bars"]:
        return _mk_neutral(symbol, "breakout", df, p["sl_mult"], p["tp_mult"], p["atr_p"], "Not enough bars")

    close = df["close"]; high = df["high"]; low = df["low"]
    cur = float(close.iloc[-1]); date = df["date"].iloc[-1]
    cur_atr = float(_atr(high, low, close, p["atr_p"]).iloc[-1])
    cur_adx = float(_adx(high, low, close).iloc[-1])
    cur_rsi = float(_rsi(close).iloc[-1])
    cur_vr  = float(_volr(df["volume"]).iloc[-1])

    n   = p["window"]
    h20 = float(close.iloc[-(n+1):-1].max()) if len(close) > n else float(close.max())
    l20 = float(close.iloc[-(n+1):-1].min()) if len(close) > n else float(close.min())

    vals = {"high20": round(h20,2), "low20": round(l20,2),
            "atr": round(cur_atr,2), "adx": round(cur_adx,1),
            "rsi": round(cur_rsi,1), "vol_ratio": round(cur_vr,2)}

    bo  = cur >= h20 * 0.998
    bd  = cur <= l20 * 1.002
    vok = cur_vr >= p["vol_confirm"]
    aok = cur_adx >= p["adx_min"]

    if bo and vok and aok and cur_rsi < p["rsi_max"]:
        sl = cur - cur_atr * p["sl_mult"]; tp = cur + cur_atr * p["tp_mult"]
        risk = cur - sl
        conf = min(1.0, 0.4 + min(1,(cur_vr-p["vol_confirm"])/p["vol_confirm"])*0.3
                   + min(1,(cur_adx-p["adx_min"])/20)*0.2
                   + max(0,(75-cur_rsi)/25)*0.1)
        return SignalResult(symbol, "breakout", date, cur, 1, round(conf,3),
            cur, round(sl,2), round(tp,2),
            round((tp-cur)/risk,2) if risk>0 else 0,
            f"Breakout>{h20:.0f} | Vol={cur_vr:.1f}x | ADX={cur_adx:.1f} | RSI={cur_rsi:.1f}", vals)

    if bd and vok and aok and cur_rsi > p["rsi_min"]:
        sl = cur + cur_atr * p["sl_mult"]; tp = cur - cur_atr * p["tp_mult"]
        risk = sl - cur
        conf = min(1.0, 0.4 + min(1,(cur_vr-p["vol_confirm"])/p["vol_confirm"])*0.3
                   + min(1,(cur_adx-p["adx_min"])/20)*0.2
                   + max(0,(cur_rsi-25)/25)*0.1)
        return SignalResult(symbol, "breakout", date, cur, -1, round(conf,3),
            cur, round(sl,2), round(tp,2),
            round((cur-tp)/risk,2) if risk>0 else 0,
            f"Breakdown<{l20:.0f} | Vol={cur_vr:.1f}x | ADX={cur_adx:.1f} | RSI={cur_rsi:.1f}", vals)

    missing = []
    if not bo and not bd: missing.append(f"in range [{l20:.0f},{h20:.0f}]")
    if not vok: missing.append(f"vol={cur_vr:.1f}x<{p['vol_confirm']}x")
    if not aok: missing.append(f"ADX={cur_adx:.1f}<{p['adx_min']}")
    return _mk_neutral(symbol, "breakout", df, p["sl_mult"], p["tp_mult"], p["atr_p"],
                       " | ".join(missing), vals)


# ── Strategy 2: Short-term Momentum ──────────────────────────

MOMENTUM_PARAMS = {
    "mom_window": 5, "mom_thresh": 0.03,
    "rsi_bull_lo": 52, "rsi_bull_hi": 72,
    "rsi_bear_lo": 28, "rsi_bear_hi": 48,
    "vol_min": 1.2, "tp_pct": 0.05, "sl_pct": 0.03,
    "atr_p": 14, "min_bars": 20,
}

def _momentum(symbol: str, df: pd.DataFrame, p: dict = None) -> SignalResult:
    p = {**MOMENTUM_PARAMS, **(p or {})}
    if len(df) < p["min_bars"]:
        return _mk_neutral(symbol, "momentum", df, p["sl_pct"], p["tp_pct"], p["atr_p"], "Not enough bars")

    close = df["close"]; date = df["date"].iloc[-1]
    cur   = float(close.iloc[-1])
    prev  = float(close.iloc[-2])
    past  = float(close.iloc[-(p["mom_window"]+1)])
    cur_rsi = float(_rsi(close).iloc[-1])
    cur_vr  = float(_volr(df["volume"], 10).iloc[-1])

    mom     = (cur - past) / past if past != 0 else 0.0
    day_chg = (cur - prev) / prev if prev != 0 else 0.0

    vals = {"mom_5d_pct": round(mom*100,2), "day_chg_pct": round(day_chg*100,2),
            "rsi": round(cur_rsi,1), "vol_ratio": round(cur_vr,2)}

    bull = (mom >= p["mom_thresh"] and day_chg >= 0
            and p["rsi_bull_lo"] <= cur_rsi <= p["rsi_bull_hi"]
            and cur_vr >= p["vol_min"])
    bear = (mom <= -p["mom_thresh"] and day_chg <= 0
            and p["rsi_bear_lo"] <= cur_rsi <= p["rsi_bear_hi"]
            and cur_vr >= p["vol_min"])

    if bull:
        sl = cur*(1-p["sl_pct"]); tp = cur*(1+p["tp_pct"])
        conf = min(1.0, 0.4 + min(1,(mom-p["mom_thresh"])/p["mom_thresh"])*0.35
                   + min(1,(cur_vr-1))*0.15 + max(0,(72-cur_rsi)/20)*0.1)
        return SignalResult(symbol, "momentum", date, cur, 1, round(conf,3),
            cur, round(sl,2), round(tp,2), round(p["tp_pct"]/p["sl_pct"],2),
            f"Mom+{mom*100:.1f}%/5d | Day={day_chg*100:+.1f}% | RSI={cur_rsi:.1f} | Vol={cur_vr:.1f}x", vals)

    if bear:
        sl = cur*(1+p["sl_pct"]); tp = cur*(1-p["tp_pct"])
        conf = min(1.0, 0.4 + min(1,(abs(mom)-p["mom_thresh"])/p["mom_thresh"])*0.35
                   + min(1,(cur_vr-1))*0.15 + max(0,(cur_rsi-28)/20)*0.1)
        return SignalResult(symbol, "momentum", date, cur, -1, round(conf,3),
            cur, round(sl,2), round(tp,2), round(p["tp_pct"]/p["sl_pct"],2),
            f"Mom{mom*100:.1f}%/5d | Day={day_chg*100:+.1f}% | RSI={cur_rsi:.1f} | Vol={cur_vr:.1f}x", vals)

    missing = []
    if abs(mom) < p["mom_thresh"]: missing.append(f"mom={mom*100:.1f}%<{p['mom_thresh']*100:.0f}%")
    if not (p["rsi_bull_lo"]<=cur_rsi<=p["rsi_bull_hi"]) and not (p["rsi_bear_lo"]<=cur_rsi<=p["rsi_bear_hi"]):
        missing.append(f"RSI={cur_rsi:.1f} out of zone")
    return _mk_neutral(symbol, "momentum", df, p["sl_pct"], p["tp_pct"], p["atr_p"],
                       " | ".join(missing), vals)


# ── Strategy 3: Mean Reversion ────────────────────────────────

MEAN_REV_PARAMS = {
    "bb_period": 20, "bb_std": 2.0,
    "rsi_os": 35, "rsi_ob": 65,
    "ema_gap_max": 0.05, "vol_max": 1.2,
    "atr_sl_mult": 1.0, "atr_p": 14, "min_bars": 30,
}

def _mean_reversion(symbol: str, df: pd.DataFrame, p: dict = None) -> SignalResult:
    p = {**MEAN_REV_PARAMS, **(p or {})}
    if len(df) < p["min_bars"]:
        return _mk_neutral(symbol, "mean_reversion", df, p["atr_sl_mult"], 1.5, p["atr_p"], "Not enough bars")

    close = df["close"]; date = df["date"].iloc[-1]
    cur   = float(close.iloc[-1])
    cur_rsi = float(_rsi(close).iloc[-1])
    cur_atr = float(_atr(df["high"], df["low"], close, p["atr_p"]).iloc[-1])
    cur_vr  = float(_volr(df["volume"]).iloc[-1])

    bb_mid   = close.rolling(p["bb_period"]).mean()
    bb_std_s = close.rolling(p["bb_period"]).std()
    bb_u = float((bb_mid + p["bb_std"]*bb_std_s).iloc[-1])
    bb_m = float(bb_mid.iloc[-1])
    bb_l = float((bb_mid - p["bb_std"]*bb_std_s).iloc[-1])

    e20 = float(_ema(close, 20).iloc[-1])
    e50 = float(_ema(close, 50).iloc[-1])
    gap = abs(e20-e50)/e50 if e50!=0 else 1.0
    sideways = gap < p["ema_gap_max"]

    vals = {"bb_upper": round(bb_u,2), "bb_mid": round(bb_m,2), "bb_lower": round(bb_l,2),
            "rsi": round(cur_rsi,1), "vol_ratio": round(cur_vr,2),
            "ema_gap_pct": round(gap*100,2)}

    at_lo = cur <= bb_l * 1.005
    at_hi = cur >= bb_u * 0.995
    lo_vl = cur_vr <= p["vol_max"]

    if at_lo and cur_rsi <= p["rsi_os"] and lo_vl and sideways:
        sl   = round(bb_l - cur_atr * p["atr_sl_mult"], 2)
        tp   = round(bb_m, 2)
        risk = cur - sl
        conf = min(1.0, 0.4 + max(0,(p["rsi_os"]-cur_rsi)/p["rsi_os"])*0.3
                   + max(0,1-cur_vr)*0.2 + max(0,1-gap/0.05)*0.1)
        return SignalResult(symbol, "mean_reversion", date, cur, 1, round(conf,3),
            cur, sl, tp, round((tp-cur)/risk,2) if risk>0 else 0,
            f"BB lower({bb_l:.0f}) | RSI={cur_rsi:.1f}(OS) | Vol={cur_vr:.1f}x(quiet) | Sideways", vals)

    if at_hi and cur_rsi >= p["rsi_ob"] and lo_vl and sideways:
        sl   = round(bb_u + cur_atr * p["atr_sl_mult"], 2)
        tp   = round(bb_m, 2)
        risk = sl - cur
        conf = min(1.0, 0.4 + max(0,(cur_rsi-p["rsi_ob"])/(100-p["rsi_ob"]))*0.3
                   + max(0,1-cur_vr)*0.2 + max(0,1-gap/0.05)*0.1)
        return SignalResult(symbol, "mean_reversion", date, cur, -1, round(conf,3),
            cur, sl, tp, round((cur-tp)/risk,2) if risk>0 else 0,
            f"BB upper({bb_u:.0f}) | RSI={cur_rsi:.1f}(OB) | Vol={cur_vr:.1f}x(quiet) | Sideways", vals)

    missing = []
    if not sideways: missing.append(f"trending(EMA gap={gap*100:.1f}%)")
    elif not at_lo and not at_hi:
        pct = (cur-bb_l)/(bb_u-bb_l)*100 if (bb_u-bb_l)>0 else 50
        missing.append(f"mid-range({pct:.0f}% of BB)")
    if cur_rsi > p["rsi_os"] and cur_rsi < p["rsi_ob"]: missing.append(f"RSI={cur_rsi:.1f} not extreme")
    return _mk_neutral(symbol, "mean_reversion", df, p["atr_sl_mult"], 1.5, p["atr_p"],
                       " | ".join(missing), vals)


# ── Neutral helper ────────────────────────────────────────────

def _mk_neutral(symbol, strategy, df, sl_m, tp_m, atr_p, reason, values=None):
    cur  = float(df["close"].iloc[-1])
    date = df["date"].iloc[-1]
    a    = float(_atr(df["high"], df["low"], df["close"], atr_p).iloc[-1])
    return SignalResult(symbol, strategy, date, cur, 0, 0.0,
        cur, round(cur-a*sl_m,2), round(cur+a*tp_m,2), round(tp_m/sl_m,2),
        f"{strategy.upper()} NEUTRAL — {reason}", values or {})


# ── Aggregator ────────────────────────────────────────────────

def compute_all(symbol: str, df: pd.DataFrame) -> AggregatedSignal:
    """Chạy cả 3 strategies, trả về AggregatedSignal."""
    signals = []
    buy_count = sell_count = 0
    for fn in [_breakout, _momentum, _mean_reversion]:
        try:
            sig = fn(symbol, df)
            signals.append(sig)
            if sig.signal == 1:  buy_count  += 1
            elif sig.signal == -1: sell_count += 1
        except Exception as e:
            logger.error(f"[{symbol}] {fn.__name__} error: {e}", exc_info=True)
    return AggregatedSignal(
        symbol=symbol, date=df["date"].iloc[-1],
        close=float(df["close"].iloc[-1]),
        signals=signals, buy_count=buy_count, sell_count=sell_count,
    )


# ── Telegram formatters ───────────────────────────────────────

_STRATEGY_EMOJI = {"breakout": "🚀", "momentum": "⚡", "mean_reversion": "🔄"}
_STRATEGY_NAME  = {"breakout": "Breakout", "momentum": "Momentum", "mean_reversion": "Mean Rev"}

def format_signal_telegram(agg: AggregatedSignal) -> str:
    active = [s for s in agg.signals if s.signal != 0]
    if not active:
        return ""
    lines = [f"📊 {agg.symbol}  |  Gia: {agg.close:,.0f}"]
    for sig in active:
        direction = "MUA" if sig.signal == 1 else "BAN"
        arrow     = "🟢" if sig.signal == 1 else "🔴"
        em  = _STRATEGY_EMOJI.get(sig.strategy, "•")
        nm  = _STRATEGY_NAME.get(sig.strategy, sig.strategy)
        bar = "█" * int(sig.confidence*5) + "░" * (5-int(sig.confidence*5))
        lines += [
            f"",
            f"{em} {nm} — {arrow} {direction}  [{bar} {sig.confidence:.0%}]",
            f"  Entry:{sig.entry_price:,.0f}  SL:{sig.sl_price:,.0f}  TP:{sig.tp_price:,.0f}  (RR {sig.rr:.1f}:1)",
            f"  {sig.reason}",
        ]
    return "\n".join(lines)


def format_scan_summary(results: dict) -> list:
    return [format_signal_telegram(agg) for agg in results.values()
            if format_signal_telegram(agg)]
