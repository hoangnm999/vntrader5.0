# ============================================================
# VN TRADER BOT V5 — config.py
# Architecture: Trend + Momentum + Volume (3-layer)
# Primary metric: Profit Factor
# ============================================================

import os

# ── Telegram ─────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Data source ───────────────────────────────────────────────
DEFAULT_RESOLUTION = "D"   # daily candles
DEFAULT_LOOKBACK   = 200   # bars fetched — needs 63+ for RS filter, 50+ for EMA

# ── 28-symbol watchlist ───────────────────────────────────────
WATCHLIST = [
    "ACB", "BCM", "BID", "BVH", "CTG",
    "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW",
    "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB",
    "VIC", "VNM", "VPB",
]

# ── Global indicator defaults ─────────────────────────────────
# These are the baseline values; per-symbol overrides below.
INDICATOR_DEFAULTS = {
    # Trend
    "ema_fast":        20,
    "ema_slow":        50,
    "adx_period":      14,
    "adx_min":         18,      # lowered from 20 — VN market runs lower ADX

    # Momentum
    "rsi_period":      14,
    "rsi_min":         50,      # RSI floor: 50 (default) or 55 (strict) — backtest to compare
    "rsi_ob":          70,      # overbought — no new longs above
    "rsi_os":          30,      # oversold   — no new shorts below
    "macd_fast":       12,
    "macd_slow":       26,
    "macd_signal":      9,

    # Volume  (REQUIRED layer)
    "vol_ma_period":   20,
    "vol_multiplier":  1.5,     # raised from 1.2 — require clear surge
    "use_vwap":        True,    # always on — VWAP is mandatory, not opt-in

    # Signal gate
    "signal_mode":     "vol_required",  # Volume must be in every signal
    "min_profit_factor": 1.5,
}

# ── Per-symbol overrides ──────────────────────────────────────
# Only specify keys that differ from INDICATOR_DEFAULTS.
# Example: symbols with thin volume need a lower vol_multiplier.
SYMBOL_CONFIG = {
    # rsi_min=55 = strict mode for symbols with cleaner momentum patterns
    "VCB": {"adx_min": 18, "vol_multiplier": 1.4},
    "HPG": {"adx_min": 20, "vol_multiplier": 1.6, "rsi_min": 55},
    "MWG": {"rsi_min": 55},
    "SSI": {"rsi_min": 55, "vol_multiplier": 1.6},
    "FPT": {"ema_fast": 15, "ema_slow": 45, "rsi_min": 55},
    "TCB": {"adx_min": 18},
    "VHM": {"vol_multiplier": 1.5},
    "ACB": {"adx_min": 18},
    "MBB": {"adx_min": 18},
    "STB": {"vol_multiplier": 1.6},
    "VPB": {"adx_min": 18},
    "BID": {"adx_min": 18, "vol_multiplier": 1.4},
    "CTG": {"adx_min": 18, "vol_multiplier": 1.4},
    "GAS": {"rsi_min": 55},
    "VNM": {"ema_fast": 15, "ema_slow": 50},
}


def get_symbol_config(symbol: str) -> dict:
    """
    Return merged config for a symbol.
    Per-symbol keys override INDICATOR_DEFAULTS.
    """
    cfg = INDICATOR_DEFAULTS.copy()
    cfg.update(SYMBOL_CONFIG.get(symbol, {}))
    return cfg


# ── Backtest settings ─────────────────────────────────────────
BACKTEST_CONFIG = {
    "initial_capital":   100_000_000,   # VND
    "position_size_pct": 0.10,          # 10% per trade
    "commission_pct":    0.0015,        # 0.15% per side (VN standard)
    "slippage_pct":      0.001,         # 0.1% slippage estimate
    "stop_loss_atr_mult": 2.0,          # SL = entry - 2×ATR
    "take_profit_rr":    2.0,           # TP = entry + RR × risk
    "max_hold_days":     20,            # force-exit after N days
    "walk_forward_folds": 4,            # WF validation folds
}

# ── Schedule (Railway cron) ───────────────────────────────────
SCAN_HOUR_HCM   = 15   # 15:05 ICT — after ATC
SCAN_MINUTE_HCM = 5
TIMEZONE        = "Asia/Ho_Chi_Minh"
