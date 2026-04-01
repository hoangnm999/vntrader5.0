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
DEFAULT_RESOLUTION = "D"    # daily candles
DEFAULT_LOOKBACK   = 1250   # ~5 năm trading days — bao phủ 1 full market cycle

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
    "adx_min":         18,

    # Momentum (v5.1: bỏ RSI slope requirement)
    "rsi_period":      14,
    "rsi_min":         50,      # 50 default, 55 strict — per symbol
    "rsi_ob":          70,
    "rsi_os":          30,
    "macd_fast":       12,
    "macd_slow":       26,
    "macd_signal":      9,

    # Volume (v5.1: giảm 1.5x → 1.2x, VWAP tham khảo)
    "vol_ma_period":   20,
    "vol_multiplier":  1.2,     # v5.1: nới lỏng từ 1.5

    # Breakout layer (NEW v5.1)
    "breakout_window":      20,   # highest/lowest N bars
    "momentum_window":      10,   # price momentum lookback
    "momentum_threshold": 0.03,   # 3% gain in 10 days = bull momentum

    # Signal gate (v5.1: 2of3 default)
    "signal_mode":     "2of3",    # nới lỏng từ vol_required
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
    "initial_capital":    100_000_000,  # VND
    "position_size_pct":  0.10,         # 10% per trade
    "commission_pct":     0.0015,       # 0.15% per side (VN standard)
    "slippage_pct":       0.001,        # 0.1% slippage estimate
    "stop_loss_atr_mult": 2.0,          # SL = entry - 2×ATR
    "take_profit_rr":     2.0,          # TP = entry + RR × risk
    "max_hold_days":      20,           # force-exit after N days
    "walk_forward_folds": 3,            # 3 folds — đủ data mỗi fold
    "bt_lookback_bars":   1250,          # ~5 năm — COVID crash + bull 2021 + bear 2022 + recovery 2023-2024
    "wf_min_trades":      5,            # min trades để fold có ý nghĩa
}

# ── Schedule (Railway cron) ───────────────────────────────────
SCAN_HOUR_HCM   = 15   # 15:05 ICT — after ATC
SCAN_MINUTE_HCM = 5
TIMEZONE        = "Asia/Ho_Chi_Minh"
