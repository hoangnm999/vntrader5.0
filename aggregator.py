# ============================================================
# signals/aggregator.py — Chạy cả 3 strategies, tổng hợp output
# ============================================================

import logging
from typing import Optional
import pandas as pd

from .base import SignalResult, AggregatedSignal
from . import breakout, momentum, mean_reversion

logger = logging.getLogger(__name__)

# Strategy registry — dễ enable/disable
STRATEGIES = {
    "breakout":       breakout.compute,
    "momentum":       momentum.compute,
    "mean_reversion": mean_reversion.compute,
}

# Per-strategy custom params (override defaults nếu cần)
STRATEGY_PARAMS: dict[str, dict] = {
    "breakout":       {},
    "momentum":       {},
    "mean_reversion": {},
}


def compute_all(
    symbol: str,
    df: pd.DataFrame,
    enabled: list[str] = None,
    custom_params: dict = None,
) -> AggregatedSignal:
    """
    Chạy tất cả (hoặc subset) strategies cho 1 symbol.

    Args:
        symbol:        Mã CK
        df:            OHLCV DataFrame
        enabled:       Danh sách strategy muốn chạy (None = tất cả)
        custom_params: Override params per strategy

    Returns:
        AggregatedSignal với tất cả kết quả
    """
    enabled = enabled or list(STRATEGIES.keys())
    params  = {**STRATEGY_PARAMS, **(custom_params or {})}

    signals  = []
    buy_count = sell_count = 0

    for name in enabled:
        if name not in STRATEGIES:
            logger.warning(f"Unknown strategy: {name}")
            continue
        try:
            sig = STRATEGIES[name](symbol, df, params.get(name, {}))
            signals.append(sig)
            if sig.signal == 1:
                buy_count += 1
            elif sig.signal == -1:
                sell_count += 1
        except Exception as e:
            logger.error(f"[{symbol}] {name} error: {e}", exc_info=True)

    cur  = float(df["close"].iloc[-1])
    date = df["date"].iloc[-1]

    return AggregatedSignal(
        symbol=symbol, date=date, close=cur,
        signals=signals, buy_count=buy_count, sell_count=sell_count,
    )


def format_signal_telegram(agg: AggregatedSignal) -> str:
    """
    Format AggregatedSignal thành Telegram message.
    Chỉ hiển thị strategies có signal (buy hoặc sell).
    """
    active = [s for s in agg.signals if s.signal != 0]
    if not active:
        return ""

    strategy_emoji = {
        "breakout":       "🚀",
        "momentum":       "⚡",
        "mean_reversion": "🔄",
    }
    strategy_name = {
        "breakout":       "Breakout",
        "momentum":       "Momentum",
        "mean_reversion": "Mean Rev",
    }

    lines = [f"📊 *{agg.symbol}*  |  Giá: {agg.close:,.0f}"]

    for sig in active:
        direction = "🟢 MUA" if sig.signal == 1 else "🔴 BÁN"
        em  = strategy_emoji.get(sig.strategy, "•")
        nm  = strategy_name.get(sig.strategy, sig.strategy)
        conf_bar = "█" * int(sig.confidence * 5) + "░" * (5 - int(sig.confidence * 5))

        lines += [
            f"",
            f"{em} *{nm}* — {direction}  [{conf_bar} {sig.confidence:.0%}]",
            f"  Entry: {sig.entry_price:,.0f}  "
            f"SL: {sig.sl_price:,.0f}  "
            f"TP: {sig.tp_price:,.0f}  "
            f"(RR {sig.rr:.1f}:1)",
            f"  {sig.reason}",
        ]

    return "\n".join(lines)


def format_scan_summary(results: dict[str, AggregatedSignal]) -> list[str]:
    """
    Format kết quả scan nhiều symbols.
    Trả về list messages (1 message per symbol có signal).
    """
    messages = []
    for sym, agg in results.items():
        msg = format_signal_telegram(agg)
        if msg:
            messages.append(msg)
    return messages
