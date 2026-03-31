# ============================================================
# VN TRADER BOT V5 — telegram_bot.py
# Commands: /scan /bt [SYM] /status /help
# Scheduled scan: daily at 15:05 ICT
# ============================================================

import logging
import asyncio
from datetime import datetime
import pytz

from telegram import Update, Bot
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, WATCHLIST,
    BACKTEST_CONFIG, SCAN_HOUR_HCM, SCAN_MINUTE_HCM, TIMEZONE,
    get_symbol_config,
)
from data_fetcher import fetch_all_symbols, fetch_vni, fetch_ohlcv
from indicators import compute_signal
from backtest import run_symbol, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Formatting helpers ────────────────────────────────────────

def _signal_emoji(s: int) -> str:
    return {1: "🟢", -1: "🔴", 0: "⚪"}.get(s, "⚪")


def _format_signal_message(sig, cfg: dict) -> str:
    e = _signal_emoji(sig.final_signal)
    direction = "MUA" if sig.final_signal == 1 else ("BÁN" if sig.final_signal == -1 else "NEUTRAL")

    lines = [
        f"{e} *{sig.symbol}* — {direction}",
        f"Giá: `{sig.close:,.0f}` | {sig.date.strftime('%d/%m')}",
        f"Confidence: `{sig.confidence:.1%}`",
        "",
        f"📈 *Trend:* {sig.trend.reason}",
        f"⚡ *Momentum:* {sig.momentum.reason}",
        f"📊 *Volume:* {sig.volume.reason}",
    ]
    return "\n".join(lines)


def _format_bt_message(symbol: str, result: dict) -> str:
    bt  = result.get("backtest")
    wf  = result.get("walk_forward")

    if not bt:
        return f"❌ Không có dữ liệu backtest cho {symbol}"

    verdict_emoji = {"ROBUST": "✅", "MARGINAL": "🟡", "WEAK": "❌", "THIN_DATA": "⚠️"}.get(bt.verdict, "")
    wf_str = f"\n🔄 *Walk-Forward:* {wf.verdict} (avg PF={wf.avg_pf:.2f}±{wf.std_pf:.2f})" if wf else ""

    lines = [
        f"📋 *Backtest {symbol}*",
        f"{verdict_emoji} Verdict: `{bt.verdict}`",
        "",
        f"Profit Factor: `{bt.profit_factor:.2f}`",
        f"Win Rate:      `{bt.win_rate*100:.1f}%`",
        f"Số lệnh:       `{bt.total_trades}`",
        f"Avg Win:       `+{bt.avg_win_pct:.2f}%`",
        f"Avg Loss:      `{bt.avg_loss_pct:.2f}%`",
        f"Max Drawdown:  `{bt.max_drawdown_pct:.1f}%`",
        f"Total Return:  `{bt.total_return_pct:.1f}%`",
        wf_str,
    ]
    return "\n".join(lines)


# ── Core scan logic ───────────────────────────────────────────

async def run_scan(symbols: list[str] = None) -> list[str]:
    """
    Scan symbols, return list of formatted signal messages.
    Only emits non-neutral signals.
    """
    if symbols is None:
        symbols = WATCHLIST

    fetch_vni()  # refresh VNI cache
    data = fetch_all_symbols(symbols)

    messages = []
    for sym, df in data.items():
        cfg = get_symbol_config(sym)
        sig = compute_signal(sym, df, cfg)
        if sig and sig.final_signal != 0:
            messages.append(_format_signal_message(sig, cfg))

    return messages


# ── Command handlers ──────────────────────────────────────────

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/scan — Run full watchlist scan now."""
    await update.message.reply_text("🔍 Đang quét 28 symbols...")
    try:
        msgs = await run_scan()
        if msgs:
            for m in msgs:
                await update.message.reply_text(m, parse_mode="Markdown")
            await update.message.reply_text(f"✅ Tìm thấy {len(msgs)} tín hiệu.")
        else:
            await update.message.reply_text("⚪ Không có tín hiệu rõ ràng hôm nay.")
    except Exception as e:
        logger.error(f"/scan error: {e}")
        await update.message.reply_text(f"❌ Lỗi khi quét: {e}")


async def cmd_bt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/bt [SYMBOL] — Backtest một symbol."""
    args = context.args
    if not args:
        await update.message.reply_text("Dùng: /bt VCB (hoặc tên symbol bất kỳ)")
        return

    symbol = args[0].upper()
    await update.message.reply_text(f"⏳ Đang backtest {symbol}...")

    try:
        df = fetch_ohlcv(symbol, count=200)
        if df.empty:
            await update.message.reply_text(f"❌ Không lấy được dữ liệu {symbol}")
            return

        result = run_symbol(symbol, df=df, use_walk_forward=True)
        msg = _format_bt_message(symbol, result)
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"/bt {symbol} error: {e}")
        await update.message.reply_text(f"❌ Lỗi backtest {symbol}: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — Bot health check."""
    tz = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    msg = (
        f"✅ *VN Trader Bot V5*\n"
        f"Thời gian: `{now.strftime('%H:%M %d/%m/%Y')} ICT`\n"
        f"Symbols: `{len(WATCHLIST)}`\n"
        f"Scan lúc: `{SCAN_HOUR_HCM:02d}:{SCAN_MINUTE_HCM:02d} ICT` mỗi ngày\n"
        f"Architecture: `Trend · Momentum · Volume`\n"
        f"Metric: `Profit Factor`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — Danh sách lệnh."""
    msg = (
        "*VN Trader Bot V5 — Lệnh*\n\n"
        "/scan — Quét toàn bộ watchlist\n"
        "/bt [SYM] — Backtest symbol (vd: /bt VCB)\n"
        "/status — Trạng thái bot\n"
        "/help — Danh sách lệnh này\n\n"
        "_Tín hiệu phát khi cả 3 layers đồng thuận:_\n"
        "📈 Trend (EMA + ADX)\n"
        "⚡ Momentum (RSI + MACD)\n"
        "📊 Volume (Vol surge + VWAP)"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


# ── Scheduled daily scan ──────────────────────────────────────

async def scheduled_scan(bot: Bot) -> None:
    """Run daily scan and push signals to chat."""
    logger.info("Scheduled scan starting...")
    try:
        msgs = await run_scan()
        tz  = pytz.timezone(TIMEZONE)
        now = datetime.now(tz).strftime("%d/%m/%Y")

        if msgs:
            header = f"📡 *Tín hiệu ngày {now}* ({len(msgs)} symbols)\n{'─'*30}"
            await bot.send_message(TELEGRAM_CHAT_ID, header, parse_mode="Markdown")
            for m in msgs:
                await bot.send_message(TELEGRAM_CHAT_ID, m, parse_mode="Markdown")
        else:
            await bot.send_message(
                TELEGRAM_CHAT_ID,
                f"⚪ *{now}* — Không có tín hiệu rõ ràng.",
                parse_mode="Markdown",
            )
        logger.info(f"Scheduled scan done: {len(msgs)} signals")
    except Exception as e:
        logger.error(f"Scheduled scan error: {e}")
        await bot.send_message(TELEGRAM_CHAT_ID, f"❌ Lỗi scan định kỳ: {e}")


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    # ── Validate env vars — exit(1) cleanly instead of crash-looping ──
    missing = []
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if missing:
        logger.critical(
            "═══════════════════════════════════════════════\n"
            "  ❌ MISSING ENVIRONMENT VARIABLES ON RAILWAY\n"
            "═══════════════════════════════════════════════\n"
            f"  Variables not set: {', '.join(missing)}\n\n"
            "  HOW TO FIX:\n"
            "  Railway Dashboard → Project → Variables → Add:\n"
            f"  {'  '.join(f'{v} = <your value>' for v in missing)}\n"
            "═══════════════════════════════════════════════"
        )
        import sys
        sys.exit(1)   # exit cleanly — Railway will NOT restart on exit(1)

    logger.info("✅ Env vars OK — starting VN Trader Bot V5...")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("scan",   cmd_scan))
    app.add_handler(CommandHandler("bt",     cmd_bt))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("help",   cmd_help))

    # Scheduler
    scheduler = AsyncIOScheduler(timezone=TIMEZONE)
    scheduler.add_job(
        scheduled_scan,
        trigger="cron",
        hour=SCAN_HOUR_HCM,
        minute=SCAN_MINUTE_HCM,
        args=[app.bot],
    )
    scheduler.start()
    logger.info(f"Scheduler started — daily scan at {SCAN_HOUR_HCM:02d}:{SCAN_MINUTE_HCM:02d} ICT")

    logger.info("🤖 VN Trader Bot V5 running — polling for commands...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
