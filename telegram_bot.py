# ============================================================
# VN TRADER BOT V5 — telegram_bot.py
# Compatible: python-telegram-bot==21.9, Python 3.11/3.13
# Scheduler: PTB built-in JobQueue (không dùng APScheduler)
# Commands: /scan /bt [SYM] /wf [SYM] /optimize [SYM] /status /help
# ============================================================

import sys
import asyncio
import logging
from datetime import datetime
import pytz

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder,
    CommandHandler, ContextTypes,
)

from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, WATCHLIST,
    SCAN_HOUR_HCM, SCAN_MINUTE_HCM, TIMEZONE,
    get_symbol_config, BACKTEST_CONFIG,
)
from data_fetcher import fetch_all_symbols, fetch_vni, fetch_ohlcv
from indicators import compute_signal
from backtest import (
    run_symbol, format_bt_telegram, SymbolResult,
    _run_backtest_on_df, _walk_forward, WalkForwardResult,
)
from optimize import run_optimize, format_optimize_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Validate env vars sớm — fail fast & clean ────────────────

def _check_env() -> None:
    missing = []
    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if missing:
        logger.critical(
            "\n══════════════════════════════════════════\n"
            "  ❌ MISSING RAILWAY ENVIRONMENT VARIABLES\n"
            "══════════════════════════════════════════\n"
            f"  Thiếu: {', '.join(missing)}\n"
            "  Fix: Railway Dashboard → Variables → Add\n"
            "══════════════════════════════════════════"
        )
        sys.exit(1)


# ── Formatting ────────────────────────────────────────────────

def _fmt_signal(sig) -> str:
    e = "🟢" if sig.final_signal == 1 else "🔴"
    direction = "MUA" if sig.final_signal == 1 else "BÁN"
    rs_str = f"RS 3M={sig.rs_3m:+.1f}% 1M={sig.rs_1m:+.1f}%" if hasattr(sig, "rs_3m") else ""
    return "\n".join([
        f"{e} *{sig.symbol}* — {direction}  {rs_str}",
        f"Giá: `{sig.close:,.0f}` | {sig.date.strftime('%d/%m')}",
        f"Confidence: `{sig.confidence:.1%}` | Layers: `{sig.layers_agree}/3`",
        "",
        f"📈 Trend: {sig.trend.reason}",
        f"⚡ Momentum: {sig.momentum.reason}",
        f"📊 Volume: {sig.volume.reason}",
    ])


# ── Walk-Forward formatter ────────────────────────────────────

def format_wf_telegram(symbol: str, wf50: WalkForwardResult, wf55: WalkForwardResult) -> str:
    """
    Định dạng kết quả walk-forward cho Telegram.
    Hiển thị 2 chế độ RSI-50 và RSI-55 side-by-side.
    """
    verdict_icon = {
        "ROBUST":       "✅",
        "INCONSISTENT": "🔄",
        "WEAK":         "❌",
        "THIN_DATA":    "⚠️",
    }

    def _fold_lines(wf: WalkForwardResult, label: str) -> list:
        if wf is None:
            return [f"[{label}] Khong co du lieu WF"]

        vi = verdict_icon.get(wf.verdict, "")
        lines = [
            f"[{label}]  {vi} {wf.verdict}",
            f"  avg PF={wf.avg_pf:.2f}  std={wf.std_pf:.2f}  min={wf.min_pf:.2f}",
        ]

        if not wf.folds:
            lines.append("  (khong du fold de phan tich)")
            return lines

        for i, f in enumerate(wf.folds, 1):
            fi = verdict_icon.get(f.verdict, "")
            lines.append(
                f"  Fold {i}: {fi} PF={f.profit_factor:.2f}  "
                f"WR={f.win_rate*100:.1f}%  T={f.total_trades}  "
                f"DD={f.max_drawdown_pct:.1f}%"
            )
        return lines

    def _wf_interpretation(wf: WalkForwardResult) -> str:
        if wf is None:
            return ""
        if wf.verdict == "ROBUST":
            return "Chien luoc on dinh qua cac giai doan khac nhau."
        elif wf.verdict == "INCONSISTENT":
            return "Chien luoc co loi nhuan nhung bien dong giua cac giai doan."
        elif wf.verdict == "WEAK":
            return "Chien luoc yeu — can xem xet lai tham so."
        else:
            return "Khong du du lieu de danh gia."

    n_folds = BACKTEST_CONFIG["walk_forward_folds"]
    sl_m    = BACKTEST_CONFIG["stop_loss_atr_mult"]
    tp_rr   = BACKTEST_CONFIG["take_profit_rr"]
    hold    = BACKTEST_CONFIG["max_hold_days"]

    lines = [
        f"🔄 Walk-Forward {symbol}  ({n_folds} folds)",
        f"⚙️  SL={sl_m}xATR  TP={tp_rr}:1  Hold<={hold}d",
        f"",
    ]

    lines += _fold_lines(wf50, "RSI-50")
    if wf50:
        interp = _wf_interpretation(wf50)
        if interp:
            lines.append(f"  → {interp}")
    lines.append("")

    lines += _fold_lines(wf55, "RSI-55")
    if wf55:
        interp = _wf_interpretation(wf55)
        if interp:
            lines.append(f"  → {interp}")

    # Kết luận
    lines.append("")
    if wf50 and wf55:
        both_robust = wf50.verdict == "ROBUST" and wf55.verdict == "ROBUST"
        either_weak = wf50.verdict == "WEAK" or wf55.verdict == "WEAK"
        if both_robust:
            lines.append("💪 Ket luan: Ca 2 che do deu ROBUST — tin cay cao de live trade.")
        elif either_weak:
            lines.append("⚠️  Ket luan: Co it nhat 1 che do WEAK — can them dieu kien loc.")
        else:
            winner = "RSI-50" if (wf50.avg_pf or 0) >= (wf55.avg_pf or 0) else "RSI-55"
            lines.append(f"📊 Ket luan: {winner} on dinh hon — uu tien su dung che do nay.")

    lines.append("")
    lines.append(f"Dung /bt {symbol} de xem full backtest + WF.")

    return "\n".join(lines)


# ── Scan logic ────────────────────────────────────────────────

async def _do_scan(symbols: list = None) -> list:
    if symbols is None:
        symbols = WATCHLIST
    df_vni = fetch_vni()
    data   = fetch_all_symbols(symbols)
    msgs = []
    for sym, df in data.items():
        cfg = get_symbol_config(sym)
        sig = compute_signal(sym, df, cfg, df_vni=df_vni, mode="vol_required")
        if sig and sig.final_signal != 0:
            msgs.append(_fmt_signal(sig))
    return msgs


# ── Command handlers ──────────────────────────────────────────

async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🔍 Đang quét 28 symbols...")
    try:
        msgs = await _do_scan()
        if msgs:
            for m in msgs:
                await update.message.reply_text(m, parse_mode="Markdown")
            await update.message.reply_text(f"✅ {len(msgs)} tín hiệu tìm thấy.")
        else:
            await update.message.reply_text("⚪ Không có tín hiệu rõ ràng hôm nay.")
    except Exception as e:
        logger.error(f"/scan error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Lỗi: {e}")


async def cmd_bt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Dung: /bt VCB")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(f"⏳ Dang backtest {symbol} (~5 nam, vui long cho ~30s)...")
    try:
        df = fetch_ohlcv(symbol, count=BACKTEST_CONFIG["bt_lookback_bars"])
        if df is None or df.empty:
            await update.message.reply_text(
                f"❌ Khong lay duoc du lieu {symbol}\n"
                f"Kiem tra lai ma CK hoac thu lai sau."
            )
            return

        result = await asyncio.to_thread(run_symbol, symbol, df)
        msg = format_bt_telegram(symbol, result)

        if len(msg) > 4000:
            mid = msg.find("[ RSI-55 ]")
            await update.message.reply_text(msg[:mid].strip())
            await update.message.reply_text(msg[mid:].strip())
        else:
            await update.message.reply_text(msg)
    except Exception as e:
        logger.error(f"/bt {symbol} error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Loi backtest {symbol}: {e}")


async def cmd_wf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /wf VCB — Walk-Forward analysis riêng cho 1 symbol.

    Chạy walk-forward 3 folds cho cả RSI-50 và RSI-55.
    Nhanh hơn /bt vì không cần chạy full backtest + comparison table.
    Dùng để kiểm tra tính ổn định trước khi live trade.
    """
    if not context.args:
        await update.message.reply_text(
            "Dung: /wf VCB\n"
            "Walk-forward chia du lieu thanh 3 fold, test tung fold.\n"
            "Ket qua on dinh = chien luoc khong bi overfit."
        )
        return

    symbol = context.args[0].upper()
    n_folds = BACKTEST_CONFIG["walk_forward_folds"]

    await update.message.reply_text(
        f"🔄 Walk-Forward {symbol} ({n_folds} folds × 2 RSI modes)...\n"
        f"Vui long cho ~20-40 giay."
    )

    try:
        df = fetch_ohlcv(symbol, count=BACKTEST_CONFIG["bt_lookback_bars"])
        if df is None or df.empty:
            await update.message.reply_text(
                f"❌ Khong lay duoc du lieu {symbol}\n"
                f"Kiem tra lai ma CK hoac thu lai sau."
            )
            return

        df_vni = fetch_vni()
        cfg    = get_symbol_config(symbol)

        # CPU-bound: chạy trong thread để không block event loop
        def _run_wf():
            wf50 = _walk_forward(symbol, df, cfg, BACKTEST_CONFIG, df_vni, "rsi50", n_folds)
            wf55 = _walk_forward(symbol, df, cfg, BACKTEST_CONFIG, df_vni, "rsi55", n_folds)
            return wf50, wf55

        wf50, wf55 = await asyncio.to_thread(_run_wf)
        msg = format_wf_telegram(symbol, wf50, wf55)

        if len(msg) > 4000:
            await update.message.reply_text(msg[:4000])
            await update.message.reply_text(msg[4000:])
        else:
            await update.message.reply_text(msg)

    except Exception as e:
        logger.error(f"/wf {symbol} error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Loi walk-forward {symbol}: {e}")


async def cmd_optimize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Dung: /optimize VCB")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(
        f"⚙️ Dang optimize {symbol} (64 combo SL x TP x Hold — parallel)...\n"
        f"Vui long cho ~30-60 giay (nhanh hon v1 nho chay song song).\n"
        f"Bot van nhan lenh khac trong luc nay."
    )
    try:
        df = fetch_ohlcv(symbol, count=BACKTEST_CONFIG["bt_lookback_bars"])
        if df is None or df.empty:
            await update.message.reply_text(f"❌ Khong lay duoc du lieu {symbol}")
            return

        # CPU-heavy grid search: chạy trong thread riêng
        opt = await asyncio.to_thread(run_optimize, symbol, df)
        msg = format_optimize_telegram(symbol, opt)

        if len(msg) > 4000:
            await update.message.reply_text(msg[:4000])
            await update.message.reply_text(msg[4000:])
        else:
            await update.message.reply_text(msg)
    except Exception as e:
        logger.error(f"/optimize {symbol} error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Loi optimize {symbol}: {e}")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tz  = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    await update.message.reply_text(
        f"✅ *VN Trader Bot V5*\n"
        f"Thời gian: `{now.strftime('%H:%M %d/%m/%Y')} ICT`\n"
        f"Symbols: `{len(WATCHLIST)}`\n"
        f"Scan: `{SCAN_HOUR_HCM:02d}:{SCAN_MINUTE_HCM:02d} ICT` mỗi ngày\n"
        f"Architecture: `Trend · Momentum · Volume + RS`\n"
        f"Metric: `Profit Factor`",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*VN Trader Bot V5*\n\n"
        "/scan — Quét toàn bộ watchlist\n"
        "/bt VCB — Backtest đầy đủ (RSI-50 vs RSI-55 + WF)\n"
        "/wf VCB — Walk-Forward riêng (kiểm tra overfit)\n"
        "/optimize VCB — Tìm SL/TP/Hold tối ưu (64 combo)\n"
        "/status — Trạng thái bot\n"
        "/help — Lệnh này\n\n"
        "Luồng đề xuất:\n"
        "  1. /bt → xem tổng quan\n"
        "  2. /wf → kiểm tra ổn định\n"
        "  3. /optimize → tìm params tốt nhất\n\n"
        "Signal khi cả 3 layers đồng thuận + RS vượt VNI:\n"
        "Trend: EMA20/50 + ADX\n"
        "Momentum: RSI slope + MACD\n"
        "Volume: surge + VWAP",
        parse_mode="Markdown",
    )


# ── Scheduled scan (PTB JobQueue — không cần APScheduler) ────

async def _scheduled_scan_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info("Scheduled scan triggered")
    try:
        msgs = await _do_scan()
        tz  = pytz.timezone(TIMEZONE)
        now = datetime.now(tz).strftime("%d/%m/%Y")
        if msgs:
            await context.bot.send_message(
                TELEGRAM_CHAT_ID,
                f"📡 *Tín hiệu ngày {now}* ({len(msgs)} symbols)",
                parse_mode="Markdown",
            )
            for m in msgs:
                await context.bot.send_message(TELEGRAM_CHAT_ID, m, parse_mode="Markdown")
        else:
            await context.bot.send_message(
                TELEGRAM_CHAT_ID,
                f"⚪ *{now}* — Không có tín hiệu.",
                parse_mode="Markdown",
            )
        logger.info(f"Scheduled scan done: {len(msgs)} signals")
    except Exception as e:
        logger.error(f"Scheduled scan error: {e}", exc_info=True)
        try:
            await context.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Lỗi scan định kỳ: {e}")
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    _check_env()
    logger.info("Env vars OK — building app...")

    app: Application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("scan",     cmd_scan))
    app.add_handler(CommandHandler("bt",       cmd_bt))
    app.add_handler(CommandHandler("wf",       cmd_wf))
    app.add_handler(CommandHandler("optimize", cmd_optimize))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("help",     cmd_help))

    # PTB JobQueue — chạy mỗi ngày lúc SCAN_HOUR_HCM:SCAN_MINUTE_HCM ICT
    tz = pytz.timezone(TIMEZONE)
    scan_time = datetime.now(tz).replace(
        hour=SCAN_HOUR_HCM,
        minute=SCAN_MINUTE_HCM,
        second=0,
        microsecond=0,
    ).timetz()

    app.job_queue.run_daily(
        _scheduled_scan_job,
        time=scan_time,
        name="daily_scan",
    )
    logger.info(f"Scheduled scan: {SCAN_HOUR_HCM:02d}:{SCAN_MINUTE_HCM:02d} ICT daily")
    logger.info("VN Trader Bot V5 — polling started")

    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
