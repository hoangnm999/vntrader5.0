# ============================================================
# VN TRADER BOT V5 — telegram_bot.py
# Compatible: python-telegram-bot==21.9, Python 3.11/3.13
# Commands: /scan /bt /wf /wfo /optimize /status /help
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
    BACKTEST_CONFIG,
)
from data_fetcher import fetch_all_symbols, fetch_vni, fetch_ohlcv
from signals.aggregator import compute_all, format_signal_telegram, format_scan_summary
from signals import breakout, momentum, mean_reversion
from backtest import (
    run_symbol, format_bt_telegram, SymbolResult,
    _walk_forward, WalkForwardResult,
)
from optimize import run_optimize, run_wfo, format_optimize_telegram, format_wfo_telegram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Validate env vars ─────────────────────────────────────────

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


# ── Formatting helpers ────────────────────────────────────────

# _fmt_signal replaced by signals/aggregator.format_signal_telegram


def format_wf_telegram(symbol: str, wf50: WalkForwardResult, wf55: WalkForwardResult) -> str:
    """Format walk-forward validation (params cố định) cho Telegram."""
    verdict_icon = {
        "ROBUST": "✅", "INCONSISTENT": "🔄",
        "WEAK": "❌", "THIN_DATA": "⚠️",
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
            lines.append("  (khong du fold)")
            return lines
        for i, f in enumerate(wf.folds, 1):
            fi = verdict_icon.get(f.verdict, "")
            lines.append(
                f"  Fold {i}: {fi} PF={f.profit_factor:.2f}  "
                f"WR={f.win_rate*100:.1f}%  T={f.total_trades}  "
                f"DD={f.max_drawdown_pct:.1f}%"
            )
        return lines

    n_folds = BACKTEST_CONFIG["walk_forward_folds"]
    sl_m    = BACKTEST_CONFIG["stop_loss_atr_mult"]
    tp_rr   = BACKTEST_CONFIG["take_profit_rr"]
    hold    = BACKTEST_CONFIG["max_hold_days"]

    lines = [
        f"🔄 Walk-Forward {symbol}  ({n_folds} folds, params co dinh)",
        f"⚙️  SL={sl_m}xATR  TP={tp_rr}:1  Hold<={hold}d",
        f"",
    ]
    lines += _fold_lines(wf50, "RSI-50")
    lines.append("")
    lines += _fold_lines(wf55, "RSI-55")
    lines += [
        f"",
        f"Dung /wfo {symbol} de tim params toi uu tren moi fold (WFO thuc su).",
    ]
    return "\n".join(lines)


# ── Scan logic ────────────────────────────────────────────────

async def _do_scan(symbols: list = None) -> list:
    """Scan dung 3 strategies: breakout + momentum + mean_reversion."""
    if symbols is None:
        symbols = WATCHLIST
    data = fetch_all_symbols(symbols)
    results = {}
    for sym, df in data.items():
        try:
            agg = compute_all(sym, df)
            if agg.buy_count > 0 or agg.sell_count > 0:
                results[sym] = agg
        except Exception as e:
            logger.warning(f"[{sym}] scan error: {e}")
    return format_scan_summary(results)


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
    await update.message.reply_text(f"⏳ Dang backtest {symbol} (~5 nam)...")
    try:
        df = fetch_ohlcv(symbol, count=BACKTEST_CONFIG["bt_lookback_bars"])
        if df is None or df.empty:
            await update.message.reply_text(f"❌ Khong lay duoc du lieu {symbol}")
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
    /wf VCB — Walk-forward validation với params CỐ ĐỊNH (không optimize).
    Kiểm tra xem strategy có ổn định qua các giai đoạn không.
    Dùng /wfo VCB để tìm params tốt nhất trên từng fold.
    """
    if not context.args:
        await update.message.reply_text(
            "Dung: /wf VCB\n"
            "Chia du lieu thanh 3 fold, test params co dinh tren tung fold.\n"
            "Dung /wfo VCB de optimize params tren tung fold (manh hon)."
        )
        return
    symbol = context.args[0].upper()
    n_folds = BACKTEST_CONFIG["walk_forward_folds"]
    await update.message.reply_text(f"🔄 Walk-Forward {symbol} ({n_folds} folds)...")
    try:
        df = fetch_ohlcv(symbol, count=BACKTEST_CONFIG["bt_lookback_bars"])
        if df is None or df.empty:
            await update.message.reply_text(f"❌ Khong lay duoc du lieu {symbol}")
            return
        df_vni = fetch_vni()
        cfg    = get_symbol_config(symbol)

        def _run():
            wf50 = _walk_forward(symbol, df, cfg, BACKTEST_CONFIG, df_vni, "rsi50", n_folds)
            wf55 = _walk_forward(symbol, df, cfg, BACKTEST_CONFIG, df_vni, "rsi55", n_folds)
            return wf50, wf55

        wf50, wf55 = await asyncio.to_thread(_run)
        msg = format_wf_telegram(symbol, wf50, wf55)
        if len(msg) > 4000:
            await update.message.reply_text(msg[:4000])
            await update.message.reply_text(msg[4000:])
        else:
            await update.message.reply_text(msg)
    except Exception as e:
        logger.error(f"/wf {symbol} error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Loi walk-forward {symbol}: {e}")


async def cmd_wfo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /wfo VCB — Walk-Forward OPTIMIZATION.

    Khác với /wf (params cố định), /wfo:
      1. Chia data thành folds rolling (IS=60%, OOS=20%, Step=20%)
      2. Mỗi fold: grid search 64 combo trên IS → tìm best params
      3. Test best params trên OOS → OOS PF
      4. Tính overfit ratio = IS_PF / OOS_PF
      5. Tổng hợp consensus params + verdict

    → Trả lời: "params tốt nhất có thực sự tốt trên data mới không?"

    Thời gian: ~3–5 phút (2 folds × 64 combo × song song).
    Bot vẫn nhận lệnh khác trong lúc chạy.
    """
    if not context.args:
        await update.message.reply_text(
            "Dung: /wfo VCB\n\n"
            "Walk-Forward Optimization:\n"
            "  IS=60% → grid search 64 combo tim best params\n"
            "  OOS=20% → test params do tren data moi\n"
            "  Step=20% → 2 folds rolling\n\n"
            "Ket qua: OOS PF + overfit ratio + consensus params\n"
            "Thoi gian: ~3-5 phut"
        )
        return

    symbol = context.args[0].upper()

    # Cho phép chọn RSI mode: /wfo VCB rsi55
    rsi_mode = "rsi50"
    if len(context.args) > 1 and context.args[1].lower() in ("rsi55", "55"):
        rsi_mode = "rsi55"

    await update.message.reply_text(
        f"🔬 Walk-Forward Optimize {symbol}  [{rsi_mode.upper()}]\n"
        f"\n"
        f"Cau truc: IS=60%  OOS=20%  Step=20%  →  2 folds\n"
        f"Moi fold: grid 64 combo (song song) → test OOS\n"
        f"\n"
        f"Vui long cho ~3-5 phut...\n"
        f"Bot van nhan lenh khac trong luc nay."
    )

    try:
        df = fetch_ohlcv(symbol, count=BACKTEST_CONFIG["bt_lookback_bars"])
        if df is None or df.empty:
            await update.message.reply_text(f"❌ Khong lay duoc du lieu {symbol}")
            return

        # CPU-heavy: chạy trong thread riêng để không block event loop
        wfo = await asyncio.to_thread(run_wfo, symbol, df, None, rsi_mode)
        msg = format_wfo_telegram(symbol, wfo)

        if len(msg) > 4000:
            # Tách tại "── Tong hop"
            split_at = msg.find("── Tong hop")
            if split_at > 0:
                await update.message.reply_text(msg[:split_at].strip())
                await update.message.reply_text(msg[split_at:].strip())
            else:
                await update.message.reply_text(msg[:4000])
                await update.message.reply_text(msg[4000:])
        else:
            await update.message.reply_text(msg)

    except Exception as e:
        logger.error(f"/wfo {symbol} error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Loi WFO {symbol}: {e}")


async def cmd_optimize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /optimize VCB — Grid search đơn trên toàn bộ data.
    Nhanh (~30-60s). Dùng /wfo để kiểm tra params có bị overfit không.
    """
    if not context.args:
        await update.message.reply_text("Dung: /optimize VCB")
        return
    symbol = context.args[0].upper()
    await update.message.reply_text(
        f"⚙️ Dang optimize {symbol} (64 combo, song song)...\n"
        f"Vui long cho ~30-60 giay.\n"
        f"Sau do chay /wfo {symbol} de kiem tra overfit."
    )
    try:
        df = fetch_ohlcv(symbol, count=BACKTEST_CONFIG["bt_lookback_bars"])
        if df is None or df.empty:
            await update.message.reply_text(f"❌ Khong lay duoc du lieu {symbol}")
            return
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


async def cmd_scanbt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /scanbt — Backtest nhanh toàn bộ 28 symbols, xếp hạng theo PF.

    Dùng vectorized engine (O(N)) → mỗi symbol ~1-2s → toàn bộ ~3-5 phút.
    Chạy trong asyncio.to_thread → bot vẫn nhận lệnh khác trong lúc chạy.

    Output: bảng xếp hạng PF, WR, verdict + danh sách ROBUST/MARGINAL.
    Tip: sau đó dùng /bt SYM để xem chi tiết, /optimize SYM để tối ưu.
    """
    await update.message.reply_text(
        "📊 Dang chay /scanbt cho 28 symbols (~3-5 phut)...\n"
        "Bot van nhan lenh khac trong luc nay."
    )
    try:
        from backtest import run_all, comparison_table

        def _run_scanbt():
            df_vni = fetch_vni()
            return run_all(symbols=None, verbose=False)

        results = await asyncio.to_thread(_run_scanbt)

        if not results:
            await update.message.reply_text("❌ Khong co ket qua scanbt")
            return

        df = comparison_table(results)

        def _ve(v):
            return {
                "ROBUST": "✅", "MARGINAL": "🟡",
                "WEAK": "❌", "THIN_DATA": "⚠️", "INCONSISTENT": "🔄",
            }.get(v, " ")

        # Header
        lines = [
            f"📊 SCANBT — {len(results)} symbols | 5 nam | xep hang PF_50",
            f"{'':1}{'SYM':<5}  {'PF50':>5}  {'PF55':>5}  {'WR%':>5}  {'T':>3}  {'Verdict':<10}  {'Best'}",
            "─" * 50,
        ]

        # Rows — sorted by PF_50 desc
        for _, row in df.iterrows():
            v   = _ve(row["V_50"])
            sym = row["symbol"]
            pf50 = f"{row['PF_50']:.2f}"
            pf55 = f"{row['PF_55']:.2f}"
            wr   = f"{row['WR%_50']:.1f}"
            t    = str(int(row["T_50"]))
            vrd  = row["V_50"]
            best = row["best_RSI"]
            lines.append(f"{v} {sym:<5}  {pf50:>5}  {pf55:>5}  {wr:>5}  {t:>3}  {vrd:<10}  {best}")

        # Summary
        robust   = df[df["V_50"] == "ROBUST"]["symbol"].tolist()
        marginal = df[df["V_50"] == "MARGINAL"]["symbol"].tolist()
        weak     = df[df["V_50"] == "WEAK"]["symbol"].tolist()

        lines += [
            "─" * 50,
            f"✅ ROBUST   ({len(robust):2d}): {', '.join(robust) or 'none'}",
            f"🟡 MARGINAL ({len(marginal):2d}): {', '.join(marginal) or 'none'}",
            f"❌ WEAK     ({len(weak):2d}): {', '.join(weak) or 'none'}",
            "",
            "Tip: /bt SYM chi tiet | /optimize SYM toi uu | /wfo SYM kiem tra overfit",
        ]

        msg = "\n".join(lines)

        # Telegram limit 4096 chars — split tại summary nếu cần
        if len(msg) > 4000:
            split_at = msg.rfind("─" * 10, 0, 4000)
            await update.message.reply_text(msg[:split_at].strip())
            await update.message.reply_text(msg[split_at:].strip())
        else:
            await update.message.reply_text(msg)

    except Exception as e:
        logger.error(f"/scanbt error: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Loi scanbt: {e}")


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
        "/scanbt — Backtest nhanh 28 symbols, xếp hạng PF\n"
        "/bt VCB — Backtest đầy đủ (RSI-50 vs RSI-55 + WF)\n"
        "/wf VCB — Walk-Forward params cố định\n"
        "/wfo VCB — Walk-Forward Optimization (IS/OOS split)\n"
        "/optimize VCB — Grid search toàn bộ data\n"
        "/status — Trạng thái bot\n"
        "/help — Lệnh này\n\n"
        "Luồng phân tích đề xuất:\n"
        "  1. /scanbt → xem symbol nào có PF tốt\n"
        "  2. /bt SYM → xem chi tiết symbol đó\n"
        "  3. /optimize SYM → tìm params tốt nhất\n"
        "  4. /wfo SYM → kiểm tra overfit (IS vs OOS)\n\n"
        "Lưu ý /wfo:\n"
        "  /wfo VCB → dùng RSI-50\n"
        "  /wfo VCB rsi55 → dùng RSI-55\n"
        "  Thời gian ~3-5 phút",
        parse_mode="Markdown",
    )


# ── Scheduled scan ────────────────────────────────────────────

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
    app.add_handler(CommandHandler("wfo",      cmd_wfo))
    app.add_handler(CommandHandler("optimize", cmd_optimize))
    app.add_handler(CommandHandler("scanbt",   cmd_scanbt))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("help",     cmd_help))

    tz = pytz.timezone(TIMEZONE)
    scan_time = datetime.now(tz).replace(
        hour=SCAN_HOUR_HCM, minute=SCAN_MINUTE_HCM,
        second=0, microsecond=0,
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
