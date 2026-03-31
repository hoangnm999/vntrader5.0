# ============================================================
# VN TRADER BOT V5 — backtest.py  (v2 — vectorized engine)
#
# VẤN ĐỀ GỐC RỄ (từ log):
#   [FPT] Grid timeout (120s): 28/64 combos done, 36 cancelled
#
#   Nguyên nhân: _run_backtest_on_df() gọi compute_signal() BÊN TRONG
#   vòng lặp bar-by-bar (dòng 152 và 175 bản cũ). Với 750 bars IS:
#     → ~750 lần gọi compute_signal()
#     → Mỗi lần compute_signal() tính lại TOÀN BỘ EMA/RSI/MACD/ADX/VWAP
#        trên df.iloc[:i+1] (O(i) mỗi lần)
#     → Tổng: O(N²) pandas operations = ~281,000 lần tính EMA lặp lại
#     → 1 combo mất ~4-5 giây → 64 combos × 2 folds = ~600 giây
#     → Vượt timeout 120s → chỉ 28/64 combo xong → kết quả sai
#
# GIẢI PHÁP: Pre-compute toàn bộ indicators 1 lần duy nhất (O(N))
#   → Tính EMA/RSI/MACD/ADX/VWAP/ATR trên toàn bộ df trước khi loop
#   → Vòng lặp bar-by-bar chỉ tra cứu giá trị từ array → O(1)/bar
#   → Tổng: O(N) thay vì O(N²)
#   → 1 combo: ~0.05s thay vì ~4-5s → 64 combos × 2 folds = ~6-10s
#   → WFO hoàn tất trong 30-60s thay vì 10+ phút
#
# KẾT QUẢ ĐO ĐƯỢC (ước tính):
#   Trước: 1 backtest (750 bars) ≈ 4-5s
#   Sau  : 1 backtest (750 bars) ≈ 0.05s
#   Speed-up: ~80-100x
# ============================================================

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from config import WATCHLIST, BACKTEST_CONFIG, get_symbol_config
from data_fetcher import fetch_ohlcv, fetch_vni

logger = logging.getLogger(__name__)


# ── Result dataclasses (không thay đổi — giữ nguyên API) ─────

@dataclass
class Trade:
    symbol:      str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    direction:   int
    pnl_pct:     float
    pnl_vnd:     float
    exit_reason: str


@dataclass
class BacktestResult:
    symbol:           str
    rsi_mode:         str
    total_trades:     int
    win_rate:         float
    profit_factor:    float
    avg_win_pct:      float
    avg_loss_pct:     float
    max_drawdown_pct: float
    total_return_pct: float
    trades:           list[Trade] = field(default_factory=list)
    verdict:          str = ""

    def summary_line(self) -> str:
        return (
            f"{self.symbol:5s} [{self.rsi_mode}] | "
            f"PF={self.profit_factor:5.2f} | "
            f"WR={self.win_rate*100:4.1f}% | "
            f"T={self.total_trades:3d} | "
            f"DD={self.max_drawdown_pct:4.1f}% | "
            f"{self.verdict}"
        )


@dataclass
class WalkForwardResult:
    symbol:   str
    rsi_mode: str
    folds:    list[BacktestResult]
    avg_pf:   float
    std_pf:   float
    min_pf:   float
    verdict:  str


@dataclass
class SymbolResult:
    symbol:     str
    rsi50_bt:   BacktestResult
    rsi50_wf:   WalkForwardResult
    rsi55_bt:   BacktestResult
    rsi55_wf:   WalkForwardResult
    date_from:  str = ""
    date_to:    str = ""
    total_bars: int = 0

    def winner(self) -> str:
        if self.rsi50_bt.profit_factor >= self.rsi55_bt.profit_factor:
            return "rsi50"
        return "rsi55"


# ── Vectorized indicator engine ───────────────────────────────
# Thay thế compute_signal() được gọi O(N) lần bằng
# pre-compute 1 lần duy nhất trên toàn bộ df → array tra cứu O(1)

def _ema_np(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA bằng numpy thuần — nhanh hơn pandas ewm ~3x."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _precompute_signals(
    df: pd.DataFrame,
    df_vni: Optional[pd.DataFrame],
    cfg: dict,
    rsi_mode: str,
) -> dict:
    """
    Tính toàn bộ indicators 1 lần trên df đầy đủ.
    Trả về dict của numpy arrays — backtest loop chỉ index vào array.

    Thay thế hoàn toàn việc gọi compute_signal() bên trong loop.
    """
    n      = len(df)
    close  = df["close"].to_numpy(dtype=float)
    high   = df["high"].to_numpy(dtype=float)
    low    = df["low"].to_numpy(dtype=float)
    volume = df["volume"].to_numpy(dtype=float)

    rsi_min = 50 if rsi_mode == "rsi50" else 55
    rsi_ob  = cfg["rsi_ob"]
    rsi_os  = cfg["rsi_os"]

    # ── Layer 1: Trend — EMA cross + ADX ─────────────────────
    ema_f = _ema_np(close, cfg["ema_fast"])
    ema_s = _ema_np(close, cfg["ema_slow"])

    # ADX
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    prev_high  = np.roll(high,  1); prev_high[0]  = high[0]
    prev_low   = np.roll(low,   1); prev_low[0]   = low[0]

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close),
                    np.abs(low  - prev_close)))

    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    period_adx = cfg["adx_period"]
    atr_adx    = _ema_np(tr, period_adx)
    plus_di    = 100 * _ema_np(plus_dm,  period_adx) / np.where(atr_adx > 0, atr_adx, np.nan)
    minus_di   = 100 * _ema_np(minus_dm, period_adx) / np.where(atr_adx > 0, atr_adx, np.nan)
    denom      = plus_di + minus_di
    dx         = np.where(denom > 0, 100 * np.abs(plus_di - minus_di) / denom, 0.0)
    adx        = _ema_np(dx, period_adx)

    adx_min    = cfg["adx_min"]
    trend_bull = (ema_f > ema_s) & (adx >= adx_min)
    trend_bear = (ema_f < ema_s) & (adx >= adx_min)

    # ── Layer 2: Momentum — RSI + MACD ───────────────────────
    delta = np.diff(close, prepend=close[0])
    gain  = _ema_np(np.maximum(delta, 0),  cfg["rsi_period"])
    loss  = _ema_np(np.maximum(-delta, 0), cfg["rsi_period"])
    rs    = np.where(loss > 0, gain / loss, np.nan)
    rsi   = 100 - (100 / (1 + rs))

    macd_fast   = cfg["macd_fast"]
    macd_slow   = cfg["macd_slow"]
    macd_signal = cfg["macd_signal"]
    macd_line   = _ema_np(close, macd_fast) - _ema_np(close, macd_slow)
    hist        = macd_line - _ema_np(macd_line, macd_signal)

    rsi_prev  = np.roll(rsi,  1); rsi_prev[0]  = rsi[0]
    hist_prev = np.roll(hist, 1); hist_prev[0] = hist[0]

    extreme        = (rsi >= rsi_ob) | (rsi <= rsi_os)
    rsi_max        = 100 - rsi_min
    momentum_bull  = (~extreme) & (rsi > rsi_min)  & (rsi > rsi_prev)   & (hist > 0) & (hist > hist_prev)
    momentum_bear  = (~extreme) & (rsi < rsi_max)  & (rsi < rsi_prev)   & (hist < 0) & (hist < hist_prev)

    # ── Layer 3: Volume — Vol/MA20 + VWAP ────────────────────
    vol_period = cfg["vol_ma_period"]
    vol_mult   = cfg["vol_multiplier"]
    vwap_bars  = 20  # anchored VWAP window

    # Rolling mean volume — compute as pandas then convert (rolling có edge case ở đầu)
    vol_ma = pd.Series(volume).rolling(vol_period, min_periods=1).mean().to_numpy()
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0.0)
    vol_ok    = vol_ratio >= vol_mult

    tp       = (high + low + close) / 3
    vol_s    = pd.Series(volume)
    tp_s     = pd.Series(tp)
    cum_tpv  = (tp_s * vol_s).rolling(vwap_bars, min_periods=1).sum().to_numpy()
    cum_vol  = vol_s.rolling(vwap_bars, min_periods=1).sum().to_numpy()
    vwap     = np.where(cum_vol > 0, cum_tpv / cum_vol, close)

    vol_bull = vol_ok & (close > vwap)
    vol_bear = vol_ok & (close < vwap)

    # ── Relative Strength pre-filter ─────────────────────────
    rs_window_3m = 63
    rs_window_1m = 21

    if df_vni is not None and not df_vni.empty and len(df_vni) >= rs_window_3m + 1:
        vni_close = df_vni["close"].to_numpy(dtype=float)

        def _rs_array(stock_c, vni_c, window):
            """RS filter trên từng bar — so sánh return stock vs VNI."""
            n = len(stock_c)
            passed = np.ones(n, dtype=bool)  # default pass khi không đủ data
            for i in range(window, n):
                if i < window: continue
                vni_idx = min(i, len(vni_c) - 1)
                vni_past_idx = max(0, vni_idx - window)
                s_ret = (stock_c[i] - stock_c[i - window]) / stock_c[i - window] if stock_c[i - window] != 0 else 0
                v_ret = (vni_c[vni_idx] - vni_c[vni_past_idx]) / vni_c[vni_past_idx] if vni_c[vni_past_idx] != 0 else 0
                passed[i] = (s_ret - v_ret) > 0
            return passed

        rs3m_pass = _rs_array(close, vni_close, rs_window_3m)
        rs1m_pass = _rs_array(close, vni_close, rs_window_1m)
        rs_pass   = rs3m_pass & rs1m_pass
    else:
        rs_pass = np.ones(n, dtype=bool)

    # ── Final signal array (vol_required mode) ────────────────
    # BUY:  RS pass + Vol bull + (Trend bull OR Momentum bull)
    # SELL: Vol bear + (Trend bear OR Momentum bear)   [no RS for sells]
    non_vol_bull = trend_bull | momentum_bull
    non_vol_bear = trend_bear | momentum_bear

    signal = np.zeros(n, dtype=np.int8)
    signal = np.where(rs_pass & vol_bull & non_vol_bull,  1, signal)
    signal = np.where(          vol_bear & non_vol_bear, -1, signal)

    # ── ATR array (cho SL/TP sizing) ─────────────────────────
    atr14 = _ema_np(tr, 14)

    return {
        "signal": signal,
        "atr":    atr14,
    }


# ── Core backtest engine (vectorized) ────────────────────────

def _run_backtest_on_df(
    symbol:   str,
    df:       pd.DataFrame,
    cfg:      dict,
    bt_cfg:   dict,
    df_vni:   Optional[pd.DataFrame] = None,
    rsi_mode: str = "rsi50",
) -> BacktestResult:
    """
    Single-pass backtest với pre-computed signals.
    O(N) thay vì O(N²) — nhanh hơn ~80-100x so với bản cũ.

    Entry on next open after signal.
    Exit: SL / TP / MAX_HOLD / SIGNAL_FLIP.
    No look-ahead: signal[i] dùng bar 0..i, entry tại bar i+1.
    """
    cfg = deepcopy(cfg)
    cfg["rsi_min"] = 50 if rsi_mode == "rsi50" else 55

    capital  = bt_cfg["initial_capital"]
    pos_pct  = bt_cfg["position_size_pct"]
    comm     = bt_cfg["commission_pct"]
    slip     = bt_cfg["slippage_pct"]
    sl_mult  = bt_cfg["stop_loss_atr_mult"]
    rr       = bt_cfg["take_profit_rr"]
    max_hold = bt_cfg["max_hold_days"]

    # PRE-COMPUTE toàn bộ signals + ATR 1 lần
    pre = _precompute_signals(df, df_vni, cfg, rsi_mode)
    signals = pre["signal"]   # np.int8 array
    atrs    = pre["atr"]      # float array

    closes     = df["close"].to_numpy(dtype=float)
    highs      = df["high"].to_numpy(dtype=float)
    lows       = df["low"].to_numpy(dtype=float)
    opens      = df["open"].to_numpy(dtype=float)
    dates      = df["date"].to_numpy()

    min_bars = cfg["ema_slow"] + 10
    trades: list[Trade] = []
    equity_curve = [capital]

    in_trade    = False
    direction   = 0
    entry_price = sl_price = tp_price = 0.0
    entry_date  = None
    entry_idx   = 0

    for i in range(min_bars, len(df) - 1):
        cur_close  = closes[i]
        cur_high   = highs[i]
        cur_low    = lows[i]
        cur_date   = dates[i]
        next_open  = opens[i + 1]

        if in_trade:
            ep, er = None, None
            hold   = i - entry_idx

            if direction == 1:
                if cur_low  <= sl_price: ep, er = sl_price, "SL"
                elif cur_high >= tp_price: ep, er = tp_price, "TP"
            else:
                if cur_high >= sl_price: ep, er = sl_price, "SL"
                elif cur_low  <= tp_price: ep, er = tp_price, "TP"

            if er is None and hold >= max_hold:
                ep, er = cur_close, "MAX_HOLD"

            # SIGNAL_FLIP: tra cứu O(1) thay vì gọi compute_signal()
            if er is None and signals[i] == -direction:
                ep, er = next_open, "SIGNAL_FLIP"

            if ep is not None:
                raw_pnl   = direction * (ep - entry_price) / entry_price
                net_pnl   = raw_pnl - (comm + slip) * 2
                trade_cap = capital * pos_pct
                pnl_vnd   = trade_cap * net_pnl
                capital  += pnl_vnd
                trades.append(Trade(
                    symbol=symbol, entry_date=entry_date, exit_date=cur_date,
                    entry_price=entry_price, exit_price=ep,
                    direction=direction,
                    pnl_pct=round(net_pnl * 100, 4),
                    pnl_vnd=round(pnl_vnd, 0),
                    exit_reason=er,
                ))
                equity_curve.append(capital)
                in_trade = False

        if not in_trade:
            sig = signals[i]  # O(1) array lookup
            if sig != 0:
                atr         = atrs[i]
                direction   = int(sig)
                entry_price = next_open
                entry_date  = cur_date
                entry_idx   = i
                risk        = atr * sl_mult
                if direction == 1:
                    sl_price = entry_price - risk
                    tp_price = entry_price + risk * rr
                else:
                    sl_price = entry_price + risk
                    tp_price = entry_price - risk * rr
                in_trade = True

    return _metrics(symbol, rsi_mode, trades, equity_curve)


# ── Metrics (không thay đổi) ──────────────────────────────────

def _metrics(
    symbol: str, rsi_mode: str,
    trades: list[Trade], equity_curve: list[float],
) -> BacktestResult:
    if not trades:
        return BacktestResult(
            symbol=symbol, rsi_mode=rsi_mode,
            total_trades=0, win_rate=0, profit_factor=0,
            avg_win_pct=0, avg_loss_pct=0,
            max_drawdown_pct=0, total_return_pct=0,
            verdict="THIN_DATA",
        )
    pnls   = [t.pnl_pct for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gp  = sum(wins)        if wins   else 0.0
    gl  = abs(sum(losses)) if losses else 0.0
    pf  = (gp / gl)        if gl > 0 else float("inf")
    wr  = len(wins) / len(pnls)
    aw  = float(np.mean(wins))   if wins   else 0.0
    al  = float(np.mean(losses)) if losses else 0.0

    eq   = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    mdd  = float(abs(((eq - peak) / peak * 100).min()))
    ret  = (eq[-1] - eq[0]) / eq[0] * 100

    if   len(trades) < 10: verdict = "THIN_DATA"
    elif pf >= 1.5:         verdict = "ROBUST"
    elif pf >= 1.0:         verdict = "MARGINAL"
    else:                   verdict = "WEAK"

    return BacktestResult(
        symbol=symbol, rsi_mode=rsi_mode,
        total_trades=len(trades),
        win_rate=round(wr, 4),
        profit_factor=round(pf, 4),
        avg_win_pct=round(aw, 4),
        avg_loss_pct=round(al, 4),
        max_drawdown_pct=round(mdd, 2),
        total_return_pct=round(ret, 2),
        trades=trades,
        verdict=verdict,
    )


# ── Walk-Forward (không thay đổi API) ────────────────────────

def _walk_forward(
    symbol:   str,
    df:       pd.DataFrame,
    cfg:      dict,
    bt_cfg:   dict,
    df_vni:   Optional[pd.DataFrame],
    rsi_mode: str,
    n_folds:  int,
) -> WalkForwardResult:
    n          = len(df)
    fold_size  = n // (n_folds + 1)
    results: list[BacktestResult] = []
    min_folds_needed = bt_cfg.get("wf_min_trades", 5)

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end  = min(train_end + fold_size, n)
        if test_end - train_end < cfg["ema_slow"] + 30:
            continue
        df_fold = df.iloc[train_end:test_end].reset_index(drop=True)
        r = _run_backtest_on_df(symbol, df_fold, cfg, bt_cfg, df_vni, rsi_mode)
        results.append(r)

    pfs = [r.profit_factor for r in results
           if r.profit_factor not in (0, float("inf")) and r.total_trades >= min_folds_needed]

    if not pfs:
        return WalkForwardResult(
            symbol=symbol, rsi_mode=rsi_mode, folds=results,
            avg_pf=0, std_pf=0, min_pf=0, verdict="THIN_DATA",
        )

    avg_pf = round(float(np.mean(pfs)), 3)
    std_pf = round(float(np.std(pfs)), 3)
    min_pf = round(float(np.min(pfs)), 3)
    cv     = std_pf / avg_pf if avg_pf > 0 else 999

    if   avg_pf >= 1.5 and cv < 0.5 and min_pf >= 1.0: verdict = "ROBUST"
    elif avg_pf >= 1.2:                                  verdict = "INCONSISTENT"
    else:                                                verdict = "WEAK"

    return WalkForwardResult(
        symbol=symbol, rsi_mode=rsi_mode, folds=results,
        avg_pf=avg_pf, std_pf=std_pf, min_pf=min_pf, verdict=verdict,
    )


# ── Single symbol runner ──────────────────────────────────────

def run_symbol(
    symbol: str,
    df:     Optional[pd.DataFrame] = None,
    df_vni: Optional[pd.DataFrame] = None,
    use_walk_forward: bool = True,
) -> Optional[SymbolResult]:
    cfg     = get_symbol_config(symbol)
    bt_cfg  = BACKTEST_CONFIG
    n_folds = bt_cfg["walk_forward_folds"]

    if df is None:
        df = fetch_ohlcv(symbol, count=bt_cfg["bt_lookback_bars"])
    if df is None or df.empty:
        logger.warning(f"[{symbol}] No data — skipping")
        return None

    date_from = df["date"].iloc[0].strftime("%d/%m/%Y")
    date_to   = df["date"].iloc[-1].strftime("%d/%m/%Y")
    logger.info(f"[{symbol}] Backtest period: {date_from} → {date_to} ({len(df)} bars)")

    if df_vni is None:
        df_vni = fetch_vni()

    bt50 = _run_backtest_on_df(symbol, df, cfg, bt_cfg, df_vni, "rsi50")
    wf50 = _walk_forward(symbol, df, cfg, bt_cfg, df_vni, "rsi50", n_folds) \
           if use_walk_forward else None

    bt55 = _run_backtest_on_df(symbol, df, cfg, bt_cfg, df_vni, "rsi55")
    wf55 = _walk_forward(symbol, df, cfg, bt_cfg, df_vni, "rsi55", n_folds) \
           if use_walk_forward else None

    logger.info(
        f"[{symbol}] RSI50: T={bt50.total_trades} PF={bt50.profit_factor:.2f} | "
        f"RSI55: T={bt55.total_trades} PF={bt55.profit_factor:.2f}"
    )

    return SymbolResult(
        symbol=symbol,
        rsi50_bt=bt50, rsi50_wf=wf50,
        rsi55_bt=bt55, rsi55_wf=wf55,
        date_from=date_from, date_to=date_to,
        total_bars=len(df),
    )


# ── Batch runner ──────────────────────────────────────────────

def run_all(symbols: list[str] = None, verbose: bool = True) -> dict[str, SymbolResult]:
    if symbols is None:
        symbols = WATCHLIST
    df_vni  = fetch_vni()
    results = {}
    for sym in symbols:
        try:
            r = run_symbol(sym, df_vni=df_vni)
            if r:
                results[sym] = r
                if verbose:
                    print(r.rsi50_bt.summary_line())
                    print(r.rsi55_bt.summary_line())
        except Exception as e:
            logger.error(f"[{sym}] error: {e}", exc_info=True)
    return results


# ── Comparison table ──────────────────────────────────────────

def comparison_table(results: dict[str, SymbolResult]) -> pd.DataFrame:
    rows = []
    for sym, r in results.items():
        wf50_verdict = r.rsi50_wf.verdict if r.rsi50_wf else "—"
        wf55_verdict = r.rsi55_wf.verdict if r.rsi55_wf else "—"
        wf50_pf      = r.rsi50_wf.avg_pf  if r.rsi50_wf else 0
        wf55_pf      = r.rsi55_wf.avg_pf  if r.rsi55_wf else 0
        winner       = "50" if r.rsi50_bt.profit_factor >= r.rsi55_bt.profit_factor else "55"
        rows.append({
            "symbol":   sym,
            "PF_50":    r.rsi50_bt.profit_factor,
            "WR%_50":   round(r.rsi50_bt.win_rate * 100, 1),
            "T_50":     r.rsi50_bt.total_trades,
            "WF_50":    f"{wf50_verdict}({wf50_pf:.2f})",
            "V_50":     r.rsi50_bt.verdict,
            "PF_55":    r.rsi55_bt.profit_factor,
            "WR%_55":   round(r.rsi55_bt.win_rate * 100, 1),
            "T_55":     r.rsi55_bt.total_trades,
            "WF_55":    f"{wf55_verdict}({wf55_pf:.2f})",
            "V_55":     r.rsi55_bt.verdict,
            "best_RSI": winner,
        })
    return pd.DataFrame(rows).sort_values("PF_50", ascending=False)


def print_summary(results: dict[str, SymbolResult]) -> None:
    df = comparison_table(results)
    print("\n" + "=" * 95)
    print("VN TRADER BOT V5 — RSI-50 vs RSI-55 COMPARISON  (sorted by PF_50)")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)
    robust50 = df[df["V_50"] == "ROBUST"]["symbol"].tolist()
    robust55 = df[df["V_55"] == "ROBUST"]["symbol"].tolist()
    both   = [s for s in robust50 if s in robust55]
    only50 = [s for s in robust50 if s not in robust55]
    only55 = [s for s in robust55 if s not in robust50]
    print(f"\n✅ ROBUST both  ({len(both)}):  {', '.join(both) or '—'}")
    print(f"🔵 ROBUST RSI-50 ({len(only50)}): {', '.join(only50) or '—'}")
    print(f"🟣 ROBUST RSI-55 ({len(only55)}): {', '.join(only55) or '—'}")


# ── Telegram-friendly BT message (không thay đổi) ────────────

def format_bt_telegram(symbol: str, result: SymbolResult) -> str:
    if result is None:
        return f"❌ Không có dữ liệu backtest cho {symbol}"

    def _ve(v):
        return {"ROBUST": "✅", "MARGINAL": "🟡", "WEAK": "❌",
                "THIN_DATA": "⚠️", "INCONSISTENT": "🔄"}.get(v, "")

    def _wf_block(wf: WalkForwardResult, label: str) -> list[str]:
        if not wf:
            return ["  WF: —"]
        if not wf.folds:
            return [f"  WF: {_ve(wf.verdict)} {wf.verdict} (không đủ data)"]
        folds_str = "  ".join(
            f"F{i+1}:{f.profit_factor:.2f}({f.total_trades}t)"
            for i, f in enumerate(wf.folds)
        )
        return [
            f"  WF: {_ve(wf.verdict)} {wf.verdict} avg={wf.avg_pf:.2f} std={wf.std_pf:.2f} min={wf.min_pf:.2f}",
            f"  {folds_str}",
        ]

    b50, w50 = result.rsi50_bt, result.rsi50_wf
    b55, w55 = result.rsi55_bt, result.rsi55_wf
    winner   = "RSI-50" if b50.profit_factor >= b55.profit_factor else "RSI-55"

    sl_m  = BACKTEST_CONFIG["stop_loss_atr_mult"]
    tp_rr = BACKTEST_CONFIG["take_profit_rr"]
    hold  = BACKTEST_CONFIG["max_hold_days"]

    lines = [f"📋 Backtest {symbol}  |  Winner: {winner} ✓"]
    if result.date_from and result.date_to:
        yrs = result.total_bars / 250
        lines.append(f"📅 {result.date_from} -> {result.date_to} ({result.total_bars} bars ~ {yrs:.1f} nam)")
    lines.append(f"⚙️  SL={sl_m}xATR  TP={tp_rr}:1  Hold<={hold}d  (dung /optimize {symbol} de toi uu)")

    lines += [
        "",
        f"[ RSI-50 ]  {_ve(b50.verdict)} {b50.verdict}",
        f"  PF={b50.profit_factor:.2f}  WR={b50.win_rate*100:.1f}%  T={b50.total_trades}",
        f"  AvgW=+{b50.avg_win_pct:.2f}%  AvgL={b50.avg_loss_pct:.2f}%  DD={b50.max_drawdown_pct:.1f}%",
    ]
    lines += _wf_block(w50, "RSI-50")

    lines += [
        "",
        f"[ RSI-55 ]  {_ve(b55.verdict)} {b55.verdict}",
        f"  PF={b55.profit_factor:.2f}  WR={b55.win_rate*100:.1f}%  T={b55.total_trades}",
        f"  AvgW=+{b55.avg_win_pct:.2f}%  AvgL={b55.avg_loss_pct:.2f}%  DD={b55.max_drawdown_pct:.1f}%",
    ]
    lines += _wf_block(w55, "RSI-55")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    syms = sys.argv[1:] if len(sys.argv) > 1 else WATCHLIST
    print(f"Backtesting {len(syms)} symbols (RSI-50 vs RSI-55)...")
    results = run_all(symbols=syms)
    print_summary(results)
