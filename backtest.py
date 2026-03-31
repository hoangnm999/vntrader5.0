# ============================================================
# VN TRADER BOT V5 — backtest.py
# Primary metric: Profit Factor
# Includes: single-symbol backtest, walk-forward validation,
#           multi-symbol batch runner, summary report.
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from config import WATCHLIST, BACKTEST_CONFIG, get_symbol_config
from data_fetcher import fetch_ohlcv, fetch_vni
from indicators import compute_signal, compute_atr

logger = logging.getLogger(__name__)


# ── Result dataclasses ────────────────────────────────────────

@dataclass
class Trade:
    symbol:      str
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    direction:   int        # +1 long, -1 short
    pnl_pct:     float      # net % after commission + slippage
    pnl_vnd:     float
    exit_reason: str        # "SL" | "TP" | "MAX_HOLD" | "SIGNAL_FLIP"


@dataclass
class BacktestResult:
    symbol:          str
    total_trades:    int
    win_rate:        float
    profit_factor:   float
    avg_win_pct:     float
    avg_loss_pct:    float
    max_drawdown_pct: float
    total_return_pct: float
    trades:          list[Trade] = field(default_factory=list)
    verdict:         str = ""   # ROBUST / THIN_DATA / WEAK

    def summary_line(self) -> str:
        return (
            f"{self.symbol:6s} | "
            f"PF={self.profit_factor:.2f} | "
            f"WR={self.win_rate*100:.1f}% | "
            f"Trades={self.total_trades} | "
            f"DD={self.max_drawdown_pct:.1f}% | "
            f"{self.verdict}"
        )


# ── Core backtest engine ──────────────────────────────────────

def _run_backtest_on_df(
    symbol: str,
    df: pd.DataFrame,
    cfg: dict,
    bt_cfg: dict,
) -> BacktestResult:
    """
    Run a single-pass backtest on a given DataFrame slice.
    Signals are generated bar-by-bar (no look-ahead).
    Entry on next open after signal. Exit via SL / TP / max hold / signal flip.
    """
    capital  = bt_cfg["initial_capital"]
    pos_size = bt_cfg["position_size_pct"]
    comm     = bt_cfg["commission_pct"]
    slip     = bt_cfg["slippage_pct"]
    sl_mult  = bt_cfg["stop_loss_atr_mult"]
    rr       = bt_cfg["take_profit_rr"]
    max_hold = bt_cfg["max_hold_days"]

    trades: list[Trade] = []
    equity_curve = [capital]

    in_trade    = False
    direction   = 0
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    entry_date  = None
    entry_idx   = 0

    min_bars = cfg["ema_slow"] + 10

    for i in range(min_bars, len(df)):
        row       = df.iloc[i]
        hist_slice = df.iloc[: i + 1]

        cur_close = float(row["close"])
        cur_open  = float(row["open"])
        cur_high  = float(row["high"])
        cur_low   = float(row["low"])
        cur_date  = row["date"]

        # ── Manage open trade ──
        if in_trade:
            hold_days = i - entry_idx
            exit_price  = None
            exit_reason = None

            if direction == 1:
                if cur_low <= sl_price:
                    exit_price  = sl_price
                    exit_reason = "SL"
                elif cur_high >= tp_price:
                    exit_price  = tp_price
                    exit_reason = "TP"
            else:  # short
                if cur_high >= sl_price:
                    exit_price  = sl_price
                    exit_reason = "SL"
                elif cur_low <= tp_price:
                    exit_price  = tp_price
                    exit_reason = "TP"

            if exit_reason is None and hold_days >= max_hold:
                exit_price  = cur_close
                exit_reason = "MAX_HOLD"

            if exit_reason is None:
                # Check signal flip
                sig = compute_signal(symbol, hist_slice, cfg)
                if sig and sig.final_signal == -direction:
                    exit_price  = cur_open
                    exit_reason = "SIGNAL_FLIP"

            if exit_price is not None:
                # Calculate PnL with costs
                raw_pnl_pct = direction * (exit_price - entry_price) / entry_price
                cost_pct    = (comm + slip) * 2
                net_pnl_pct = raw_pnl_pct - cost_pct
                trade_capital = capital * pos_size
                pnl_vnd     = trade_capital * net_pnl_pct
                capital    += pnl_vnd

                trades.append(Trade(
                    symbol=symbol,
                    entry_date=entry_date,
                    exit_date=cur_date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    direction=direction,
                    pnl_pct=round(net_pnl_pct * 100, 4),
                    pnl_vnd=round(pnl_vnd, 0),
                    exit_reason=exit_reason,
                ))
                equity_curve.append(capital)
                in_trade = False

        # ── Check for new entry ──
        if not in_trade:
            sig = compute_signal(symbol, hist_slice, cfg)
            if sig and sig.final_signal != 0:
                atr = compute_atr(hist_slice)
                entry_price = float(df.iloc[i + 1]["open"]) if i + 1 < len(df) else cur_close
                direction   = sig.final_signal
                entry_date  = cur_date
                entry_idx   = i

                risk = atr * sl_mult
                if direction == 1:
                    sl_price = entry_price - risk
                    tp_price = entry_price + risk * rr
                else:
                    sl_price = entry_price + risk
                    tp_price = entry_price - risk * rr

                in_trade = True

    return _compute_metrics(symbol, trades, equity_curve)


def _compute_metrics(
    symbol: str,
    trades: list[Trade],
    equity_curve: list[float],
) -> BacktestResult:
    if not trades:
        return BacktestResult(
            symbol=symbol, total_trades=0, win_rate=0.0, profit_factor=0.0,
            avg_win_pct=0.0, avg_loss_pct=0.0, max_drawdown_pct=0.0,
            total_return_pct=0.0, trades=[], verdict="THIN_DATA",
        )

    pnls = [t.pnl_pct for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    gross_profit = sum(wins)   if wins   else 0.0
    gross_loss   = abs(sum(losses)) if losses else 0.0

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate      = len(wins) / len(pnls)
    avg_win       = np.mean(wins)   if wins   else 0.0
    avg_loss      = np.mean(losses) if losses else 0.0

    # Max drawdown from equity curve
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / peak * 100
    max_dd = float(abs(dd.min()))

    total_return = (eq[-1] - eq[0]) / eq[0] * 100

    # Verdict
    if len(trades) < 10:
        verdict = "THIN_DATA"
    elif profit_factor >= 1.5:
        verdict = "ROBUST"
    elif profit_factor >= 1.0:
        verdict = "MARGINAL"
    else:
        verdict = "WEAK"

    return BacktestResult(
        symbol=symbol,
        total_trades=len(trades),
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3),
        avg_win_pct=round(avg_win, 4),
        avg_loss_pct=round(avg_loss, 4),
        max_drawdown_pct=round(max_dd, 2),
        total_return_pct=round(total_return, 2),
        trades=trades,
        verdict=verdict,
    )


# ── Walk-forward validation ───────────────────────────────────

@dataclass
class WalkForwardResult:
    symbol:   str
    folds:    list[BacktestResult]
    avg_pf:   float
    std_pf:   float
    verdict:  str   # ROBUST / INCONSISTENT / THIN_DATA


def walk_forward(
    symbol: str,
    df: pd.DataFrame,
    cfg: dict,
    bt_cfg: dict,
    n_folds: int = 4,
) -> WalkForwardResult:
    """
    Split df into n_folds time-sequential folds.
    Run backtest on each fold independently.
    Verdict based on consistency of profit factor across folds.
    """
    fold_size = len(df) // n_folds
    results   = []

    for i in range(n_folds):
        start = i * fold_size
        end   = start + fold_size if i < n_folds - 1 else len(df)
        fold_df = df.iloc[start:end].reset_index(drop=True)

        if len(fold_df) < cfg["ema_slow"] + 20:
            continue

        r = _run_backtest_on_df(symbol, fold_df, cfg, bt_cfg)
        results.append(r)

    if not results:
        return WalkForwardResult(symbol=symbol, folds=[], avg_pf=0.0, std_pf=0.0, verdict="THIN_DATA")

    pfs    = [r.profit_factor for r in results if r.profit_factor != float("inf")]
    avg_pf = float(np.mean(pfs)) if pfs else 0.0
    std_pf = float(np.std(pfs))  if pfs else 0.0
    cv     = std_pf / avg_pf if avg_pf > 0 else 999

    if avg_pf >= 1.5 and cv < 0.5:
        verdict = "ROBUST"
    elif avg_pf >= 1.2:
        verdict = "INCONSISTENT"
    else:
        verdict = "WEAK"

    return WalkForwardResult(
        symbol=symbol,
        folds=results,
        avg_pf=round(avg_pf, 3),
        std_pf=round(std_pf, 3),
        verdict=verdict,
    )


# ── Single symbol runner ──────────────────────────────────────

def run_symbol(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    use_walk_forward: bool = True,
) -> dict:
    """
    Full backtest for one symbol.
    Returns dict with 'backtest' and optionally 'walk_forward' keys.
    """
    cfg    = get_symbol_config(symbol)
    bt_cfg = BACKTEST_CONFIG

    if df is None:
        df = fetch_ohlcv(symbol, count=200)

    if df.empty:
        logger.warning(f"[{symbol}] No data for backtest")
        return {}

    result = _run_backtest_on_df(symbol, df, cfg, bt_cfg)
    out = {"backtest": result}

    if use_walk_forward:
        wf = walk_forward(symbol, df, cfg, bt_cfg, n_folds=bt_cfg["walk_forward_folds"])
        out["walk_forward"] = wf

    return out


# ── Batch runner ──────────────────────────────────────────────

def run_all(
    symbols: list[str] = None,
    use_walk_forward: bool = True,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run backtest for all symbols. Returns {symbol: result_dict}."""
    if symbols is None:
        symbols = WATCHLIST

    _vni = fetch_vni()   # preload VNI cache

    results = {}
    for sym in symbols:
        try:
            r = run_symbol(sym, use_walk_forward=use_walk_forward)
            if r:
                results[sym] = r
                if verbose:
                    bt: BacktestResult = r["backtest"]
                    print(bt.summary_line())
        except Exception as e:
            logger.error(f"[{sym}] Backtest error: {e}")

    return results


# ── Summary report ────────────────────────────────────────────

def print_summary(results: dict[str, dict]) -> None:
    """Print ranked summary sorted by profit factor."""
    rows = []
    for sym, r in results.items():
        bt: BacktestResult = r.get("backtest")
        wf: WalkForwardResult = r.get("walk_forward")
        if bt:
            rows.append({
                "symbol":   sym,
                "PF":       bt.profit_factor,
                "WR%":      round(bt.win_rate * 100, 1),
                "trades":   bt.total_trades,
                "DD%":      bt.max_drawdown_pct,
                "return%":  bt.total_return_pct,
                "verdict":  bt.verdict,
                "WF":       wf.verdict if wf else "—",
            })

    df = pd.DataFrame(rows).sort_values("PF", ascending=False)

    print("\n" + "=" * 70)
    print("VN TRADER BOT V5 — BACKTEST SUMMARY (ranked by Profit Factor)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    robust = df[df["verdict"] == "ROBUST"]
    print(f"\n✅ ROBUST symbols ({len(robust)}): {', '.join(robust['symbol'].tolist())}")
    weak   = df[df["verdict"] == "WEAK"]
    print(f"❌ WEAK symbols   ({len(weak)}): {', '.join(weak['symbol'].tolist())}")


# ── CLI entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) > 1:
        syms = sys.argv[1:]
        print(f"Running backtest for: {syms}")
        results = run_all(symbols=syms)
    else:
        print("Running full 28-symbol backtest...")
        results = run_all()

    print_summary(results)
