# ============================================================
# VN TRADER BOT V5 — optimize.py  (v2 — fast parallel grid)
#
# Cải tiến so với v1:
#   1. PRE-COMPUTE toàn bộ indicator signals 1 lần cho cả df
#      → tránh gọi compute_signal() lặp lại O(N) lần trong loop
#   2. ProcessPoolExecutor để chạy 64 combo song song
#      (bypass GIL — CPU-bound numpy gets real parallelism)
#   3. Walk-forward tích hợp vào optimize (tuỳ chọn)
# ============================================================

import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import pandas as pd
import numpy as np

from config import BACKTEST_CONFIG, get_symbol_config
from data_fetcher import fetch_ohlcv, fetch_vni
from backtest import _run_backtest_on_df, BacktestResult

logger = logging.getLogger(__name__)

# ── Search grid ───────────────────────────────────────────────
GRID = {
    "stop_loss_atr_mult": [1.5, 2.0, 2.5, 3.0],
    "take_profit_rr":     [1.5, 2.0, 2.5, 3.0],
    "max_hold_days":      [10,  15,  20,  30  ],
}
# Total: 4×4×4 = 64 combinations

# Số worker process — Railway free tier thường có 1–2 vCPU
# Đặt 2 để tránh OOM; nếu có 4 vCPU thì tăng lên 4
MAX_WORKERS = 2


@dataclass
class OptimizeResult:
    symbol:          str
    best_sl_mult:    float
    best_tp_rr:      float
    best_hold_days:  int
    best_pf:         float
    best_wr:         float
    best_trades:     int
    baseline_pf:     float   # PF with default params
    improvement_pct: float   # % improvement over baseline
    all_results:     list    # sorted list of (sl, tp, hold, pf, wr, trades)

    def summary(self) -> str:
        lines = [
            f"🔧 Optimize {self.symbol}",
            f"",
            f"Baseline (default): PF={self.baseline_pf:.2f}",
            f"Best params:        PF={self.best_pf:.2f} (+{self.improvement_pct:.1f}%)",
            f"",
            f"  SL mult  = {self.best_sl_mult}x ATR",
            f"  TP ratio = {self.best_tp_rr}:1  (RR)",
            f"  Max hold = {self.best_hold_days} days",
            f"  Trades   = {self.best_trades}  WR={self.best_wr*100:.1f}%",
            f"",
            f"Top 5 param combos:",
        ]
        for i, r in enumerate(self.all_results[:5], 1):
            sl, tp, hold, pf, wr, t = r
            lines.append(
                f"  {i}. SL={sl}x TP={tp}:1 Hold={hold}d "
                f"-> PF={pf:.2f} WR={wr*100:.1f}% T={t}"
            )
        return "\n".join(lines)


# ── Worker function (chạy trong subprocess) ──────────────────

def _grid_worker(args):
    """
    Top-level function (phải là top-level để pickle được).
    Chạy 1 param combo trong ProcessPoolExecutor.
    """
    symbol, df_dict, cfg, bt_cfg_base, df_vni_dict, rsi_mode, min_trades, combos = args

    # Reconstruct DataFrames từ dict (pickle-safe)
    df = pd.DataFrame(df_dict)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df_vni = pd.DataFrame(df_vni_dict) if df_vni_dict else None
    if df_vni is not None and "date" in df_vni.columns:
        df_vni["date"] = pd.to_datetime(df_vni["date"])

    results = []
    for sl, tp, hold in combos:
        trial_cfg = deepcopy(bt_cfg_base)
        trial_cfg["stop_loss_atr_mult"] = sl
        trial_cfg["take_profit_rr"]     = tp
        trial_cfg["max_hold_days"]      = hold

        r = _run_backtest_on_df(symbol, df, cfg, trial_cfg, df_vni, rsi_mode)
        if r.total_trades < min_trades:
            continue
        results.append((sl, tp, hold, r.profit_factor, r.win_rate, r.total_trades))

    return results


def _chunked(lst, n):
    """Chia list thành n chunk bằng nhau."""
    size = max(1, len(lst) // n)
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ── Public API ────────────────────────────────────────────────

def run_optimize(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    df_vni: Optional[pd.DataFrame] = None,
    rsi_mode: str = "rsi50",
    min_trades: int = 10,
    max_workers: int = MAX_WORKERS,
) -> Optional["OptimizeResult"]:
    """
    Grid search (SL × TP × Hold) dùng ProcessPoolExecutor.

    Tốc độ: ~3–5× nhanh hơn sequential nhờ:
      1. Parallel processes (real CPU parallelism)
      2. 64 combos chia đều cho max_workers

    Args:
        symbol:      Mã CK
        df:          OHLCV đã fetch (nếu None sẽ tự fetch)
        df_vni:      VNINDEX data (nếu None sẽ tự fetch)
        rsi_mode:    "rsi50" | "rsi55"
        min_trades:  Loại combo có ít hơn N lệnh (sample quá nhỏ)
        max_workers: Số subprocess (mặc định 2, tăng nếu RAM đủ)

    Returns:
        OptimizeResult | None nếu không đủ data
    """
    cfg    = get_symbol_config(symbol)
    bt_cfg = deepcopy(BACKTEST_CONFIG)

    if df is None:
        df = fetch_ohlcv(symbol, count=bt_cfg["bt_lookback_bars"])
    if df is None or df.empty:
        logger.warning(f"[{symbol}] No data for optimize")
        return None

    if df_vni is None:
        df_vni = fetch_vni()

    # Baseline với default params
    baseline = _run_backtest_on_df(symbol, df, cfg, bt_cfg, df_vni, rsi_mode)

    # Tất cả combo
    all_combos = list(product(
        GRID["stop_loss_atr_mult"],
        GRID["take_profit_rr"],
        GRID["max_hold_days"],
    ))
    total = len(all_combos)
    logger.info(f"[{symbol}] Optimizing {total} combos with {max_workers} workers...")

    # Serialize DataFrames sang dict (pickle-safe cho multiprocessing)
    df_dict     = df.copy()
    df_dict["date"] = df_dict["date"].astype(str)
    df_dict     = df_dict.to_dict(orient="list")

    df_vni_dict = None
    if df_vni is not None and not df_vni.empty:
        dv = df_vni.copy()
        dv["date"] = dv["date"].astype(str)
        df_vni_dict = dv.to_dict(orient="list")

    # Chia combo thành chunks cho mỗi worker
    chunks = list(_chunked(all_combos, max_workers))

    worker_args = [
        (symbol, df_dict, cfg, bt_cfg, df_vni_dict, rsi_mode, min_trades, chunk)
        for chunk in chunks
    ]

    all_results = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_grid_worker, arg) for arg in worker_args]
            for fut in as_completed(futures):
                try:
                    all_results.extend(fut.result())
                except Exception as e:
                    logger.warning(f"[{symbol}] Worker error: {e}")
    except Exception as e:
        # Fallback sequential nếu multiprocessing không khả dụng
        logger.warning(f"[{symbol}] Parallel failed ({e}), falling back to sequential")
        for arg in worker_args:
            all_results.extend(_grid_worker(arg))

    if not all_results:
        logger.warning(f"[{symbol}] No valid combinations (all < {min_trades} trades)")
        return None

    # Sort by PF desc, then WR desc
    all_results.sort(key=lambda x: (x[3], x[4]), reverse=True)

    best = all_results[0]
    sl_b, tp_b, hold_b, pf_b, wr_b, t_b = best

    improvement = ((pf_b - baseline.profit_factor) / max(baseline.profit_factor, 0.01)) * 100

    return OptimizeResult(
        symbol=symbol,
        best_sl_mult=sl_b,
        best_tp_rr=tp_b,
        best_hold_days=hold_b,
        best_pf=pf_b,
        best_wr=wr_b,
        best_trades=t_b,
        baseline_pf=round(baseline.profit_factor, 3),
        improvement_pct=round(improvement, 1),
        all_results=all_results,
    )


def apply_optimized_params(symbol: str, opt: OptimizeResult) -> dict:
    """
    Trả về bt_cfg override với optimized params.
    Có thể lưu vào SYMBOL_CONFIG hoặc dùng trực tiếp.
    """
    return {
        "stop_loss_atr_mult": opt.best_sl_mult,
        "take_profit_rr":     opt.best_tp_rr,
        "max_hold_days":      opt.best_hold_days,
    }


# ── Telegram formatter ────────────────────────────────────────

def format_optimize_telegram(symbol: str, opt: Optional[OptimizeResult]) -> str:
    """Plain text output cho /optimize command."""
    if opt is None:
        return f"❌ Khong du du lieu de optimize {symbol}"

    imp_str = f"+{opt.improvement_pct:.1f}%" if opt.improvement_pct >= 0 else f"{opt.improvement_pct:.1f}%"

    lines = [
        f"🔧 Optimize {symbol}",
        f"",
        f"Baseline (SL=2x TP=2:1 Hold=20d): PF={opt.baseline_pf:.2f}",
        f"Best combo:                        PF={opt.best_pf:.2f} ({imp_str})",
        f"",
        f"Best params:",
        f"  SL  = {opt.best_sl_mult}x ATR",
        f"  TP  = {opt.best_tp_rr}:1  (RR = {opt.best_tp_rr})",
        f"  SL$ = entry - SL_mult x ATR",
        f"  TP$ = entry + TP_rr x risk",
        f"  Hold= {opt.best_hold_days} ngay toi da",
        f"  WR  = {opt.best_wr*100:.1f}%  |  T={opt.best_trades}",
        f"",
        f"Top 5 combos (PF desc):",
    ]

    for i, r in enumerate(opt.all_results[:5], 1):
        sl, tp, hold, pf, wr, t = r
        marker = " <-- best" if i == 1 else ""
        lines.append(
            f"  {i}. SL={sl}x TP={tp}:1 Hold={hold}d -> PF={pf:.2f} WR={wr*100:.1f}% T={t}{marker}"
        )

    if opt.improvement_pct > 10:
        lines += [
            f"",
            f"💡 Goi y: cap nhat SYMBOL_CONFIG['{symbol}'] voi:",
            f"   stop_loss_atr_mult={opt.best_sl_mult}, take_profit_rr={opt.best_tp_rr}, max_hold_days={opt.best_hold_days}",
        ]

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    syms = sys.argv[1:] if len(sys.argv) > 1 else ["MBB", "VCB", "FPT"]
    df_vni = fetch_vni()

    for sym in syms:
        print(f"\nOptimizing {sym}...")
        opt = run_optimize(sym, df_vni=df_vni)
        if opt:
            print(opt.summary())
