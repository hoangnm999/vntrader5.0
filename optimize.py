# ============================================================
# VN TRADER BOT V5 — optimize.py  (v3 — WFO)
#
# 2 tính năng chính:
#
#   1. run_optimize()  — Grid search parallel (64 combo, ProcessPoolExecutor)
#                        Dùng cho /optimize — nhanh ~30-60s
#
#   2. run_wfo()       — Walk-Forward Optimization thực sự
#      Rolling folds:  IS=60%  OOS=20%  Step=20%  → 2 folds tự nhiên
#      Mỗi fold:
#        a. Grid search 64 combo trên IS → tìm best params
#        b. Test best params trên OOS → lấy OOS PF
#        c. Ghi nhận IS_PF vs OOS_PF → tính overfit ratio
#      Tổng hợp:
#        - OOS PF trung bình, std, min
#        - Overfit ratio = IS_PF / OOS_PF  (lý tưởng < 1.5)
#        - Consensus params (bộ params xuất hiện nhiều nhất)
#        - Verdict: ROBUST / MARGINAL / OVERFIT / WEAK / THIN_DATA
# ============================================================

import logging
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import pandas as pd
import numpy as np

from config import BACKTEST_CONFIG, get_symbol_config
from data_fetcher import fetch_ohlcv, fetch_vni
from backtest import _run_backtest_on_df

logger = logging.getLogger(__name__)


# ── Search grid ───────────────────────────────────────────────
GRID = {
    "stop_loss_atr_mult": [1.5, 2.0, 2.5, 3.0],
    "take_profit_rr":     [1.5, 2.0, 2.5, 3.0],
    "max_hold_days":      [10,  15,  20,  30  ],
}
# Total: 4×4×4 = 64 combinations per fold

# WFO rolling structure (tỷ lệ trên tổng bars)
WFO_IS_RATIO   = 0.60   # 60% in-sample
WFO_OOS_RATIO  = 0.20   # 20% out-of-sample
WFO_STEP_RATIO = 0.20   # bước trượt → tự nhiên ra 2 folds đầy đủ IS+OOS

MAX_WORKERS = 2          # subprocess cho grid (Railway free tier 1-2 vCPU)


# ── Result dataclasses ────────────────────────────────────────

@dataclass
class OptimizeResult:
    """Kết quả grid search đơn (không WFO)."""
    symbol:          str
    best_sl_mult:    float
    best_tp_rr:      float
    best_hold_days:  int
    best_pf:         float
    best_wr:         float
    best_trades:     int
    baseline_pf:     float
    improvement_pct: float
    all_results:     list    # sorted [(sl, tp, hold, pf, wr, trades), ...]

    def summary(self) -> str:
        lines = [
            f"🔧 Optimize {self.symbol}",
            f"",
            f"Baseline (default): PF={self.baseline_pf:.2f}",
            f"Best params:        PF={self.best_pf:.2f} (+{self.improvement_pct:.1f}%)",
            f"",
            f"  SL mult  = {self.best_sl_mult}x ATR",
            f"  TP ratio = {self.best_tp_rr}:1",
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


@dataclass
class WFOFold:
    """Kết quả 1 fold trong WFO."""
    fold_num:       int
    is_bars:        int
    oos_bars:       int
    is_date_from:   str
    is_date_to:     str
    oos_date_from:  str
    oos_date_to:    str
    # Best params từ IS grid search
    best_sl_mult:   float
    best_tp_rr:     float
    best_hold_days: int
    is_pf:          float       # PF tốt nhất trên IS
    is_trades:      int
    # OOS result dùng đúng IS best params
    oos_pf:         float
    oos_wr:         float
    oos_trades:     int
    overfit_ratio:  float       # IS_PF / OOS_PF  (lý tưởng ≤ 1.5)


@dataclass
class WFOResult:
    """Kết quả toàn bộ Walk-Forward Optimization."""
    symbol:            str
    rsi_mode:          str
    n_folds:           int
    folds:             list       # list[WFOFold]
    # OOS tổng hợp
    avg_oos_pf:        float
    std_oos_pf:        float
    min_oos_pf:        float
    avg_overfit_ratio: float
    # Consensus params
    consensus_sl:      float
    consensus_tp:      float
    consensus_hold:    int
    consensus_count:   int        # số fold đồng thuận
    # Verdict
    verdict:           str        # ROBUST | MARGINAL | OVERFIT | WEAK | THIN_DATA


# ── Grid parallel worker ──────────────────────────────────────

def _grid_worker(args):
    """Top-level worker (phải top-level để pickle được với ProcessPoolExecutor)."""
    symbol, df_dict, cfg, bt_cfg_base, df_vni_dict, rsi_mode, min_trades, combos = args

    df = pd.DataFrame(df_dict)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df_vni = None
    if df_vni_dict:
        df_vni = pd.DataFrame(df_vni_dict)
        if "date" in df_vni.columns:
            df_vni["date"] = pd.to_datetime(df_vni["date"])

    results = []
    for sl, tp, hold in combos:
        trial = deepcopy(bt_cfg_base)
        trial["stop_loss_atr_mult"] = sl
        trial["take_profit_rr"]     = tp
        trial["max_hold_days"]      = hold
        r = _run_backtest_on_df(symbol, df, cfg, trial, df_vni, rsi_mode)
        if r.total_trades < min_trades:
            continue
        results.append((sl, tp, hold, r.profit_factor, r.win_rate, r.total_trades))
    return results


def _chunked(lst, n):
    size = max(1, len(lst) // n)
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _df_to_dict(df: pd.DataFrame) -> dict:
    d = df.copy()
    if "date" in d.columns:
        d["date"] = d["date"].astype(str)
    return d.to_dict(orient="list")


def _run_grid_parallel(
    symbol: str,
    df: pd.DataFrame,
    cfg: dict,
    bt_cfg: dict,
    df_vni: Optional[pd.DataFrame],
    rsi_mode: str,
    min_trades: int,
    max_workers: int,
) -> list:
    """Chạy 64-combo grid search song song. Trả về list sorted by PF desc."""
    all_combos = list(product(
        GRID["stop_loss_atr_mult"],
        GRID["take_profit_rr"],
        GRID["max_hold_days"],
    ))
    df_dict     = _df_to_dict(df)
    df_vni_dict = _df_to_dict(df_vni) if (df_vni is not None and not df_vni.empty) else None
    chunks      = list(_chunked(all_combos, max_workers))
    worker_args = [
        (symbol, df_dict, cfg, bt_cfg, df_vni_dict, rsi_mode, min_trades, chunk)
        for chunk in chunks
    ]

    results = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_grid_worker, arg) for arg in worker_args]
            for fut in as_completed(futures):
                try:
                    results.extend(fut.result())
                except Exception as e:
                    logger.warning(f"[{symbol}] Worker error: {e}")
    except Exception as e:
        logger.warning(f"[{symbol}] Parallel failed ({e}), fallback sequential")
        for arg in worker_args:
            results.extend(_grid_worker(arg))

    results.sort(key=lambda x: (x[3], x[4]), reverse=True)
    return results


# ── Grid search đơn (toàn bộ data) ───────────────────────────

def run_optimize(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    df_vni: Optional[pd.DataFrame] = None,
    rsi_mode: str = "rsi50",
    min_trades: int = 10,
    max_workers: int = MAX_WORKERS,
) -> Optional[OptimizeResult]:
    """
    Grid search đơn trên toàn bộ data.
    Dùng cho /optimize — nhanh ~30-60s.
    Không có IS/OOS split → dùng /wfo để kiểm tra overfit.
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

    baseline    = _run_backtest_on_df(symbol, df, cfg, bt_cfg, df_vni, rsi_mode)
    all_results = _run_grid_parallel(
        symbol, df, cfg, bt_cfg, df_vni, rsi_mode, min_trades, max_workers
    )

    if not all_results:
        logger.warning(f"[{symbol}] No valid combinations (all < {min_trades} trades)")
        return None

    best = all_results[0]
    sl_b, tp_b, hold_b, pf_b, wr_b, t_b = best
    improvement = ((pf_b - baseline.profit_factor) / max(baseline.profit_factor, 0.01)) * 100

    return OptimizeResult(
        symbol=symbol,
        best_sl_mult=sl_b, best_tp_rr=tp_b, best_hold_days=hold_b,
        best_pf=pf_b, best_wr=wr_b, best_trades=t_b,
        baseline_pf=round(baseline.profit_factor, 3),
        improvement_pct=round(improvement, 1),
        all_results=all_results,
    )


# ── Walk-Forward Optimization ─────────────────────────────────

def run_wfo(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    df_vni: Optional[pd.DataFrame] = None,
    rsi_mode: str = "rsi50",
    min_trades_is: int = 10,
    min_trades_oos: int = 3,
    max_workers: int = MAX_WORKERS,
    is_ratio: float = WFO_IS_RATIO,
    oos_ratio: float = WFO_OOS_RATIO,
    step_ratio: float = WFO_STEP_RATIO,
) -> Optional[WFOResult]:
    """
    Walk-Forward Optimization — rolling IS/OOS folds.

    Cấu trúc (mặc định với 1250 bars):
      Fold 1: IS=[0   → 750]   OOS=[750 → 1000]   (~3yr IS, ~1yr OOS)
      Fold 2: IS=[250 → 1000]  OOS=[1000 → 1250]  (~3yr IS, ~1yr OOS)
      Fold 3: IS=[500 → 1250]  → không đủ OOS → dừng

    Mỗi fold:
      1. Grid search 64 combo trên IS → best params (song song)
      2. Chạy backtest OOS với đúng best IS params → OOS PF
      3. overfit_ratio = IS_PF / OOS_PF

    Tổng hợp:
      - avg/std/min OOS PF
      - avg overfit ratio
      - Consensus params (params được chọn nhiều nhất qua các fold)
      - Verdict

    Tiêu chí verdict:
      ROBUST   : OOS avg PF ≥ 1.5, overfit ≤ 1.5, OOS min PF ≥ 1.0
      MARGINAL : OOS avg PF ≥ 1.2, overfit ≤ 2.0
      OVERFIT  : overfit ratio > 2.0 (IS tốt hơn OOS gấp đôi)
      WEAK     : OOS avg PF < 1.2
      THIN_DATA: Không đủ folds/trades
    """
    cfg    = get_symbol_config(symbol)
    bt_cfg = deepcopy(BACKTEST_CONFIG)

    if df is None:
        df = fetch_ohlcv(symbol, count=bt_cfg["bt_lookback_bars"])
    if df is None or df.empty:
        logger.warning(f"[{symbol}] No data for WFO")
        return None
    if df_vni is None:
        df_vni = fetch_vni()

    n         = len(df)
    is_bars   = int(n * is_ratio)
    oos_bars  = int(n * oos_ratio)
    step_bars = int(n * step_ratio)
    min_fold  = cfg["ema_slow"] + 60   # bars tối thiểu để tính được indicator

    logger.info(
        f"[{symbol}] WFO start: {n} bars | "
        f"IS={is_bars} OOS={oos_bars} step={step_bars}"
    )

    folds: list[WFOFold] = []
    fold_num = 0
    start    = 0

    while True:
        is_end  = start + is_bars
        oos_end = is_end + oos_bars

        # Dừng khi không đủ bars cho cả IS lẫn OOS
        if oos_end > n:
            break

        fold_num += 1
        df_is  = df.iloc[start:is_end].reset_index(drop=True)
        df_oos = df.iloc[is_end:oos_end].reset_index(drop=True)

        if len(df_is) < min_fold:
            logger.debug(f"[{symbol}] Fold {fold_num}: IS quá ngắn ({len(df_is)} bars)")
            start += step_bars
            continue

        is_from  = df_is["date"].iloc[0].strftime("%d/%m/%Y")
        is_to    = df_is["date"].iloc[-1].strftime("%d/%m/%Y")
        oos_from = df_oos["date"].iloc[0].strftime("%d/%m/%Y")
        oos_to   = df_oos["date"].iloc[-1].strftime("%d/%m/%Y")

        logger.info(
            f"[{symbol}] Fold {fold_num}: "
            f"IS=[{is_from}→{is_to}] OOS=[{oos_from}→{oos_to}]"
        )

        # ── Bước 1: Grid search song song trên IS ────────────
        is_results = _run_grid_parallel(
            symbol, df_is, cfg, bt_cfg, df_vni,
            rsi_mode, min_trades_is, max_workers,
        )

        if not is_results:
            logger.warning(f"[{symbol}] Fold {fold_num}: IS grid vô kết quả, bỏ fold")
            start += step_bars
            continue

        best_is = is_results[0]
        sl_b, tp_b, hold_b, is_pf, _, is_t = best_is

        # ── Bước 2: Test IS best params trên OOS ─────────────
        oos_cfg = deepcopy(bt_cfg)
        oos_cfg["stop_loss_atr_mult"] = sl_b
        oos_cfg["take_profit_rr"]     = tp_b
        oos_cfg["max_hold_days"]      = hold_b

        oos_r = _run_backtest_on_df(symbol, df_oos, cfg, oos_cfg, df_vni, rsi_mode)

        # Tránh division by zero khi tính overfit
        oos_pf_safe   = oos_r.profit_factor if oos_r.profit_factor not in (0, float("inf")) else 0.01
        overfit_ratio = round(is_pf / oos_pf_safe, 3)

        folds.append(WFOFold(
            fold_num=fold_num,
            is_bars=len(df_is),
            oos_bars=len(df_oos),
            is_date_from=is_from,
            is_date_to=is_to,
            oos_date_from=oos_from,
            oos_date_to=oos_to,
            best_sl_mult=sl_b,
            best_tp_rr=tp_b,
            best_hold_days=hold_b,
            is_pf=round(is_pf, 3),
            is_trades=is_t,
            oos_pf=round(oos_r.profit_factor, 3),
            oos_wr=round(oos_r.win_rate, 4),
            oos_trades=oos_r.total_trades,
            overfit_ratio=overfit_ratio,
        ))

        start += step_bars

    # ── Tổng hợp ─────────────────────────────────────────────
    if not folds:
        return WFOResult(
            symbol=symbol, rsi_mode=rsi_mode, n_folds=0, folds=[],
            avg_oos_pf=0, std_oos_pf=0, min_oos_pf=0,
            avg_overfit_ratio=0,
            consensus_sl=0, consensus_tp=0, consensus_hold=0, consensus_count=0,
            verdict="THIN_DATA",
        )

    # Chỉ tính stats từ fold có đủ OOS trades
    valid = [f for f in folds if f.oos_trades >= min_trades_oos]
    if not valid:
        valid = folds  # fallback: dùng tất cả

    oos_pfs  = [f.oos_pf for f in valid if f.oos_pf != float("inf")]
    of_ratios = [f.overfit_ratio for f in valid]

    avg_oos_pf  = round(float(np.mean(oos_pfs)), 3)  if oos_pfs  else 0.0
    std_oos_pf  = round(float(np.std(oos_pfs)), 3)   if oos_pfs  else 0.0
    min_oos_pf  = round(float(np.min(oos_pfs)), 3)   if oos_pfs  else 0.0
    avg_overfit = round(float(np.mean(of_ratios)), 3) if of_ratios else 0.0

    # Consensus params: bộ (sl, tp, hold) được chọn nhiều nhất qua các fold
    param_counter = Counter(
        (f.best_sl_mult, f.best_tp_rr, f.best_hold_days) for f in valid
    )
    (c_sl, c_tp, c_hold), c_count = param_counter.most_common(1)[0]

    # Verdict
    if not oos_pfs:
        verdict = "THIN_DATA"
    elif avg_overfit > 2.0:
        verdict = "OVERFIT"
    elif avg_oos_pf >= 1.5 and avg_overfit <= 1.5 and min_oos_pf >= 1.0:
        verdict = "ROBUST"
    elif avg_oos_pf >= 1.2:
        verdict = "MARGINAL"
    else:
        verdict = "WEAK"

    return WFOResult(
        symbol=symbol,
        rsi_mode=rsi_mode,
        n_folds=len(folds),
        folds=folds,
        avg_oos_pf=avg_oos_pf,
        std_oos_pf=std_oos_pf,
        min_oos_pf=min_oos_pf,
        avg_overfit_ratio=avg_overfit,
        consensus_sl=c_sl,
        consensus_tp=c_tp,
        consensus_hold=c_hold,
        consensus_count=c_count,
        verdict=verdict,
    )


# ── Telegram formatters ───────────────────────────────────────

def format_optimize_telegram(symbol: str, opt: Optional[OptimizeResult]) -> str:
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
        f"  TP  = {opt.best_tp_rr}:1",
        f"  Hold= {opt.best_hold_days} ngay toi da",
        f"  WR  = {opt.best_wr*100:.1f}%  |  T={opt.best_trades}",
        f"",
        f"Top 5 combos (PF desc):",
    ]
    for i, r in enumerate(opt.all_results[:5], 1):
        sl, tp, hold, pf, wr, t = r
        marker = " <-- best" if i == 1 else ""
        lines.append(
            f"  {i}. SL={sl}x TP={tp}:1 Hold={hold}d "
            f"-> PF={pf:.2f} WR={wr*100:.1f}% T={t}{marker}"
        )
    if opt.improvement_pct > 10:
        lines += [
            f"",
            f"💡 Goi y: cap nhat SYMBOL_CONFIG['{symbol}'] voi:",
            f"   stop_loss_atr_mult={opt.best_sl_mult}, take_profit_rr={opt.best_tp_rr}, max_hold_days={opt.best_hold_days}",
            f"",
            f"Chay /wfo {symbol} de kiem tra params nay co bi overfit khong.",
        ]
    return "\n".join(lines)


def format_wfo_telegram(symbol: str, wfo: Optional[WFOResult]) -> str:
    """
    Định dạng WFOResult cho Telegram.
    Output đầy đủ: fold-by-fold IS vs OOS, overfit ratio, consensus params, verdict.
    """
    if wfo is None:
        return f"❌ Khong du du lieu de WFO {symbol}"

    vi = {
        "ROBUST":    "✅",
        "MARGINAL":  "🟡",
        "OVERFIT":   "🔴",
        "WEAK":      "❌",
        "THIN_DATA": "⚠️",
    }.get(wfo.verdict, "")

    lines = [
        f"🔬 Walk-Forward Optimize {symbol}  [{wfo.rsi_mode.upper()}]",
        f"",
        f"Cau truc: IS=60%  OOS=20%  Step=20%  →  {wfo.n_folds} fold",
        f"",
    ]

    for f in wfo.folds:
        if f.oos_pf >= 1.5 and f.overfit_ratio <= 1.5:
            fi = "✅"
        elif f.oos_pf >= 1.2 and f.overfit_ratio <= 2.0:
            fi = "🟡"
        elif f.overfit_ratio > 2.5:
            fi = "🔴"
        else:
            fi = "❌"

        of_comment = (
            "(ok — on dinh)"        if f.overfit_ratio <= 1.5 else
            "(chap nhan duoc)"      if f.overfit_ratio <= 2.0 else
            "(cao — co the overfit)"
        )

        lines += [
            f"── Fold {f.fold_num} {fi} ──────────────────────────────",
            f"  IS : {f.is_date_from} → {f.is_date_to}  ({f.is_bars} bars)",
            f"  OOS: {f.oos_date_from} → {f.oos_date_to}  ({f.oos_bars} bars)",
            f"",
            f"  Best IS params:  SL={f.best_sl_mult}x  TP={f.best_tp_rr}:1  Hold={f.best_hold_days}d",
            f"  IS  PF={f.is_pf:.2f}  T={f.is_trades}",
            f"",
            f"  OOS (dung IS params): PF={f.oos_pf:.2f}  WR={f.oos_wr*100:.1f}%  T={f.oos_trades}",
            f"  Overfit ratio = {f.overfit_ratio:.2f}x  {of_comment}",
            f"",
        ]

    lines += [
        f"── Tong hop ─────────────────────────────────",
        f"  OOS PF : avg={wfo.avg_oos_pf:.2f}  std={wfo.std_oos_pf:.2f}  min={wfo.min_oos_pf:.2f}",
        f"  Overfit: avg={wfo.avg_overfit_ratio:.2f}x",
        f"  Verdict: {vi} {wfo.verdict}",
        f"",
        f"── Consensus Params ─────────────────────────",
        f"  Params chon nhieu nhat qua {wfo.n_folds} fold:",
        f"  SL={wfo.consensus_sl}x  TP={wfo.consensus_tp}:1  Hold={wfo.consensus_hold}d",
        f"  ({wfo.consensus_count}/{wfo.n_folds} fold dong thuan)",
        f"",
        f"── Phan tich ────────────────────────────────",
    ]

    explanation = {
        "ROBUST":    "OOS PF tot, overfit thap. An toan de live trade voi consensus params.",
        "MARGINAL":  "OOS co loi nhung chua on dinh. Giam position size 50% khi live.",
        "OVERFIT":   "IS PF >> OOS PF. Params qua fit vao data cu — KHONG nen live trade.",
        "WEAK":      "OOS PF thap — chien luoc khong co loi tren data moi. Xem xet lai.",
        "THIN_DATA": "Khong du folds/trades. Can them data lich su hoac giam min_trades.",
    }
    lines.append(f"  {explanation.get(wfo.verdict, '')}")

    if wfo.verdict in ("ROBUST", "MARGINAL") and wfo.consensus_count >= 2:
        lines += [
            f"",
            f"💡 Cap nhat SYMBOL_CONFIG['{symbol}']:",
            f"   stop_loss_atr_mult = {wfo.consensus_sl}",
            f"   take_profit_rr     = {wfo.consensus_tp}",
            f"   max_hold_days      = {wfo.consensus_hold}",
        ]

    return "\n".join(lines)


def apply_optimized_params(symbol: str, opt: OptimizeResult) -> dict:
    return {
        "stop_loss_atr_mult": opt.best_sl_mult,
        "take_profit_rr":     opt.best_tp_rr,
        "max_hold_days":      opt.best_hold_days,
    }


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cmd  = sys.argv[1] if len(sys.argv) > 1 else "wfo"
    syms = sys.argv[2:] if len(sys.argv) > 2 else ["VCB", "FPT"]
    df_vni = fetch_vni()

    for sym in syms:
        if cmd == "optimize":
            print(f"\n── Optimize {sym} ──")
            opt = run_optimize(sym, df_vni=df_vni)
            if opt:
                print(opt.summary())
        else:
            print(f"\n── WFO {sym} ──")
            wfo = run_wfo(sym, df_vni=df_vni)
            if wfo:
                print(format_wfo_telegram(sym, wfo))
