import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import csv
from config_thresholds import BACKTEST_CFG, REGIME_DEFAULTS

from utils.filters import MLFilter
from core.feature_pipeline import compute_triangle_features, Z_SCORE_WINDOW
from core.execution_engine import calculate_dynamic_sl_tp

np.random.seed(BACKTEST_CFG.get("defaults", {}).get("seed", 42))

TICK_CSV = "data/triangular_ticks.csv"
df = pd.read_csv(TICK_CSV)

# --- Date filtering logic added ---
DATE_START = '2022-01-01'
DATE_END = '2022-02-01'
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df = df[(df['timestamp'] >= DATE_START) & (df['timestamp'] < DATE_END)].copy()

N_TUNE_ROWS = None   # Set None for all data, or pick a number for speed
if N_TUNE_ROWS:
    df = df.iloc[:N_TUNE_ROWS].copy()

model = MLFilter(BACKTEST_CFG.get("defaults", {}).get("model_path", "ml_model/triangular_rf_model.json"))

# rolling windows for feature generation
windows_template = {
    "spread": deque(maxlen=Z_SCORE_WINDOW),
    "btc": deque(maxlen=Z_SCORE_WINDOW),
    "eth": deque(maxlen=Z_SCORE_WINDOW),
    "ethbtc": deque(maxlen=Z_SCORE_WINDOW),
}

def run_backtest(
    df,
    base_thr_sell,
    thr_buy,
    sl_percent,
    tp_percent,
    version="old",
    writer=None,
):
    trades = []
    windows = {k: deque(maxlen=Z_SCORE_WINDOW) for k in windows_template}
    for idx in range(len(df) - 1):
        row = df.iloc[idx]

        features = compute_triangle_features(
            row["btc_price"], row["eth_price"], row["ethbtc_price"], windows
        )
        confidence, signal = model.predict_with_confidence(pd.DataFrame([features]))

        direction = 0
        if signal == -1 and confidence > base_thr_sell:
            direction = -1
        elif signal == 1 and confidence > thr_buy:
            direction = 1

        if direction != 0:
            next_row = df.iloc[idx + 1]
            next_features = compute_triangle_features(
                next_row["btc_price"], next_row["eth_price"], next_row["ethbtc_price"], windows
            )

            dyn_sl, dyn_tp = calculate_dynamic_sl_tp(
                spread_zscore=features["spread_zscore"],
                vol_spread=features["vol_spread"],
                confidence=confidence,
            )
            stop_loss_pct = sl_percent * (dyn_sl / 0.3)
            take_profit_pct = tp_percent * (dyn_tp / 0.5)

            price_change_pct = (
                (next_row["eth_price"] - row["eth_price"]) / row["eth_price"] * 100
            )
            pnl = price_change_pct * direction
            rr = np.abs(pnl) / stop_loss_pct if stop_loss_pct else 0.0
            if pnl >= take_profit_pct:
                exit_reason = "TP"
            elif pnl <= -stop_loss_pct:
                exit_reason = "SL"
            else:
                exit_reason = "HOLD"
            trades.append((pnl, rr, direction, exit_reason))
            if writer:
                writer.writerow({
                    "version": version,
                    "base_thr_sell": base_thr_sell,
                    "thr_buy": thr_buy,
                    "sl_percent": sl_percent,
                    "tp_percent": tp_percent,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                })
    if not trades:
        return {
            "rr_realized": 0,
            "sharpe": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "max_drawdown": 0,
            "n_trades": 0,
            "n_buys": 0,
            "n_sells": 0,
            "tp_count": 0,
            "sl_count": 0,
            "version": version,
            "base_thr_sell": base_thr_sell,
            "thr_buy": thr_buy,
            "sl_percent": sl_percent,
            "tp_percent": tp_percent,
        }
    pnls, rrs, directions, reasons = zip(*trades)
    rr_realized = np.mean(rrs)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9)
    win_rate = np.mean([p > 0 for p in pnls])
    avg_pnl = np.mean(pnls)
    equity = np.cumsum(pnls)
    drawdown = np.minimum.accumulate(equity) - equity
    max_drawdown = drawdown.min() if len(drawdown) else 0
    n_trades = len(trades)
    n_buys = np.sum([d == 1 for d in directions])
    n_sells = np.sum([d == -1 for d in directions])
    tp_count = reasons.count("TP")
    sl_count = reasons.count("SL")
    return {
        "rr_realized": rr_realized,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "version": version,
        "base_thr_sell": base_thr_sell,
        "thr_buy": thr_buy,
        "sl_percent": sl_percent,
        "tp_percent": tp_percent,
    }

# Define sweep ranges from config to ensure reproducibility
back_cfg = BACKTEST_CFG.get("grid", {})
sell_thresholds = back_cfg.get("sell_thresholds", [0.80, 0.83, 0.86])
buy_thresholds = back_cfg.get("buy_thresholds", [0.70, 0.73, 0.76])
sl_percents = back_cfg.get("sl_percents", [0.15, 0.17])
tp_percents = back_cfg.get("tp_percents", [0.54, 0.61])

sell_thresholds_new = back_cfg.get("sell_thresholds_new", sell_thresholds)
buy_thresholds_new = back_cfg.get("buy_thresholds_new", buy_thresholds)
sl_percents_new = back_cfg.get("sl_percents_new", sl_percents)
tp_percents_new = back_cfg.get("tp_percents_new", tp_percents)

if BACKTEST_CFG.get("defaults", {}).get("use_regime_defaults", False):
    base = REGIME_DEFAULTS.get("flat", {})
    sell_thresholds = [base.get("base_thr_sell", 0.9)]
    buy_thresholds = [base.get("thr_buy", 0.7)]
    sl_percents = [base.get("sl_percent", 0.3)]
    tp_percents = [base.get("tp_percent", 0.85)]
    sell_thresholds_new = sell_thresholds
    buy_thresholds_new = buy_thresholds
    sl_percents_new = sl_percents
    tp_percents_new = tp_percents

combos_old = list(itertools.product(sell_thresholds, buy_thresholds, sl_percents, tp_percents))
combos_new = list(itertools.product(sell_thresholds_new, buy_thresholds_new, sl_percents_new, tp_percents_new))

results = []
log_path = "logs/backtest_trades.csv"
with open(log_path, "w", newline="") as f:
    fieldnames = ["version", "base_thr_sell", "thr_buy", "sl_percent", "tp_percent", "pnl", "exit_reason"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    print("\n==== Threshold Performance Summary (OLD) ====")
    print("SELL_THR  BUY_THR   SL%   TP%   WinRate  AvgPnL  MaxDD  Trades")
    print("-" * 80)
    for thr_sell, thr_buy, sl, tp in combos_old:
        res = run_backtest(df, thr_sell, thr_buy, sl, tp, version="old", writer=writer)
        print(f"{thr_sell:7.2f}  {thr_buy:7.2f}  {sl:4.2f}  {tp:4.2f}   {res['win_rate']:7.2f}  {res['avg_pnl']:7.2f}  {res['max_drawdown']:7.2f}  {res['n_trades']:6d}")
        results.append(res)

    print("\n==== Threshold Performance Summary (NEW) ====")
    print("SELL_THR  BUY_THR   SL%   TP%   WinRate  AvgPnL  MaxDD  Trades")
    print("-" * 80)
    for thr_sell, thr_buy, sl, tp in combos_new:
        res = run_backtest(df, thr_sell, thr_buy, sl, tp, version="new", writer=writer)
        print(f"{thr_sell:7.2f}  {thr_buy:7.2f}  {sl:4.2f}  {tp:4.2f}   {res['win_rate']:7.2f}  {res['avg_pnl']:7.2f}  {res['max_drawdown']:7.2f}  {res['n_trades']:6d}")
        results.append(res)

df_results = pd.DataFrame(results)
df_results.to_csv("quant_sweep_summary_full.csv", index=False)

summary = df_results.groupby("version")[["win_rate", "avg_pnl", "max_drawdown"]].mean()
print("\n==== Summary by Version ====")
print(summary)

tp_sl = df_results.groupby("version")[["tp_count", "sl_count"]].sum()
tp_sl.plot(kind="bar")
plt.title("TP vs SL Counts")
plt.ylabel("Count")
plt.savefig("tp_sl_counts.png")

if set(summary.index) >= {"old", "new"}:
    edge_improved = summary.loc["new", "avg_pnl"] > summary.loc["old", "avg_pnl"]
    print("\nEdge Improved:" , edge_improved)

# --- Print best by Sharpe and R:R ---
best_sharpe = df_results.loc[df_results.sharpe.idxmax()]
best_rr = df_results.loc[df_results.rr_realized.idxmax()]

print("\n[QUANT] Best by Sharpe:")
print(f"SELL={best_sharpe['base_thr_sell']:.2f}  BUY={best_sharpe['thr_buy']:.2f}  SL={best_sharpe['sl_percent']:.2f}%  TP={best_sharpe['tp_percent']:.2f}%  "
      f"R:R={best_sharpe['rr_realized']:.2f}  Sharpe={best_sharpe['sharpe']:.2f}  WinRate={best_sharpe['win_rate']:.2f}  Trades={int(best_sharpe['n_trades'])}")

print("[QUANT] Best by R:R:")
print(f"SELL={best_rr['base_thr_sell']:.2f}  BUY={best_rr['thr_buy']:.2f}  SL={best_rr['sl_percent']:.2f}%  TP={best_rr['tp_percent']:.2f}%  "
      f"R:R={best_rr['rr_realized']:.2f}  Sharpe={best_rr['sharpe']:.2f}  WinRate={best_rr['win_rate']:.2f}  Trades={int(best_rr['n_trades'])}")

# --- Optional: Heatmap for best SELL threshold at best BUY threshold ---
best_sell = df_results.groupby("base_thr_sell")["sharpe"].mean().idxmax()
best_buy = df_results.groupby("thr_buy")["sharpe"].mean().idxmax()
pivot = df_results[(df_results.base_thr_sell == best_sell) & (df_results.thr_buy == best_buy)] \
    .pivot(index='sl_percent', columns='tp_percent', values='sharpe')
plt.figure(figsize=(10,6))
plt.title(f"Sharpe Ratio Heatmap (SELL={best_sell:.2f}, BUY={best_buy:.2f})")
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu")
plt.ylabel("SL (%)")
plt.xlabel("TP (%)")
plt.show()
