import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

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

def run_backtest(
    df,
    base_thr_sell,
    thr_buy,
    sl_percent,
    tp_percent,
):
    trades = []
    for idx, row in df.iterrows():
        # Example placeholder logic (replace with real model logic if available)
        model_confidence_sell = row.get('conf_sell', np.random.uniform(0.6, 1.0))
        model_confidence_buy  = row.get('conf_buy', np.random.uniform(0.6, 1.0))
        direction = 0
        if model_confidence_sell > base_thr_sell:
            direction = -1  # SELL
        elif model_confidence_buy > thr_buy:
            direction = 1   # BUY
        if direction != 0:
            pnl = np.random.normal(loc=0.10*direction, scale=0.05)
            rr = np.abs(pnl) / (sl_percent/100)
            trades.append((pnl, rr, direction))
    if not trades:
        return 0, 0, 0, 0, 0, 0
    pnls, rrs, directions = zip(*trades)
    rr_realized = np.mean(rrs)
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9)
    win_rate = np.mean([p > 0 for p in pnls])
    n_trades = len(trades)
    n_buys = np.sum([d == 1 for d in directions])
    n_sells = np.sum([d == -1 for d in directions])
    return rr_realized, sharpe, win_rate, n_trades, n_buys, n_sells

# Define sweep ranges for all 4 thresholds (match your Optuna or gridsearch ranges)
sell_thresholds = np.arange(0.80, 0.99, 0.03)     # 0.80, 0.83, ..., 0.98
buy_thresholds  = np.arange(0.70, 0.88, 0.03)      # 0.70, 0.73, ..., 0.85
sl_percents     = np.arange(0.15, 0.21, 0.02)      # 0.15, 0.17, 0.19
tp_percents     = np.arange(0.54, 0.76, 0.07)      # 0.54, 0.61, ..., 0.75

# Cartesian product of all threshold combos
threshold_combos = list(itertools.product(sell_thresholds, buy_thresholds, sl_percents, tp_percents))

results = []
print("\n==== Threshold Performance Summary ====")
print("SELL_THR  BUY_THR   SL%   TP%   R:R     Sharpe   WinRate  Trades  Buys  Sells")
print("-"*80)
for thr_sell, thr_buy, sl, tp in threshold_combos:
    rr, sharpe, win_rate, n_trades, n_buys, n_sells = run_backtest(
        df,
        base_thr_sell=thr_sell,
        thr_buy=thr_buy,
        sl_percent=sl,
        tp_percent=tp,
    )
    print(f"{thr_sell:7.2f}  {thr_buy:7.2f}  {sl:4.2f}  {tp:4.2f}   {rr:6.2f}   {sharpe:7.2f}   {win_rate:7.2f}  {n_trades:6d}  {n_buys:5d}  {n_sells:5d}")
    results.append({
        "base_thr_sell": thr_sell,
        "thr_buy": thr_buy,
        "sl_percent": sl,
        "tp_percent": tp,
        "rr_realized": rr,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "n_buys": n_buys,
        "n_sells": n_sells,
    })

df_results = pd.DataFrame(results)
df_results.to_csv("quant_sweep_summary_full.csv", index=False)

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
