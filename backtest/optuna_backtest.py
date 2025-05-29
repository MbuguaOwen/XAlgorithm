import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import pandas as pd
from collections import deque
import numpy as np
from main import process_tick

TICK_CSV = "data/triangular_ticks.csv"
N_TUNE_ROWS = 3500    # Subsample for speed—adjust higher/lower for balance

# ----------- Load Data Once, Globally -----------
df_full = pd.read_csv(TICK_CSV)
df_tune = df_full.iloc[:N_TUNE_ROWS].copy()

def best_entry_in_window(entry_price, direction, prices, sl_percent, tp_percent):
    best_r_r = -np.inf
    best_pnl = 0
    best_entry = entry_price
    n = len(prices)
    for i in range(n):
        trial_entry = prices[i]
        sl_level = trial_entry * (1 - sl_percent / 100) if direction == 1 else trial_entry * (1 + sl_percent / 100)
        tp_level = trial_entry * (1 + tp_percent / 100) if direction == 1 else trial_entry * (1 - tp_percent / 100)
        got_tp = got_sl = False
        for price in prices[i+1:]:
            if direction == 1:
                if price <= sl_level:
                    got_sl = True
                    break
                if price >= tp_level:
                    got_tp = True
                    break
            elif direction == -1:
                if price >= sl_level:
                    got_sl = True
                    break
                if price <= tp_level:
                    got_tp = True
                    break
        risk = abs(trial_entry - sl_level)
        reward = abs(tp_level - trial_entry)
        r_r = reward / risk if risk > 0 else -np.inf
        if got_tp and not got_sl and r_r > best_r_r:
            best_r_r = r_r
            best_pnl = reward
            best_entry = trial_entry
    return best_entry, best_pnl, best_r_r

def run_backtest(
    df,
    base_thr_sell,
    thr_buy,
    sl_percent,
    tp_percent,
    sniper_window=12,
    cluster_size_sell=17,
    cluster_size_buy=4,
):
    # Initialize clusters for test run
    signal_cluster_buy = deque(maxlen=cluster_size_buy)
    signal_cluster_sell = deque(maxlen=cluster_size_sell)
    trades = []
    n_rows = len(df)

    for i, row in df.iterrows():
        # Pass thresholds into process_tick (must update main.py to accept them as arguments if not already)
        trade = process_tick(
            row['timestamp'], row['btc_price'], row['eth_price'], row['ethbtc_price'],
            base_thr_sell=base_thr_sell,
            thr_buy=thr_buy,
            sl_percent=sl_percent,
            tp_percent=tp_percent,
            cluster_buy=cluster_size_buy,
            cluster_sell=cluster_size_sell,
            signal_cluster_buy=signal_cluster_buy,
            signal_cluster_sell=signal_cluster_sell,
        )
        if trade is not None:
            idx = i
            future_idx = min(idx + sniper_window, n_rows - 1)
            prices = df.iloc[idx:future_idx+1][
                'btc_price' if trade['pair']=='BTCUSDT' else 'eth_price'
            ].values
            best_entry, realized_pnl, best_r_r = best_entry_in_window(
                trade['entry_price'], trade['direction'], prices, sl_percent, tp_percent
            )
            if best_r_r <= 0:
                continue
            trades.append({
                **trade,
                "entry_price": best_entry,
                "pnl": realized_pnl,
                "r_r": best_r_r
            })
    if not trades:
        return 0, 0, 0
    pnls = [t['pnl'] for t in trades]
    returns = np.array(pnls)
    sharpe = np.mean(returns) / (np.std(returns) + 1e-9)
    win_rate = np.mean([p > 0 for p in pnls])
    rr_realized = np.nanmean([t['r_r'] for t in trades if not np.isnan(t['r_r'])])
    return rr_realized, sharpe, win_rate

def objective(trial):
    df = df_tune
    # Only tune the 4 trade gates!
    base_thr_sell = trial.suggest_float('base_thr_sell', 0.80, 0.99)
    thr_buy       = trial.suggest_float('thr_buy', 0.70, 0.88)
    sl_percent    = trial.suggest_float('sl_percent', 0.15, 0.20)
    tp_percent    = trial.suggest_float('tp_percent', 0.54, 0.75)
    rr_realized, sharpe, win_rate = run_backtest(
        df,
        base_thr_sell,
        thr_buy,
        sl_percent,
        tp_percent
    )
    # Prune poor trials: abort if no trades or R:R < 1
    if rr_realized < 1 or sharpe < 0.5 or win_rate < 0.3:
        raise optuna.TrialPruned()
    return rr_realized

if __name__ == "__main__":
    print(f"\nLoaded {len(df_tune):,} rows for parameter tuning (fast mode)")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, n_jobs=2, show_progress_bar=True)

    print("\n==== Top Trials ====")
    df_trials = study.trials_dataframe()
    print(df_trials.sort_values("value", ascending=False).head(10))
    df_trials.to_csv("optuna_backtest_results.csv", index=False)
    print("\nBest params:")
    print(study.best_params)
