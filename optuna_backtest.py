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

def get_regime(row):
    returns = row['btc_price'] - row.get('btc_price_prev', row['btc_price'])
    if returns > 0.5:
        return 'bull'
    elif returns < -0.5:
        return 'bear'
    else:
        return 'neutral'

def run_backtest(
    df,
    base_thr_sell,
    thr_buy,
    sl_percent,
    tp_percent,
    conviction_weight,
    coint_weight,
    zscore_weight,
    slope_weight,
    sniper_window,
    base_cl_sell=17,
    cl_buy=4,
    min_spread=0.00008
):
    import main
    main.CONVICTION_WEIGHT = conviction_weight
    main.COINT_WEIGHT = coint_weight
    main.ZSCORE_WEIGHT = zscore_weight
    main.SLOPE_WEIGHT = slope_weight
    main.MASTER_CONVICTION_THRESHOLD_BUY = thr_buy
    main.CLUSTER_SIZE_BUY = cl_buy
    main.MIN_SPREAD_MAGNITUDE = min_spread
    main.signal_cluster_buy = deque(maxlen=cl_buy)
    main.veto_counters = {k: 0 for k in main.veto_counters}

    trades = []
    n_rows = len(df)
    prev_btc = None
    for i, row in df.iterrows():
        if prev_btc is not None:
            row['btc_price_prev'] = prev_btc
        regime = get_regime(row)
        prev_btc = row['btc_price']
        if regime == 'bull':
            master_threshold_sell = base_thr_sell
            cluster_size_sell = base_cl_sell
        elif regime == 'bear':
            master_threshold_sell = max(0.55, base_thr_sell - 0.2)
            cluster_size_sell = max(3, base_cl_sell - 10)
        else:
            master_threshold_sell = base_thr_sell - 0.1
            cluster_size_sell = max(5, base_cl_sell - 5)
        main.MASTER_CONVICTION_THRESHOLD_SELL = master_threshold_sell
        main.CLUSTER_SIZE_SELL = cluster_size_sell
        main.signal_cluster_sell = deque(maxlen=cluster_size_sell)

        trade = main.process_tick(
            row['timestamp'], row['btc_price'], row['eth_price'], row['ethbtc_price']
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
    else:
        pnls = [t['pnl'] for t in trades]
        returns = np.array(pnls)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9)
        win_rate = np.mean([p > 0 for p in pnls])
        rr_realized = np.nanmean([t['r_r'] for t in trades if not np.isnan(t['r_r'])])
        return rr_realized, sharpe, win_rate

def objective(trial):
    # Use data slice for fast tuning
    df = df_tune
    conviction_weight = trial.suggest_float('conviction_weight', 0.10, 0.20)
    coint_weight      = trial.suggest_float('coint_weight', 0.35, 0.60)
    zscore_weight     = trial.suggest_float('zscore_weight', 0.30, 0.60)
    slope_weight      = trial.suggest_float('slope_weight', 0.10, 0.20)
    sniper_window     = trial.suggest_int('sniper_window', 5, 13)
    base_thr_sell     = trial.suggest_float('base_thr_sell', 0.70, 0.95)
    thr_buy           = trial.suggest_float('thr_buy', 0.62, 0.80)
    sl_percent        = trial.suggest_float('sl_percent', 0.15, 0.20)
    tp_percent        = trial.suggest_float('tp_percent', 0.40, 0.75)

    rr_realized, sharpe, win_rate = run_backtest(
        df,
        base_thr_sell,
        thr_buy,
        sl_percent,
        tp_percent,
        conviction_weight,
        coint_weight,
        zscore_weight,
        slope_weight,
        sniper_window
    )
    # Prune poor trials: abort if no trades or R:R < 1
    if rr_realized < 1 or sharpe < 0.5 or win_rate < 0.3:
        raise optuna.TrialPruned()
    return rr_realized

if __name__ == "__main__":
    print(f"\nLoaded {len(df_tune):,} rows for parameter tuning (fast mode)")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=40, n_jobs=2, show_progress_bar=True)  # 2 jobs = faster startup

    print("\n==== Top Trials ====")
    df_trials = study.trials_dataframe()
    print(df_trials.sort_values("value", ascending=False).head(10))
    df_trials.to_csv("optuna_backtest_results.csv", index=False)
    print("\nBest params:")
    print(study.best_params)
