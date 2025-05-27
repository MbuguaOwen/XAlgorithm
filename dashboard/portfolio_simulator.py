# portfolio_simulator.py – Phase 2: Regime-Aware Virtual Balance & PnL Tracker

import pandas as pd
import os
from datetime import datetime

class VirtualPortfolio:
    def __init__(self, initial_balance=10000.0):
        self.usdt_balance = initial_balance
        self.position = 0  # +1 = long, -1 = short, 0 = flat
        self.entry_price = None
        self.active_trade = None
        self.trades = []

    def execute(self, signal, price, timestamp, pair, regime="default"):
        tp_sl_map = {
            "volatile": {"tp": 5, "sl": 3},
            "flat": {"tp": 2, "sl": 2},
            "trending": {"tp": 15, "sl": 7},
            "default": {"tp": 4, "sl": 3},
        }

        log = {
            "timestamp": timestamp,
            "pair": pair,
            "price": price,
            "signal": signal,
            "regime": regime,
            "usdt_balance": self.usdt_balance,
            "position_before": self.position
        }

        if signal == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
            self.active_trade = {
                "entry": price,
                "tp": price + tp_sl_map[regime]["tp"],
                "sl": price - tp_sl_map[regime]["sl"]
            }

        elif signal == -1 and self.position == 1:
            pnl = price - self.entry_price
            self.usdt_balance += pnl * 1000
            self.position = 0
            self.entry_price = None
            log["exit_price"] = price
            log["pnl"] = pnl * 1000
            self.active_trade = None
        else:
            log["pnl"] = 0

        log["position_after"] = self.position
        log["usdt_balance_after"] = self.usdt_balance
        self.trades.append(log)

    def export(self, path="logs/portfolio_log.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(self.trades)
        df.to_csv(path, index=False)
        print(f"✅ Portfolio log saved to: {path}")

# Example Usage
if __name__ == "__main__":
    vp = VirtualPortfolio()
    vp.execute(signal=1, price=1000.0, timestamp=str(datetime.now()), pair="ETH/USDT", regime="trending")
    vp.execute(signal=-1, price=1010.0, timestamp=str(datetime.now()), pair="ETH/USDT", regime="trending")
    vp.export()
