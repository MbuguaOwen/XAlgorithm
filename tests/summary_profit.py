import os
import sys
import pandas as pd

# ─── Auto-Resolved Path to Logs ────────────────────────────────────────
def get_log_path(filename="signal_log.csv"):
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go from /tests/ to root
    return os.path.join(base_dir, "logs", filename)

# ─── Summary Function ───────────────────────────────────────────────────
def summarize_signal_log(log_path=None):
    if not log_path:
        log_path = get_log_path()

    if not os.path.exists(log_path):
        print(f"❌ Log file not found at: {log_path}")
        return

    df = pd.read_csv(log_path)
    if "profit" not in df.columns:
        print("⚠️  'profit' column not found in log file.")
        return

    df = df.dropna(subset=["profit"])
    df["profit"] = df["profit"].astype(float)

    win_df = df[df["profit"] > 0]
    loss_df = df[df["profit"] < 0]

    print("📊 Summary of Spread Reversion Trades:")
    print("──────────────────────────────────────────────")
    print(f"Total Trades:         {len(df)}")
    print(f"Profitable Trades:    {len(win_df)}")
    print(f"Loss Trades:          {len(loss_df)}")
    print(f"Win Rate:             {len(win_df) / len(df):.2%}")
    print(f"Average Profit:       {df['profit'].mean():.4f}")
    print(f"Max Profit:           {df['profit'].max():.4f}")
    print(f"Max Loss:             {df['profit'].min():.4f}")
    print("──────────────────────────────────────────────")

# ─── CLI Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    custom_log = sys.argv[1] if len(sys.argv) > 1 else None
    summarize_signal_log(custom_log)
