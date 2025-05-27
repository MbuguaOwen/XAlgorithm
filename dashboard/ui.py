# ui.py – XAlgo Core: Streamlit Dashboard with Health Check & Safe Paths

import os
import time
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="📈 XAlgo Core Dashboard", layout="wide")
st.title("📊 Live Signal & PnL Monitor – XAlgo Core")

# 🩺 Health Check for Log Files
def log_status(path):
    return os.path.exists(path) and os.path.getsize(path) > 10

# 🔁 Auto Refresh Toggle
auto_refresh = st.sidebar.checkbox("🔁 Auto Refresh Every 2s", value=True)
if auto_refresh:
    time.sleep(2)
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()
    st.stop()

# ✅ Use relative paths from project root
signal_path = "logs/signal_log.csv"
exec_path = "logs/execution_log.csv"
portfolio_path = "logs/portfolio_log.csv"

# 🔎 File Status Display
with st.sidebar:
    st.markdown("### 🔎 Log File Status")
    st.markdown(f"**Signals:** {'✅ Found' if log_status(signal_path) else '❌ Missing'}")
    st.markdown(f"**Executions:** {'✅ Found' if log_status(exec_path) else '❌ Missing'}")
    st.markdown(f"**Portfolio:** {'✅ Found' if log_status(portfolio_path) else '❌ Missing'}")

# ─────────────────────────────────────────────
# 📡 Live Signals
# ─────────────────────────────────────────────
st.subheader("📡 ML Signal Log")
if os.path.exists(signal_path):
    try:
        df_signal = pd.read_csv(signal_path)
        df_signal["timestamp"] = pd.to_datetime(df_signal["timestamp"])
        df_signal = df_signal.sort_values("timestamp", ascending=False)
        st.dataframe(df_signal.head(50), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load signals: {e}")
else:
    st.warning("No signal_log.csv file found.")

# ─────────────────────────────────────────────
# 🚀 Executions
# ─────────────────────────────────────────────
st.subheader("🚀 Execution Log")
if os.path.exists(exec_path):
    try:
        df_exec = pd.read_csv(exec_path)
        df_exec["timestamp"] = pd.to_datetime(df_exec["timestamp"])
        df_exec = df_exec.sort_values("timestamp", ascending=False)
        st.dataframe(df_exec.head(50), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load execution log: {e}")
else:
    st.warning("No execution_log.csv file found.")

# ─────────────────────────────────────────────
# 💰 Portfolio Equity Curve
# ─────────────────────────────────────────────
st.subheader("💰 Virtual Portfolio Tracker")
if os.path.exists(portfolio_path):
    try:
        df_port = pd.read_csv(portfolio_path)
        df_port["timestamp"] = pd.to_datetime(df_port["timestamp"])
        df_port = df_port.sort_values("timestamp")

        st.line_chart(df_port[["timestamp", "usdt_balance_after"]].set_index("timestamp"))
        latest = df_port.iloc[-1]
        st.metric("Current Balance", f"${latest['usdt_balance_after']:.2f}")
        st.metric("Current Regime", latest["regime"].capitalize())

    except Exception as e:
        st.error(f"Failed to load portfolio log: {e}")
else:
    st.warning("No portfolio_log.csv file found.")
