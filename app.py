from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Large-Cap AI CSP Income Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Large-Cap AI Cash-Secured Put (CSP) Income Screener")
st.markdown(
    "Scan large-cap equities for high-efficiency income near technical support"
    " with advanced risk filters."
)

# --- SIDEBAR CONFIGURATION (Inputs) ---
st.sidebar.header("Screener Parameters")

max_pe = st.sidebar.number_input("Max P/E Ratio", value=100.0, step=1.0)

max_pct_support = st.sidebar.number_input(
    "Max % Above 200-Day SMA (Support)",
    value=50.0,
    step=1.0,
    help="Filters for stocks trading within X% of their long-term 200-day moving average support level.",
)

target_delta = st.sidebar.number_input(
    "Target Put Delta (OTM)", value=0.20, step=0.05, format="%.2f"
)

# Refinement Toggles & Inputs
st.sidebar.markdown("---")
st.sidebar.header("Advanced Risk Controls")
exclude_earnings = st.sidebar.checkbox(
    "Exclude Earnings Within Expiration Cycle",
    value=True,
    help=(
        "Filters out tickers that have an earnings release scheduled before"
        " the option expires."
    ),
)
max_spread_pct = st.sidebar.number_input(
    "Max Bid-Ask Spread (%)",
    value=15.0,
    step=1.0,
    help=(
        "Filters out option contracts where the percentage spread exceeds this"
        " threshold to ensure liquidity."
    ),
)

use_custom_exp = st.sidebar.checkbox("Specify Exact Expiration Date")
target_expiration = None
if use_custom_exp:
  exp_date_input = st.sidebar.date_input("Target Expiration Date")
  target_expiration = exp_date_input.strftime("%Y-%m-%d")

min_revenue = 10.0 * 1e9
min_cash = 1.0 * 1e9

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Hard
