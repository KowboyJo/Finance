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
st.sidebar.markdown("**Active Hardcoded Filters:**")
st.sidebar.text(f"• Min Revenue: ${min_revenue / 1e9:.1f}B")
st.sidebar.text(f"• Min Cash: ${min_cash / 1e9:.1f}B")
st.sidebar.text("• Quote Type: EQUITY")

run_button = st.sidebar.button("Run Scanner", type="primary")


# --- CORE LOGIC FUNCTIONS ---
@st.cache_data(ttl=86400)
def fetch_universe():
  try:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()
  except Exception:
    return ["UBER", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "WMT"]


def find_target_expirations(available_dates):
  today = datetime.today()
  best_weekly = None
  best_monthly = None
  min_w_diff, min_m_diff = float("inf"), float("inf")

  for date_str in available_dates:
    exp_date = datetime.strptime(date_str, "%Y-%m-%d")
    diff_days = (exp_date - today).days

    if diff_days < 1:
      continue

    if abs(diff_days - 5) < min_w_diff and diff_days <= 10:
      min_w_diff = abs(diff_days - 5)
      best_weekly = date_str

    if abs(diff_days - 30) < min_m_diff and 21 <= diff_days <= 45:
      min_m_diff = abs(diff_days - 30)
      best_monthly = date_str

  if not best_weekly and len(available_dates) > 0:
    best_weekly = available_dates[0]
  if not best_monthly and len(available_dates) > 1:
    best_monthly = available_dates[min(4, len(available_dates) - 1)]

  return best_weekly, best_monthly


def check_earnings_conflict(ticker_symbol, exp_date):
  """Checks if earnings fall on or before the option expiration date."""
  try:
    tk = yf.Ticker(ticker_symbol)
    cal = tk.calendar
    earnings_date = None

    if isinstance(cal, dict) and "Earnings Date" in cal:
      ed_list = cal["Earnings Date"]
      if ed_list:
        earnings_date = pd.to_datetime(ed_list[0])
    elif isinstance(cal, pd.DataFrame) and not cal.empty:
      if "Earnings Date" in cal.index:
        earnings_date = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])

    if earnings_date is not None:
      exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
      if datetime.today() <= earnings
