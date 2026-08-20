import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI CSP Income Screener", page_icon="📈", layout="wide"
)

st.title("🤖 AI Cash-Secured Put (CSP) Income Screener")
st.markdown(
    "Scan the market for high-probability, high-yield Cash-Secured Put opportunities using custom fundamental, technical, and options parameters."
)

# --- SIDEBAR CONFIGURATION (Inputs) ---
st.sidebar.header("Screener Parameters")

# Updated default values as requested
max_pe = st.sidebar.number_input("Max P/E Ratio", value=40.0, step=1.0)

max_pct_support = st.sidebar.number_input(
    "Max % Above 200-Day SMA (Support)",
    value=20.0,
    step=1.0,
    help="Filters for stocks trading within X% of their long-term 200-day moving average support level.",
)

target_delta = st.sidebar.number_input(
    "Target Put Delta (OTM)", value=0.30, step=0.05, format="%.2f"
)

min_iv = st.sidebar.slider(
    "Min Implied Volatility (%)", min_value=10, max_value=150, value=30, step=5
)

min_yield = st.sidebar.slider(
    "Min Annualized Return (%)", min_value=5, max_value=100, value=15, step=5
)

# Sample Ticker Universe (Can be expanded or loaded dynamically)
default_tickers = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, AMD, META, NFLX, SPY, QQQ, IWM, JPM, BAC, XOM, CVX"
ticker_input = st.sidebar.text_area(
    "Ticker Universe (comma-separated)", default_tickers
)

run_scan = st.sidebar.button("Run CSP Scan", type="primary")

# --- MAIN LOGIC ---
if run_scan:
  tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

  st.write(
      f"Scanning **{len(tickers)} tickers** with Max P/E: **{max_pe}**, Target"
      f" Delta: **{target_delta}**, and Max 200 SMA Distance:"
      f" **{max_pct_support}%**..."
  )

  progress_bar = st.progress(0)
  results = []

  for i, ticker in enumerate(tickers):
    try:
      stock = yf.Ticker(ticker)
      hist = stock.history(period="1y")

      if hist.empty or len(hist) < 200:
        continue

      current_price = hist["Close"].iloc[-1]
      sma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]
      pct_above_sma = ((current_price - sma_200) / sma_200) * 100

      # Basic fundamentals extraction
      info = stock.info
      pe_ratio = info.get("trailingPE", np.nan)

      # Apply fundamental and technical filters
      if not np.isnan(pe_ratio) and pe_ratio > max_pe:
        continue
      if pct_above_sma > max_pct_support:
        continue

      results.append({
          "Ticker": ticker,
          "Price": round(current_price, 2),
          "
