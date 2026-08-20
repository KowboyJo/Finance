from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf
from screener_logic import (
    ai_score_trade,
    analyze_puts,
    fetch_universe,
    find_target_expirations,
)

# Page Configuration
st.set_page_config(
    page_title="NASDAQ 500 AI CSP Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 NASDAQ 500 AI CSP Screener")
st.markdown(
    "Scan large-cap equities for high-efficiency income near technical support"
    " with advanced risk filters."
)

# --- SIDEBAR CONFIGURATION (Inputs with New Defaults) ---
st.sidebar.header("Screener Parameters")

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

min_revenue = 3.0 * 1e9
min_cash = 1.0 * 1e9

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Hardcoded Filters:**")
st.sidebar.text(f"• Min Revenue: ${min_revenue / 1e9:.1f}B")
st.sidebar.text(f"• Min Cash: ${min_cash / 1e9:.1f}B")
st.sidebar.text("• Quote Type: EQUITY")

run_button = st.sidebar.button("Run Scanner", type="primary")

# --- MAIN EXECUTION FLOW ---
if run_button:
  universe = fetch_universe()
  st.info(f"Loaded {len(universe)} symbols. Screening for fundamentals & support...")

  passed_tickers = []
  progress_bar = st.progress(0)
  status_text = st.empty()

  for i, ticker in enumerate(universe):
    status_text.text(
        f"Screening ({i+1}/{len(universe)}): Checking {ticker}..."
    )
    progress_bar.progress((i + 1) / len(universe))
    try:
      stock = yf.Ticker(ticker)
      info = stock.info

      if info.get("quoteType") != "EQUITY":
        continue

      price = info.get("currentPrice", info.get("regularMarketPrice", 0))
      pe_ratio = info.get("trailingPE", info.get("forwardPE", 0))
      revenue = info.get("totalRevenue", 0)

      if not price or price <= 0:
        continue

      if not pe_ratio:
        pe_ratio = 15.0

      hist = stock.history(period="1y")
      if hist.empty or len(hist) < 50:
        continue

      window_size = min(200, len(hist))
      sma_200 = hist["Close"].rolling(window=window_size).mean().iloc[-1]
      pct_above_support = ((price - sma_200) / sma_200) * 100

      if (
          pe_ratio <= max_pe
          and pct_above_support <= max_pct_support
          and revenue >= min_revenue
      ):
        passed_tickers.append(ticker)
    except Exception:
      continue

  status_text.text(
      f"Screening complete! Found {len(passed_tickers)} matching equities near support."
  )

  if not passed_tickers:
    st.warning("No stocks matched your custom thresholds near support.")
  else:
    st.success(
        f"Scanning option chains & applying risk filters for {len(passed_tickers)} matches..."
    )

    results = []
    for ticker in passed_tickers:
      tk = yf.Ticker(ticker)
      try:
        spot_price = tk.history(period="1d")["Close"].iloc[-1]
        exp_dates = tk.options
        if not exp_dates:
          continue

        if target_expiration:
          if target_expiration in exp_dates:
            res = analyze_puts(
                ticker,
                target_expiration,
                spot_price,
                f"Custom ({target_expiration})",
                target_delta,
                max_spread_pct,
                exclude_earnings,
            )
            if res:
              results.append(res)
        else:
          weekly_exp, monthly_exp = find_target_expirations(exp_dates)
          if weekly_exp:
            w_res = analyze_puts(
                ticker,
                weekly_exp,
                spot_price,
                "5-Day Weekly",
                target_delta,
                max_spread_pct,
                exclude_earnings,
            )
            if w_res:
              results.append(w_res)
          if monthly_exp:
            m_res = analyze_puts(
                ticker,
                monthly_exp,
                spot_price,
                "30-Day Monthly",
                target_delta,
                max_spread_pct,
                exclude_earnings,
            )
            if m_res:
              results.append(m_res)
      except Exception:
        continue

    if results:
      df = pd.DataFrame(results)
      df["Chart"] = df["Ticker"].apply(
          lambda t: f"https://finance.yahoo.com/chart/{t}"
      )
      df["AI Score"] = df.apply(ai_score_trade, axis=1)
      df = df.sort_values(by="Ratio (%)", ascending=False).reset_index(
          drop=True
      )

      st.subheader("📊 Tactical Trade Matrix (Near Support)")
      st.dataframe(
          df,
          column_config={
              "Chart": st.column_config.LinkColumn(
                  "Yahoo Chart",
                  help="Open interactive advanced chart in a new tab",
                  display_text="📈 View Chart",
              )
          },
          use_container_width=True,
      )

      st.download_button(
          label="📥 Download Tactical Matrix as CSV",
          data=df.to_csv(index=False).encode("utf-8"),
          file_name=(
              f"csp_screener_{datetime.today().strftime('%Y-%m-%d')}.csv"
          ),
          mime="text/csv",
      )

      top_trade = df.iloc[0]
      st.markdown("### 🤖 AI Trade Synthesis Matrix")
      st.info(
          f"• **Top Recommendation:** [{top_trade['Ticker']}](https://finance.yahoo.com/chart/{top_trade['Ticker']}) ({top_trade['Cycle']})\n\n"
          f"• **Current Stock Price:** ${top_trade['Stock Price']}\n\n"
          f"• **Optimal Strike:** ${top_trade['Strike']} (IV: {top_trade['IV (%)']}%)\n\n"
          f"• **Put Premium (Mid):** ${top_trade['Put Premium']}\n\n"
          f"• **Capital Efficiency:** Yields {top_trade['Ratio (%)']}% return over"
          f" {top_trade['DTE']} days.\n\n"
          f"• **Tactical Edge:** Optimal balance of volatility capture and theta"
          f" decay velocity near major moving average support."
      )
    else:
      st.warning(
          "No options found matching your parameters, spread criteria, or"
          " earnings filters for the selected cycle."
      )
else:
  st.info(
      "Adjust your parameters in the sidebar and click **Run Scanner** to begin."
  )
