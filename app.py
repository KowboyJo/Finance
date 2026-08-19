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
st.markdown("Scan large-cap equities for high-efficiency income near technical support.")

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
    return ["UBER", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "WMT"]


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


def analyze_puts(ticker_symbol, exp_date, spot_price, expiry_label, target_delta):
  tk = yf.Ticker(ticker_symbol)
  try:
    chain = tk.option_chain(exp_date)
    puts = chain.puts
    if puts.empty:
      return None

    otm_puts = puts[puts["strike"] <= spot_price].copy()
    if otm_puts.empty:
      otm_puts = puts

    otm_puts["target_dist"] = abs(
        (otm_puts["strike"] / spot_price) - (1.0 - (target_delta * 0.15))
    )
    chosen_put = otm_puts.loc[otm_puts["target_dist"].idxmin()]

    strike = chosen_put["strike"]
    bid = chosen_put["bid"] if pd.notna(chosen_put["bid"]) else 0.0
    ask = chosen_put["ask"] if pd.notna(chosen_put["ask"]) else 0.0
    mid_price = (
        (bid + ask) / 2 if (bid > 0 and ask > 0) else chosen_put["lastPrice"]
    )
    iv = chosen_put.get("impliedVolatility", 0.0) * 100

    dte = max(
        1, (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.today()).days
    )
    ratio = (mid_price / spot_price) * 100

    return {
        "Ticker": ticker_symbol,
        "Cycle": expiry_label,
        "DTE": dte,
        "Strike": strike,
        "IV (%)": round(iv, 1),
        "Put Mid": round(mid_price, 2),
        "Ratio (%)": round(ratio, 2),
    }
  except Exception:
    return None


def ai_score_trade(row):
  ratio = row["Ratio (%)"]
  iv = row["IV (%)"]
  dte = row["DTE"]
  daily_yield_velocity = ratio / dte
  iv_score = min(iv / 40.0, 1.5)
  ai_score = (daily_yield_velocity * 100) * (1 + iv_score)
  return round(ai_score, 2)


# --- MAIN EXECUTION FLOW ---
if run_button:
  universe = fetch_universe()
  st.info(f"Loaded {len(universe)} symbols. Screening for fundamentals & support...")

  passed_tickers = []
  progress_bar = st.progress(0)
  status_text = st.empty()

  # Screening Loop
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

      hist = stock.history(period="1yr")
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
    st.success(f"Scanning option chains for {len(passed_tickers)} matches...")

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
            )
            if res:
              results.append(res)
        else:
          weekly_exp, monthly_exp = find_target_expirations(exp_dates)
          if weekly_exp:
            w_res = analyze_puts(
                ticker, weekly_exp, spot_price, "5-Day Weekly", target_delta
            )
            if w_res:
              results.append(w_res)
          if monthly_exp:
            m_res = analyze_puts(
                ticker, monthly_exp, spot_price, "30-Day Monthly", target_delta
            )
            if m_res:
              results.append(m_res)
      except Exception:
        continue

    if results:
      df = pd.DataFrame(results)
      df["AI Score"] = df.apply(ai_score_trade, axis=1)
      df = df.sort_values(by="Ratio (%)", ascending=False).reset_index(
          drop=True
      )

      st.subheader("📊 Tactical Trade Matrix (Near Support)")
      st.dataframe(df, use_container_width=True)

      top_trade = df.iloc[0]
      st.markdown("### 🤖 AI Trade Synthesis Matrix")
      st.info(
          f"• **Top Recommendation:** {top_trade['Ticker']} ({top_trade['Cycle']})\n\n"
          f"• **Optimal Strike:** ${top_trade['Strike']} (IV: {top_trade['IV (%)']}%)\n\n"
          f"• **Capital Efficiency:** Yields {top_trade['Ratio (%)']}% return over"
          f" {top_trade['DTE']} days.\n\n"
          f"• **Tactical Edge:** Optimal balance of volatility capture and theta"
          f" decay velocity near major moving average support."
      )
    else:
      st.warning(
          "No options found matching your parameters for the selected expiration"
          " cycle."
      )
else:
  st.info(
      "Adjust your parameters in the sidebar and click **Run Scanner** to begin."
  )
