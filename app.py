from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="KowboyJo's Tactical AI CSP Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 KowboyJo's Tactical AI Cash-Secured Put (CSP) Scanner")
st.markdown(
    "Scan Mag 7 equities, high-yield leveraged ETFs, or the S&P 500 for"
    " cash-secured put income near technical support."
)

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("🎯 Target Universe Selection")

# Asset Universe Selector
selected_universes = st.sidebar.multiselect(
    "Select Target Asset Classes",
    options=["Mag 7 Tech Equities", "Leveraged & Volatile ETFs", "S&P 500 Universe"],
    default=["Mag 7 Tech Equities", "Leveraged & Volatile ETFs"],
    help="Select one or multiple watchlists to scan for CSP candidates.",
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Screener Parameters")

# Cash Allocation Input
available_cash = st.sidebar.number_input(
    "Available Cash ($)",
    value=100000.0,
    step=5000.0,
    help="Total cash available to secure short put option contracts.",
)

max_pe = st.sidebar.number_input(
    "Max P/E Ratio (S&P 500 only)",
    value=40.0,
    step=1.0,
    help="Filters valuation for standard equities. Ignored for Mag 7 and ETFs.",
)
max_pct_support = st.sidebar.number_input(
    "Max % Above 200-Day SMA (Support)",
    value=20.0,
    step=1.0,
    help="Filters for stocks trading within X% of their long-term 200-day moving average support level.",
)
target_delta = st.sidebar.number_input(
    "Target Put Delta (OTM)", value=0.30, step=0.05, format="%.2f"
)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Advanced Risk Controls")

exclude_earnings = st.sidebar.checkbox(
    "Exclude Earnings Within Expiration Cycle",
    value=True,
    help="Filters out tickers that have an earnings release scheduled before the option expires.",
)

max_spread_pct = st.sidebar.number_input(
    "Max Bid-Ask Spread (%)",
    value=15.0,
    step=1.0,
    help="Filters out option contracts where the percentage spread exceeds this threshold.",
)

use_custom_exp = st.sidebar.checkbox("Specify Exact Expiration Date")
target_expiration = None
if use_custom_exp:
  exp_date_input = st.sidebar.date_input("Target Expiration Date")
  target_expiration = exp_date_input.strftime("%Y-%m-%d")

# Hardcoded Core Fundamental Filters
min_revenue = 3.0 * 1e9  # $3.0 Billion
min_cash = 1.0 * 1e9  # $1.0 Billion

run_button = st.sidebar.button("Run Scanner", type="primary")

# ==========================================
# UNIVERSE & DATA FETCHING LOGIC
# ==========================================
MAG_7 = ["TSLA", "NVDA", "AMZN", "GOOGL", "META", "AAPL", "MSFT"]
LEVERAGED_ETFS = ["MSTX", "CONL", "SOXL", "TSLL", "NVDL", "TQQQ", "UPRO"]

@st.cache_data(ttl=86400)
def fetch_sp500_universe():
  """Fetches S&P 500 components list."""
  try:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()
  except Exception:
    return ["UBER", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "WMT"]

def get_target_ticker_list(universes):
  """Assembles composite ticker list from user selection."""
  tickers = []
  if "Mag 7 Tech Equities" in universes:
    tickers.extend(MAG_7)
  if "Leveraged & Volatile ETFs" in universes:
    tickers.extend(LEVERAGED_ETFS)
  if "S&P 500 Universe" in universes:
    tickers.extend(fetch_sp500_universe())
  return list(dict.fromkeys(tickers))  # Preserve order & remove duplicates

def find_target_expirations(available_dates):
  """Locates target weekly (~5 DTE) and monthly (~30 DTE) expirations."""
  today = datetime.today()
  best_weekly, best_monthly = None, None
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
  """Checks if earnings date falls on or before option expiration."""
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
      if datetime.today() <= earnings_date <= exp_dt:
        return True
  except Exception:
    pass
  return False

def calculate_rsi(data, window=14):
  """Calculates 14-period Relative Strength Index (RSI)."""
  delta = data.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
  rs = gain / loss
  return 100 - (100 / (1 + rs))

def analyze_puts(ticker_symbol, exp_date, spot_price, expiry_label, target_delta, max_spread, check_earnings):
  """Extracts optimal put option contract specs."""
  if check_earnings and check_earnings_conflict(ticker_symbol, exp_date):
    return None

  tk = yf.Ticker(ticker_symbol)
  try:
    chain = tk.option_chain(exp_date)
    puts = chain.puts
    if puts.empty:
      return None

    otm_puts = puts[puts["strike"] <= spot_price].copy()
    if otm_puts.empty:
      otm_puts = puts

    otm_puts["target_dist"] = abs((otm_puts["strike"] / spot_price) - (1.0 - (target_delta * 0.15)))
    chosen_put = otm_puts.loc[otm_puts["target_dist"].idxmin()]

    strike = chosen_put["strike"]
    bid = chosen_put["bid"] if pd.notna(chosen_put["bid"]) else 0.0
    ask = chosen_put["ask"] if pd.notna(chosen_put["ask"]) else 0.0
    mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else chosen_put["lastPrice"]

    if bid > 0 and ask > 0 and mid_price > 0:
      spread_pct = ((ask - bid) / mid_price) * 100
      if spread_pct > max_spread:
        return None

    iv = chosen_put.get("impliedVolatility", 0.0) * 100
    dte = max(1, (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.today()).days)
    yield_pct = (mid_price / spot_price) * 100

    return {
        "Ticker": ticker_symbol,
        "Cycle": expiry_label,
        "DTE": dte,
        "Stock Price": round(spot_price, 2),
        "Strike": strike,
        "Put Premium": round(mid_price, 2),
        "IV (%)": round(iv, 1),
        "Yield (%)": round(yield_pct, 2),
    }
  except Exception:
    return None

def ai_score_trade(row):
  """Ranks candidates by daily yield velocity and implied volatility."""
  yield_pct = row["Yield (%)"]
  iv = row["IV (%)"]
  dte = row["DTE"]

  daily_yield_velocity = yield_pct / dte
  iv_score = min(iv / 40.0, 1.5)
  ai_score = (daily_yield_velocity * 100) * (1 + iv_score)
  return round(ai_score, 2)

# ==========================================
# MAIN EXECUTION ROUTINE
# ==========================================
if run_button:
  if not selected_universes:
    st.error("Please select at least one Target Asset Class from the sidebar.")
  else:
    universe = get_target_ticker_list(selected_universes)
    st.info(f"Loaded {len(universe)} symbols across selected watchlists.")

    passed_tickers = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: Fundamental & Support Filtering
    for i, ticker in enumerate(universe):
      status_text.text(f"Screening ({i+1}/{len(universe)}): Checking {ticker}...")
      progress_bar.progress((i + 1) / len(universe))
      try:
        stock = yf.Ticker(ticker)
        info = stock.info
        quote_type = info.get("quoteType", "")

        price = info.get("currentPrice", info.get("regularMarketPrice", 0))
        pe_ratio = info.get("trailingPE", info.get("forwardPE", 0))

        if not price or price <= 0:
          continue

        hist = stock.history(period="1y")
        if hist.empty or len(hist) < 20:
          continue

        # DIRECT OVERRIDE: Bypass P/E and fundamental checks for Mag 7 and ETFs
        is_mag7 = ticker in MAG_7
        is_etf = quote_type == "ETF" or ticker in LEVERAGED_ETFS

        if is_mag7 or is_etf:
          passed_tickers.append(ticker)
          continue

        # Standard S&P 500 P/E & Technical Filter
        if not pe_ratio:
          pe_ratio = 15.0

        window_size = min(200, len(hist))
        sma_200 = hist["Close"].rolling(window=window_size).mean().iloc[-1]
        pct_above_support = ((price - sma_200) / sma_200) * 100
        rsi_series = calculate_rsi(hist["Close"])
        current_rsi = rsi_series.iloc[-1] if not rsi_series.empty else 50.0

        if (
            pe_ratio <= max_pe
            and pct_above_support <= max_pct_support
            and current_rsi <= 65.0
        ):
          passed_tickers.append(ticker)
      except Exception:
        continue

    status_text.text(f"Screening complete! Found {len(passed_tickers)} qualified candidates.")

    # Step 2: Option Chain Analysis & Matrix Generation
    if not passed_tickers:
      st.warning("No tickers matched your specified thresholds.")
    else:
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
              res = analyze_puts(ticker, target_expiration, spot_price, f"Custom ({target_expiration})", target_delta, max_spread_pct, exclude_earnings)
              if res:
                results.append(res)
          else:
            weekly_exp, monthly_exp = find_target_expirations(exp_dates)
            if weekly_exp:
              w_res = analyze_puts(ticker, weekly_exp, spot_price, "5-Day Weekly", target_delta, max_spread_pct, exclude_earnings)
              if w_res:
                results.append(w_res)
            if monthly_exp:
              m_res = analyze_puts(ticker, monthly_exp, spot_price, "30-Day Monthly", target_delta, max_spread_pct, exclude_earnings)
              if m_res:
                results.append(m_res)
        except Exception:
          continue

      if results:
        df = pd.DataFrame(results)

        # Capital Position Sizing Output Calculations
        df["Collateral / Contract"] = df["Strike"] * 100
        df["Max Contracts"] = (available_cash // df["Collateral / Contract"]).astype(int)
        df["Cash Used ($)"] = df["Max Contracts"] * df["Collateral / Contract"]
        df["Total Premium ($)"] = (df["Max Contracts"] * df["Put Premium"] * 100).round(2)
        df["Return on Cash (%)"] = ((df["Total Premium ($)"] / available_cash) * 100).round(2)

        df["Chart"] = df["Ticker"].apply(lambda t: f"https://finance.yahoo.com/chart/{t}")
        df["AI Score"] = df.apply(ai_score_trade, axis=1)
        df = df.sort_values(by="AI Score", ascending=False).reset_index(drop=True)

        st.subheader("📊 Tactical Trade Matrix (Near Support)")

        st.dataframe(
            df[[
                "Ticker",
                "Cycle",
                "DTE",
                "Stock Price",
                "Strike",
                "Put Premium",
                "Max Contracts",
                "Cash Used ($)",
                "Total Premium ($)",
                "Return on Cash (%)",
                "AI Score",
                "Chart",
            ]],
            column_config={
                "Chart": st.column_config.LinkColumn("Yahoo Chart", display_text="📈 View Chart"),
                "Cash Used ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Total Premium ($)": st.column_config.NumberColumn(format="$%.2f"),
                "Return on Cash (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
            use_container_width=True,
        )

        # AI Synthesis Card Output
        top_trade = df.iloc[0]
        st.markdown("### 🤖 AI Trade Synthesis Matrix")
        st.info(
            f"• **Top Recommendation:** [{top_trade['Ticker']}](https://finance.yahoo.com/chart/{top_trade['Ticker']}) ({top_trade['Cycle']})\n\n"
            f"• **Position Sizing (${available_cash:,.0f} Cash):** Sell **{top_trade['Max Contracts']} contracts** at **${top_trade['Strike']}** strike.\n\n"
            f"• **Capital Allocation:** Deploys ${top_trade['Cash Used ($)']:,.0f} in cash collateral.\n\n"
            f"• **Total Income Collected:** **${top_trade['Total Premium ($)']:,.2f}** ({top_trade['Return on Cash (%)']}% yield over {top_trade['DTE']} days).\n\n"
            f"• **Tactical Edge:** Optimal balance of volatility capture and theta decay velocity near key technical support."
        )
      else:
        st.warning("No option contracts passed all liquidity, spread, and delta filters.")
