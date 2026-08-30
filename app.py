import os
import io
import datetime
import pandas as pd
import streamlit as st
import yfinance as yf

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Options Income Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stMetric {
        background-color: #1e222d;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #2a2e39;
    }
    .stDataFrame {
        border: 1px solid #2a2e39;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONSTANTS & ASSET UNIVERSES
# ==========================================
UNIVERSES = {
    "Mag 7 & Tech Leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "PLTR", "SOUN", "UBER"],
    "High-Beta & Crypto-Linked": ["MSTR", "COIN", "MARA", "RIOT", "RKLB", "ABTC"],
    "Income & Dividend ETFs": ["SPYI", "QQQI", "BTCI", "JEPI", "JEPQ"],
    "Industrial & Materials": ["GLW"],
    "Custom List": []
}

# ==========================================
# HELPER FUNCTIONS & SCREENING ENGINE
# ==========================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker_symbol):
    """Fetch underlying stock price, 200 SMA, and RSI."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 200:
            return None
        
        current_price = hist["Close"].iloc[-1]
        sma_200 = hist["Close"].rolling(window=200).mean().iloc[-1]
        
        # 14-day RSI calculation
        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]

        # Upcoming earnings date check
        earnings_date = "N/A"
        try:
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                if "Earnings Date" in cal.index:
                    earnings_date = str(cal.loc["Earnings Date"].iloc[0].date())
        except Exception:
            pass

        return {
            "ticker": ticker_symbol,
            "price": current_price,
            "sma_200": sma_200,
            "above_200_sma": current_price >= sma_200,
            "rsi_14": rsi_14,
            "earnings_date": earnings_date,
            "ticker_obj": ticker
        }
    except Exception:
        return None

def scan_cash_secured_puts(stock_info, max_dte, min_dte, max_put_delta, min_premium_ratio, min_open_interest, max_spread_pct):
    """Filter options chain for Cash-Secured Puts (CSPs)."""
    results = []
    ticker = stock_info["ticker_obj"]
    current_price = stock_info["price"]

    try:
        expirations = ticker.expirations
    except Exception:
        return results

    today = datetime.date.today()

    for exp in expirations:
        exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
        dte = (exp_date - today).days

        if not (min_dte <= dte <= max_dte):
            continue

        try:
            opt_chain = ticker.option_chain(exp)
            puts = opt_chain.puts
        except Exception:
            continue

        if puts.empty:
            continue

        for _, put in puts.iterrows():
            strike = put["strike"]
            bid = put.get("bid", 0.0)
            ask = put.get("ask", 0.0)
            open_interest = put.get("openInterest", 0)
            delta = abs(put.get("delta", 0.0)) if "delta" in put and not pd.isna(put["delta"]) else None

            # Basic liquidity & pricing sanity filters
            if bid <= 0 or ask <= 0:
                continue

            mid_price = (bid + ask) / 2.0
            spread_pct = ((ask - bid) / mid_price) * 100.0

            if open_interest < min_open_interest or spread_pct > max_spread_pct:
                continue

            # Option metrics calculation
            discount_to_market = ((current_price - strike) / current_price) * 100.0
            collateral = strike * 100
            total_premium = mid_price * 100
            premium_ratio = (mid_price / strike) * 100.0  # Return on collateral per cycle
            annualized_return = (premium_ratio / dte) * 365.0

            # Delta threshold check (if delta available)
            if delta is not None and delta > max_put_delta:
                continue

            # Premium ratio threshold
            if premium_ratio < min_premium_ratio:
                continue

            results.append({
                "Ticker": stock_info["ticker"],
                "Stock Price": round(current_price, 2),
                "Strike": strike,
                "Type": "PUT (CSP)",
                "Expiration": exp,
                "DTE": dte,
                "Bid": round(bid, 2),
                "Ask": round(ask, 2),
                "Mid Premium": round(mid_price, 2),
                "Spread %": round(spread_pct, 1),
                "Discount %": round(discount_to_market, 2),
                "Premium %": round(premium_ratio, 2),
                "Ann. Return %": round(annualized_return, 2),
                "Delta": round(delta, 2) if delta else "N/A",
                "Open Interest": int(open_interest),
                "200 SMA Support": "YES" if stock_info["above_200_sma"] else "NO",
                "RSI (14)": round(stock_info["rsi_14"], 1),
                "Earnings Date": stock_info["earnings_date"]
            })

    return results

# ==========================================
# SIDEBAR CONTROLS & PARAMETERS
# ==========================================
st.sidebar.title("🔍 Screener Filters")

# 1. Universe Selection
universe_choice = st.sidebar.selectbox("Select Asset Universe", list(UNIVERSES.keys()))

if universe_choice == "Custom List":
    custom_raw = st.sidebar.text_input("Enter Tickers (comma separated)", "TSLA, PLTR, MSTR, AMZN")
    ticker_list = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
else:
    ticker_list = UNIVERSES[universe_choice]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Strategy & Option Rules")

# 2. Options Parameter Inputs
col_dte1, col_dte2 = st.sidebar.columns(2)
with col_dte1:
    min_dte = st.number_input("Min DTE", min_value=1, max_value=90, value=14, step=1)
with col_dte2:
    max_dte = st.number_input("Max DTE", min_value=1, max_value=180, value=45, step=1)

max_put_delta = st.sidebar.slider("Max Put Delta (|Δ|)", min_value=0.05, max_value=0.50, value=0.30, step=0.01)
min_premium_ratio = st.sidebar.number_input("Min Return on Collateral %", min_value=0.1, max_value=10.0, value=1.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Liquidity & Technical Controls")

min_open_interest = st.sidebar.number_input("Min Open Interest", min_value=0, max_value=10000, value=100, step=50)
max_spread_pct = st.sidebar.slider("Max Bid-Ask Spread %", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
require_above_sma = st.sidebar.checkbox("Require Stock > 200-Day SMA", value=False)

# ==========================================
# MAIN INTERFACE & PROCESSING
# ==========================================
st.title("📈 Options Income Screener")
st.caption("Cash-Secured Put (CSP) Scanner | Real-time Option Chains & Technical Filters")

if not ticker_list:
    st.warning("Please select or enter at least one valid ticker symbol to scan.")
    st.stop()

st.info(f"Targeting **{len(ticker_list)}** symbols in universe: `{', '.join(ticker_list)}`")

run_scan = st.button("🚀 Run Options Scan", type="primary")

if run_scan:
    all_opportunities = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker_symbol in enumerate(ticker_list):
        status_text.text(f"Fetching market data for {ticker_symbol} ({idx+1}/{len(ticker_list)})...")
        stock_info = fetch_stock_data(ticker_symbol)

        if not stock_info:
            continue

        if require_above_sma and not stock_info["above_200_sma"]:
            continue

        status_text.text(f"Scanning option chains for {ticker_symbol}...")
        opportunities = scan_cash_secured_puts(
            stock_info,
            max_dte=max_dte,
            min_dte=min_dte,
            max_put_delta=max_put_delta,
            min_premium_ratio=min_premium_ratio,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct
        )
        all_opportunities.extend(opportunities)
        progress_bar.progress((idx + 1) / len(ticker_list))

    progress_bar.empty()
    status_text.empty()

    # Display Results
    if all_opportunities:
        df = pd.DataFrame(all_opportunities)
        
        # Metric Summary Cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Puts Found", len(df))
        col_m2.metric("Avg Annualized Return", f"{df['Ann. Return %'].mean():.1f}%")
        col_m3.metric("Avg Discount to Price", f"{df['Discount %'].mean():.1f}%")
        col_m4.metric("Highest Premium Ratio", f"{df['Premium %'].max():.2f}%")

        st.markdown("### 📋 Screened Put Selling Opportunities")
        
        # Dataframe Sorting & Display
        df_sorted = df.sort_values(by="Ann. Return %", ascending=False)
        st.dataframe(df_sorted, use_container_width=True)

        # CSV Export Option
        csv_buffer = io.BytesIO()
        df_sorted.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Screened Results to CSV",
            data=csv_buffer.getvalue(),
            file_name=f"options_scan_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No option contracts met your criteria. Try widening DTE ranges, lowering minimum return thresholds, or allowing higher bid-ask spreads.")
