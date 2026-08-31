from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Large-Cap & ETF CSP Income Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Large-Cap & Leveraged ETF Cash-Secured Put (CSP) Income Screener")
st.caption("Scan equities and bull/leveraged ETFs for high-efficiency income near technical support.")

# --- ASSET UNIVERSE PRESETS (LONG ONLY) ---
MAG_7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# Bullish Index & Sector ETFs
INDEX_SECTOR_LEVERAGED_ETFS = [
    "SOXL", "SMH", "TECL",  # Semiconductors & Tech
    "TQQQ", "QQQ", "SPY", "UPRO", "IWM", "TNA",  # Indexes
    "FAS", "LABU", "XLE", "ERX", "NUGT", "BOIL"   # Sectors & Commodities
]

# Bullish Single-Stock & Crypto 2x ETFs
SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS = [
    "NVDL", "NVDX",  # NVDA 2x Bull
    "TSLL", "TSLR",  # TSLA 2x Bull
    "MSTU", "MSTX", "MSTP",  # MSTR 2x Bull
    "CONL",  # COIN 2x Bull
    "PTIR", "PLTU",  # PLTR 2x Bull
    "AMZZ", "AMZU",  # AMZN 2x Bull
    "UBRL",  # UBER 2x Bull
    "BITX", "BTCI"   # Bitcoin 2x / High Yield
]

# Keywords used to exclude inverse/short funds dynamically
INVERSE_KEYWORDS = ["BEAR", "SHORT", "INVERSE", "$-1X", "$-2X", "$-3X", "-1X", "-2X", "-3X"]


# --- CORE LOGIC FUNCTIONS ---
@st.cache_data(ttl=86400)
def fetch_universe():
    """Fetches the S&P 500 constituents ticker symbols, falling back to core list if offline."""
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return ["UBER", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "WMT", "AMD"]


def find_target_expirations(available_dates):
    """Finds target option expiration dates: ~5 DTE (weekly) and ~30 DTE (monthly)."""
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
    """Safely checks if an upcoming earnings date occurs on or before expiration."""
    try:
        tk = yf.Ticker(ticker_symbol)
        cal = tk.calendar
        earnings_date = None

        if isinstance(cal, dict):
            if "Earnings Date" in cal and cal["Earnings Date"]:
                earnings_date = pd.to_datetime(cal["Earnings Date"][0])
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                earnings_date = pd.to_datetime(cal.loc["Earnings Date"].iloc[0])

        if earnings_date is not None:
            earnings_date = earnings_date.tz_localize(None)
            today_date = pd.to_datetime(datetime.today().date())
            exp_dt = pd.to_datetime(exp_date)

            if today_date <= earnings_date <= exp_dt:
                return True
    except Exception:
        return False

    return False


def analyze_puts(
    ticker_symbol,
    exp_date,
    spot_price,
    expiry_label,
    target_delta,
    max_spread=15.0,
    check_earnings=False,
    min_oi=100,
):
    """Analyzes put option chains and calculates yield and risk metrics."""
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

        otm_puts["target_dist"] = abs(
            (otm_puts["strike"] / spot_price) - (1.0 - (target_delta * 0.15))
        )
        chosen_put = otm_puts.loc[otm_puts["target_dist"].idxmin()]

        raw_oi = chosen_put.get("openInterest", 0)
        oi = 0 if pd.isna(raw_oi) or raw_oi is None else int(raw_oi)

        if oi < min_oi:
            return None

        strike = chosen_put["strike"]
        bid = chosen_put["bid"] if pd.notna(chosen_put["bid"]) else 0.0
        ask = chosen_put["ask"] if pd.notna(chosen_put["ask"]) else 0.0
        mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else chosen_put.get("lastPrice", 0.0)

        if mid_price <= 0:
            return None

        if bid > 0 and ask > 0:
            spread_pct = ((ask - bid) / mid_price) * 100
            if spread_pct > max_spread:
                return None

        iv = chosen_put.get("impliedVolatility", 0.0) * 100
        dte = max(1, (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.today()).days)
        yield_pct = (mid_price / strike) * 100

        return {
            "Ticker": ticker_symbol,
            "Cycle": expiry_label,
            "DTE": dte,
            "Stock Price": round(spot_price, 2),
            "Strike": strike,
            "Put Premium": round(mid_price, 2),
            "IV (%)": round(iv, 1),
            "Yield (%)": round(yield_pct, 2),
            "Open Interest": oi,
        }
    except Exception:
        return None


def ai_score_trade(row):
    """Calculates AI Trade Score based on yield velocity and IV efficiency."""
    yield_pct = row.get("Yield (%)", 0)
    iv = row.get("IV (%)", 0)
    dte = row.get("DTE", 1)

    daily_yield_velocity = yield_pct / dte
    iv_score = min(iv / 40.0, 1.5)
    ai_score = (daily_yield_velocity * 100) * (1 + iv_score)
    return round(ai_score, 2)


# --- CONDENSED SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Core Controls")

scan_mode = st.sidebar.radio(
    "Target Universe",
    options=[
        "Mag 7 Equities",
        "Single-Stock 2x ETFs",
        "Index & Sector ETFs",
        "Combined Long ETFs",
        "Full Universe",
    ],
    index=0,
)

custom_tickers = st.sidebar.text_input(
    "Add Custom Tickers (comma separated)",
    placeholder="e.g. SOUN, RKLB, SPYI",
    help="Add specific tickers to your active scan.",
)
extra_symbols = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

col1, col2 = st.sidebar.columns(2)
with col1:
    max_pe = st.number_input("Max P/E", value=40.0, step=1.0)
with col2:
    target_delta = st.number_input("Put Delta", value=0.30, step=0.05, format="%.2f")

max_pct_support = st.sidebar.number_input(
    "Max % Above 200D SMA",
    value=20.0,
    step=1.0,
    help="Filters for stocks/ETFs trading within X% of their 200-day moving average.",
)

if scan_mode != "Full Universe":
    st.sidebar.caption("💡 *P/E and Revenue filters are bypassed for ETF and Mag 7 modes.*")

with st.sidebar.expander("🛠️ Advanced Risk & Options Filters", expanded=False):
    exclude_earnings = st.checkbox(
        "Exclude Earnings Cycles",
        value=True,
        help="Filters out tickers with earnings before expiration (ignored for ETFs).",
    )
    max_spread_pct = st.number_input(
        "Max Spread (%)",
        value=15.0,
        step=1.0,
        help="Filters out illiquid option contracts.",
    )
    min_open_interest = st.number_input(
        "Min Open Interest",
        value=100,
        step=25,
        help="Filters out option contracts with low open interest.",
    )

    use_custom_exp = st.checkbox("Specify Exact Expiration")
    target_expiration = None
    if use_custom_exp:
        exp_date_input = st.date_input("Target Expiration")
        target_expiration = exp_date_input.strftime("%Y-%m-%d")

min_revenue = 3.0 * 1e9
min_cash = 1.0 * 1e9

with st.sidebar.expander("📊 Active Quality Thresholds", expanded=False):
    st.text(f"• Min Revenue: ${min_revenue / 1e9:.1f}B (Equities)")
    st.text(f"• Min Cash: ${min_cash / 1e9:.1f}B (Equities)")
    st.text("• Allowed Quote Types: EQUITY, ETF")

run_button = st.sidebar.button("🚀 Run Scanner", type="primary", use_container_width=True)

# --- MAIN EXECUTION FLOW ---
if run_button:
    with st.spinner("🔍 Running deep scan across market universe & option chains..."):

        if scan_mode == "Mag 7 Equities":
            base_universe = MAG_7_TICKERS
        elif scan_mode == "Single-Stock 2x ETFs":
            base_universe = SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS
        elif scan_mode == "Index & Sector ETFs":
            base_universe = INDEX_SECTOR_LEVERAGED_ETFS
        elif scan_mode == "Combined Long ETFs":
            base_universe = list(set(SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS + INDEX_SECTOR_LEVERAGED_ETFS))
        else:
            base_universe = list(set(fetch_universe() + SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS + INDEX_SECTOR_LEVERAGED_ETFS))

        universe = list(set(base_universe + extra_symbols))

        st.info(f"Loaded {len(universe)} symbols ({scan_mode}). Screening fundamentals & support...")

        passed_tickers = []
        pe_map = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, ticker in enumerate(universe):
            status_text.text(f"Screening ({i+1}/{len(universe)}): Checking {ticker}...")
            progress_bar.progress((i + 1) / len(universe))
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                quote_type = info.get("quoteType", "")

                if quote_type not in ["EQUITY", "ETF"]:
                    continue

                is_etf = (quote_type == "ETF")
                long_name = info.get("longName", "").upper()

                if is_etf and any(kw in long_name for kw in INVERSE_KEYWORDS):
                    continue

                price = info.get("currentPrice", info.get("regularMarketPrice", 0))

                if is_etf:
                    pe_ratio = None
                    revenue = min_revenue
                else:
                    pe_ratio = info.get("trailingPE", info.get("forwardPE", 15.0))
                    revenue = info.get("totalRevenue", 0)
                    if not revenue or revenue == 0:
                        try:
                            fin = stock.financials
                            if not fin.empty and "Total Revenue" in fin.index:
                                revenue = fin.loc["Total Revenue"].iloc[0]
                        except Exception:
                            revenue = 0

                if not price or price <= 0:
                    continue

                hist = stock.history(period="1y")
                if hist.empty or len(hist) < 50:
                    continue

                window_size = min(200, len(hist))
                sma_200 = hist["Close"].rolling(window=window_size).mean().iloc[-1]
                pct_above_support = ((price - sma_200) / sma_200) * 100

                pe_pass = True if (is_etf or scan_mode in ["Mag 7 Equities", "Single-Stock 2x ETFs", "Index & Sector ETFs", "Combined Long ETFs"]) else (pe_ratio <= max_pe)

                if (
                    pe_pass
                    and pct_above_support <= max_pct_support
                    and revenue >= min_revenue
                ):
                    passed_tickers.append(ticker)
                    pe_map[ticker] = "ETF" if is_etf else pe_ratio
            except Exception:
                continue

        status_text.text(
            f"Screening complete! Found {len(passed_tickers)} matching assets near support."
        )

        if not passed_tickers:
            st.warning("No stocks or ETFs matched your custom thresholds near support.")
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

                    is_etf_symbol = (pe_map.get(ticker) == "ETF")
                    effective_exclude_earnings = False if is_etf_symbol else exclude_earnings

                    if target_expiration:
                        if target_expiration in exp_dates:
                            res = analyze_puts(
                                ticker,
                                target_expiration,
                                spot_price,
                                f"Custom ({target_expiration})",
                                target_delta,
                                max_spread=max_spread_pct,
                                check_earnings=effective_exclude_earnings,
                                min_oi=min_open_interest,
                            )
                            if res:
                                res["P/E"] = pe_map.get(ticker, "N/A")
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
                                max_spread=max_spread_pct,
                                check_earnings=effective_exclude_earnings,
                                min_oi=min_open_interest,
                            )
                            if w_res:
                                w_res["P/E"] = pe_map.get(ticker, "N/A")
                                results.append(w_res)
                        if monthly_exp:
                            m_res = analyze_puts(
                                ticker,
                                monthly_exp,
                                spot_price,
                                "30-Day Monthly",
                                target_delta,
                                max_spread=max_spread_pct,
                                check_earnings=effective_exclude_earnings,
                                min_oi=min_open_interest,
                            )
                            if m_res:
                                m_res["P/E"] = pe_map.get(ticker, "N/A")
                                results.append(m_res)
                except Exception:
                    continue

            if results:
                df = pd.DataFrame(results)
                df["Chart"] = df["Ticker"].apply(
                    lambda t: f"https://finance.yahoo.com/chart/{t}"
                )
                df["AI Score"] = df.apply(ai_score_trade, axis=1)
                df = df.sort_values(by="Yield (%)", ascending=False).reset_index(
                    drop=True
                )

                col_order = ["Ticker", "P/E"] + [c for c in df.columns if c not in ["Ticker", "P/E"]]
                df = df[col_order]

                st.subheader("📊 Tactical Trade Matrix (Near Support)")
                st.dataframe(
                    df,
                    column_config={
                        "Chart": st.column_config.LinkColumn(
                            "Yahoo Chart",
                            help="Open interactive advanced chart in a new tab",
                            display_text="📈 View Chart",
                        ),
                        "P/E": st.column_config.TextColumn(
                            "P/E Ratio",
                        ),
                    },
                    use_container_width=True,
                )

                st.download_button(
                    label="📥 Download Tactical Matrix as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"csp_screener_{datetime.today().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                )

                top_trade = df.iloc[0]
                st.markdown("### 🤖 AI Trade Synthesis Matrix")
                pe_val = top_trade.get("P/E", "N/A")
                pe_str = f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else str(pe_val)

                st.info(
                    f"• **Top Recommendation:** [{top_trade['Ticker']}](https://finance.yahoo.com/chart/{top_trade['Ticker']}) ({top_trade['Cycle']})\n\n"
                    f"• **Current Stock Price:** ${top_trade['Stock Price']} (P/E Ratio: {pe_str})\n\n"
                    f"• **Optimal Strike:** ${top_trade['Strike']} (IV: {top_trade['IV (%)']}%)\n\n"
                    f"• **Put Premium (Mid):** ${top_trade['Put Premium']}\n\n"
                    f"• **Capital Efficiency:** Yields {top_trade['Yield (%)']}% return over {top_trade['DTE']} days.\n\n"
                    f"• **Tactical Edge:** Optimal balance of volatility capture and theta decay velocity near major moving average support."
                )
            else:
                st.warning(
                    "No options found matching your parameters, spread criteria, or earnings filters for the selected cycle."
                )
else:
    st.info("Adjust your parameters in the sidebar and click **Run Scanner** to begin.")
