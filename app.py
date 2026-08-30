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
    page_title="Tactical Income & Leveraged ETF CSP Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Tactical Income & Leveraged ETF Cash-Secured Put Screener")
st.caption("Scan Mag 7 equities, index leverage, and 2x single-stock/crypto ETFs for high-yield CSP trades.")

# --- ASSET UNIVERSE PRESETS ---
MAG_7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

INDEX_SECTOR_LEVERAGED_ETFS = [
    "SOXL", "SOXS", "SMH", "TECL", "TECS",  # Semiconductors & Tech
    "TQQQ", "SQQQ", "QQQ", "SPY", "UPRO", "SPXU", "IWM", "TNA", "TZA",  # Indexes
    "FAS", "FAZ", "LABU", "LABD", "XLE", "ERX", "NUGT", "BOIL", "KOLD"   # Sectors & Commodities
]

SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS = [
    "NVDL", "NVDX",  # NVDA 2x Bull
    "TSLL", "TSLR",  # TSLA 2x Bull
    "MSTU", "MSTX", "MSTP",  # MSTR 2x Bull
    "CONL",  # COIN 2x Bull
    "PTIR", "PLTU",  # PLTR 2x Bull
    "AMZZ", "AMZU",  # AMZN 2x Bull
    "UBRL",  # UBER 2x Bull
    "BITX", "BTCI"   # Bitcoin 2x / High Income
]

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Core Controls")

# Multi-option asset target switch
scan_mode = st.sidebar.radio(
    "Target Asset Universe",
    options=[
        "Mag 7 Equities",
        "Single-Stock & Crypto 2x ETFs",
        "Index & Sector Leveraged ETFs",
        "All Leveraged ETFs Combined",
        "Full Universe (Equities + ETFs)",
    ],
    index=1,
    help="Select your target trading universe. Fundamental checks (P/E & Revenue) are automatically bypassed for ETFs.",
)

# Custom ticker overlay
custom_tickers = st.sidebar.text_input(
    "Add Custom Tickers (comma separated)",
    placeholder="e.g. SOUN, RKLB, SPYI",
    help="Add specific tickers to your active scan.",
)

extra_symbols = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

# Strategy Parameters
col1, col2 = st.sidebar.columns(2)
with col1:
    max_pe = st.number_input("Max P/E (Equities)", value=40.0, step=1.0)
with col2:
    target_delta = st.number_input("Target Put Delta", value=0.25, step=0.05, format="%.2f")

max_pct_support = st.sidebar.number_input(
    "Max % Above 200D SMA",
    value=40.0,
    step=5.0,
    help="Increase this threshold (e.g. 40%-60%) when scanning high-beta/leveraged ETFs.",
)

if "ETF" in scan_mode or scan_mode == "Mag 7 Equities":
    st.sidebar.caption("💡 *P/E and Revenue quality thresholds are automatically bypassed for ETF instruments.*")

# Advanced Risk Controls
with st.sidebar.expander("🛠️ Advanced Risk & Options Filters", expanded=False):
    exclude_earnings = st.checkbox(
        "Exclude Earnings Cycles",
        value=True,
        help="Filters out equities with earnings before expiration (automatically ignored for ETFs).",
    )
    max_spread_pct = st.number_input(
        "Max Option Spread (%)",
        value=15.0,
        step=1.0,
        help="Filters out illiquid option contracts with wide bid-ask spreads.",
    )

    use_custom_exp = st.checkbox("Specify Exact Expiration")
    target_expiration = None
    if use_custom_exp:
        exp_date_input = st.date_input("Target Expiration Date")
        target_expiration = exp_date_input.strftime("%Y-%m-%d")

# Quality Threshold Display
min_revenue = 3.0 * 1e9
min_cash = 1.0 * 1e9

with st.sidebar.expander("📊 Quality Threshold Settings", expanded=False):
    st.text(f"• Min Revenue: ${min_revenue / 1e9:.1f}B (Equities only)")
    st.text(f"• Min Cash: ${min_cash / 1e9:.1f}B (Equities only)")
    st.text("• Quote Types Allowed: EQUITY, ETF")

run_button = st.sidebar.button("🚀 Run Options Scanner", type="primary", use_container_width=True)

# --- MAIN EXECUTION FLOW ---
if run_button:
    with st.spinner("🔍 Executing options chain scan across selected universe..."):
        
        # Route selection based on active scan_mode
        if scan_mode == "Mag 7 Equities":
            base_universe = MAG_7_TICKERS
        elif scan_mode == "Single-Stock & Crypto 2x ETFs":
            base_universe = SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS
        elif scan_mode == "Index & Sector Leveraged ETFs":
            base_universe = INDEX_SECTOR_LEVERAGED_ETFS
        elif scan_mode == "All Leveraged ETFs Combined":
            base_universe = list(set(SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS + INDEX_SECTOR_LEVERAGED_ETFS))
        else:
            base_universe = list(set(fetch_universe() + SINGLE_STOCK_CRYPTO_LEVERAGED_ETFS + INDEX_SECTOR_LEVERAGED_ETFS))

        universe = list(set(base_universe + extra_symbols))

        st.info(f"Loaded {len(universe)} symbols for **{scan_mode}**. Screening support levels...")

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

                # Skip non-equity/non-ETF assets
                if quote_type not in ["EQUITY", "ETF"]:
                    continue

                is_etf = (quote_type == "ETF")
                price = info.get("currentPrice", info.get("regularMarketPrice", 0))

                # Handle P/E and Revenue for ETFs vs Equities
                if is_etf:
                    pe_ratio = None
                    revenue = min_revenue  # Bypass revenue requirement for ETFs
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

                # Bypass P/E requirement for Mag 7 or ETFs
                pe_pass = True if (is_etf or scan_mode != "Full Universe (Equities + ETFs)") else (pe_ratio <= max_pe)

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
            f"Screening complete! Found {len(passed_tickers)} matching assets near technical support."
        )

        if not passed_tickers:
            st.warning("No tickers matched your moving average parameters or support constraints.")
        else:
            st.success(
                f"Fetching options chains & applying delta/spread filters for {len(passed_tickers)} candidates..."
            )

            results = []
            for ticker in passed_tickers:
                tk = yf.Ticker(ticker)
                try:
                    spot_price = tk.history(period="1d")["Close"].iloc[-1]
                    exp_dates = tk.options
                    if not exp_dates:
                        continue

                    # ETF earnings exclusion bypass
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
                                max_spread_pct,
                                effective_exclude_earnings,
                            )
                            if res:
                                res["P/E"] = pe_map.get(ticker, "ETF")
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
                                effective_exclude_earnings,
                            )
                            if w_res:
                                w_res["P/E"] = pe_map.get(ticker, "ETF")
                                results.append(w_res)
                        if monthly_exp:
                            m_res = analyze_puts(
                                ticker,
                                monthly_exp,
                                spot_price,
                                "30-Day Monthly",
                                target_delta,
                                max_spread_pct,
                                effective_exclude_earnings,
                            )
                            if m_res:
                                m_res["P/E"] = pe_map.get(ticker, "ETF")
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

                # Format and reorder table columns
                col_order = ["Ticker", "P/E"] + [c for c in df.columns if c not in ["Ticker", "P/E"]]
                df = df[col_order]

                st.subheader("📊 Tactical Trade Matrix (Near Support)")
                
                st.dataframe(
                    df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "P/E": st.column_config.TextColumn("P/E", width="small"),
                        "Stock Price": st.column_config.NumberColumn("Stock Price", format="$%.2f", width="small"),
                        "Strike": st.column_config.NumberColumn("Strike", format="$%.2f", width="small"),
                        "Put Premium": st.column_config.NumberColumn("Premium", format="$%.2f", width="small"),
                        "Yield (%)": st.column_config.NumberColumn("Yield %", format="%.2f%%", width="small"),
                        "DTE": st.column_config.NumberColumn("DTE", width="small"),
                        "AI Score": st.column_config.NumberColumn("AI Score", format="%.1f", width="small"),
                        "Chart": st.column_config.LinkColumn(
                            "Chart",
                            help="Open interactive Yahoo Finance chart",
                            display_text="📈 View",
                            width="small",
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
                pe_val = top_trade["P/E"]
                pe_str = f"{pe_val:.2f}" if isinstance(pe_val, (int, float)) else str(pe_val)

                st.info(
                    f"• **Top Recommendation:** [{top_trade['Ticker']}](https://finance.yahoo.com/chart/{top_trade['Ticker']}) ({top_trade['Cycle']})\n\n"
                    f"• **Current Price:** ${top_trade['Stock Price']} | **P/E Ratio:** {pe_str}\n\n"
                    f"• **Optimal Strike:** ${top_trade['Strike']} (IV: {top_trade['IV (%)']}%)\n\n"
                    f"• **Put Premium (Mid):** ${top_trade['Put Premium']}\n\n"
                    f"• **Capital Efficiency:** Yields {top_trade['Yield (%)']}% return over {top_trade['DTE']} days.\n\n"
                    f"• **Tactical Edge:** Optimal balance of volatility capture and theta decay velocity near major moving average support."
                )
            else:
                st.warning(
                    "No option contracts found matching your delta, target expiration, or bid-ask spread limits."
                )
else:
    st.info("Select your asset universe and parameters in the sidebar, then click **Run Options Scanner**.")
