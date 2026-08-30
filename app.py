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

# --- CONDENSED SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Core Controls")

# Universe Selection Mode
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

# Custom Ticker Overlay
custom_tickers = st.sidebar.text_input(
    "Add Custom Tickers (comma separated)",
    placeholder="e.g. SOUN, RKLB, SPYI",
    help="Add specific tickers to your active scan.",
)
extra_symbols = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]

# Core Strategy Parameters
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

# Advanced Risk Controls inside an Expander
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

    use_custom_exp = st.checkbox("Specify Exact Expiration")
    target_expiration = None
    if use_custom_exp:
        exp_date_input = st.date_input("Target Expiration")
        target_expiration = exp_date_input.strftime("%Y-%m-%d")

# Active Hardcoded Constraints inside an Expander
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
        
        # Route universe selection
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

                # Allow EQUITY and ETF types
                if quote_type not in ["EQUITY", "ETF"]:
                    continue

                is_etf = (quote_type == "ETF")
                long_name = info.get("longName", "").upper()

                # Dynamic Inverse/Short ETF Guardrail
                if is_etf and any(kw in long_name for kw in INVERSE_KEYWORDS):
                    continue

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

                # Bypass P/E requirement for ETFs or Mag 7
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
                                max_spread_pct,
                                effective_exclude_earnings,
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
                                max_spread_pct,
                                effective_exclude_earnings,
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

                # Reorder columns to place P/E right after Ticker
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
