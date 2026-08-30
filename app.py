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
    page_title="Large-Cap AI CSP Income Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Large-Cap AI Cash-Secured Put (CSP) Income Screener")
st.caption("Scan large-cap equities for high-efficiency income near technical support.")

# --- MAG 7 CONSTANT ---
MAG_7_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]

# --- CONDENSED SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Core Controls")

# Universe Selection Mode
scan_mode = st.sidebar.segmented_control(
    "Target Universe",
    options=["Mag 7 Only", "Full Universe"],
    default="Mag 7 Only",
)

# Core Strategy Parameters (Side-by-Side to save height)
col1, col2 = st.sidebar.columns(2)
with col1:
    max_pe = st.number_input("Max P/E", value=40.0, step=1.0)
with col2:
    target_delta = st.number_input("Put Delta", value=0.30, step=0.05, format="%.2f")

max_pct_support = st.sidebar.number_input(
    "Max % Above 200D SMA",
    value=20.0,
    step=1.0,
    help="Filters for stocks trading within X% of their 200-day moving average.",
)

if scan_mode == "Mag 7 Only":
    st.sidebar.caption("💡 *P/E filter is bypassed for Mag 7 (displayed in table).*")

# Advanced Risk Controls inside an Expander
with st.sidebar.expander("🛠️ Advanced Risk & Options Filters", expanded=False):
    exclude_earnings = st.checkbox(
        "Exclude Earnings Cycles",
        value=True,
        help="Filters out tickers with earnings before expiration.",
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
    st.text(f"• Min Revenue: ${min_revenue / 1e9:.1f}B")
    st.text(f"• Min Cash: ${min_cash / 1e9:.1f}B")
    st.text("• Quote Type: EQUITY")

run_button = st.sidebar.button("🚀 Run Scanner", type="primary", use_container_width=True)

# --- MAIN EXECUTION FLOW ---
if run_button:
    with st.spinner("🔍 Running deep scan across market universe & option chains..."):
        if scan_mode == "Mag 7 Only":
            universe = MAG_7_TICKERS
        else:
            universe = fetch_universe()

        st.info(f"Loaded {len(universe)} symbols. Screening fundamentals & support...")

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

                if info.get("quoteType") != "EQUITY":
                    continue

                price = info.get("currentPrice", info.get("regularMarketPrice", 0))
                pe_ratio = info.get("trailingPE", info.get("forwardPE", 0))

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

                if not pe_ratio:
                    pe_ratio = 15.0

                hist = stock.history(period="1y")
                if hist.empty or len(hist) < 50:
                    continue

                window_size = min(200, len(hist))
                sma_200 = hist["Close"].rolling(window=window_size).mean().iloc[-1]
                pct_above_support = ((price - sma_200) / sma_200) * 100

                # Bypass P/E requirement when scanning Mag 7 mode
                pe_pass = True if scan_mode == "Mag 7 Only" else (pe_ratio <= max_pe)

                if (
                    pe_pass
                    and pct_above_support <= max_pct_support
                    and revenue >= min_revenue
                ):
                    passed_tickers.append(ticker)
                    pe_map[ticker] = pe_ratio
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
                                res["P/E"] = pe_map.get(ticker, 15.0)
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
                                w_res["P/E"] = pe_map.get(ticker, 15.0)
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
                                m_res["P/E"] = pe_map.get(ticker, 15.0)
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

                # Reorder columns to ensure P/E appears right after Ticker
                col_order = ["Ticker", "P/E"] + [c for c in df.columns if c not in ["Ticker", "P/E"]]
                df = df[col_order]

                st.subheader("📊 Tactical Trade Matrix (Near Support)")
                
                # Condensed table width optimization for desktop PC viewports
                st.dataframe(
                    df,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "P/E": st.column_config.NumberColumn("P/E", format="%.2f", width="small"),
                        "Stock Price": st.column_config.NumberColumn("Stock Price", format="$%.2f", width="small"),
                        "Strike": st.column_config.NumberColumn("Strike", format="$%.2f", width="small"),
                        "Put Premium": st.column_config.NumberColumn("Premium", format="$%.2f", width="small"),
                        "Yield (%)": st.column_config.NumberColumn("Yield %", format="%.2f%%", width="small"),
                        "DTE": st.column_config.NumberColumn("DTE", width="small"),
                        "AI Score": st.column_config.NumberColumn("AI Score", format="%.1f", width="small"),
                        "Chart": st.column_config.LinkColumn(
                            "Chart",
                            help="Open interactive advanced chart",
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
                pe_val = f"{top_trade['P/E']:.2f}" if "P/E" in top_trade else "N/A"

                st.info(
                    f"• **Top Recommendation:** [{top_trade['Ticker']}](https://finance.yahoo.com/chart/{top_trade['Ticker']}) ({top_trade['Cycle']})\n\n"
                    f"• **Current Stock Price:** ${top_trade['Stock Price']} | **P/E Ratio:** {pe_val}\n\n"
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
