from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf

from screener_logic import (
    ai_score_trade,
    analyze_puts,
    fetch_universe,
    find_target_expirations,
)

# ------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Large-Cap AI CSP Income Screener",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Large-Cap AI Cash-Secured Put (CSP) Income Screener")
st.markdown(
    "Scan large-cap equities for **high-efficiency income** near technical support "
    "with advanced risk and balance-sheet filters.  \n"
    "Goal: **Highest yield with the lowest realistic chance of assignment**, "
    "or assignment is acceptable when fundamentals strongly support an upward move."
)

# ------------------------------------------------------------------
# SIDEBAR (compact version)
# ------------------------------------------------------------------
st.sidebar.header("Screener Parameters")

max_pe = st.sidebar.number_input("Max P/E", value=40.0, step=1.0)
max_pct_support = st.sidebar.number_input("Max % Above 200-SMA", value=20.0, step=1.0)
target_delta = st.sidebar.number_input("Target Delta (OTM)", value=0.25, step=0.05, format="%.2f")

with st.sidebar.expander("Balance Sheet Filters", expanded=False):
    max_debt_equity = st.number_input("Max Debt/Equity (%)", value=150.0, step=10.0)
    require_positive_fcf = st.checkbox("Require Positive FCF", value=True)

with st.sidebar.expander("Risk Controls", expanded=False):
    exclude_earnings = st.checkbox("Exclude Earnings in Cycle", value=True)
    max_spread_pct = st.number_input("Max Bid-Ask Spread (%)", value=12.0, step=1.0)
    use_custom_exp = st.checkbox("Custom Expiration")
    target_expiration = None
    if use_custom_exp:
        exp_date_input = st.date_input("Expiration Date")
        target_expiration = exp_date_input.strftime("%Y-%m-%d")

st.sidebar.caption("Hard filters: Rev ≥ $3B | Cash ≥ $1B | Equity only")
run_button = st.sidebar.button("Run Scanner", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if run_button:
    with st.spinner("🔍 Running deep scan across market universe & option chains..."):
        universe = fetch_universe()
        st.info(f"Loaded {len(universe)} symbols. Screening for fundamentals & support...")

        passed_tickers = []
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

                price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                pe_ratio = info.get("trailingPE") or info.get("forwardPE") or 15.0

                revenue = info.get("totalRevenue") or 0
                if not revenue:
                    try:
                        fin = stock.financials
                        if not fin.empty and "Total Revenue" in fin.index:
                            revenue = fin.loc["Total Revenue"].iloc[0]
                    except Exception:
                        revenue = 0

                if not price or price <= 0:
                    continue

                total_cash = info.get("totalCash") or 0
                free_cash_flow = info.get("freeCashflow") or 0
                debt_to_equity = info.get("debtToEquity")

                fcf_pass = (free_cash_flow > 0) if require_positive_fcf else True
                de_pass = (debt_to_equity is None) or (debt_to_equity <= max_debt_equity)

                hist = stock.history(period="1y")
                if hist.empty or len(hist) < 50:
                    continue

                window_size = min(200, len(hist))
                sma_200 = hist["Close"].rolling(window=window_size).mean().iloc[-1]
                pct_above_support = ((price - sma_200) / sma_200) * 100

                if (
                    pe_ratio <= max_pe
                    and pct_above_support <= max_pct_support
                    and revenue >= 3.0e9
                    and total_cash >= 1.0e9
                    and fcf_pass
                    and de_pass
                ):
                    passed_tickers.append(ticker)

            except Exception:
                continue

        status_text.text(
            f"Screening complete! Found {len(passed_tickers)} matching equities near support."
        )
        progress_bar.empty()

        if not passed_tickers:
            st.warning("No stocks matched your custom thresholds near support.")
        else:
            st.success(
                f"Scanning option chains & applying risk filters for {len(passed_tickers)} matches..."
            )

            results = []
            for ticker in passed_tickers:
                try:
                    tk = yf.Ticker(ticker)
                    hist = tk.history(period="5d")
                    if hist.empty:
                        continue
                    spot_price = hist["Close"].iloc[-1]

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
                                "5-12 Day Weekly",
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
                                "25-45 Day Monthly",
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
                df["AI Score"] = df.apply(ai_score_trade, axis=1)
                df = df.sort_values(by="AI Score", ascending=False).reset_index(drop=True)
                df["Chart"] = df["Ticker"].apply(
                    lambda t: f"https://finance.yahoo.com/chart/{t}"
                )

                st.subheader("📊 Tactical Trade Matrix (Near Support)")
                st.dataframe(
                    df,
                    column_config={
                        "Chart": st.column_config.LinkColumn(
                            "Yahoo Chart",
                            display_text="📈 View Chart",
                        ),
                        "AI Score": st.column_config.NumberColumn(format="%.1f"),
                        "Prob Assign %": st.column_config.NumberColumn(format="%.1f"),
                        "Ann. Yield (%)": st.column_config.NumberColumn(format="%.1f"),
                    },
                    use_container_width=True,
                    height=500,
                )

                st.download_button(
                    label="📥 Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"csp_screener_{datetime.today().strftime('%Y-%m-%d')}.csv",
                    mime="text/csv",
                )

                top = df.iloc[0]
                st.markdown("### 🤖 Top Recommendation")
                st.info(
                    f"**[{top['Ticker']}](https://finance.yahoo.com/chart/{top['Ticker']})** — {top['Cycle']}\n\n"
                    f"- **Stock Price:** ${top['Stock Price']}\n"
                    f"- **Strike:** ${top['Strike']}  |  **OTM:** {top.get('OTM %', 'N/A')}%\n"
                    f"- **Premium (mid):** ${top['Put Premium']}\n"
                    f"- **Yield:** {top['Yield (%)']}%  →  **Annualized:** {top.get('Ann. Yield (%)', 'N/A')}%\n"
                    f"- **Approx. Assignment Probability:** {top.get('Prob Assign %', 'N/A')}%\n"
                    f"- **DTE:** {top['DTE']} days  |  **IV:** {top['IV (%)']}%\n"
                    f"- **AI Score:** {top['AI Score']}"
                )
            else:
                st.warning(
                    "No options found matching your parameters, spread criteria, or earnings filters."
                )
else:
    st.info("Adjust your parameters in the sidebar and click **Run Scanner** to begin.")
