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


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="CSP Income Screener",
    page_icon="📈",
    layout="wide",
)


# ================================================================
# HEADER
# ================================================================

st.title(
    "📈 Large-Cap Cash-Secured Put Screener"
)

st.markdown(
    """
    Screen large-cap equities for cash-secured put opportunities
    using fundamentals, technical support, option liquidity,
    probability estimates, and risk-adjusted scoring.
    """
)


# ================================================================
# SIDEBAR
# ================================================================

st.sidebar.header("Screener Parameters")


# Fundamental filters
st.sidebar.subheader("Fundamentals")

max_pe = st.sidebar.number_input(
    "Maximum P/E",
    min_value=1.0,
    max_value=200.0,
    value=40.0,
    step=1.0,
)

max_pct_support = st.sidebar.number_input(
    "Maximum % Above 200-SMA",
    min_value=-50.0,
    max_value=100.0,
    value=20.0,
    step=1.0,
)

min_revenue = st.sidebar.number_input(
    "Minimum Revenue ($B)",
    min_value=0.0,
    value=3.0,
    step=0.5,
)

min_cash = st.sidebar.number_input(
    "Minimum Cash ($B)",
    min_value=0.0,
    value=1.0,
    step=0.5,
)

max_debt_equity = st.sidebar.number_input(
    "Maximum Debt/Equity (%)",
    min_value=0.0,
    value=150.0,
    step=10.0,
)

require_positive_fcf = st.sidebar.checkbox(
    "Require Positive FCF",
    value=True,
)


# Option filters
st.sidebar.subheader("Option Selection")

target_delta = st.sidebar.number_input(
    "Target Put Delta",
    min_value=0.05,
    max_value=0.80,
    value=0.25,
    step=0.05,
    format="%.2f",
)

max_spread_pct = st.sidebar.number_input(
    "Maximum Bid/Ask Spread (%)",
    min_value=1.0,
    max_value=50.0,
    value=12.0,
    step=1.0,
)

min_open_interest = st.sidebar.number_input(
    "Minimum Open Interest",
    min_value=0,
    value=100,
    step=50,
)

min_volume = st.sidebar.number_input(
    "Minimum Daily Option Volume",
    min_value=0,
    value=5,
    step=5,
)


# Risk
st.sidebar.subheader("Risk Controls")

exclude_earnings = st.sidebar.checkbox(
    "Exclude Earnings in Cycle",
    value=True,
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=4.0,
    step=0.25,
) / 100.0


# Expiration
st.sidebar.subheader("Expiration")

use_custom_exp = st.sidebar.checkbox(
    "Use Custom Expiration",
    value=False,
)

target_expiration = None

if use_custom_exp:

    exp_date_input = st.sidebar.date_input(
        "Expiration Date"
    )

    target_expiration = (
        exp_date_input.strftime("%Y-%m-%d")
    )


st.sidebar.caption(
    "Equity filters: revenue, cash, FCF, debt/equity, "
    "valuation, and proximity to the 200-SMA."
)


run_button = st.sidebar.button(
    "🚀 Run Scanner",
    type="primary",
    use_container_width=True,
)


# ================================================================
# FUNDAMENTAL SCREEN
# ================================================================

if run_button:

    with st.spinner(
        "🔍 Screening the S&P 500..."
    ):

        universe = fetch_universe()

        st.info(
            f"Loaded {len(universe)} symbols."
        )

        passed_tickers = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        for i, ticker in enumerate(universe):

            status_text.text(
                f"Fundamental screening "
                f"({i + 1}/{len(universe)}): {ticker}"
            )

            progress_bar.progress(
                (i + 1) / len(universe)
            )

            try:

                stock = yf.Ticker(ticker)

                info = stock.info

                # ------------------------------------------------
                # Equity check
                # ------------------------------------------------

                if info.get("quoteType") != "EQUITY":
                    continue

                # ------------------------------------------------
                # Price
                # ------------------------------------------------

                price = (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or 0
                )

                if not price or price <= 0:
                    continue

                # ------------------------------------------------
                # Valuation
                # ------------------------------------------------

                pe_ratio = (
                    info.get("trailingPE")
                    or info.get("forwardPE")
                )

                if pe_ratio is None:
                    continue

                # ------------------------------------------------
                # Revenue
                # ------------------------------------------------

                revenue = (
                    info.get("totalRevenue")
                    or 0
                )

                if revenue <= 0:

                    try:

                        financials = stock.financials

                        if (
                            not financials.empty
                            and "Total Revenue"
                            in financials.index
                        ):
                            revenue = float(
                                financials.loc[
                                    "Total Revenue"
                                ].iloc[0]
                            )

                    except Exception:
                        revenue = 0

                # ------------------------------------------------
                # Balance sheet
                # ------------------------------------------------

                total_cash = (
                    info.get("totalCash")
                    or 0
                )

                free_cash_flow = (
                    info.get("freeCashflow")
                    or 0
                )

                debt_to_equity = (
                    info.get("debtToEquity")
                )

                # ------------------------------------------------
                # FCF
                # ------------------------------------------------

                if require_positive_fcf:
                    if free_cash_flow <= 0:
                        continue

                # ------------------------------------------------
                # Debt
                # ------------------------------------------------

                if debt_to_equity is not None:

                    if debt_to_equity > max_debt_equity:
                        continue

                # ------------------------------------------------
                # Historical price
                # ------------------------------------------------

                hist = stock.history(
                    period="1y"
                )

                if hist.empty or len(hist) < 50:
                    continue

                close = hist["Close"].dropna()

                if len(close) < 50:
                    continue

                # ------------------------------------------------
                # 200-day SMA
                #
                # If less than 200 observations exist, use the
                # available history rather than rejecting the stock.
                # ------------------------------------------------

                window_size = min(
                    200,
                    len(close),
                )

                sma_200 = (
                    close
                    .rolling(window=window_size)
                    .mean()
                    .iloc[-1]
                )

                if pd.isna(sma_200) or sma_200 <= 0:
                    continue

                pct_above_support = (
                    (float(price) - sma_200)
                    / sma_200
                    * 100
                )

                # ------------------------------------------------
                # Final fundamental screen
                # ------------------------------------------------

                if (
                    float(pe_ratio) <= max_pe
                    and pct_above_support <= max_pct_support
                    and revenue >= min_revenue * 1e9
                    and total_cash >= min_cash * 1e9
                ):

                    passed_tickers.append(ticker)

            except Exception:
                continue

        status_text.text(
            f"Fundamental screening complete: "
            f"{len(passed_tickers)} stocks passed."
        )

        progress_bar.empty()


    # ============================================================
    # NO MATCHES
    # ============================================================

    if not passed_tickers:

        st.warning(
            "No stocks matched the current fundamental "
            "and technical filters."
        )

        st.stop()


    # ============================================================
    # OPTION SCREEN
    # ============================================================

    st.success(
        f"Found {len(passed_tickers)} stocks. "
        "Now scanning option chains..."
    )

    results = []

    option_progress = st.progress(0)

    option_status = st.empty()


    for i, ticker in enumerate(passed_tickers):

        option_status.text(
            f"Options ({i + 1}/{len(passed_tickers)}): {ticker}"
        )

        option_progress.progress(
            (i + 1) / len(passed_tickers)
        )

        try:

            tk = yf.Ticker(ticker)

            hist = tk.history(
                period="5d"
            )

            if hist.empty:
                continue

            spot_price = float(
                hist["Close"].dropna().iloc[-1]
            )

            if spot_price <= 0:
                continue

            exp_dates = tk.options

            if not exp_dates:
                continue


            # ----------------------------------------------------
            # Custom expiration
            # ----------------------------------------------------

            if target_expiration:

                if target_expiration not in exp_dates:
                    continue

                result = analyze_puts(
                    ticker=ticker,
                    expiration=target_expiration,
                    spot_price=spot_price,
                    cycle_label=(
                        f"Custom "
                        f"({target_expiration})"
                    ),
                    target_delta=target_delta,
                    max_spread_pct=max_spread_pct,
                    exclude_earnings=exclude_earnings,
                    min_open_interest=min_open_interest,
                    min_volume=min_volume,
                    risk_free_rate=risk_free_rate,
                )

                if result:
                    results.append(result)


            # ----------------------------------------------------
            # Automatic weekly/monthly
            # ----------------------------------------------------

            else:

                weekly_exp, monthly_exp = (
                    find_target_expirations(
                        exp_dates
                    )
                )

                # Weekly
                if weekly_exp:

                    result = analyze_puts(
                        ticker=ticker,
                        expiration=weekly_exp,
                        spot_price=spot_price,
                        cycle_label="Weekly",
                        target_delta=target_delta,
                        max_spread_pct=max_spread_pct,
                        exclude_earnings=exclude_earnings,
                        min_open_interest=min_open_interest,
                        min_volume=min_volume,
                        risk_free_rate=risk_free_rate,
                    )

                    if result:
                        results.append(result)


                # Monthly
                if monthly_exp:

                    result = analyze_puts(
                        ticker=ticker,
                        expiration=monthly_exp,
                        spot_price=spot_price,
                        cycle_label="Monthly",
                        target_delta=target_delta,
                        max_spread_pct=max_spread_pct,
                        exclude_earnings=exclude_earnings,
                        min_open_interest=min_open_interest,
                        min_volume=min_volume,
                        risk_free_rate=risk_free_rate,
                    )

                    if result:
                        results.append(result)

        except Exception:
            continue


    option_status.text(
        f"Option scan complete: "
        f"{len(results)} trade candidates."
    )

    option_progress.empty()


    # ============================================================
    # RESULTS
    # ============================================================

    if not results:

        st.warning(
            "No option contracts matched the current "
            "delta, liquidity, spread, or earnings filters."
        )

        st.stop()


    df = pd.DataFrame(results)


    # ============================================================
    # SCORE
    # ============================================================

    df["AI Score"] = df.apply(
        ai_score_trade,
        axis=1,
    )


    df = (
        df.sort_values(
            by="AI Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    # Rank
    df.insert(
        0,
        "Rank",
        range(1, len(df) + 1),
    )


    # Yahoo chart
    df["Chart"] = df["Ticker"].apply(
        lambda ticker:
            f"https://finance.yahoo.com/chart/{ticker}"
    )


    # ============================================================
    # TOP RECOMMENDATION
    # ============================================================

    top = df.iloc[0]


    st.subheader(
        "🏆 Top CSP Candidate"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Ticker",
            top["Ticker"],
        )

        st.caption(
            f"{top['Cycle']} • "
            f"{top['Expiration']}"
        )


    with col2:

        st.metric(
            "AI Score",
            f"{top['AI Score']:.1f}",
        )

        st.caption(
            "Risk-adjusted ranking"
        )


    with col3:

        st.metric(
            "Annualized Yield",
            f"{top['Ann. Yield (%)']:.1f}%",
        )

        st.caption(
            f"{top['Yield (%)']:.2f}% actual cycle yield"
        )


    with col4:

        st.metric(
            "Assignment Probability",
            f"{top['Prob Assign %']:.1f}%",
        )

        st.caption(
            "Model estimate"
        )


    st.info(
        f"""
        **{top['Ticker']} ${top['Strike']:.2f} put**

        Stock: **${top['Stock Price']:.2f}**

        Premium: **${top['Put Premium']:.2f}**

        Breakeven: **${top['Breakeven']:.2f}**

        Breakeven distance: **{top['Breakeven OTM %']:.1f}%**

        Delta: **{top['Delta']:.3f}**

        IV: **{top['IV (%)']:.1f}%**

        DTE: **{top['DTE']} days**

        Liquidity Score: **{top['Liquidity Score']:.1f}/100**
        """
    )


    # ============================================================
    # TRADE MATRIX
    # ============================================================

    st.subheader(
        "📊 CSP Trade Matrix"
    )


    display_columns = [
        "Rank",
        "Ticker",
        "Cycle",
        "Expiration",
        "DTE",
        "Stock Price",
        "Strike",
        "OTM %",
        "Breakeven",
        "Put Premium",
        "Yield (%)",
        "Ann. Yield (%)",
        "IV (%)",
        "Delta",
        "Prob Assign %",
        "Spread (%)",
        "Volume",
        "Open Interest",
        "Liquidity Score",
        "AI Score",
        "Chart",
    ]


    display_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]


    st.dataframe(
        df[display_columns],
        column_config={

            "Chart":
                st.column_config.LinkColumn(
                    "Chart",
                    display_text="📈 View",
                ),

            "AI Score":
                st.column_config.NumberColumn(
                    "Score",
                    format="%.1f",
                ),

            "Prob Assign %":
                st.column_config.NumberColumn(
                    "Assignment %",
                    format="%.1f%%",
                ),

            "Ann. Yield (%)":
                st.column_config.NumberColumn(
                    "Ann. Yield",
                    format="%.1f%%",
                ),

            "Yield (%)":
                st.column_config.NumberColumn(
                    "Cycle Yield",
                    format="%.2f%%",
                ),

            "OTM %":
                st.column_config.NumberColumn(
                    "OTM",
                    format="%.1f%%",
                ),

            "Spread (%)":
                st.column_config.NumberColumn(
                    "Spread",
                    format="%.1f%%",
                ),

            "IV (%)":
                st.column_config.NumberColumn(
                    "IV",
                    format="%.1f%%",
                ),

            "Delta":
                st.column_config.NumberColumn(
                    "Delta",
                    format="%.3f",
                ),

            "Liquidity Score":
                st.column_config.NumberColumn(
                    "Liquidity",
                    format="%.1f",
                ),
        },

        use_container_width=True,
        height=650,
        hide_index=True,
    )


    # ============================================================
    # CSV DOWNLOAD
    # ============================================================

    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(
            index=False
        ).encode("utf-8"),
        file_name=(
            f"csp_screener_"
            f"{datetime.today().strftime('%Y-%m-%d')}.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


    # ============================================================
    # SCORE EXPLANATION
    # ============================================================

    with st.expander(
        "ℹ️ How the AI Score works"
    ):

        st.markdown(
            """
            The **AI Score is a quantitative heuristic**, not a
            prediction produced by a machine-learning model.

            It rewards:

            - Higher option premium yield
            - Greater distance below the current stock price
            - Better option liquidity
            - Reasonable implied volatility
            - Preferred DTE ranges

            It penalizes:

            - Higher modeled assignment probability
            - Wide bid/ask spreads
            - Very short-dated contracts
            - Extremely high implied volatility

            A higher score means the contract has a more attractive
            combination of income, downside cushion, liquidity, and
            modeled risk according to these rules.
            """
        )


# ================================================================
# INITIAL STATE
# ================================================================

else:

    st.info(
        "Configure the filters in the sidebar and click "
        "**🚀 Run Scanner** to begin."
    )
