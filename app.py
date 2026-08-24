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

# Hardcoded filters (kept short)
st.sidebar.caption("Hard filters: Rev ≥ $3B | Cash ≥ $1B | Equity only")

run_button = st.sidebar.button("Run Scanner", type="primary", use_container_width=True)
