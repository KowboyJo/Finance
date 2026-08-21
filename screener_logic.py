from datetime import datetime
import pandas as pd
import yfinance as yf


def fetch_universe():
    """
    Fetches the S&P 500 constituents ticker symbols.
    Falls back to a core list of large-cap tech/income stocks if offline.
    """
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return [
            "UBER", "AAPL", "MSFT", "GOOGL", "AMZN",
            "NVDA", "META", "JPM", "WMT", "AMD"
        ]


def find_target_expirations(available_dates):
    """
    Finds target option expiration dates:
    - Best weekly (~5 DTE)
    - Best monthly (~30 DTE)
    """
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
    """
    Checks if an upcoming earnings date occurs on or before the expiration date.
    Returns True if there is an earnings event within the expiration window.
    """
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


def analyze_puts(
    ticker_symbol,
    exp_date,
    spot_price,
    expiry_label,
    target_delta,
    max_spread,
    check_earnings,
    min_oi=100,
):
    """
    Analyzes put option chains for a given ticker and expiration date.
    Applies bid-ask spread filters, minimum open interest, and calculates Yield (%).
    """
    # 1. Earnings Date Check
    if check_earnings and check_earnings_conflict(ticker_symbol, exp_date):
        return None

    tk = yf.Ticker(ticker_symbol)
    try:
        chain = tk.option_chain(exp_date)
        puts = chain.puts
        if puts.empty:
            return None

        # Filter OTM puts
        otm_puts = puts[puts["strike"] <= spot_price].copy()
        if otm_puts.empty:
            otm_puts = puts

        # Target Strike Selection using Delta heuristic
        otm_puts["target_dist"] = abs(
            (otm_puts["strike"] / spot_price) - (1.0 - (target_delta * 0.15))
        )
        chosen_put = otm_puts.loc[otm_puts["target_dist"].idxmin()]

        # 2. Minimum Open Interest Filter
        oi = chosen_put.get("openInterest", 0)
        if pd.isna(oi) or oi < min_oi:
            return None

        strike = chosen_put["strike"]
        bid = chosen_put["bid"] if pd.notna(chosen_put["bid"]) else 0.0
        ask = chosen_put["ask"] if pd.notna(chosen_put["ask"]) else 0.0
        mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else chosen_put["lastPrice"]

        # 3. Maximum Bid-Ask Spread Filter
        if bid > 0 and ask > 0 and mid_price > 0:
            spread_pct = ((ask - bid) / mid_price) * 100
            if spread_pct > max_spread:
                return None

        iv = chosen_put.get("impliedVolatility", 0.0) * 100
        dte = max(1, (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.today()).days)

        # 4. Premium Return Yield Calculation
        yield_pct = (mid_price / spot_price) * 100

        # 5. Technical Support Buffer Calculation (20-period 2-std lower band)
        hist = tk.history(period="3m")
        buffer_pct = 0.0
        if not hist.empty and len(hist) >= 20:
            ma20 = hist["Close"].rolling(20).mean().iloc[-1]
            std20 = hist["Close"].rolling(20).std().iloc[-1]
            lower_band = ma20 - (2 * std20)
            buffer_pct = round(((strike - lower_band) / spot_price) * 100, 2)

        return {
            "Ticker": ticker_symbol,
            "Cycle": expiry_label,
            "DTE": dte,
            "Stock Price": round(spot_price, 2),
            "Strike": strike,
            "Put Premium": round(mid_price, 2),
            "IV (%)": round(iv, 1),
            "Yield (%)": round(yield_pct, 2),
            "Strike vs Lower Band (%)": buffer_pct,
            "Open Interest": int(oi),
        }
    except Exception:
        return None


def ai_score_trade(row):
    """
    Calculates AI Trade Score based on yield velocity and IV efficiency.
    """
    yield_pct = row["Yield (%)"]
    iv = row["IV (%)"]
    dte = row["DTE"]

    daily_yield_velocity = yield_pct / dte
    iv_score = min(iv / 40.0, 1.5)
    ai_score = (daily_yield_velocity * 100) * (1 + iv_score)
    return round(ai_score, 2)
