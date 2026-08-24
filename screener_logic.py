# screener_logic.py
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def fetch_universe() -> list[str]:
    """
    Returns the current S&P 500 constituents.
    Tries multiple reliable sources and falls back to a large static list.
    """
    # Method 1: GitHub CSV
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            return sorted(tickers)
    except Exception:
        pass

    # Method 2: Wikipedia with headers
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=10
        )
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            return sorted(tickers)
    except Exception:
        pass

    # Fallback list
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "LLY", "AVGO",
        "JPM", "XOM", "UNH", "V", "MA", "PG", "JNJ", "COST", "HD", "ABBV",
        "MRK", "CVX", "PEP", "KO", "WMT", "BAC", "CRM", "TMO", "ACN", "LIN",
        "MCD", "CSCO", "ABT", "DHR", "WFC", "TXN", "PM", "NEE", "AMD", "ORCL",
        "IBM", "QCOM", "CAT", "GE", "AMAT", "INTU", "SPGI", "ISRG", "NOW", "BKNG",
        "ADI", "AMGN", "PFE", "DIS", "NKE", "LOW", "UPS", "BA", "RTX", "HON",
        "GS", "MS", "BLK", "SCHW", "AXP", "C", "USB", "PNC", "TFC", "COF",
        "T", "VZ", "CMCSA", "TMUS", "INTC", "MU", "LRCX", "KLAC", "SNPS", "CDNS",
        "TSLA", "NFLX", "ADBE", "PYPL", "SBUX", "MDT", "SYK", "BSX", "EW", "ZTS",
        "REGN", "VRTX", "GILD", "BIIB", "MRNA", "CI", "ELV", "CVS", "HUM", "MO",
        "BTI", "UL", "CL", "KMB", "GIS", "KHC", "DE", "CMI", "PCAR", "FDX",
        "NSC", "UNP", "CSX", "WM", "RSG", "ECL", "SHW", "PPG", "APD", "GD",
        "LMT", "NOC", "MMM", "ITW", "EMR", "ROK", "PH", "DOV", "IR", "ETN",
        "CARR", "OTIS", "JCI", "TT", "AME", "FTV",
    ]


def find_target_expirations(exp_dates: list[str]) -> tuple[str | None, str | None]:
    today = datetime.now().date()
    weekly = None
    monthly = None

    for exp_str in sorted(exp_dates):
        try:
            exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp - today).days
            if 4 <= dte <= 12 and weekly is None:
                weekly = exp_str
            if 25 <= dte <= 45 and monthly is None:
                monthly = exp_str
        except ValueError:
            continue

    return weekly, monthly


def approx_prob_itm(spot: float, strike: float, dte: int, iv: float) -> float:
    if dte <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return 0.5

    if HAS_SCIPY:
        try:
            T = dte / 365.0
            sigma = iv / 100.0
            d2 = (np.log(spot / strike) + (-0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return float(norm.cdf(-d2))
        except Exception:
            pass

    moneyness = strike / spot
    if moneyness < 0.90:
        return 0.08
    elif moneyness < 0.95:
        return 0.18
    elif moneyness < 1.00:
        return 0.35
    elif moneyness < 1.05:
        return 0.55
    else:
        return 0.75


def analyze_puts(
    ticker: str,
    expiration: str,
    spot_price: float,
    cycle_label: str,
    target_delta: float = 0.25,
    max_spread_pct: float = 12.0,
    exclude_earnings: bool = True,
) -> dict | None:
    try:
        tk = yf.Ticker(ticker)

        # Earnings filter
        if exclude_earnings:
            try:
                cal = tk.calendar
                if cal is not None:
                    earn_date = None
                    if isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
                        val = cal.loc["Earnings Date"]
                        earn_date = val.iloc[0] if hasattr(val, "iloc") else val
                    elif isinstance(cal, dict) and "Earnings Date" in cal:
                        ed = cal["Earnings Date"]
                        earn_date = ed[0] if isinstance(ed, (list, tuple)) else ed

                    if earn_date is not None:
                        if isinstance(earn_date, str):
                            earn_date = datetime.strptime(str(earn_date)[:10], "%Y-%m-%d").date()
                        elif hasattr(earn_date, "date"):
                            earn_date = earn_date.date()

                        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                        if earn_date <= exp_date:
                            return None
            except Exception:
                pass

        # Option chain
        chain = tk.option_chain(expiration)
        puts = chain.puts.copy()
        if puts.empty:
            return None

        puts = puts.dropna(subset=["bid", "ask", "strike"])
        puts = puts[(puts["bid"] > 0) & (puts["ask"] > puts["bid"])]

        # === CRITICAL RULE: Only OTM or ATM puts (strike ≤ current price) ===
        puts = puts[puts["strike"] <= spot_price]

        if puts.empty:
            return None

        # Spread filter
        puts["spread_pct"] = (puts["ask"] - puts["bid"]) / ((puts["ask"] + puts["bid"]) / 2) * 100
        puts = puts[puts["spread_pct"] <= max_spread_pct]
        if puts.empty:
            return None

        puts["moneyness"] = puts["strike"] / spot_price

        if "delta" in puts.columns and puts["delta"].notna().any():
            puts["approx_delta"] = puts["delta"]
        else:
            puts["approx_delta"] = -np.clip(1.15 * (1 - puts["moneyness"]), 0.05, 0.90)

        puts["delta_diff"] = (puts["approx_delta"].abs() - target_delta).abs()
        puts = puts.sort_values("delta_diff")

        best = puts.iloc[0]

        mid = (float(best["bid"]) + float(best["ask"])) / 2
        strike = float(best["strike"])
        dte = (datetime.strptime(expiration, "%Y-%m-%d").date() - datetime.now().date()).days
        if dte <= 0:
            return None

        collateral = strike * 100
        premium = mid * 100
        yield_pct = (premium / collateral) * 100
        annualized_yield = yield_pct * (365 / max(dte, 1))

        iv = 0.0
        if "impliedVolatility" in best and pd.notna(best["impliedVolatility"]):
            iv = float(best["impliedVolatility"]) * 100

        delta_approx = float(best["approx_delta"])
        otm_pct = ((spot_price - strike) / spot_price) * 100
        prob_assign = approx_prob_itm(spot_price, strike, dte, iv) * 100

        return {
            "Ticker": ticker,
            "Cycle": cycle_label,
            "Expiration": expiration,
            "DTE": dte,
            "Stock Price": round(spot_price, 2),
            "Strike": round(strike, 2),
            "OTM %": round(otm_pct, 1),
            "Put Premium": round(mid, 2),
            "Yield (%)": round(yield_pct, 2),
            "Ann. Yield (%)": round(annualized_yield, 1),
            "IV (%)": round(iv, 1),
            "Delta (approx)": round(delta_approx, 2),
            "Prob Assign %": round(prob_assign, 1),
            "Spread (%)": round(float(best["spread_pct"]), 1),
            "Bid": round(float(best["bid"]), 2),
            "Ask": round(float(best["ask"]), 2),
        }

    except Exception:
        return None


def ai_score_trade(row: pd.Series) -> float:
    try:
        ann_yield = float(row.get("Ann. Yield (%)", 0) or 0)
        prob_assign = float(row.get("Prob Assign %", 50) or 50) / 100.0
        otm = float(row.get("OTM %", 0) or 0)
        spread = float(row.get("Spread (%)", 20) or 20)
        dte = max(float(row.get("DTE", 30) or 30), 1)

        yield_component = ann_yield * 1.6
        risk_penalty = prob_assign * 70
        safety_bonus = max(otm, 0) * 0.9
        liquidity_bonus = max(0, 12 - spread) * 1.3

        dte_factor = 1.0
        if 20 <= dte <= 40:
            dte_factor = 1.15
        elif dte < 10:
            dte_factor = 0.9

        score = (yield_component - risk_penalty + safety_bonus + liquidity_bonus) * dte_factor
        return round(max(score, 0), 2)

    except Exception:
        return 0.0
