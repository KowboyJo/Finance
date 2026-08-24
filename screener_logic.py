from __future__ import annotations

from datetime import datetime
from math import erf, exp, log, pi, sqrt
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from scipy.stats import norm

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ================================================================
# CONSTANTS
# ================================================================

TRADING_DAYS = 252
CALENDAR_DAYS = 365


# ================================================================
# BASIC MATH
# ================================================================

def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution."""
    if HAS_SCIPY:
        return float(norm.cdf(x))

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard normal probability density."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


# ================================================================
# S&P 500 UNIVERSE
# ================================================================

def fetch_universe() -> list[str]:
    """
    Retrieve the current S&P 500 constituents.

    Uses GitHub first, Wikipedia second, then a static fallback.
    """

    # ------------------------------------------------------------
    # Method 1: GitHub dataset
    # ------------------------------------------------------------
    try:
        url = (
            "https://raw.githubusercontent.com/datasets/"
            "s-and-p-500-companies/master/data/constituents.csv"
        )

        df = pd.read_csv(url)

        if "Symbol" in df.columns:
            tickers = (
                df["Symbol"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.replace(".", "-", regex=False)
                .tolist()
            )

            if len(tickers) > 400:
                return sorted(set(tickers))

    except Exception:
        pass

    # ------------------------------------------------------------
    # Method 2: Wikipedia
    # ------------------------------------------------------------
    try:
        import requests

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        }

        response = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        tables = pd.read_html(response.text)

        if tables:
            df = tables[0]

            if "Symbol" in df.columns:
                tickers = (
                    df["Symbol"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.replace(".", "-", regex=False)
                    .tolist()
                )

                if len(tickers) > 400:
                    return sorted(set(tickers))

    except Exception:
        pass

    # ------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG",
        "BRK-B", "LLY", "AVGO", "JPM", "XOM", "UNH", "V", "MA",
        "PG", "JNJ", "COST", "HD", "ABBV", "MRK", "CVX", "PEP",
        "KO", "WMT", "BAC", "CRM", "TMO", "ACN", "LIN", "MCD",
        "CSCO", "ABT", "DHR", "WFC", "TXN", "PM", "NEE", "AMD",
        "ORCL", "IBM", "QCOM", "CAT", "GE", "AMAT", "INTU", "SPGI",
        "ISRG", "NOW", "BKNG", "ADI", "AMGN", "PFE", "DIS", "NKE",
        "LOW", "UPS", "BA", "RTX", "HON", "GS", "MS", "BLK", "SCHW",
        "AXP", "C", "USB", "PNC", "TFC", "COF", "T", "VZ", "CMCSA",
        "TMUS", "INTC", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "TSLA",
        "NFLX", "ADBE", "PYPL", "SBUX", "MDT", "SYK", "BSX", "EW",
        "ZTS", "REGN", "VRTX", "GILD", "BIIB", "MRNA", "CI", "ELV",
        "CVS", "HUM", "MO", "BTI", "UL", "CL", "KMB", "GIS", "KHC",
        "DE", "CMI", "PCAR", "FDX", "NSC", "UNP", "CSX", "WM", "RSG",
        "ECL", "SHW", "PPG", "APD", "GD", "LMT", "NOC", "MMM", "ITW",
        "EMR", "ROK", "PH", "DOV", "IR", "ETN", "CARR", "OTIS", "JCI",
        "TT", "AME", "FTV",
    ]


# ================================================================
# EXPIRATIONS
# ================================================================

def find_target_expirations(
    exp_dates: list[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Find a relatively short weekly expiration and a medium-term
    expiration.

    Weekly:
        3-21 DTE

    Monthly:
        22-60 DTE
    """

    today = datetime.now().date()

    weekly = None
    monthly = None

    valid_dates = []

    for exp_str in exp_dates:
        try:
            exp = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp - today).days

            if dte > 0:
                valid_dates.append((exp, exp_str, dte))

        except (ValueError, TypeError):
            continue

    valid_dates.sort()

    for exp, exp_str, dte in valid_dates:

        if weekly is None and 3 <= dte <= 21:
            weekly = exp_str

        if monthly is None and 22 <= dte <= 60:
            monthly = exp_str

        if weekly and monthly:
            break

    return weekly, monthly


# ================================================================
# BLACK-SCHOLES
# ================================================================

def black_scholes_put_delta(
    spot: float,
    strike: float,
    dte: int,
    iv: float,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Estimate European put delta.

    Returns a negative delta between approximately -1 and 0.
    """

    if (
        spot <= 0
        or strike <= 0
        or dte <= 0
        or iv <= 0
    ):
        return -0.50

    sigma = iv / 100.0
    T = dte / CALENDAR_DAYS

    if sigma <= 0 or T <= 0:
        return -0.50

    try:
        d1 = (
            log(spot / strike)
            + (risk_free_rate + 0.5 * sigma**2) * T
        ) / (sigma * sqrt(T))

        return -normal_cdf(-d1)

    except Exception:
        return -0.50


def probability_stock_below_strike(
    spot: float,
    strike: float,
    dte: int,
    iv: float,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Estimate the risk-neutral probability that the stock finishes
    below the strike at expiration.

    This is a model estimate, NOT a guarantee of assignment.

    Result is 0-1.
    """

    if (
        spot <= 0
        or strike <= 0
        or dte <= 0
        or iv <= 0
    ):
        return 0.50

    sigma = iv / 100.0
    T = dte / CALENDAR_DAYS

    try:
        d2 = (
            log(spot / strike)
            + (risk_free_rate - 0.5 * sigma**2) * T
        ) / (sigma * sqrt(T))

        return float(normal_cdf(-d2))

    except Exception:
        return 0.50


# Backwards-compatible name
def approx_prob_itm(
    spot: float,
    strike: float,
    dte: int,
    iv: float,
) -> float:
    return probability_stock_below_strike(
        spot,
        strike,
        dte,
        iv,
    )


# ================================================================
# EARNINGS
# ================================================================

def get_earnings_date(ticker_obj) -> Optional[datetime.date]:
    """
    Safely attempt to extract the next earnings date from Yahoo.
    """

    try:
        calendar = ticker_obj.calendar

        if calendar is None:
            return None

        earnings_value = None

        if isinstance(calendar, pd.DataFrame):

            if "Earnings Date" in calendar.index:
                value = calendar.loc["Earnings Date"]

                if hasattr(value, "iloc") and len(value) > 0:
                    earnings_value = value.iloc[0]
                else:
                    earnings_value = value

        elif isinstance(calendar, dict):

            if "Earnings Date" in calendar:
                value = calendar["Earnings Date"]

                if isinstance(value, (list, tuple)):
                    if value:
                        earnings_value = value[0]
                else:
                    earnings_value = value

        if earnings_value is None:
            return None

        if isinstance(earnings_value, pd.Timestamp):
            return earnings_value.date()

        if hasattr(earnings_value, "date"):
            return earnings_value.date()

        text = str(earnings_value)

        return datetime.strptime(
            text[:10],
            "%Y-%m-%d",
        ).date()

    except Exception:
        return None


def earnings_before_expiration(
    ticker_obj,
    expiration: str,
) -> bool:
    """
    Returns True if known earnings occur on or before expiration.
    """

    earnings_date = get_earnings_date(ticker_obj)

    if earnings_date is None:
        return False

    try:
        expiration_date = datetime.strptime(
            expiration,
            "%Y-%m-%d",
        ).date()

        return earnings_date <= expiration_date

    except Exception:
        return False


# ================================================================
# OPTION LIQUIDITY
# ================================================================

def calculate_spread_pct(
    bid: float,
    ask: float,
) -> float:
    if bid <= 0 or ask <= 0 or ask < bid:
        return 999.0

    midpoint = (bid + ask) / 2.0

    if midpoint <= 0:
        return 999.0

    return ((ask - bid) / midpoint) * 100.0


def liquidity_score(
    volume: float,
    open_interest: float,
    spread_pct: float,
) -> float:
    """
    Produce a 0-100 liquidity score.
    """

    volume = max(float(volume or 0), 0)
    open_interest = max(float(open_interest or 0), 0)
    spread_pct = max(float(spread_pct or 100), 0)

    volume_score = min(volume / 1000.0, 1.0) * 35.0
    oi_score = min(open_interest / 5000.0, 1.0) * 35.0

    spread_score = max(
        0.0,
        min(30.0, (12.0 - spread_pct) / 12.0 * 30.0),
    )

    return round(
        min(100.0, volume_score + oi_score + spread_score),
        1,
    )


# ================================================================
# OPTION ANALYSIS
# ================================================================

def analyze_puts(
    ticker: str,
    expiration: str,
    spot_price: float,
    cycle_label: str,
    target_delta: float = 0.25,
    max_spread_pct: float = 12.0,
    exclude_earnings: bool = True,
    min_open_interest: int = 100,
    min_volume: int = 5,
    risk_free_rate: float = 0.04,
) -> Optional[dict]:
    """
    Analyze puts for a particular expiration.

    Selection philosophy:

    1. Only ATM/OTM puts.
    2. Remove illiquid contracts.
    3. Prefer contracts near target delta.
    4. Use liquidity as a secondary selection factor.
    """

    try:
        if spot_price <= 0:
            return None

        tk = yf.Ticker(ticker)

        # --------------------------------------------------------
        # Earnings filter
        # --------------------------------------------------------
        if exclude_earnings:
            if earnings_before_expiration(tk, expiration):
                return None

        # --------------------------------------------------------
        # Option chain
        # --------------------------------------------------------
        chain = tk.option_chain(expiration)

        if chain is None:
            return None

        puts = chain.puts.copy()

        if puts.empty:
            return None

        # --------------------------------------------------------
        # Required fields
        # --------------------------------------------------------
        required = ["bid", "ask", "strike"]

        for column in required:
            if column not in puts.columns:
                return None

        puts = puts.dropna(
            subset=["bid", "ask", "strike"]
        )

        puts = puts[
            (puts["bid"] > 0)
            & (puts["ask"] > puts["bid"])
            & (puts["strike"] > 0)
        ]

        # --------------------------------------------------------
        # Only ATM or OTM
        # --------------------------------------------------------
        puts = puts[
            puts["strike"] <= float(spot_price)
        ]

        if puts.empty:
            return None

        # --------------------------------------------------------
        # Spread
        # --------------------------------------------------------
        puts["spread_pct"] = (
            (puts["ask"] - puts["bid"])
            / ((puts["ask"] + puts["bid"]) / 2)
            * 100
        )

        puts = puts[
            puts["spread_pct"] <= max_spread_pct
        ]

        if puts.empty:
            return None

        # --------------------------------------------------------
        # Liquidity
        # --------------------------------------------------------
        if "openInterest" not in puts.columns:
            puts["openInterest"] = 0

        if "volume" not in puts.columns:
            puts["volume"] = 0

        puts["openInterest"] = pd.to_numeric(
            puts["openInterest"],
            errors="coerce",
        ).fillna(0)

        puts["volume"] = pd.to_numeric(
            puts["volume"],
            errors="coerce",
        ).fillna(0)

        # Keep contracts with meaningful liquidity.
        liquidity_mask = (
            (puts["openInterest"] >= min_open_interest)
            | (puts["volume"] >= min_volume)
        )

        puts = puts[liquidity_mask]

        if puts.empty:
            return None

        # --------------------------------------------------------
        # Basic metrics
        # --------------------------------------------------------
        puts["moneyness"] = (
            puts["strike"] / float(spot_price)
        )

        puts["otm_pct"] = (
            (float(spot_price) - puts["strike"])
            / float(spot_price)
            * 100
        )

        # --------------------------------------------------------
        # DTE
        # --------------------------------------------------------
        expiration_date = datetime.strptime(
            expiration,
            "%Y-%m-%d",
        ).date()

        dte = (
            expiration_date
            - datetime.now().date()
        ).days

        if dte <= 0:
            return None

        # --------------------------------------------------------
        # Delta
        # --------------------------------------------------------
        if "delta" in puts.columns:
            yahoo_delta = pd.to_numeric(
                puts["delta"],
                errors="coerce",
            )
        else:
            yahoo_delta = pd.Series(
                np.nan,
                index=puts.index,
            )

        puts["delta_source"] = np.where(
            yahoo_delta.notna(),
            "Yahoo",
            "Black-Scholes",
        )

        puts["approx_delta"] = yahoo_delta

        missing_delta = puts["approx_delta"].isna()

        if missing_delta.any():

            puts.loc[
                missing_delta,
                "approx_delta"
            ] = puts.loc[
                missing_delta
            ].apply(
                lambda row: black_scholes_put_delta(
                    float(spot_price),
                    float(row["strike"]),
                    dte,
                    float(row.get("impliedVolatility", 0) or 0) * 100,
                    risk_free_rate,
                ),
                axis=1,
            )

        # Still missing? Remove it.
        puts = puts[
            puts["approx_delta"].notna()
        ]

        if puts.empty:
            return None

        # --------------------------------------------------------
        # Delta distance
        # --------------------------------------------------------
        target_abs_delta = abs(float(target_delta))

        puts["delta_diff"] = (
            puts["approx_delta"].abs()
            - target_abs_delta
        ).abs()

        # --------------------------------------------------------
        # Liquidity score
        # --------------------------------------------------------
        puts["liquidity_score"] = puts.apply(
            lambda row: liquidity_score(
                row["volume"],
                row["openInterest"],
                row["spread_pct"],
            ),
            axis=1,
        )

        # --------------------------------------------------------
        # Contract selection
        #
        # Delta is primary.
        # Liquidity breaks ties.
        # --------------------------------------------------------
        puts = puts.sort_values(
            by=[
                "delta_diff",
                "spread_pct",
                "openInterest",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

        best = puts.iloc[0]

        # --------------------------------------------------------
        # Premium
        # --------------------------------------------------------
        bid = float(best["bid"])
        ask = float(best["ask"])

        mid = (bid + ask) / 2.0

        strike = float(best["strike"])

        collateral = strike * 100.0
        premium = mid * 100.0

        yield_pct = (
            premium / collateral * 100.0
        )

        annualized_yield = (
            yield_pct
            * (CALENDAR_DAYS / max(dte, 1))
        )

        # --------------------------------------------------------
        # IV
        # --------------------------------------------------------
        iv = 0.0

        if (
            "impliedVolatility" in best.index
            and pd.notna(best["impliedVolatility"])
        ):
            iv = float(best["impliedVolatility"]) * 100.0

        # --------------------------------------------------------
        # Delta
        # --------------------------------------------------------
        delta = float(best["approx_delta"])

        # --------------------------------------------------------
        # Probability of finishing ITM
        # --------------------------------------------------------
        prob_assign = (
            probability_stock_below_strike(
                float(spot_price),
                strike,
                dte,
                iv,
                risk_free_rate,
            )
            * 100.0
        )

        # --------------------------------------------------------
        # Breakeven
        # --------------------------------------------------------
        breakeven = strike - mid

        breakeven_otm = (
            (float(spot_price) - breakeven)
            / float(spot_price)
            * 100.0
        )

        # --------------------------------------------------------
        # Return on collateral
        # --------------------------------------------------------
        roc = yield_pct

        # --------------------------------------------------------
        # Liquidity
        # --------------------------------------------------------
        volume = float(best.get("volume", 0) or 0)
        open_interest = float(
            best.get("openInterest", 0) or 0
        )

        spread_pct = float(
            best["spread_pct"]
        )

        liq_score = float(
            best["liquidity_score"]
        )

        return {
            "Ticker": ticker,
            "Cycle": cycle_label,
            "Expiration": expiration,
            "DTE": dte,

            "Stock Price": round(
                float(spot_price),
                2,
            ),

            "Strike": round(
                strike,
                2,
            ),

            "OTM %": round(
                float(best["otm_pct"]),
                2,
            ),

            "Breakeven": round(
                breakeven,
                2,
            ),

            "Breakeven OTM %": round(
                breakeven_otm,
                2,
            ),

            "Put Premium": round(
                mid,
                2,
            ),

            "Yield (%)": round(
                yield_pct,
                2,
            ),

            "Ann. Yield (%)": round(
                annualized_yield,
                1,
            ),

            "IV (%)": round(
                iv,
                1,
            ),

            "Delta": round(
                delta,
                3,
            ),

            "Delta Source": str(
                best.get(
                    "delta_source",
                    "Model",
                )
            ),

            "Prob Assign %": round(
                prob_assign,
                1,
            ),

            "Spread (%)": round(
                spread_pct,
                1,
            ),

            "Bid": round(
                bid,
                2,
            ),

            "Ask": round(
                ask,
                2,
            ),

            "Volume": int(
                volume
            ),

            "Open Interest": int(
                open_interest
            ),

            "Liquidity Score": round(
                liq_score,
                1,
            ),

            "Collateral": round(
                collateral,
                2,
            ),
        }

    except Exception:
        return None


# ================================================================
# TRADE SCORE
# ================================================================

def ai_score_trade(row: pd.Series) -> float:
    """
    Risk-adjusted CSP score.

    Higher is better.

    Components:

    - Annualized yield
    - Distance OTM
    - Probability of assignment
    - Liquidity
    - Bid/ask spread
    - DTE preference
    - IV

    This is a heuristic ranking model, NOT an AI prediction model.
    """

    try:
        ann_yield = float(
            row.get("Ann. Yield (%)", 0) or 0
        )

        prob_assign = float(
            row.get("Prob Assign %", 50) or 50
        )

        otm = float(
            row.get("OTM %", 0) or 0
        )

        spread = float(
            row.get("Spread (%)", 20) or 20
        )

        dte = max(
            float(row.get("DTE", 30) or 30),
            1,
        )

        liquidity = float(
            row.get("Liquidity Score", 0) or 0
        )

        iv = float(
            row.get("IV (%)", 0) or 0
        )

        # --------------------------------------------------------
        # Yield
        # --------------------------------------------------------
        yield_score = min(
            ann_yield * 1.35,
            40.0,
        )

        # --------------------------------------------------------
        # Safety / OTM
        # --------------------------------------------------------
        safety_score = min(
            max(otm, 0) * 1.4,
            30.0,
        )

        # --------------------------------------------------------
        # Assignment penalty
        # --------------------------------------------------------
        assignment_penalty = (
            prob_assign * 0.70
        )

        # --------------------------------------------------------
        # Liquidity
        # --------------------------------------------------------
        liquidity_score_value = (
            liquidity * 0.20
        )

        # --------------------------------------------------------
        # Spread penalty
        # --------------------------------------------------------
        spread_penalty = max(
            0.0,
            spread - 3.0,
        ) * 1.5

        # --------------------------------------------------------
        # DTE preference
        #
        # Rough sweet spot:
        # 20-45 days.
        # --------------------------------------------------------
        if 20 <= dte <= 45:
            dte_multiplier = 1.10

        elif 14 <= dte < 20:
            dte_multiplier = 1.03

        elif 45 < dte <= 60:
            dte_multiplier = 1.00

        elif dte < 7:
            dte_multiplier = 0.85

        else:
            dte_multiplier = 0.95

        # --------------------------------------------------------
        # IV bonus
        #
        # Moderate IV is attractive because it increases premium,
        # but extremely high IV often represents elevated risk.
        # --------------------------------------------------------
        if 20 <= iv <= 45:
            iv_bonus = 4.0

        elif 45 < iv <= 60:
            iv_bonus = 2.0

        elif iv > 80:
            iv_bonus = -4.0

        else:
            iv_bonus = 0.0

        raw_score = (
            yield_score
            + safety_score
            + liquidity_score_value
            + iv_bonus
            - assignment_penalty
            - spread_penalty
        )

        score = raw_score * dte_multiplier

        return round(
            max(score, 0.0),
            2,
        )

    except Exception:
        return 0.0
