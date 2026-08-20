from datetime import datetime
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=86400)
def fetch_universe():
  try:
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
    df = pd.read_csv(url)
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()
  except Exception:
    return ["UBER", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "WMT"]


def find_target_expirations(available_dates):
  today = datetime.today()
  best_weekly = None
  best_monthly = None
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
  """Checks if earnings fall on or before the option expiration date."""
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


def analyze_puts(
    ticker_symbol,
    exp_date,
    spot_price,
    expiry_label,
    target_delta,
    max_spread,
    check_earnings,
):
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

    strike = chosen_put["strike"]
    bid = chosen_put["bid"] if pd.notna(chosen_put["bid"]) else 0.0
    ask = chosen_put["ask"] if pd.notna(chosen_put["ask"]) else 0.0
    mid_price = (
        (bid + ask) / 2 if (bid > 0 and ask > 0) else chosen_put["lastPrice"]
    )

    if bid > 0 and ask > 0 and mid_price > 0:
      spread_pct = ((ask - bid) / mid_price) * 100
      if spread_pct > max_spread:
        return None

    iv = chosen_put.get("impliedVolatility", 0.0) * 100
    dte = max(
        1, (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.today()).days
    )
    ratio = (mid_price / spot_price) * 100

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
        "Premium Yield (%)": round(ratio, 2),
        "Strike vs Lower Band (%)": buffer_pct,
    }
  except Exception:
    return None


def ai_score_trade(row):
  ratio = row["Ratio (%)"]
  iv = row["IV (%)"]
  dte = row["DTE"]
  daily_yield_velocity = ratio / dte
  iv_score = min(iv / 40.0, 1.5)
  ai_score = (daily_yield_velocity * 100) * (1 + iv_score)
  return round(ai_score, 2)
