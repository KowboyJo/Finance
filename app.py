# Screening Loop
  for i, ticker in enumerate(universe):
    status_text.text(
        f"Screening ({i+1}/{len(universe)}): Checking {ticker}..."
    )
    progress_bar.progress((i + 1) / len(universe))
    try:
      stock = yf.Ticker(ticker)
      info = stock.info

      if info.get("quoteType") != "EQUITY":
        continue

      price = info.get("currentPrice", info.get("regularMarketPrice", 0))
      pe_ratio = info.get("trailingPE", info.get("forwardPE", 0))
      revenue = info.get("totalRevenue", 0)
      total_cash = info.get("totalCash", 0)

      if not price or price <= 0:
        continue

      # Fallback for missing fundamental tags in yfinance
      if not pe_ratio:
        pe_ratio = 15.0  # Default neutral fallback if null

      # Fetch historical data to compute 200-day SMA
      hist = stock.history(period="1yr")
      if hist.empty or len(hist) < 50:  # Relaxed slightly to avoid drops
        continue

      # Calculate 200 SMA (or use available length if slightly under 200)
      window_size = min(200, len(hist))
      sma_200 = hist["Close"].rolling(window=window_size).mean().iloc[-1]

      pct_above_support = ((price - sma_200) / sma_200) * 100

      # Loosened filter boundaries to ensure we catch options
      if (
          pe_ratio <= max_pe
          and pct_above_support
          <= max_pct_support  # Only cap the maximum extension above support
          and revenue >= min_revenue
      ):
        passed_tickers.append(ticker)
    except Exception:
      continue
