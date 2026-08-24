def fetch_universe() -> list[str]:
    """
    Returns the current S&P 500 constituents.
    Tries multiple reliable sources and falls back to a large static list.
    """
    # --- Method 1: GitHub-maintained CSV (most reliable) ---
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        df = pd.read_csv(url)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            return sorted(tickers)
    except Exception:
        pass

    # --- Method 2: Wikipedia with better headers ---
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

    # --- Fallback: larger static list ---
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
