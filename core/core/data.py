"""Market-data layer: OKX / Bybit perpetuals (via ccxt) and Binance Spot (public API, no key)."""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import requests

STABLES = {"USDC", "USDT", "DAI", "TUSD", "FDUSD", "USDD", "PYUSD", "USDP", "EUR", "USDG",
           "USD1", "USDE", "USDS", "RLUSD", "XSOXL"}
SOURCES = {"OKX Perpetual (USDT)": "okx", "Bybit Perpetual (USDT)": "bybit", "Binance Spot (USDT)": "binancespot"}
COLS = ["ts", "open", "high", "low", "close", "vol"]


def tf_ms(tf: str) -> int:
    return int(tf[:-1]) * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[tf[-1]]


def norm(base: str) -> str:
    """1000PEPE -> PEPE so names match across exchanges."""
    return re.sub(r"^(1000000|1000|1M)", "", base.upper())


class BinanceSpot:
    """Tiny Binance Spot client (data-api.binance.vision). Same calls the app needs from ccxt."""
    BASE = "https://data-api.binance.vision/api/v3"

    def __init__(self):
        self.markets, self._tick = {}, {}

    def _get(self, path, params=None):
        r = requests.get(self.BASE + path, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_tickers(self):
        if self._tick:
            return self._tick
        data = self._get("/ticker/24hr")
        syms = {t["symbol"] for t in data}
        for t in data:
            s = t["symbol"]
            if not s.endswith("USDT"):
                continue
            base = s[:-4]
            # skip leveraged tokens (BTCUPUSDT / BTCDOWNUSDT)
            if (base.endswith("UP") and base[:-2] + "USDT" in syms) or \
               (base.endswith("DOWN") and base[:-4] + "USDT" in syms):
                continue
            self.markets[s] = {"swap": True, "linear": True, "quote": "USDT", "settle": "USDT",
                               "active": True, "base": base}
            self._tick[s] = {"quoteVolume": float(t["quoteVolume"]), "last": float(t["lastPrice"])}
        return self._tick

    def load_markets(self):
        if not self.markets:
            self.fetch_tickers()
        return self.markets

    def fetch_ohlcv(self, sym, tf, since=None, limit=300):
        time.sleep(0.03)
        params = {"symbol": sym, "interval": tf, "limit": limit}
        if since:
            params["startTime"] = int(since)
        rows = self._get("/klines", params)
        return [[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows]


def make_ex(name: str):
    if name == "binancespot":
        return BinanceSpot()
    import ccxt  # lazy import
    return getattr(ccxt, name)({"enableRateLimit": True,
                                "options": {"defaultType": "swap", "defaultSubType": "linear"}})


def candles(ex, sym: str, tf: str, limit: int = 300) -> pd.DataFrame:
    return pd.DataFrame(ex.fetch_ohlcv(sym, tf, limit=limit), columns=COLS)


def universe(ex, min_vol_usd: float = 0.0):
    """Liquid USDT markets as [(24h volume USD, symbol, base)], biggest first."""
    ex.load_markets()
    tk = ex.fetch_tickers()
    out = []
    for s, m in ex.markets.items():
        if (m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("settle") == "USDT"
                and m.get("active") is not False and m.get("base") not in STABLES and s in tk):
            t = tk[s]
            vol = t.get("quoteVolume")
            if not vol and t.get("baseVolume") and t.get("last"):
                vol = t["baseVolume"] * t["last"]
            out.append((float(vol or 0), s, m["base"]))
    if any(v > 0 for v, _, _ in out):
        out = [x for x in out if x[0] >= min_vol_usd]
    return sorted(out, reverse=True)


def find_symbol(ex, coin: str):
    ex.load_markets()
    key = norm(coin)
    for s, m in ex.markets.items():
        if (m.get("swap") and m.get("linear") and m.get("quote") == "USDT" and m.get("settle") == "USDT"
                and m.get("active") is not False and norm(m["base"]) == key):
            return s
    return None


def funding_rates(ex) -> dict:
    """{symbol: funding rate} for perpetuals. Empty dict if the exchange does not give it."""
    try:
        fr = ex.fetch_funding_rates()
        return {s: float(v["fundingRate"]) for s, v in fr.items() if v.get("fundingRate") is not None}
    except Exception:
        return {}


def funding_one(ex, sym):
    try:
        v = ex.fetch_funding_rate(sym).get("fundingRate")
        return float(v) if v is not None else np.nan
    except Exception:
        return np.nan


def fetch_history(ex, sym: str, tf: str, days: int) -> pd.DataFrame:
    """About `days` days of completed candles, page by page."""
    step = tf_ms(tf)
    now = int(time.time() * 1000)
    since = now - days * 86_400_000
    page = 1000 if isinstance(ex, BinanceSpot) else 100
    out = []
    for _ in range(800):
        rows = ex.fetch_ohlcv(sym, tf, since=since, limit=page)
        if not rows:
            break
        out.extend(rows)
        nxt = rows[-1][0] + step
        if nxt >= now or nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(out, columns=COLS).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df[df["ts"] + step <= now].reset_index(drop=True)
