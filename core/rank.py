"""Ranking scanner: scores every coin (trend, strength, volume, momentum vs BTC, setup, derivatives).
No hard filters, so it always returns a list."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data import candles, funding_rates, make_ex, universe


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(d, n=14):
    pc = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def eff_ratio(c, n=40):
    if len(c) <= n:
        return 0.0
    path = float(c.diff().abs().tail(n).sum())
    return abs(float(c.iloc[-1] - c.iloc[-1 - n])) / path if path > 0 else 0.0


def roc24(h1):
    return (float(h1.close.iloc[-1]) / float(h1.close.iloc[-25]) - 1) * 100


def score_coin(h1, h4, btc_roc):
    """h1, h4: candles incl. the unfinished one. Returns a dict or None."""
    if len(h1) < 80 or len(h4) < 80:
        return None
    price = float(h1.close.iloc[-1])
    c1, c4 = h1.iloc[:-1].reset_index(drop=True), h4.iloc[:-1].reset_index(drop=True)
    r24 = roc24(h1)
    e20_4, e50_4 = ema(c4.close, 20), ema(c4.close, 50)
    e20_1, e50_1 = ema(c1.close, 20), ema(c1.close, 50)
    trend4 = float(e20_4.iloc[-1] - e50_4.iloc[-1])
    s = 1 if trend4 > 0 else -1 if trend4 < 0 else (1 if r24 >= 0 else -1)
    a1 = atr(c1)
    if a1 <= 0:
        return None

    pts = {}
    pts["4H trend"] = (15 if s * trend4 > 0 else 0) + (10 if s * (c4.close.iloc[-1] - e50_4.iloc[-1]) > 0 else 0)
    pts["1H trend"] = 10 if (s * (e20_1.iloc[-1] - e50_1.iloc[-1]) > 0 and s * (c1.close.iloc[-1] - e50_1.iloc[-1]) > 0) else 0
    er = eff_ratio(c4.close)
    pts["Clean trend"] = round(min(er / 0.4, 1) * 15, 1)
    v = c1["vol"].values.astype(float)
    rvol = float(v[-3:].mean() / v[-23:-3].mean()) if v[-23:-3].mean() > 0 else float("nan")
    pts["Volume"] = round(min(max((rvol if rvol == rvol else 1) - 1, 0) / 1.5, 1) * 15, 1)
    rel = s * (r24 - btc_roc)
    pts["Strength vs BTC"] = round(min(max(rel / 3, 0), 1) * 10, 1)

    hi20, lo20 = c4.high.iloc[-21:-1].max(), c4.low.iloc[-21:-1].min()
    brk = (s == 1 and c4.close.iloc[-1] > hi20) or (s == -1 and c4.close.iloc[-1] < lo20)
    r1 = float(rsi(c1.close).iloc[-1])
    near = abs(price - float(e20_1.iloc[-1])) <= 1.0 * a1
    pull = near and ((s == 1 and 38 <= r1 <= 55) or (s == -1 and 45 <= r1 <= 62))
    setup = "Breakout" if brk else "Pullback" if pull else "-"
    pts["Setup"] = 25 if setup != "-" else 0

    risk = 1.5 * a1
    stop = price - s * risk
    return {"dir": "LONG" if s == 1 else "SHORT", "score": round(sum(pts.values()), 1), "setup": setup,
            "price": price, "chg24": r24, "vs_btc": r24 - btc_roc, "rvol": rvol, "rsi": r1, "er": er,
            "atr_pct": a1 / price * 100, "entry": price, "stop": stop,
            "tp1": price + s * 1.5 * risk, "tp2": price + s * 3 * risk,
            "why": ", ".join(f"{k} {v:g}" for k, v in pts.items() if v)}


def oi_change(ex, sym):
    """(OI change over 24h %, over 4h %) from hourly open-interest history. (nan, nan) if unavailable."""
    try:
        h = ex.fetch_open_interest_history(sym, "1h", limit=30)
        vals = [x.get("openInterestValue") or x.get("openInterestAmount") for x in h]
        vals = [float(v) for v in vals if v]
        if len(vals) < 25:
            return np.nan, np.nan
        return (vals[-1] / vals[-25] - 1) * 100, (vals[-1] / vals[-5] - 1) * 100
    except Exception:
        return np.nan, np.nan


def derivs_bonus(r):
    """Up to +15: OI rising with the move (+10) and funding not crowded against the trade (+5)."""
    long_ = r["dir"] == "LONG"
    pts = 0.0
    oi, price_chg = r.get("oi24"), r["chg24"]
    if oi == oi and oi is not None and oi > 0 and ((long_ and price_chg > 0) or (not long_ and price_chg < 0)):
        pts += 10
    f = r.get("funding")
    if f == f and f is not None and ((long_ and f <= 0.00005) or (not long_ and f >= 0.0001)):
        pts += 5
    return pts


def run_rank(exch, min_vol_m=20, top_n=100, on_progress=None, deriv_top=40):
    ex = make_ex(exch)
    uni = universe(ex, min_vol_m * 1_000_000)[:top_n]
    btc_sym = "BTCUSDT" if exch == "binancespot" else "BTC/USDT:USDT"
    btc = roc24(candles(ex, btc_sym, "1h", 100))

    def one(sym, base):
        r = score_coin(candles(ex, sym, "1h", 200), candles(ex, sym, "4h", 200), btc)
        return None if r is None else {"coin": base, "sym": sym, **r}

    rows, fails = [], 0
    workers = 8 if exch == "binancespot" else 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, s, b) for _, s, b in uni]
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception:
                fails += 1
            if on_progress:
                on_progress((i + 1) / len(futs) * (0.8 if exch != "binancespot" else 1.0))
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {"n": len(uni), "fails": fails, "btc_roc": btc, "derivs": False,
                    "time": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")}
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["oi24"], df["oi4"], df["funding"], df["bonus"] = np.nan, np.nan, np.nan, 0.0
    derivs = False
    if exch != "binancespot":  # open interest + funding exist only for perpetuals
        fr = funding_rates(ex)
        top = df.head(deriv_top)
        with ThreadPoolExecutor(max_workers=4) as pool:
            res = list(pool.map(lambda s: oi_change(ex, s), top["sym"]))
        for (idx, _), (o24, o4) in zip(top.iterrows(), res):
            df.at[idx, "oi24"], df.at[idx, "oi4"] = o24, o4
        df["funding"] = df["sym"].map(fr).astype(float)
        derivs = bool(df["oi24"].notna().any() or df["funding"].notna().any())
        df["bonus"] = [derivs_bonus(r) for r in df.to_dict("records")]
        df["score"] = (df["score"] + df["bonus"]).round(1)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        if on_progress:
            on_progress(1.0)
    return df, {"n": len(uni), "fails": fails, "btc_roc": btc, "derivs": derivs,
                "time": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")}
