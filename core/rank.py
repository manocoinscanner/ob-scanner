"""Ranking scanner: scores every coin 0-100 (trend, strength, volume, momentum vs BTC, setup).
No hard filters, so it always returns a list."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .data import candles, make_ex, universe


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


def run_rank(exch, min_vol_m=20, top_n=100, on_progress=None):
    ex = make_ex(exch)
    uni = universe(ex, min_vol_m * 1_000_000)[:top_n]
    btc_sym = "BTCUSDT" if exch == "binancespot" else "BTC/USDT:USDT"
    btc = roc24(candles(ex, btc_sym, "1h", 100))

    def one(sym, base):
        r = score_coin(candles(ex, sym, "1h", 200), candles(ex, sym, "4h", 200), btc)
        return None if r is None else {"coin": base, **r}

    rows, fails = [], 0
    with ThreadPoolExecutor(max_workers=8 if exch == "binancespot" else 4) as pool:
        futs = [pool.submit(one, s, b) for _, s, b in uni]
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
                if r:
                    rows.append(r)
            except Exception:
                fails += 1
            if on_progress:
                on_progress((i + 1) / len(futs))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df, {"n": len(uni), "fails": fails, "btc_roc": btc,
                "time": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")}
