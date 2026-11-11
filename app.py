"""Ranking scanner: scores every coin (trend, strength, volume, momentum vs BTC, setup, derivatives,
order book walls, taker flow). No hard filters, so it always returns a list."""
from __future__ import annotations

import time
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


# ───────────────────────── order book (DOM) ─────────────────────────
def get_book(ex, sym):
    """Order book snapshot {'bids': [[price, qty], ...], 'asks': [...]}."""
    if hasattr(ex, "BASE") and hasattr(ex, "_get"):  # the small Binance Spot client
        j = ex._get("/depth", {"symbol": sym, "limit": 1000})
        return {"bids": j["bids"], "asks": j["asks"]}
    return ex.fetch_order_book(sym, 200)


def _walls(levels, mult=8.0, min_levels=10):
    """Levels (price, qty) whose size is >= mult x the median level size. Returns [(price, size_x, qty)]."""
    if len(levels) < min_levels:
        return []
    med = float(np.median([q for _, q in levels]))
    if med <= 0:
        return []
    return [(p, q / med, q) for p, q in levels if q >= mult * med]


def book_features(book, price, side, tp1):
    """Order-book (DOM) features around the price. Uses only the range the snapshot really covers.
    Returns a dict or None when the snapshot is too thin."""
    bids = [(float(p), float(q)) for p, q in book["bids"]]
    asks = [(float(p), float(q)) for p, q in book["asks"]]
    if len(bids) < 20 or len(asks) < 20:
        return None
    cover = min(price - bids[-1][0], asks[-1][0] - price) / price  # both sides visible up to this distance
    if cover < 0.001:  # < 0.1%: snapshot too shallow (typical for BTC/ETH with tiny ticks)
        return None
    b = [(p, q) for p, q in bids if p >= price * (1 - cover)]
    a = [(p, q) for p, q in asks if p <= price * (1 + cover)]
    w = min(cover, 0.01)
    vb = sum(q for p, q in b if p >= price * (1 - w))
    va = sum(q for p, q in a if p <= price * (1 + w))
    share = vb / (vb + va) if vb + va > 0 else float("nan")
    bw, aw = _walls(b), _walls(a)
    nb = max(bw, key=lambda x: x[0]) if bw else None   # nearest bid wall (highest price below)
    na = min(aw, key=lambda x: x[0]) if aw else None   # nearest ask wall (lowest price above)
    long_ = side == "LONG"
    if long_:
        blocks = any(price < x[0] <= tp1 for x in aw)
        support = nb is not None and (price - nb[0]) / price <= 0.02
        crowd = share == share and share >= 0.58
        sup = nb if support else None
    else:
        blocks = any(tp1 <= x[0] < price for x in bw)
        support = na is not None and (na[0] - price) / price <= 0.02
        crowd = share == share and share <= 0.42
        sup = na if support else None
    dom = (5 if support else 0) + (5 if crowd else 0) - (5 if blocks else 0)
    return {"dom_share": share, "bid_wall_x": nb[1] if nb else np.nan,
            "bid_wall_dist": (price - nb[0]) / price * 100 if nb else np.nan,
            "ask_wall_x": na[1] if na else np.nan,
            "ask_wall_dist": (na[0] - price) / price * 100 if na else np.nan,
            "blocks": bool(blocks), "dom": float(dom), "cover": cover * 100,
            "sup_p": sup[0] if sup else np.nan, "sup_q": sup[2] if sup else np.nan,
            "sup_bid": ((1.0 if long_ else 0.0) if sup else np.nan)}


def dom_one(ex, r):
    try:
        return book_features(get_book(ex, r["sym"]), r["price"], r["dir"], r["tp1"])
    except Exception:
        return None


def persist_one(ex, sym, wall_p, wall_q, is_bid):
    """Second snapshot: is the support wall still there (same price, at least half of its size)?
    Returns 1.0 (still there), 0.0 (vanished / shrunk) or nan (could not check)."""
    try:
        bk = get_book(ex, sym)
        levels = bk["bids"] if is_bid else bk["asks"]
        for p, q in levels:
            if abs(float(p) - wall_p) / wall_p <= 0.0005:
                return 1.0 if float(q) >= 0.5 * wall_q else 0.0
        return 0.0
    except Exception:
        return np.nan


# ───────────────────────── taker flow (CVD) ─────────────────────────
def get_trades(ex, sym):
    """Recent trades as [(ts_ms, price, qty, taker_is_buy)]."""
    if hasattr(ex, "BASE") and hasattr(ex, "_get"):  # Binance Spot: m = buyer is the maker -> taker sold
        j = ex._get("/aggTrades", {"symbol": sym, "limit": 1000})
        return [(int(x["T"]), float(x["p"]), float(x["q"]), not x["m"]) for x in j]
    try:
        tr = ex.fetch_trades(sym, limit=500)
    except Exception:
        tr = ex.fetch_trades(sym)
    return [(int(t["timestamp"]), float(t["price"]), float(t["amount"]), t.get("side") == "buy")
            for t in tr if t.get("timestamp") and t.get("price") and t.get("amount") and t.get("side") in ("buy", "sell")]


def flow_features(trades, side):
    """Taker buy share of the traded value + points. None if too few trades or window < 1 minute."""
    if len(trades) < 50:
        return None
    ts = [t[0] for t in trades]
    win = (max(ts) - min(ts)) / 60000
    if win < 1.0:
        return None
    buy = sum(p * q for _, p, q, b in trades if b)
    tot = sum(p * q for _, p, q, _ in trades)
    if tot <= 0:
        return None
    share = buy / tot
    if side == "LONG":
        pts = 5 if share >= 0.55 else -5 if share <= 0.40 else 0
    else:
        pts = 5 if share <= 0.45 else -5 if share >= 0.60 else 0
    return {"flow_share": share, "flow_win": win, "flow": float(pts)}


def flow_one(ex, r):
    try:
        return flow_features(get_trades(ex, r["sym"]), r["dir"])
    except Exception:
        return None


def run_rank(exch, min_vol_m=20, top_n=100, on_progress=None, deriv_top=40, dom_top=40):
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
                on_progress((i + 1) / len(futs) * 0.6)
    meta = {"n": len(uni), "fails": fails, "btc_roc": btc, "derivs": False, "dom": False, "flow": False,
            "time": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")}
    df = pd.DataFrame(rows)
    if df.empty:
        return df, meta
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    for c in ("oi24", "oi4", "funding", "dom_share", "bid_wall_x", "bid_wall_dist", "ask_wall_x",
              "ask_wall_dist", "cover", "sup_p", "sup_q", "sup_bid", "flow_share", "flow_win", "persist"):
        df[c] = np.nan
    df["bonus"], df["dom"], df["flow"], df["blocks"] = 0.0, 0.0, 0.0, False

    if exch != "binancespot":  # open interest + funding exist only for perpetuals
        fr = funding_rates(ex)
        top = df.head(deriv_top)
        with ThreadPoolExecutor(max_workers=4) as pool:
            res = list(pool.map(lambda s: oi_change(ex, s), top["sym"]))
        for (idx, _), (o24, o4) in zip(top.iterrows(), res):
            df.at[idx, "oi24"], df.at[idx, "oi4"] = o24, o4
        df["funding"] = df["sym"].map(fr).astype(float)
        meta["derivs"] = bool(df["oi24"].notna().any() or df["funding"].notna().any())
        df["bonus"] = [derivs_bonus(r) for r in df.to_dict("records")]
        df["score"] = (df["score"] + df["bonus"]).round(1)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    if on_progress:
        on_progress(0.75)

    top = df.head(dom_top)
    recs = top.to_dict("records")

    # 1) order book snapshot #1
    with ThreadPoolExecutor(max_workers=4) as pool:
        res = list(pool.map(lambda r: dom_one(ex, r), recs))
    for (idx, _), f in zip(top.iterrows(), res):
        if f:
            for k, v in f.items():
                df.at[idx, k] = v
    meta["dom"] = bool(df["dom_share"].notna().any())
    if on_progress:
        on_progress(0.85)

    # 2) taker flow (CVD) from the latest trades (also gives the time gap before snapshot #2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        res = list(pool.map(lambda r: flow_one(ex, r), recs))
    for (idx, _), f in zip(top.iterrows(), res):
        if f:
            for k, v in f.items():
                df.at[idx, k] = v
    meta["flow"] = bool(df["flow_share"].notna().any())
    if on_progress:
        on_progress(0.93)

    # 3) order book snapshot #2: does the support wall still stand (real) or has it vanished (spoof?)
    jobs = [(idx, df.at[idx, "sym"], df.at[idx, "sup_p"], df.at[idx, "sup_q"], df.at[idx, "sup_bid"] == 1.0)
            for idx, _ in top.iterrows() if df.at[idx, "sup_p"] == df.at[idx, "sup_p"]]
    if jobs:
        time.sleep(1)
        with ThreadPoolExecutor(max_workers=4) as pool:
            res = list(pool.map(lambda j: persist_one(ex, j[1], j[2], j[3], j[4]), jobs))
        for (idx, *_), pr in zip(jobs, res):
            df.at[idx, "persist"] = pr
            if pr == 0.0:  # the +5 given for the support wall is taken back
                df.at[idx, "dom"] = float(df.at[idx, "dom"]) - 5

    df["score"] = (df["score"] + df["dom"] + df["flow"]).round(1)

    def why_text(w, d, b, fl, pr):
        s = w
        if d:
            s += f", DOM {d:+g}" + (" (wall blocks TP1)" if b else "")
        if pr == 0:
            s += " (support wall vanished = spoof?)"
        if fl:
            s += f", Flow {fl:+g}"
        return s

    df["why"] = [why_text(w, d, b, fl, pr) for w, d, b, fl, pr in
                 zip(df["why"], df["dom"], df["blocks"], df["flow"], df["persist"])]
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    if on_progress:
        on_progress(1.0)
    return df, meta
