"""RSI scanner: RSI overbought / oversold and RSI divergence (regular + optional hidden) on many timeframes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

TFS = ["1m", "3m", "5m", "15m", "1h", "4h"]
TF_RANK = {tf: i for i, tf in enumerate(TFS)}


@dataclass(frozen=True)
class DivCfg:
    k: int = 3                 # candles on each side of a pivot
    max_age: int = 15          # the second pivot must be at most this many closed candles ago
    min_gap: int = 5           # candles between the two pivots (min)
    max_gap: int = 60          # candles between the two pivots (max)
    min_rsi_diff: float = 2.0  # minimum RSI difference between the two pivots
    need_extreme: bool = True  # regular divergence: the first pivot's RSI must be overbought / oversold
    ob: float = 65.0
    os: float = 35.0
    hidden: bool = False       # also look for hidden divergence (trend continuation)
    period: int = 14


def rsi_wilder(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    return out.mask((dn == 0) & (up > 0), 100.0).fillna(50.0)


def pivots(v, k, kind):
    """Indexes of pivot highs ('H') or lows ('L'). A pivot needs k candles on both sides."""
    idx = []
    for i in range(k, len(v) - k):
        if kind == "H":
            if v[i] >= v[i - k:i].max() and v[i] > v[i + 1:i + k + 1].max():
                idx.append(i)
        else:
            if v[i] <= v[i - k:i].min() and v[i] < v[i + 1:i + k + 1].min():
                idx.append(i)
    return idx


def find_divergences(d, cfg: DivCfg):
    """d: CLOSED candles (open, high, low, close). Returns (rows, rsi_now)."""
    d = d.reset_index(drop=True)
    n = len(d)
    r = rsi_wilder(d["close"], cfg.period).values
    h, l = d["high"].values, d["low"].values
    rows = []

    def last_two(v, kind):
        piv = [i for i in pivots(v, cfg.k, kind) if i >= 40]  # skip the RSI warm-up
        if len(piv) < 2:
            return None
        i1, i2 = piv[-2], piv[-1]
        if n - 1 - i2 > cfg.max_age or not (cfg.min_gap <= i2 - i1 <= cfg.max_gap):
            return None
        return i1, i2

    def add(side, kind, v, i1, i2):
        rows.append({"side": side, "kind": kind, "dir": "SHORT" if side == "Bearish" else "LONG",
                     "p1": float(v[i1]), "p2": float(v[i2]), "r1": float(r[i1]), "r2": float(r[i2]),
                     "ago": int(n - 1 - i2), "gap": int(i2 - i1)})

    pr = last_two(h, "H")
    if pr:
        i1, i2 = pr
        if h[i2] > h[i1] and r[i2] <= r[i1] - cfg.min_rsi_diff and (not cfg.need_extreme or r[i1] >= cfg.ob):
            add("Bearish", "Regular", h, i1, i2)
        elif cfg.hidden and h[i2] < h[i1] and r[i2] >= r[i1] + cfg.min_rsi_diff:
            add("Bearish", "Hidden", h, i1, i2)
    pr = last_two(l, "L")
    if pr:
        i1, i2 = pr
        if l[i2] < l[i1] and r[i2] >= r[i1] + cfg.min_rsi_diff and (not cfg.need_extreme or r[i1] <= cfg.os):
            add("Bullish", "Regular", l, i1, i2)
        elif cfg.hidden and l[i2] > l[i1] and r[i2] <= r[i1] - cfg.min_rsi_diff:
            add("Bullish", "Hidden", l, i1, i2)
    return rows, float(r[-1])


def scan_coin(ex, sym, base, vol, tfs, cfg, candles):
    rows, rsis, errs = [], {}, []
    for tf in tfs:
        try:
            df = candles(ex, sym, tf, 300)
            closed = df.iloc[:-1]  # the last candle is still forming
            if len(closed) < 80:
                continue
            found, now = find_divergences(closed, cfg)
            rsis[tf] = now
            price = float(df["close"].iloc[-1])
            for f in found:
                rows.append({"coin": base, "tf": tf, "price": price, "vol_m": vol / 1e6, **f})
        except Exception as e:
            errs.append(f"{base} {tf}: {str(e)[:80]}")
    return rows, rsis, errs


def run_scan(exch, tfs, cfg, min_vol_m=20, top_n=60, on_progress=None):
    from .data import candles, make_ex, universe
    ex = make_ex(exch)
    uni = universe(ex, min_vol_m * 1_000_000)[:top_n]
    workers = 8 if exch == "binancespot" else 4
    rows, rsi_map, errs = [], {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(scan_coin, ex, s, b, v, tfs, cfg, candles): b for v, s, b in uni}
        for i, f in enumerate(as_completed(futs)):
            try:
                r, m, e = f.result()
                rows += r
                rsi_map[futs[f]] = m
                errs += e
            except Exception as e:
                errs.append(f"{futs[f]}: {str(e)[:80]}")
            if on_progress:
                on_progress((i + 1) / len(futs))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["tf_rank"] = df["tf"].map(TF_RANK)
        cnt = df.groupby(["coin", "dir"])["tf"].transform("count")
        df["tf_count"] = cnt
        df = df.sort_values(["tf_count", "tf_rank", "ago"], ascending=[False, False, True]).reset_index(drop=True)
    return {"div": df, "rsi": rsi_map, "n": len(uni), "errors": errs, "tfs": list(tfs),
            "time": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")}
