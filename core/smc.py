"""Smart-money concepts engine: swings, structure Order Blocks (with liquidity sweep), FVG,
daily bias, Turtle soup, CISD, premium/discount, volatility, volume and a simple trade plan."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Cfg:
    swing_k: int = 2          # candles on each side of a swing
    need_sweep: bool = True   # OB needs the liquidity sweep ("torch")
    max_age: int = 150        # structure break must be within this many closed candles
    conf_win: int = 3         # Turtle soup / CISD must happen within this many candles
    rule2: bool = True        # daily bias Rule 2 (rejection wick at FVG/OB)
    skip_pegged: bool = True  # skip stable-like coins
    prox: float = 3.0         # max distance to the OB (%)


def swings(d, k=2):
    """Swing highs/lows as (index, 'H'/'L', price), strictly alternating."""
    h, l = d.high.values, d.low.values
    raw = []
    for i in range(k, len(d) - k):
        is_h = h[i] >= h[i - k:i].max() and h[i] > h[i + 1:i + k + 1].max()
        is_l = l[i] <= l[i - k:i].min() and l[i] < l[i + 1:i + k + 1].min()
        if is_h == is_l:
            continue
        raw.append((i, "H" if is_h else "L", float(h[i] if is_h else l[i])))
    out = []
    for i, kind, price in raw:
        if out and out[-1][1] == kind:
            if (kind == "H" and price > out[-1][2]) or (kind == "L" and price < out[-1][2]):
                out[-1] = (i, kind, price)
            continue
        out.append((i, kind, price))
    return out


def find_structure_obs(d, cfg: Cfg):
    """Live OBs as (side, low, high, break_index).
    Bull: LH+LL down-trend, last LL sweeps the previous low and closes back above it, then a close
    above the last LH. Zone = LH high .. LL low. Bear is the mirror. A zone dies when a later candle
    closes beyond it."""
    h, l, c = d.high.values, d.low.values, d.close.values
    n = len(d)
    sw = swings(d, cfg.swing_k)
    found = []
    for t in range(3, len(sw)):
        s0, s1, s2, s3 = sw[t - 3], sw[t - 2], sw[t - 1], sw[t]
        kinds = (s0[1], s1[1], s2[1], s3[1])
        if kinds == ("H", "L", "H", "L") and s2[2] < s0[2] and s3[2] < s1[2]:
            if cfg.need_sweep and not (l[s3[0]] < s1[2] and c[s3[0]] > s1[2]):
                continue
            top, bot = s2[2], s3[2]
            for j in range(s3[0] + 1, n):
                if l[j] < bot:
                    break
                if c[j] > top:
                    found.append(("bull", bot, top, j))
                    break
        elif kinds == ("L", "H", "L", "H") and s2[2] > s0[2] and s3[2] > s1[2]:
            if cfg.need_sweep and not (h[s3[0]] > s1[2] and c[s3[0]] < s1[2]):
                continue
            bot, top = s2[2], s3[2]
            for j in range(s3[0] + 1, n):
                if h[j] > top:
                    break
                if c[j] < bot:
                    found.append(("bear", bot, top, j))
                    break
    live = []
    for side, lo, hi, j in found:
        later = c[j + 1:]
        if side == "bull" and (later < lo).any():
            continue
        if side == "bear" and (later > hi).any():
            continue
        live.append((side, float(lo), float(hi), j))
    return live


def find_fvgs(d):
    """Unfilled fair value gaps as (side, low, high)."""
    h, l, c = d.high.values, d.low.values, d.close.values
    found = []
    for i in range(2, len(d)):
        if l[i] > h[i - 2]:
            found.append(("bull", h[i - 2], l[i], i))
        if h[i] < l[i - 2]:
            found.append(("bear", h[i], l[i - 2], i))
    live = []
    for side, lo, hi, i in found:
        later = c[i + 1:]
        if side == "bull" and (later < lo).any():
            continue
        if side == "bear" and (later > hi).any():
            continue
        live.append((side, float(lo), float(hi)))
    return live


def daily_bias(d, cfg: Cfg):
    """d = daily candles including the current unfinished one. Returns (bias, reason)."""
    closed = d.iloc[:-1].reset_index(drop=True)
    if len(closed) < 6:
        return "bull", "Rule 1 (short history)"
    bias = "bull" if closed.close.iloc[-1] > closed.close.iloc[-2] else "bear"
    if not cfg.rule2:
        return bias, "Rule 1"
    prev = closed.iloc[:-1].reset_index(drop=True)
    zones = [("FVG", s, lo, hi) for s, lo, hi in find_fvgs(prev)]
    zones += [("OB", s, lo, hi) for s, lo, hi, _ in find_structure_obs(prev, cfg)]
    k = closed.iloc[-1]
    o, h, l, c = k.open, k.high, k.low, k.close
    body, rg = abs(c - o), h - l
    low_wick, up_wick = min(o, c) - l, h - max(o, c)
    if rg > 0:
        if low_wick >= 2 * body and low_wick >= 0.5 * rg:
            for kind, side, lo, hi in zones:
                if side == "bull" and l <= hi and min(o, c) >= lo:
                    return "bull", f"Rule 2: rejection at bull {kind}"
        if up_wick >= 2 * body and up_wick >= 0.5 * rg:
            for kind, side, lo, hi in zones:
                if side == "bear" and h >= lo and max(o, c) <= hi:
                    return "bear", f"Rule 2: rejection at bear {kind}"
    return bias, "Rule 1"


def turtle_soup(d, side, look=20, recent=3):
    h, l, c = d.high.values, d.low.values, d.close.values
    n = len(d)
    for k in range(max(n - recent, look), n):
        if side == "bull":
            ref = l[k - look:k].min()
            if l[k] < ref and c[k] > ref:
                return True
        else:
            ref = h[k - look:k].max()
            if h[k] > ref and c[k] < ref:
                return True
    return False


def cisd(d, side, recent=3, scan=12):
    o, c = d.open.values, d.close.values
    n = len(d)
    opposing = (c < o) if side == "bull" else (c > o)
    for e in range(n - 2, max(n - scan, 1) - 1, -1):
        if opposing[e] and not opposing[e + 1]:
            s = e
            while s > 0 and opposing[s - 1]:
                s -= 1
            run_open = o[s]
            for j in range(e + 1, n):
                broke = c[j] > run_open if side == "bull" else c[j] < run_open
                if broke:
                    return j >= n - recent
            return False
    return False


def volatility_stats(d, n=14, er_n=40):
    """(ATR price, ATR % of price, efficiency ratio 0..1)."""
    h, l, c = d.high, d.low, d.close
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = float(tr.tail(n).mean())
    price = float(c.iloc[-1])
    er = 0.0
    if len(c) > er_n:
        net = abs(float(c.iloc[-1]) - float(c.iloc[-1 - er_n]))
        path = float(c.diff().abs().tail(er_n).sum())
        er = net / path if path > 0 else 0.0
    return atr, (atr / price * 100 if price > 0 else 0.0), er


def rel_volume(d, recent=3, base=20):
    """Average volume of the last `recent` closed candles / average of the `base` before them."""
    v = d["vol"].values.astype(float)
    if len(v) < recent + base:
        return float("nan")
    b = v[-(recent + base):-recent].mean()
    return float(v[-recent:].mean() / b) if b > 0 else float("nan")


def analyze_tf(hd, bias, cfg: Cfg):
    """Best OB near the price on one timeframe (or None). hd includes the unfinished candle."""
    closed = hd.iloc[:-1].reset_index(drop=True)
    n = len(closed)
    if n < 60:
        return None
    price = float(hd.close.iloc[-1])
    best = None
    for side, lo, hi, j in find_structure_obs(closed, cfg):
        if j < n - cfg.max_age:
            continue
        if (side == "bull" and price < lo) or (side == "bear" and price > hi):
            continue  # price already beyond the zone
        dist = 0.0 if lo <= price <= hi else ((price - hi) / price * 100 if price > hi else (lo - price) / price * 100)
        key = (dist, 0 if side == bias else 1, -j)
        if dist <= cfg.prox and (best is None or key < best[0]):
            best = (key, side, lo, hi, j)
    if best is None:
        return None
    (dist, _, _), side, lo, hi, j = best

    sigs = {"ts": turtle_soup(closed, side, recent=cfg.conf_win),
            "cisd": cisd(closed, side, recent=cfg.conf_win), "fvg": False}
    for fside, flo, fhi in find_fvgs(closed.tail(100)):
        if fside != side:
            continue
        gap = 0.0 if (fhi >= lo and flo <= hi) else (flo - hi if flo > hi else lo - fhi)
        if gap / price * 100 <= 1.0:
            sigs["fvg"] = True
            break

    rng_hi, rng_lo = float(closed.high.tail(100).max()), float(closed.low.tail(100).min())
    pos = (((lo + hi) / 2 - rng_lo) / (rng_hi - rng_lo) * 100) if rng_hi > rng_lo else 50.0
    pd_ok = (side == "bull" and pos <= 50) or (side == "bear" and pos > 50)
    atr, atr_pct, er = volatility_stats(closed)

    # trade plan: limit at the 50% of the OB, stop beyond the far edge, target = nearest opposite swing
    entry = (lo + hi) / 2
    stop = lo - 0.25 * atr if side == "bull" else hi + 0.25 * atr
    risk = abs(entry - stop)
    sw = swings(closed, cfg.swing_k)
    if side == "bull":
        cand = [p for _, k, p in sw if k == "H" and p > entry]
        target = min(cand) if cand else None
    else:
        cand = [p for _, k, p in sw if k == "L" and p < entry]
        target = max(cand) if cand else None
    tgt_src = "swing"
    if target is None:
        target = entry + 2 * risk if side == "bull" else entry - 2 * risk
        tgt_src = "2R"
    rr = abs(target - entry) / risk if risk > 0 else 0.0

    return {"side": side, "lo": lo, "hi": hi, "dist": dist, "inside": dist == 0.0,
            "brk_ago": n - 1 - j, **sigs, "pos": pos, "pd_zone": "Premium" if pos > 50 else "Discount",
            "pd_ok": pd_ok, "price": price, "er": er, "atr_pct": atr_pct,
            "zone_atr": (hi - lo) / atr if atr > 0 else 0.0, "rvol": rel_volume(closed),
            "entry": entry, "stop": stop, "target": target, "tgt_src": tgt_src, "rr": rr}


def ltf_trigger(ld, side, cfg: Cfg):
    """Lower-timeframe trigger: CISD or Turtle soup in the OB direction on the last closed candles."""
    closed = ld.iloc[:-1].reset_index(drop=True)
    if len(closed) < 40:
        return False
    return bool(cisd(closed, side, recent=cfg.conf_win) or turtle_soup(closed, side, recent=cfg.conf_win))
