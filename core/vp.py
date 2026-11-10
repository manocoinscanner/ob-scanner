"""Previous-day volume profile (POC / VAH / VAL) reaction backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd


def volume_profile(h, l, v, bins=100, va=0.70):
    lo, hi = float(np.min(l)), float(np.max(h))
    if hi <= lo or float(np.sum(v)) <= 0:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for hh, ll, vv in zip(h, l, v):
        if hh <= ll:
            vol[min(int((hh - lo) / (hi - lo) * bins), bins - 1)] += vv
            continue
        inside = np.clip(edges[1:], ll, hh) - np.clip(edges[:-1], ll, hh)
        vol += vv * inside / (hh - ll)
    poc_i = int(vol.argmax())
    total, got = vol.sum(), vol[poc_i]
    lo_i = hi_i = poc_i
    while got < va * total and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1.0
        dn = vol[lo_i - 1] if lo_i > 0 else -1.0
        if up >= dn:
            hi_i += 1
            got += vol[hi_i]
        else:
            lo_i -= 1
            got += vol[lo_i]
    return {"POC": float((edges[poc_i] + edges[poc_i + 1]) / 2),
            "VAH": float(edges[hi_i + 1]), "VAL": float(edges[lo_i])}


def backtest_vp(df, tf_min, levels, stop_atr=1.0, rr=1.5, va=0.70, fee_pct=0.1, atr_n=14):
    """Each UTC day uses the PREVIOUS day's profile (no look-ahead). First touch of a level = limit
    order at the level (long if price comes down onto it, short if up). Stop = stop_atr x ATR beyond
    the level, target = rr x risk, time exit at day end. Stop assumed first when both hit in one candle."""
    n = len(df)
    if n < atr_n + 3 or not levels:
        return []
    ts = df["ts"].values.astype("int64")
    o, h, l, c, v = (df[k].values.astype(float) for k in ("open", "high", "low", "close", "vol"))
    day = ts // 86_400_000
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.full(n, np.nan)
    for i in range(atr_n, n):
        atr[i] = tr[i - atr_n:i].mean()
    per_day = 1440 / tf_min
    days = np.unique(day)
    trades = []

    def rec(d, pos, exit_px, outcome):
        risk, entry = pos["risk"], pos["entry"]
        gross = (exit_px - entry) / risk if pos["side"] == "long" else (entry - exit_px) / risk
        net = gross - (fee_pct / 100.0) * entry / risk
        trades.append({"day": pd.Timestamp(int(d) * 86_400_000, unit="ms").strftime("%Y-%m-%d"),
                       "level": pos["name"], "side": pos["side"], "entry": entry, "stop": pos["stop"],
                       "target": pos["target"], "exit": exit_px, "outcome": outcome, "R": float(net),
                       "risk_pct": float(risk / entry * 100)})

    for k in range(1, len(days)):
        prev = np.where(day == days[k - 1])[0]
        cur = np.where(day == days[k])[0]
        if len(cur) == 0 or len(prev) < 0.8 * per_day:
            continue
        prof = volume_profile(h[prev], l[prev], v[prev], va=va)
        if prof is None:
            continue
        lv = {name: prof[name] for name in levels}
        used, pos = set(), None
        for i in cur:
            if pos is not None:
                long_ = pos["side"] == "long"
                if (l[i] <= pos["stop"]) if long_ else (h[i] >= pos["stop"]):
                    rec(days[k], pos, pos["stop"], "stop"); pos = None; continue
                if (h[i] >= pos["target"]) if long_ else (l[i] <= pos["target"]):
                    rec(days[k], pos, pos["target"], "target"); pos = None; continue
            touched = []
            for name, L in lv.items():
                if name in used:
                    continue
                if o[i] > L and l[i] <= L:
                    touched.append((abs(o[i] - L), name, L, "long"))
                elif o[i] < L and h[i] >= L:
                    touched.append((abs(o[i] - L), name, L, "short"))
            if not touched:
                continue
            touched.sort()
            for t in touched:
                used.add(t[1])
            if pos is not None:
                continue
            _, name, L, side = touched[0]
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            risk = stop_atr * a
            stop = L - risk if side == "long" else L + risk
            target = L + rr * risk if side == "long" else L - rr * risk
            p = {"name": name, "side": side, "entry": L, "stop": stop, "target": target, "risk": risk}
            if (side == "long" and l[i] <= stop) or (side == "short" and h[i] >= stop):
                rec(days[k], p, stop, "stop")
                continue
            pos = p
        if pos is not None:
            rec(days[k], pos, float(c[cur[-1]]), "time")
    return trades


def summarize(trades):
    if not trades:
        return {"trades": 0, "win": 0.0, "avg_r": 0.0, "total_r": 0.0, "pf": 0.0, "max_dd": 0.0,
                "h1": float("nan"), "h2": float("nan"), "risk_pct": 0.0}
    R = np.array([t["R"] for t in trades])
    wins, losses = R[R > 0].sum(), -R[R < 0].sum()
    pf = wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0)
    cum = np.cumsum(R)
    half = len(R) // 2
    return {"trades": int(len(R)), "win": float((R > 0).mean() * 100), "avg_r": float(R.mean()),
            "total_r": float(R.sum()), "pf": float(min(pf, 99.0)),
            "max_dd": float((np.maximum.accumulate(cum) - cum).max()),
            "h1": float(R[:half].mean()) if half > 0 else float("nan"),
            "h2": float(R[half:].mean()),
            "risk_pct": float(np.mean([t["risk_pct"] for t in trades]))}
