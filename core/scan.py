"""Scanner pipeline: fetch -> detect -> features. Scoring and filters run afterwards on the table,
so moving a slider never needs a re-scan."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import candles, funding_rates, make_ex, tf_ms, universe
from .smc import Cfg, analyze_tf, daily_bias, ltf_trigger

TF_MAP = {"4H": "4h", "1H": "1h", "15m": "15m", "5m": "5m"}
CONF_COLS = {"Turtle soup": "ts", "CISD": "cisd", "FVG near OB": "fvg", "LTF trigger": "ltf"}

DEFAULT_W = {"near": 2, "inside": 1, "bias": 1, "pd": 1, "ts": 2, "cisd": 2, "fvg": 1, "ltf": 2,
             "rvol": 1, "funding": 1, "dxy": 1, "kz": 1, "rs": 1}
W_LABELS = {"near": "OB near price", "inside": "Price inside OB (extra)", "bias": "OB with daily bias",
            "pd": "Correct Premium/Discount", "ts": "Turtle soup", "cisd": "CISD", "fvg": "FVG near OB",
            "ltf": "LTF trigger", "rvol": "Relative volume high", "funding": "Funding supports",
            "dxy": "DXY supports (±)", "kz": "Inside a kill zone", "rs": "Strength vs BTC"}


@dataclass(frozen=True)
class ScanParams:
    exch: str
    tfs: tuple
    ltf: str            # "" = off
    cfg: Cfg
    min_vol_m: float
    top_n: int          # 0 = all
    pre_min_chg: float  # skip coins that moved less than this % from today's open (speed-up)


def btc_symbol(exch):
    return "BTCUSDT" if exch == "binancespot" else "BTC/USDT:USDT"


def scan_coin(ex, sym, base, p: ScanParams, funding=np.nan, src=""):
    """Rows (one per timeframe with a live OB near price) for one coin."""
    dd = candles(ex, sym, "1d", 60)
    if len(dd) < 25:
        return []
    if p.cfg.skip_pegged and (dd.high.max() - dd.low.min()) / dd.close.iloc[-1] < 0.05:
        return []
    chg_open = (float(dd.close.iloc[-1]) / float(dd.open.iloc[-1]) - 1) * 100
    if abs(chg_open) < p.pre_min_chg:
        return []
    bias, why = daily_bias(dd, p.cfg)
    rows = []
    for name in p.tfs:
        hd = candles(ex, sym, TF_MAP[name], 300)
        r = analyze_tf(hd, bias, p.cfg) if len(hd) >= 60 else None
        if not r:
            continue
        ltf = False
        if p.ltf and tf_ms(TF_MAP[p.ltf]) < tf_ms(TF_MAP[name]):
            try:
                ltf = ltf_trigger(candles(ex, sym, TF_MAP[p.ltf], 300), r["side"], p.cfg)
            except Exception:
                ltf = False
        rows.append({"coin": base, "sym": sym, "src": src, "bias": bias, "why": why, "tf": name,
                     "ltf": ltf, "chg_open": chg_open, "funding": funding, **r})
    return rows


def run_scan(p: ScanParams, on_progress=None):
    ex = make_ex(p.exch)
    uni = universe(ex, p.min_vol_m * 1_000_000)
    n_uni = len(uni)
    if p.top_n:
        uni = uni[:p.top_n]
    fr = funding_rates(ex) if p.exch != "binancespot" else {}
    rows, fails = [], 0
    workers = 8 if p.exch == "binancespot" else 4
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(scan_coin, ex, s, b, p, fr.get(s, np.nan), p.exch) for _, s, b in uni]
        for i, f in enumerate(as_completed(futs)):
            try:
                rows += f.result()
            except Exception:
                fails += 1
            if on_progress:
                on_progress((i + 1) / len(futs))
    return {"rows": rows, "fails": fails, "n": len(uni), "n_universe": n_uni, "params": p,
            "time": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")}


# ---------- scoring & filters (run on the table, instant) ----------

def score_df(df, W, ctx):
    """ctx: dxy ('bull'/'bear'/None), btc_today (float|None), in_kz (bool), rvol_min (float)."""
    df = df.copy()
    if df.empty:
        df["Score"] = []
        return df
    bull = df["side"] == "bull"
    df["dxy_ok"] = (bull & (ctx["dxy"] == "bear")) | (~bull & (ctx["dxy"] == "bull")) if ctx["dxy"] else False
    f = df["funding"].astype(float)
    df["fund_ok"] = (bull & (f <= 0)) | (~bull & (f >= 0.0001))
    bt = ctx.get("btc_today")
    df["rs_ok"] = ((bull & (df["chg_open"] > bt)) | (~bull & (df["chg_open"] < bt))) if bt is not None else False
    df["rvol_ok"] = df["rvol"].fillna(0) >= ctx["rvol_min"]
    df["with_bias"] = df["side"] == df["bias"]

    s = pd.Series(float(W["near"]), index=df.index)
    s += np.where(df["inside"], W["inside"], 0)
    s += np.where(df["with_bias"], W["bias"], 0)
    s += np.where(df["pd_ok"], W["pd"], 0)
    for key in ("ts", "cisd", "fvg", "ltf"):
        s += np.where(df[key], W[key], 0)
    s += np.where(df["rvol_ok"], W["rvol"], 0)
    s += np.where(df["fund_ok"], W["funding"], 0)
    s += np.where(df["rs_ok"], W["rs"], 0)
    if ctx["dxy"]:
        s += np.where(df["dxy_ok"], W["dxy"], -W["dxy"])
    s += W["kz"] if ctx["in_kz"] else 0
    df["Score"] = s.round(1)
    return df


def build_rules(f, dxy_known, rr_known=True):
    """Optional hard filters: (name, enabled, function(df) -> bool Series)."""
    cols = [CONF_COLS[c] for c in f["conf"]]
    return [
        (f"Score ≥ {f['min_score']}", True, lambda d: d["Score"] >= f["min_score"]),
        (f"Distance to OB ≤ {f['dist']}%", f["f_dist"], lambda d: d["dist"] <= f["dist"]),
        ("OB direction = daily bias", f["f_bias"], lambda d: d["with_bias"]),
        ("Correct Premium/Discount side", f["f_pd"], lambda d: d["pd_ok"]),
        ("Confirmation: " + " / ".join(f["conf"]), bool(cols), lambda d: d[cols].any(axis=1)),
        ("DXY supports direction", f["f_dxy"] and dxy_known, lambda d: d["dxy_ok"]),
        ("Inside a kill zone now", f["f_kz"], lambda d: pd.Series(f["_in_kz"], index=d.index)),
        ("Funding supports direction", f["f_fund"], lambda d: d["fund_ok"]),
        (f"Trend efficiency ≥ {f['min_er']}", f["min_er"] > 0, lambda d: d["er"] >= f["min_er"]),
        (f"OB size ≥ {f['min_zone_atr']} ATR", f["min_zone_atr"] > 0, lambda d: d["zone_atr"] >= f["min_zone_atr"]),
        (f"ATR % ≥ {f['min_atr_pct']}", f["min_atr_pct"] > 0, lambda d: d["atr_pct"] >= f["min_atr_pct"]),
        (f"Move from open ≥ ±{f['min_chg']}%", f["min_chg"] > 0, lambda d: d["chg_open"].abs() >= f["min_chg"]),
        (f"Relative volume ≥ {f['min_rvol']}", f["min_rvol"] > 0, lambda d: d["rvol"].fillna(0) >= f["min_rvol"]),
        (f"R:R ≥ {f['min_rr']}", f["min_rr"] > 0, lambda d: d["rr"] >= f["min_rr"]),
    ]


def apply_rules(df, rules):
    """Returns (filtered df, funnel list of (rule, coins left))."""
    funnel = [("Scan result (OB near price)", len(df))]
    for name, on, fn in rules:
        if on:
            df = df[fn(df)]
            funnel.append((name, len(df)))
    return df, funnel


def fail_reasons(df, rules):
    """Per-row text: which enabled rules the row fails ('' = passes everything)."""
    out = pd.Series([""] * len(df), index=df.index)
    for name, on, fn in rules:
        if on and len(df):
            bad = ~fn(df)
            out = out.where(~bad, out + np.where(out == "", "", "; ") + name)
    return out
