import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

from core.data import SOURCES, candles, fetch_history, find_symbol, funding_one, make_ex, tf_ms, universe
from core.scan import (CONF_COLS, DEFAULT_W, TF_MAP, W_LABELS, ScanParams, apply_rules, btc_symbol,
                       build_rules, fail_reasons, run_scan, scan_coin, score_df)
from core.smc import Cfg, analyze_tf, daily_bias
from core.vp import backtest_vp, summarize

st.set_page_config(page_title="OB Scanner Pro", page_icon="🎯", layout="wide")


# ───────────────────────── cached helpers ─────────────────────────
@st.cache_resource(ttl=3600, show_spinner=False)
def get_ex(name):
    ex = make_ex(name)
    ex.load_markets()
    return ex


@st.cache_data(ttl=3600, show_spinner=False)
def history_df(exch, sym, tf, days):
    return fetch_history(get_ex(exch), sym, tf, days)


@st.cache_data(ttl=600, show_spinner=False)
def dxy_info():
    try:
        import yfinance as yf
        d = yf.download("DX-Y.NYB", period="3mo", interval="1d", progress=False, auto_adjust=True)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna(subset=["Close"])
        if d.index[-1].date() >= dt.datetime.now(dt.timezone.utc).date():
            d = d.iloc[:-1]
        last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
        return {"bias": "bull" if last > prev else "bear", "pct": (last / prev - 1) * 100}
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def btc_overview(exch, cfg):
    ex = get_ex(exch)
    dd = candles(ex, btc_symbol(exch), "1d", 60)
    bias, why = daily_bias(dd, cfg)
    done = dd.iloc[:-1]
    return {"bias": bias, "why": why, "day_pct": (done.close.iloc[-1] / done.close.iloc[-2] - 1) * 100,
            "today_pct": (float(dd.close.iloc[-1]) / float(dd.open.iloc[-1]) - 1) * 100,
            "price": float(dd.close.iloc[-1])}


def fp(x):
    return "" if x is None or x != x else f"{x:.6g}"


def kill_zone_now(zones):
    h = dt.datetime.now(dt.timezone.utc).hour
    for name, a, b in zones:
        if a <= h < b:
            return name
    return None


def send_telegram(text):
    try:
        tok, chat = st.secrets["TELEGRAM_TOKEN"], st.secrets["TELEGRAM_CHAT_ID"]
    except Exception:
        return "Telegram not configured. Add TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in Secrets."
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=20)
    return "Sent ✅" if r.ok else f"Telegram error: {r.text[:200]}"


# ───────────────────────── sidebar: scan settings (need a re-scan) ─────────────────────────
with st.sidebar:
    st.title("🎯 OB Scanner Pro")
    st.caption("Settings here change what is scanned (press Scan after changing them).")
    exch_name = st.selectbox("Data source", list(SOURCES),
                             help="Binance Futures API is blocked on many cloud servers. OKX, Bybit and Binance Spot work.")
    exch = SOURCES[exch_name]
    tf_choice = st.multiselect("OB timeframes", list(TF_MAP), default=["4H", "1H"])
    ltf_name = st.selectbox("LTF trigger timeframe", ["off", "1H", "15m", "5m"], index=2,
                            help="Lower-timeframe CISD / Turtle soup inside the higher-timeframe OB = precise trigger.")
    min_vol = st.slider("Min 24h volume (million USD)", 0, 500, 20)
    scan_all = st.checkbox("Scan ALL matching coins", False)
    top_n = st.slider("Top N coins by volume", 20, 600, 100, 10, disabled=scan_all)
    prox = st.slider("Max distance to OB zone (%)", 0.5, 10.0, 3.0, 0.5)
    pre_chg = st.slider("Speed-up: skip coins that moved less than ±% from today's open (0 = off)", 0.0, 10.0, 0.0, 0.5)
    with st.expander("Structure rules (all optional)"):
        need_sweep = st.checkbox("OB needs liquidity sweep (torch)", True)
        swing_k = st.slider("Swing sensitivity (candles each side)", 1, 5, 2)
        max_age = st.slider("OB structure-break max age (candles)", 10, 300, 150, 10)
        conf_win = st.slider("Turtle soup / CISD window (candles)", 1, 10, 3)
        rule2 = st.checkbox("Daily bias Rule 2 (rejection wick at FVG/OB)", True)
        skip_pegged = st.checkbox("Skip stable-like coins", True)
    with st.expander("Kill zones (UTC hours)"):
        lo_a, lo_b = st.slider("London", 0, 24, (7, 10))
        ny_a, ny_b = st.slider("New York", 0, 24, (12, 15))
        zones = [("London", lo_a, lo_b), ("New York", ny_a, ny_b)]
    scan_btn = st.button("🔍 Scan now", type="primary", use_container_width=True)

cfg = Cfg(swing_k=swing_k, need_sweep=need_sweep, max_age=max_age, conf_win=conf_win, rule2=rule2,
          skip_pegged=skip_pegged, prox=prox)

# ───────────────────────── market overview ─────────────────────────
st.title("Crypto OB Scanner Pro")
kz = kill_zone_now(zones)
now_utc = dt.datetime.now(dt.timezone.utc)
now_lk = now_utc + dt.timedelta(hours=5, minutes=30)
o1, o2, o3, o4 = st.columns(4)
btc, dxy_i = None, dxy_info()
try:
    btc = btc_overview(exch, cfg)
except Exception as e:
    o1.warning(f"BTC overview unavailable: {e}")
if btc:
    o1.metric("BTC daily bias", "🟢 Bullish" if btc["bias"] == "bull" else "🔴 Bearish",
              f"{btc['day_pct']:+.2f}% last closed day")
    o1.caption(f"{btc['why']} | price {btc['price']:,.1f} | today {btc['today_pct']:+.2f}%")
if dxy_i:
    o2.metric("DXY daily bias", "🟢 Bullish ↑" if dxy_i["bias"] == "bull" else "🔴 Bearish ↓",
              f"{dxy_i['pct']:+.2f}%", delta_color="inverse")
    o2.caption("DXY ↓ is usually supportive for crypto, DXY ↑ is pressure (not always).")
else:
    o2.warning("DXY unavailable")
o3.metric("Kill zone", kz or "none now", f"Sri Lanka {now_lk:%H:%M} | UTC {now_utc:%H:%M}", delta_color="off")
if btc and dxy_i:
    if btc["bias"] == "bull" and dxy_i["bias"] == "bear":
        o4.success("BTC ↑ + DXY ↓: long setups favoured.")
    elif btc["bias"] == "bear" and dxy_i["bias"] == "bull":
        o4.warning("BTC ↓ + DXY ↑: short setups favoured, be careful with longs.")
    else:
        o4.info("BTC and DXY disagree: be selective.")

# ───────────────────────── run scan ─────────────────────────
if scan_btn:
    if not tf_choice:
        st.warning("Select at least one OB timeframe.")
    else:
        p = ScanParams(exch, tuple(tf_choice), "" if ltf_name == "off" else ltf_name, cfg, float(min_vol),
                       0 if scan_all else int(top_n), float(pre_chg))
        bar = st.progress(0.0, text="Scanning ...")
        try:
            st.session_state["scan"] = run_scan(p, lambda x: bar.progress(x, text=f"Scanning {x:.0%}"))
        except Exception as e:
            st.error(f"Scan failed: {e}. Try another data source.")
        bar.empty()

tab_set, tab_chk, tab_vp, tab_guide = st.tabs(["🎯 Setups", "🔎 Coin check", "🧪 VP backtest", "📖 Guide"])

# ───────────────────────── presets & filters ─────────────────────────
PRESETS = {
    "Strict (entry ready)": dict(min_score=7, f_dist=True, dist=0.3, f_bias=True, f_pd=True,
                                 conf=list(CONF_COLS), f_dxy=True, f_kz=False, f_fund=False, min_er=0.2,
                                 min_zone_atr=1.0, min_atr_pct=0.0, min_chg=0.0, min_rvol=0.0, min_rr=1.5),
    "Balanced": dict(min_score=5, f_dist=True, dist=1.0, f_bias=False, f_pd=False, conf=[], f_dxy=False,
                     f_kz=False, f_fund=False, min_er=0.15, min_zone_atr=0.5, min_atr_pct=0.0, min_chg=0.0,
                     min_rvol=0.0, min_rr=1.0),
    "Wide (watchlist)": dict(min_score=3, f_dist=False, dist=3.0, f_bias=False, f_pd=False, conf=[],
                             f_dxy=False, f_kz=False, f_fund=False, min_er=0.0, min_zone_atr=0.0,
                             min_atr_pct=0.0, min_chg=0.0, min_rvol=0.0, min_rr=0.0),
}

with tab_set:
    preset = st.radio("Preset (every filter below can still be changed or removed)", list(PRESETS), horizontal=True)
    P = PRESETS[preset]
    k = lambda name: f"{preset}|{name}"

    with st.expander("⚙️ Filters (each one is optional: untick it or set the slider to 0)", expanded=True):
        a, b, c = st.columns(3)
        f = {}
        f["min_score"] = a.slider("Minimum score", 0, 16, P["min_score"], key=k("min_score"))
        f["f_dist"] = a.checkbox("Price must be close to the OB", P["f_dist"], key=k("f_dist"))
        f["dist"] = a.slider("…how close (%) (0 = inside the zone)", 0.0, 5.0, float(P["dist"]), 0.1, key=k("dist"))
        f["f_bias"] = a.checkbox("OB direction = daily bias", P["f_bias"], key=k("f_bias"))
        f["f_pd"] = a.checkbox("Correct Premium/Discount side", P["f_pd"], key=k("f_pd"))
        f["conf"] = b.multiselect("Required confirmation (any one; empty = off)", list(CONF_COLS),
                                  default=P["conf"], key=k("conf"))
        f["f_dxy"] = b.checkbox("DXY supports the direction", P["f_dxy"], key=k("f_dxy"))
        f["f_kz"] = b.checkbox("Only inside a kill zone", P["f_kz"], key=k("f_kz"))
        f["f_fund"] = b.checkbox("Funding rate supports direction", P["f_fund"], key=k("f_fund"),
                                 help="Long: funding ≤ 0. Short: funding ≥ 0.01%. Perpetual sources only.")
        f["min_er"] = c.slider("Min trend efficiency (ER)", 0.0, 0.6, float(P["min_er"]), 0.05, key=k("min_er"))
        f["min_zone_atr"] = c.slider("Min OB size (ATR)", 0.0, 3.0, float(P["min_zone_atr"]), 0.25, key=k("zone"))
        f["min_atr_pct"] = c.slider("Min ATR % per candle", 0.0, 3.0, float(P["min_atr_pct"]), 0.1, key=k("atr"))
        f["min_chg"] = c.slider("Min move from today's open (±%)", 0.0, 20.0, float(P["min_chg"]), 0.5, key=k("chg"))
        f["min_rvol"] = c.slider("Min relative volume (x)", 0.0, 5.0, float(P["min_rvol"]), 0.1, key=k("rvol"))
        f["min_rr"] = c.slider("Min reward:risk", 0.0, 5.0, float(P["min_rr"]), 0.25, key=k("rr"))
        f["_in_kz"] = kz is not None
        news = st.checkbox("High-impact news today (raises the minimum score by 2)", False)
        if news:
            f["min_score"] += 2

    with st.expander("🧮 Score weights (set a weight to 0 to switch that item off)"):
        wcols = st.columns(4)
        W = {key: wcols[i % 4].number_input(W_LABELS[key], 0.0, 5.0, float(DEFAULT_W[key]), 0.5, key=f"w_{key}")
             for i, key in enumerate(DEFAULT_W)}
        rvol_hi = st.slider("'Relative volume high' means at least (x)", 1.0, 5.0, 1.5, 0.1)

    ctx = {"dxy": dxy_i["bias"] if dxy_i else None, "btc_today": btc["today_pct"] if btc else None,
           "in_kz": kz is not None, "rvol_min": rvol_hi}
    rules = build_rules(f, dxy_i is not None)

    res = st.session_state.get("scan")
    if not res:
        st.info("Press **🔍 Scan now** in the sidebar.")
    else:
        rp = res["params"]
        st.caption(f"{res['n']} of {res['n_universe']} coins scanned on {rp.exch} "
                   f"at {res['time']} UTC | OB TFs {', '.join(rp.tfs)} | LTF trigger {rp.ltf or 'off'} | failed {res['fails']}")
        if rp.cfg != cfg:
            st.warning("Structure settings in the sidebar changed after this scan. Press Scan again to apply them.")
        df = pd.DataFrame(res["rows"])
        if df.empty:
            st.info("No OB near price in this scan. Increase the max distance, add timeframes, or lower the volume filter.")
        else:
            df = score_df(df, W, ctx)
            out, funnel = apply_rules(df, rules)
            st.markdown("**Filter funnel** (shows where coins are lost)")
            st.dataframe(pd.DataFrame(funnel, columns=["Step", "Coins left"]), hide_index=True)
            if out.empty:
                st.info("Nothing passes all enabled filters. Look at the funnel: the first step where it drops to 0 "
                        "is the filter to relax. Or switch to the 'Wide (watchlist)' preset.")
            else:
                out = out.sort_values(["Score", "dist", "brk_ago"], ascending=[False, True, True]).copy()
                show = pd.DataFrame({
                    "Coin": out["coin"], "Dir": np.where(out["side"] == "bull", "🟢 LONG", "🔴 SHORT"),
                    "Score": out["Score"], "TF": out["tf"],
                    "Setup": np.where(out["with_bias"], "With bias", "Pullback"),
                    "OB zone": [f"{fp(a)} – {fp(b)}" for a, b in zip(out["lo"], out["hi"])],
                    "Dist %": out["dist"].round(2), "Price": out["price"].map(fp),
                    "Entry (50% OB)": out["entry"].map(fp), "Stop": out["stop"].map(fp),
                    "Target": out["target"].map(fp), "R:R": out["rr"].round(1),
                    "Confirmations": [", ".join(n for n, col in (("Turtle soup", r.ts), ("CISD", r.cisd),
                                      ("FVG", r.fvg), ("LTF", r.ltf)) if col) or "-" for r in out.itertuples()],
                    "P/D": [f"{z} {p:.0f}%" + (" ✅" if ok else " ⚠️") for z, p, ok in
                            zip(out["pd_zone"], out["pos"], out["pd_ok"])],
                    "RVOL": out["rvol"].round(1), "Funding %": (out["funding"] * 100).round(4),
                    "ER": out["er"].round(2), "Chg open %": out["chg_open"].round(2),
                    "Break ago": out["brk_ago"], "Daily bias": np.where(out["bias"] == "bull", "🟢", "🔴"),
                })
                st.success(f"{len(show)} setups")
                st.dataframe(show, hide_index=True)
                d1, d2 = st.columns([1, 3])
                d1.download_button("⬇️ Download CSV", show.to_csv(index=False), "setups.csv", "text/csv")
                if d2.button("📨 Send top 10 to Telegram"):
                    msg = "OB Scanner setups\n" + "\n".join(
                        f"{r['Dir']} {r['Coin']} {r['TF']} score {r['Score']} | zone {r['OB zone']} | "
                        f"entry {r['Entry (50% OB)']} stop {r['Stop']} tp {r['Target']} (R:R {r['R:R']})"
                        for _, r in show.head(10).iterrows())
                    d2.write(send_telegram(msg))
                st.caption("Entry = 50% of the OB, stop = beyond the far edge + 0.25 ATR, target = nearest opposite swing "
                           "(or 2R if none). Candidates only, not trade advice: confirm on the chart first.")

# ───────────────────────── coin check ─────────────────────────
with tab_chk:
    st.caption("Shows the nearest live OB on every timeframe and tells you exactly which of your filters it fails.")
    c1, c2 = st.columns([3, 1])
    one = c1.text_input("Coin (e.g. BTC, SOL, PEPE)", "").strip().upper()
    c2.write("")
    if c2.button("Check coin") and one:
        try:
            ex1 = get_ex(exch)
            sym = find_symbol(ex1, one)
            src = exch
            if sym is None and exch != "binancespot":
                ex1, src = get_ex("binancespot"), "binancespot"
                sym = find_symbol(ex1, one)
                if sym:
                    st.info(f"{one} is not on {exch_name}; using Binance Spot data.")
            if sym is None:
                st.warning(f"{one} not found.")
            else:
                wide = ScanParams(src, tuple(TF_MAP), "" if ltf_name == "off" else ltf_name,
                                  Cfg(**{**cfg.__dict__, "prox": 50.0, "skip_pegged": False}), 0, 0, 0.0)
                fund = funding_one(ex1, sym) if src != "binancespot" else np.nan
                rows = scan_coin(ex1, sym, one, wide, fund, src)
                if not rows:
                    st.info("No live OB found on any timeframe.")
                else:
                    d = score_df(pd.DataFrame(rows), W, ctx)
                    d["Why not"] = fail_reasons(d, rules)
                    t = pd.DataFrame({
                        "TF": d["tf"], "Dir": np.where(d["side"] == "bull", "🟢 LONG", "🔴 SHORT"),
                        "OB zone": [f"{fp(a)} – {fp(b)}" for a, b in zip(d["lo"], d["hi"])],
                        "Dist %": d["dist"].round(2), "Score": d["Score"], "Entry": d["entry"].map(fp),
                        "Stop": d["stop"].map(fp), "Target": d["target"].map(fp), "R:R": d["rr"].round(1),
                        "ER": d["er"].round(2), "RVOL": d["rvol"].round(1),
                        "Scanner": np.where(d["Why not"] == "", "✅ passes", "❌ " + d["Why not"])})
                    st.write(f"**{one}** daily bias: {'🟢 Bullish' if d['bias'].iloc[0] == 'bull' else '🔴 Bearish'} "
                             f"({d['why'].iloc[0]}) | price {fp(d['price'].iloc[0])} | "
                             f"{d['chg_open'].iloc[0]:+.2f}% from today's open")
                    st.dataframe(t, hide_index=True)
        except Exception as e:
            st.error(f"Could not check {one}: {e}")

# ───────────────────────── VP backtest ─────────────────────────
with tab_vp:
    st.caption("Each UTC day: build the volume profile of the PREVIOUS day, trade the first touch of POC/VAH/VAL "
               "(limit at the level, ATR stop, fixed R target, exit at day end). Tip: Binance Spot downloads fastest.")
    p1, p2, p3, p4 = st.columns(4)
    months = p1.slider("Months", 1, 12, 3)
    vtf = p2.selectbox("Candle size", ["15m", "30m", "1h", "5m"])
    levels = p3.multiselect("Levels", ["POC", "VAH", "VAL"], default=["POC", "VAH", "VAL"])
    va = p4.slider("Value area %", 60, 80, 70)
    p5, p6, p7, p8 = st.columns(4)
    v_stop = p5.slider("Stop (x ATR beyond level)", 0.25, 3.0, 1.0, 0.25)
    v_rr = p6.slider("Target (R)", 0.5, 4.0, 1.5, 0.25)
    v_fee = p7.slider("Fees + slippage (%)", 0.0, 0.3, 0.1, 0.01)
    v_usd = p8.slider("Example risk per trade (USD)", 1, 1000, 10)

    def vp_eval(coin):
        ex1 = get_ex(exch)
        sym = find_symbol(ex1, coin)
        if sym is None:
            return None
        df_ = history_df(exch, sym, vtf, months * 30 + 1)
        tr = backtest_vp(df_, tf_ms(vtf) // 60_000, levels, v_stop, v_rr, va / 100, v_fee)
        return {"coin": coin, "trades": tr, "sum": summarize(tr), "days": int(len(np.unique(df_["ts"].values // 86_400_000)))}

    def vp_row(r):
        z = r["sum"]
        return {"Coin": r["coin"], "Days": r["days"], "Trades": z["trades"], "Win %": round(z["win"], 1),
                "Avg R": round(z["avg_r"], 2), "Profit factor": round(z["pf"], 2), "Total R": round(z["total_r"], 1),
                f"Total USD @{v_usd}": round(z["total_r"] * v_usd), "Max DD (R)": round(z["max_dd"], 1),
                "1st half R": None if z["h1"] != z["h1"] else round(z["h1"], 2), "2nd half R": round(z["h2"], 2)}

    st.markdown("**1. Backtest coins** (one or two, comma separated, e.g. `SOL` or `SOL, WIF`)")
    coins = [x.strip().upper() for x in st.text_input("Coins", "SOL", key="vp_coins").split(",") if x.strip()][:2]
    if st.button("Run backtest") and coins and levels:
        outs = []
        with st.spinner("Downloading candles and backtesting ..."):
            for cn in coins:
                try:
                    r = vp_eval(cn)
                    outs.append(r) if r else st.warning(f"{cn} not found on {exch_name}.")
                except Exception as e:
                    st.error(f"{cn}: {e}")
        st.session_state["vp_out"] = outs
    outs = st.session_state.get("vp_out") or []
    if outs:
        st.dataframe(pd.DataFrame([vp_row(r) for r in outs]), hide_index=True)
        for r in outs:
            if r["sum"]["trades"] < 20:
                st.warning(f"{r['coin']}: only {r['sum']['trades']} trades, too few to trust.")
        curves = []
        for r in outs:
            if r["trades"]:
                t = pd.DataFrame(r["trades"])
                s_ = t["R"].cumsum().reset_index(drop=True)
                s_.name = r["coin"]
                curves.append(s_)
        if curves:
            st.line_chart(pd.concat(curves, axis=1).ffill())
        with st.expander("All trades"):
            for r in outs:
                st.write(r["coin"])
                st.dataframe(pd.DataFrame(r["trades"]), hide_index=True)

    st.markdown("**2. Find coins that respect the volume profile** (backtests many coins)")
    q1, q2, q3 = st.columns(3)
    n_c = q1.slider("Coins to test (top by volume)", 10, 100, 30, 10)
    m_tr = q2.slider("Min trades", 5, 100, 20)
    m_avg = q2.slider("Min avg R", -0.5, 1.0, 0.1, 0.05)
    m_pf = q3.slider("Min profit factor", 0.0, 3.0, 1.2, 0.1)
    both = q3.checkbox("Both halves profitable", True)
    if st.button("Scan coins for VP edge") and levels:
        try:
            uni = universe(get_ex(exch), 0)[:n_c]
        except Exception as e:
            uni = []
            st.error(f"Could not load coins: {e}")
        bar, rows_, fails = st.progress(0.0), [], 0
        for i, (_, s_, b_) in enumerate(uni):
            bar.progress((i + 1) / len(uni), text=b_)
            try:
                df_ = history_df(exch, s_, vtf, months * 30 + 1)
                tr = backtest_vp(df_, tf_ms(vtf) // 60_000, levels, v_stop, v_rr, va / 100, v_fee)
                rows_.append({"coin": b_, "trades": tr, "sum": summarize(tr),
                              "days": int(len(np.unique(df_["ts"].values // 86_400_000)))})
            except Exception:
                fails += 1
        bar.empty()
        st.session_state["vp_scan"] = {"rows": rows_, "fails": fails}
    rs = st.session_state.get("vp_scan")
    if rs:
        def ok(z):
            r_ = z["trades"] >= m_tr and z["avg_r"] >= m_avg and z["pf"] >= m_pf
            return r_ and (not both or (z["h1"] == z["h1"] and z["h1"] > 0 and z["h2"] > 0))
        good = [vp_row(r) for r in rs["rows"] if ok(r["sum"])]
        if good:
            st.dataframe(pd.DataFrame(good).sort_values(["Avg R", "Trades"], ascending=False), hide_index=True)
        else:
            st.info("No coin passed. Lower the limits.")
        st.caption("Picking the best of many coins from one period flatters the numbers. Treat this as a candidate "
                   "list and re-test each coin on a longer period. Past results do not guarantee the future.")

# ───────────────────────── guide ─────────────────────────
with tab_guide:
    st.markdown("""
### භාවිතය (සරලව)
1. වම් sidebar එකේ data source, timeframes, volume තෝරලා **Scan now** එබුවාම coins scan වෙනවා.
2. **Setups** tab එකේ preset එකක් තෝරන්න: **Strict** (දැන්ම entry), **Balanced**, **Wide** (watchlist).
3. හැම filter එකක්ම optional. ටික් අයින් කරන්න, නැත්නම් slider එක 0 කරන්න. **Filter funnel** table එකෙන් coins ඔක්කොම නැති වෙන්නේ කුමන filter එකෙන්ද කියලා පේනවා.
4. Filters වෙනස් කළාම නැවත scan කරන්න ඕන නැහැ. Sidebar එකේ structure settings වෙනස් කළොත් විතරයි නැවත scan.

### Score එක
OB එක ළඟ (+2), zone එක ඇතුළේ (+1), daily bias එකට ගැළපෙනවා (+1), හරි Premium/Discount (+1), Turtle soup (+2), CISD (+2), FVG (+1), **LTF trigger** (+2), වැඩි volume (RVOL) (+1), Funding හිතකරයි (+1), DXY හිතකරයි (±1), Kill zone (+1), BTC ට වඩා ශක්තිමත් (+1). Weights ඔයාටම වෙනස් කරන්න පුළුවන්.

### Telegram alerts
Streamlit Cloud → App settings → Secrets වල `TELEGRAM_TOKEN` සහ `TELEGRAM_CHAT_ID` දාන්න.

### අවවාදය
මේ candidates ලැයිස්තුවක් විතරයි, trade advice නෙවෙයි. Chart එකේ තහවුරු කරලා, backtest කරලා පාවිච්චි කරන්න.
""")
st.caption("Candidates only, not trade advice. Confirm on the chart and backtest before trading real money.")
