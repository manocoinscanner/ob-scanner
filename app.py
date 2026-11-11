import time

import numpy as np
import pandas as pd
import streamlit as st

from core.ai import providers, review_coin
from core.data import SOURCES
from core.rank import run_rank

st.set_page_config(page_title="Coin Ranker", page_icon="🏆", layout="wide")
st.title("🏆 Coin Ranker")
st.caption("හැම coin එකකටම ලකුණු දීලා ඉහළම ඒවා පෙන්නනවා. Filters නැහැ, ඒ නිසා ලැයිස්තුව කවදාවත් හිස් වෙන්නේ නැහැ.")

c1, c2, c3, c4 = st.columns(4)
exch_name = c1.selectbox("Data source", list(SOURCES)[::-1])
exch = SOURCES[exch_name]
min_vol = c2.slider("Min 24h volume (million USD)", 0, 200, 20)
top_n = c3.slider("Coins to scan (by volume)", 30, 300, 100, 10)
show_n = c4.slider("Show top", 10, 100, 25, 5)

if st.button("🔍 Scan now", type="primary"):
    bar = st.progress(0.0, text="Scanning ...")
    try:
        st.session_state["rank"] = run_rank(exch, min_vol, top_n, lambda x: bar.progress(min(x, 1.0), text=f"{x:.0%}"))
    except Exception as e:
        st.error(f"Scan failed: {e}. වෙනත් Data source එකක් තෝරන්න.")
    bar.empty()

res = st.session_state.get("rank")
if res:
    df, meta = res
    df = df.copy()
    # පරණ scan ප්‍රතිඵලයක columns නැති වුණත් නොකැඩෙන්න
    for col, default in (("oi24", np.nan), ("oi4", np.nan), ("funding", np.nan), ("bonus", 0.0), ("sym", ""),
                         ("dom_share", np.nan), ("bid_wall_x", np.nan), ("bid_wall_dist", np.nan),
                         ("ask_wall_x", np.nan), ("ask_wall_dist", np.nan), ("blocks", False), ("dom", 0.0),
                         ("cover", np.nan), ("flow_share", np.nan), ("flow_win", np.nan), ("flow", 0.0),
                         ("persist", np.nan)):
        if col not in df.columns:
            df[col] = default
    st.caption(f"{meta['n']} coins scanned at {meta['time']} UTC | BTC 24h {meta['btc_roc']:+.2f}% | failed {meta['fails']}")
    if df.empty:
        st.warning("දත්ත ලැබුණේ නැහැ. වෙනත් Data source එකක් තෝරන්න.")
    else:
        f1, f2 = st.columns(2)
        side = f1.radio("Direction", ["All", "LONG", "SHORT"], horizontal=True)
        only_setup = f2.checkbox("Breakout / Pullback setup තියෙන ඒවා විතරයි (optional)", False)
        d = df
        if side != "All":
            d = d[d["dir"] == side]
        if only_setup:
            d = d[d["setup"] != "-"]
        d = d.head(show_n)
        fmt = lambda x: f"{x:.6g}"
        persist_txt = d["persist"].astype(float).map(lambda x: "" if x != x else ("✅ yes" if x == 1 else "⚠️ vanished (spoof?)"))
        table = pd.DataFrame({
            "#": range(1, len(d) + 1), "Coin": d["coin"],
            "Dir": d["dir"].map({"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}), "Score": d["score"],
            "Setup": d["setup"], "Price": d["price"].map(fmt), "24h %": d["chg24"].round(2),
            "vs BTC %": d["vs_btc"].round(2), "RVOL": d["rvol"].round(1), "RSI 1H": d["rsi"].round(0),
            "Trend ER": d["er"].round(2),
            "OI 24h %": d["oi24"].astype(float).round(1), "OI 4h %": d["oi4"].astype(float).round(1),
            "Funding %": (d["funding"].astype(float) * 100).round(4), "Derivs +": d["bonus"],
            "Bid share %": (d["dom_share"].astype(float) * 100).round(0),
            "Bid wall (× / dist%)": [("" if x != x else f"{x:.0f}× / {y:.2f}%") for x, y in zip(d["bid_wall_x"].astype(float), d["bid_wall_dist"].astype(float))],
            "Ask wall (× / dist%)": [("" if x != x else f"{x:.0f}× / {y:.2f}%") for x, y in zip(d["ask_wall_x"].astype(float), d["ask_wall_dist"].astype(float))],
            "Wall blocks TP1": np.where(d["blocks"].astype(bool), "⚠️ yes", ""),
            "Wall persists": persist_txt, "DOM +": d["dom"],
            "Taker buy %": (d["flow_share"].astype(float) * 100).round(0),
            "Flow window (min)": d["flow_win"].astype(float).round(1), "Flow +": d["flow"],
            "Entry": d["entry"].map(fmt), "Stop": d["stop"].map(fmt),
            "TP1 (1.5R)": d["tp1"].map(fmt), "TP2 (3R)": d["tp2"].map(fmt), "ලකුණු බෙදීම": d["why"],
        })
        if not meta.get("dom"):
            st.info("Order book (DOM) data ගන්න බැරි වුණා, නැත්නම් ඒ coins වල book එක ඉතා තුනී. ඒ නිසා DOM ලකුණු 0.")
        if not meta.get("flow"):
            st.info("Taker flow (CVD) data ගන්න බැරි වුණා, නැත්නම් ගනුදෙනු කාල පරාසය ඉතා කෙටියි. ඒ නිසා Flow ලකුණු 0.")
        if not meta.get("derivs"):
            st.info("OI / Funding data නැහැ. Data source එක **OKX** හෝ **Bybit** තෝරන්න (Binance Spot වල derivatives නැහැ).")
        st.dataframe(table, hide_index=True)
        st.download_button("⬇️ Download CSV", table.to_csv(index=False), "ranking.csv", "text/csv")
        st.markdown("""
**ලකුණු (උපරිමය 135)**: 4H trend 25 + 1H trend 10 + පිරිසිදු trend 15 + Volume 15 + BTC ට වඩා ශක්තිමත් 10 + Setup 25 (Breakout හෝ Pullback) + Derivs 15 + DOM 10 + Flow 5 + (wall persistence).
**Derivs +**: Open Interest වැඩි වෙමින් මිලත් ඒ දිශාවටම යනවා නම් +10, Funding rate එක ඒ දිශාවට හොඳයි නම් +5 (OKX / Bybit වලට විතරයි).
**DOM +** (order book): ඔයාගේ දිශාවට උපකාරක විශාල limit order "බිත්තියක්" (wall) මිල ළඟ තියෙනවා නම් +5, bid/ask ප්‍රමාණය ඔයාගේ දිශාවට බරයි නම් +5, TP1 ට යන පාරේ විරුද්ධ wall එකක් තියෙනවා නම් −5. Wall = ඒ පැත්තේ මධ්‍යස්ථ order එකට වඩා 8 ගුණයක් විශාල order එකක්.
**Wall persists**: Order book එක දෙපාරක් (තත්පර කීපයක් පරතරයකින්) ගන්නවා. උපකාරක wall එක එහෙමම තියෙනවා නම් ✅. අතුරුදහන් වෙලා නම් "vanished (spoof?)" කියලා පෙන්නලා ඒ +5 අයින් කරනවා.
**Flow +** (taker flow / CVD): අන්තිම ගනුදෙනු වලින් buyers ආක්‍රමණශීලීව ගහපු කොටස. Long එකකට ≥55% නම් +5, ≤40% නම් −5 (Short එකට අනිත් පැත්ත). මේක අන්තිම ගනුදෙනු 500-1000ක කෙටි කාලයක් විතරයි (Flow window column එකේ පේනවා).
**Stop** = 1H ATR එකෙන් 1.5ක්, **TP1** = 1.5R, **TP2** = 3R.
**Breakout** = 4H එකේ අන්තිම 20 candles ඉහළම/පහළම කඩලා close වීම. **Pullback** = trend එක තියෙද්දී 1H EMA20 ළඟට ආපහු ඇවිත් RSI මධ්‍යස්ථව තියෙන තැන.
""")

        # ───────────── AI review (optional, free AIs with automatic fallback) ─────────────
        st.subheader("🤖 AI review (optional)")

        def get_secret(name):
            try:
                return st.secrets.get(name)
            except Exception:
                return None

        provs = providers(get_secret)
        if not provs:
            st.info("AI keys තවම දාලා නැහැ. Streamlit → Manage app → Settings → Secrets වල අවම වශයෙන් එක key එකක් දාන්න "
                    "(`GEMINI_API_KEY`, `GROQ_API_KEY`). ඒවා නොමිලේ ගන්න පුළුවන්.")
        else:
            st.caption("Connected: " + " → ".join(p["name"] for p in provs) +
                       " (එකක් limit වුණොත් ඊළඟ එකට ස්වයංක්‍රීයව මාරු වෙනවා)")
            a1, a2 = st.columns([3, 1])
            k_ai = a1.slider("Code එකෙන් ඉහළට ආපු coins කීයක් AI එකෙන් check කරන්නද", 1, 25, 10)
            a2.write("")
            if a2.button("🤖 Run AI review"):
                cache = st.session_state.setdefault("ai_cache", {})
                bar2 = st.progress(0.0, text="AI reviewing ...")
                top_rows = d.head(k_ai).to_dict("records")
                outs = []
                for i, r in enumerate(top_rows):
                    key = f"{r['coin']}|{r['dir']}|{r['score']}|{r['setup']}"
                    o = cache.get(key)
                    if o is None:  # same coin + same signal is not asked twice (saves free limits)
                        o = review_coin(r, provs)
                        if o.get("verdict") != "ERROR":
                            cache[key] = o
                        time.sleep(2)
                    outs.append(o)
                    bar2.progress((i + 1) / len(top_rows))
                bar2.empty()
                st.session_state["ai"] = outs

        ai_out = st.session_state.get("ai")
        if ai_out:
            icon = {"AGREE": "🟢 AGREE", "CAUTION": "🟡 CAUTION", "REJECT": "🔴 REJECT", "ERROR": "⚠️ ERROR"}
            order = {"AGREE": 0, "CAUTION": 1, "REJECT": 2, "ERROR": 3}

            st.markdown("### ✅ Final list (Code + AI දෙකටම අනුව)")
            g1, g2, g3 = st.columns(3)
            min_checks = g1.slider("Code checks අවම (5න්)", 0, 5, 4)
            min_conf = g2.slider("AI confidence අවම", 0, 100, 50)
            show_rej = g3.checkbox("REJECT / ERROR ඒවාත් පෙන්නන්න", False)
            final = []
            for o in ai_out:
                v = str(o.get("verdict")).upper()
                passed = sum(ok for _, ok in o["sanity"])
                conf = o.get("confidence") or 0
                if show_rej or (v in ("AGREE", "CAUTION") and passed >= min_checks and conf >= min_conf):
                    final.append((order.get(v, 3), -conf, -(o["row"].get("score") or 0), o))
            final.sort(key=lambda x: x[:3])
            if not final:
                st.info("Code + AI දෙකම පාස් වුණ coin එකක් නැහැ. Sliders ලිහිල් කරන්න, නැත්නම් REJECT ඒවාත් පෙන්නන්න.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Coin": o["coin"], "Dir": o["dir"], "Final": icon.get(str(o.get("verdict")).upper(), o.get("verdict")),
                    "AI confidence": o.get("confidence"), "Code score": o["row"].get("score"),
                    "Code checks": f"{sum(ok for _, ok in o['sanity'])}/{len(o['sanity'])}",
                    "Entry": fmt(o["row"]["entry"]), "Stop": fmt(o["row"]["stop"]),
                    "TP1": fmt(o["row"]["tp1"]), "TP2": fmt(o["row"]["tp2"]),
                    "AI": o.get("provider"), "AI summary": o.get("summary")} for *_, o in final]), hide_index=True)

            with st.expander("AI results (සියල්ල) සහ විස්තර"):
                st.dataframe(pd.DataFrame([{
                    "Coin": o["coin"], "Dir": o["dir"],
                    "AI verdict": icon.get(str(o.get("verdict")).upper(), o.get("verdict")),
                    "Confidence": o.get("confidence"), "AI": o.get("provider"),
                    "Code checks": f"{sum(ok for _, ok in o['sanity'])}/{len(o['sanity'])}",
                    "Summary": o.get("summary")} for o in ai_out]), hide_index=True)
                for o in ai_out:
                    st.markdown(f"**{o['coin']} {o['dir']}**")
                    for name, ok in o["sanity"]:
                        st.write(("✅ " if ok else "❌ ") + name)
                    for c in o.get("checks") or []:
                        st.write(("✅ " if c.get("ok") else "❌ ") + f"{c.get('rule')}: {c.get('note')}")
                    if o.get("risks"):
                        st.write("Risks: " + "; ".join(map(str, o["risks"])))
                    if o.get("errors"):
                        st.caption("Provider notes: " + " | ".join(o["errors"]))
            st.caption("AI එකට පේන්නේ scanner දත්ත විතරයි, chart එක නෙවෙයි. එයාට වැරදිය හැකියි. Final list එක candidates ලැයිස්තුවක්, trade advice නෙවෙයි.")
else:
    st.info("**Scan now** එබුවාම ලැයිස්තුව එනවා.")

st.caption("Candidates only, not trade advice. Chart එකේ තහවුරු කරලා, ඔයාගේ risk එකට අනුව පාවිච්චි කරන්න. ලාභය සහතික නැහැ.")
