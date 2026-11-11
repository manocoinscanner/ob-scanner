import time

import numpy as np
import pandas as pd
import streamlit as st

from core.ai import providers, review_coin
from core.data import SOURCES
from core.rank import run_rank

st.set_page_config(page_title="Coin Ranker", page_icon="🏆", layout="wide")
st.title("🏆 Coin Ranker")
st.caption("Premium/Discount + DOM (order book) + order flow confirm වුණ coins විතරක් තෝරලා, ඊට පස්සේ technical analysis බලනවා.")

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
                         ("persist", np.nan), ("pd_pos", np.nan), ("pd_zone", ""), ("pd_ok", True)):
        if col not in df.columns:
            df[col] = default
    st.caption(f"{meta['n']} coins scanned at {meta['time']} UTC | BTC 24h {meta['btc_roc']:+.2f}% | failed {meta['fails']}")
    if df.empty:
        st.warning("දත්ත ලැබුණේ නැහැ. වෙනත් Data source එකක් තෝරන්න.")
    else:
        # technical score = total score without the DOM / Flow / Derivs parts (max 100)
        df["tech"] = (df["score"] - df["bonus"].astype(float) - df["dom"].astype(float) - df["flow"].astype(float)).round(1)

        st.markdown("### 🔒 Confirm නීති (Premium/Discount + DOM + Flow, ඊට පස්සේ Technical)")
        r0, r1, r2, r3 = st.columns(4)
        need_pd = r0.checkbox("Premium/Discount අනිවාර්යයි", True,
                              help="Long = මිල range එකේ Discount (පහළ භාගය) තුළ. Short = Premium (ඉහළ භාගය) තුළ. Range = අන්තිම 4H candles 100.")
        need_dom = r1.checkbox("DOM (order book) confirm අනිවාර්යයි", True,
                               help="Order book data ලැබෙන්න ඕන, DOM ලකුණු ධනාත්මක, TP1 ට හරහට wall නැහැ, support wall එක අතුරුදහන් වෙලා නැහැ.")
        need_flow = r2.checkbox("Order flow (taker flow) confirm අනිවාර්යයි", True,
                                help="Taker flow data ලැබෙන්න ඕන සහ Flow ලකුණු ධනාත්මක.")
        need_setup = r3.checkbox("Technical: Breakout / Pullback setup අනිවාර්යයි", True)
        min_tech = st.slider("Technical ලකුණු අවම (100න්, DOM/Flow/Derivs නැතුව)", 0, 100, 50)

        side = st.radio("Direction", ["All", "LONG", "SHORT"], horizontal=True)

        work = df
        funnel = [("Scan result (සියලු coins)", len(work))]
        if side != "All":
            work = work[work["dir"] == side]
            funnel.append((f"Direction = {side}", len(work)))
        if need_pd:
            work = work[work["pd_ok"].astype(bool)]
            funnel.append(("Premium/Discount හරි (Long=Discount, Short=Premium) ✅", len(work)))
        if need_dom:
            work = work[(work["dom_share"].astype(float).notna()) & (work["dom"].astype(float) > 0)
                        & (~work["blocks"].astype(bool)) & (work["persist"].astype(float).fillna(1.0) != 0.0)]
            funnel.append(("DOM confirm ✅", len(work)))
        if need_flow:
            work = work[(work["flow_share"].astype(float).notna()) & (work["flow"].astype(float) > 0)]
            funnel.append(("Order flow confirm ✅", len(work)))
        if need_setup:
            work = work[work["setup"] != "-"]
            funnel.append(("Technical: setup තියෙනවා", len(work)))
        if min_tech > 0:
            work = work[work["tech"] >= min_tech]
            funnel.append((f"Technical ලකුණු ≥ {min_tech}", len(work)))
        work = work.sort_values("score", ascending=False)

        st.markdown("**Filter funnel** (coins ඉතුරු වෙන ප්‍රමාණය පියවරෙන් පියවර)")
        st.dataframe(pd.DataFrame(funnel, columns=["පියවර", "Coins left"]), hide_index=True)
        if not meta.get("dom"):
            st.warning("මේ scan එකේ Order book (DOM) data කිසිම coin එකකට ලැබුණේ නැහැ. DOM confirm අනිවාර්ය නම් ලැයිස්තුව හිස් වෙනවා.")
        if not meta.get("flow"):
            st.warning("මේ scan එකේ Taker flow data කිසිම coin එකකට ලැබුණේ නැහැ. Flow confirm අනිවාර්ය නම් ලැයිස්තුව හිස් වෙනවා.")

        d = work.head(show_n)
        if d.empty:
            st.info("Confirm වුණ coin එකක් නැහැ. Funnel table එකේ Coins left 0 වෙන පළවෙනි පියවර බලලා, ඒ නීතියේ ටික් එක අයින් කරන්න "
                    "(නැත්නම් Technical ලකුණු අවම අඩු කරන්න). DOM/Flow data එන්නේ ලකුණු අනුව ඉහළම coins 40ට විතරයි.")
        else:
            fmt = lambda x: f"{x:.6g}"
            persist_txt = d["persist"].astype(float).map(lambda x: "" if x != x else ("✅ yes" if x == 1 else "⚠️ vanished (spoof?)"))
            pd_txt = [("" if p != p else f"{z} {p:.0f}%" + (" ✅" if ok else " ⚠️ වැරදි පැත්ත"))
                      for z, p, ok in zip(d["pd_zone"], d["pd_pos"].astype(float), d["pd_ok"].astype(bool))]
            table = pd.DataFrame({
                "#": range(1, len(d) + 1), "Coin": d["coin"],
                "Dir": d["dir"].map({"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}), "P/D zone": pd_txt, "Score": d["score"],
                "Tech (100)": d["tech"], "Setup": d["setup"], "Price": d["price"].map(fmt), "24h %": d["chg24"].round(2),
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
            st.success(f"{len(d)} coins confirm වුණා")
            st.dataframe(table, hide_index=True)
            st.download_button("⬇️ Download CSV", table.to_csv(index=False), "ranking.csv", "text/csv")

        st.markdown("""
**ලකුණු (උපරිමය 135)**: 4H trend 25 + 1H trend 10 + පිරිසිදු trend 15 + Volume 15 + BTC ට වඩා ශක්තිමත් 10 + Setup 25 (Breakout හෝ Pullback) + Derivs 15 + DOM 10 + Flow 5. **Tech (100)** = Derivs, DOM, Flow අයින් කළාම ඉතුරු ලකුණු.
**Premium/Discount**: අන්තිම 4H candles 100ක ඉහළම/පහළම මිල අතර range එකේ, මිල 50%ට පහළ නම් **Discount** (Long entry වලට හරි), 50%ට ඉහළ නම් **Premium** (Short entry වලට හරි). මේක ලකුණු වලට එකතු වෙන්නේ නැහැ, අනිවාර්ය නීතියක් (ටික් කරලා අයින් කරන්න පුළුවන්).
**DOM confirm**: Order book data ලැබිලා, ඔයාගේ දිශාවට support wall එකක් (මිලට 2% ඇතුළත) හෝ bid/ask බර තියෙන්න ඕන; TP1 ට හරහට විරුද්ධ wall එකක් නොතියෙන්න ඕන; support wall එක දෙවෙනි snapshot එකේ අතුරුදහන් වෙලා නොතියෙන්න ඕන.
**Order flow confirm**: Taker flow data ලැබිලා, Long එකකට buyers ≥55% හෝ Short එකකට sellers ආක්‍රමණශීලී (buyers ≤45%) වෙන්න ඕන. මේක අන්තිම ගනුදෙනු 500ක කෙටි කාලයක් විතරයි (Flow window column එකේ පේනවා).
**Derivs +**: Open Interest වැඩි වෙමින් මිලත් ඒ දිශාවටම යනවා නම් +10, Funding rate එක ඒ දිශාවට හොඳයි නම් +5 (OKX / Bybit වලට විතරයි).
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
        elif d.empty:
            st.info("AI review කරන්න coins නැහැ (confirm වුණ coin එකක් නැහැ).")
        else:
            st.caption("Connected: " + " → ".join(p["name"] for p in provs) +
                       " (එකක් limit වුණොත් ඊළඟ එකට ස්වයංක්‍රීයව මාරු වෙනවා) | AI review කරන්නේ confirm වුණ coins විතරයි")
            a1, a2 = st.columns([3, 1])
            k_ai = a1.slider("Confirm වුණ coins කීයක් AI එකෙන් check කරන්නද", 1, 25, min(10, max(1, len(d))))
            a2.write("")
            if a2.button("🤖 Run AI review"):
                cache = st.session_state.setdefault("ai_cache", {})
                bar2 = st.progress(0.0, text="AI reviewing ...")
                top_rows = d.head(k_ai).to_dict("records")
                outs = []
                for i, r in enumerate(top_rows):
                    key = f"{r['coin']}|{r['dir']}|{r['score']}|{r['setup']}|{r['dom']}|{r['flow']}|{r['pd_zone']}"
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

            st.markdown("### ✅ Final list (P/D + DOM + Flow + Technical + AI)")
            g1, g2, g3 = st.columns(3)
            n_checks = max((len(o["sanity"]) for o in ai_out), default=6)
            min_checks = g1.slider(f"Code checks අවම ({n_checks}න්)", 0, n_checks, min(n_checks - 1, 5))
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
                st.info("AI + Code දෙකම පාස් වුණ coin එකක් නැහැ. Sliders ලිහිල් කරන්න, නැත්නම් REJECT ඒවාත් පෙන්නන්න.")
            else:
                st.dataframe(pd.DataFrame([{
                    "Coin": o["coin"], "Dir": o["dir"],
                    "P/D": (f"{o['row'].get('pd_zone')} {o['row'].get('pd_pos'):.0f}%" if o["row"].get("pd_pos") is not None else ""),
                    "Final": icon.get(str(o.get("verdict")).upper(), o.get("verdict")),
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
