import numpy as np
import pandas as pd
import streamlit as st

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
    for col, default in (("oi24", np.nan), ("oi4", np.nan), ("funding", np.nan), ("bonus", 0.0), ("sym", "")):
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
        table = pd.DataFrame({
            "#": range(1, len(d) + 1), "Coin": d["coin"],
            "Dir": d["dir"].map({"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}), "Score": d["score"],
            "Setup": d["setup"], "Price": d["price"].map(fmt), "24h %": d["chg24"].round(2),
            "vs BTC %": d["vs_btc"].round(2), "RVOL": d["rvol"].round(1), "RSI 1H": d["rsi"].round(0),
            "Trend ER": d["er"].round(2),
            "OI 24h %": d["oi24"].astype(float).round(1), "OI 4h %": d["oi4"].astype(float).round(1),
            "Funding %": (d["funding"].astype(float) * 100).round(4), "Derivs +": d["bonus"],
            "Entry": d["entry"].map(fmt), "Stop": d["stop"].map(fmt),
            "TP1 (1.5R)": d["tp1"].map(fmt), "TP2 (3R)": d["tp2"].map(fmt), "ලකුණු බෙදීම": d["why"],
        })
        if not meta.get("derivs"):
            st.info("OI / Funding data නැහැ. Data source එක **OKX** හෝ **Bybit** තෝරන්න (Binance Spot වල derivatives නැහැ).")
        st.dataframe(table, hide_index=True)
        st.download_button("⬇️ Download CSV", table.to_csv(index=False), "ranking.csv", "text/csv")
        st.markdown("""
**ලකුණු (උපරිමය 115)**: 4H trend 25 + 1H trend 10 + පිරිසිදු trend 15 + Volume 15 + BTC ට වඩා ශක්තිමත් 10 + Setup 25 (Breakout හෝ Pullback) + Derivs 15.
**Derivs +**: Open Interest වැඩි වෙමින් මිලත් ඒ දිශාවටම යනවා නම් +10, Funding rate එක ඒ දිශාවට හොඳයි නම් +5 (OKX / Bybit වලට විතරයි).
**Stop** = 1H ATR එකෙන් 1.5ක්, **TP1** = 1.5R, **TP2** = 3R.
**Breakout** = 4H එකේ අන්තිම 20 candles ඉහළම/පහළම කඩලා close වීම. **Pullback** = trend එක තියෙද්දී 1H EMA20 ළඟට ආපහු ඇවිත් RSI මධ්‍යස්ථව තියෙන තැන.
""")
else:
    st.info("**Scan now** එබුවාම ලැයිස්තුව එනවා.")

st.caption("Candidates only, not trade advice. Chart එකේ තහවුරු කරලා, ඔයාගේ risk එකට අනුව පාවිච්චි කරන්න. ලාභය සහතික නැහැ.")
