import numpy as np
import pandas as pd
import streamlit as st

from core.data import SOURCES
from core.rsi_div import TFS, DivCfg, run_scan

st.set_page_config(page_title="RSI Divergence Scanner", page_icon="📉", layout="wide")
st.title("📉 RSI Divergence Scanner")
st.caption("Timeframes ගණනාවක RSI divergence සහ overbought / oversold coins හොයනවා. "
           "මෙය වෙනම scanner එකක්, Coin Ranker එකට බලපාන්නේ නැහැ.")

a1, a2, a3, a4 = st.columns(4)
exch_name = a1.selectbox("Data source", list(SOURCES)[::-1])
exch = SOURCES[exch_name]
min_vol = a2.slider("Min 24h volume (million USD)", 0, 200, 20)
top_n = a3.slider("Coins to scan (by volume)", 20, 200, 60, 10)
tfs = a4.multiselect("Timeframes", TFS, default=["5m", "15m", "1h", "4h"])

with st.expander("⚙️ Divergence සැකසුම් (සියල්ල වෙනස් කරන්න පුළුවන්)", expanded=False):
    b1, b2, b3 = st.columns(3)
    k = b1.slider("Pivot සංවේදීතාව (candles දෙපැත්තේ)", 2, 6, 3,
                  help="වැඩි නම් විශාල swing විතරයි. අඩු නම් වැඩියෙන් සංඥා එනවා.")
    max_age = b1.slider("අන්තිම pivot එක උපරිම candles කීයක් පරණද", 3, 40, 15)
    min_gap = b2.slider("Pivots දෙක අතර අවම candles", 3, 20, 5)
    max_gap = b2.slider("Pivots දෙක අතර උපරිම candles", 20, 120, 60)
    min_diff = b3.slider("RSI වෙනස අවම (pivots දෙක අතර)", 0.0, 10.0, 2.0, 0.5)
    hidden = b3.checkbox("Hidden divergence ත් හොයන්න (trend continuation)", False)
    need_ext = st.checkbox("Regular divergence එකට පළමු pivot එකේ RSI overbought / oversold වෙන්න ඕන", True)
    e1, e2 = st.columns(2)
    ob = e1.slider("Overbought මට්ටම", 60, 85, 65)
    os_ = e2.slider("Oversold මට්ටම", 15, 40, 35)

cfg = DivCfg(k=k, max_age=max_age, min_gap=min_gap, max_gap=max_gap, min_rsi_diff=float(min_diff),
             need_extreme=need_ext, ob=float(ob), os=float(os_), hidden=hidden)

if st.button("🔍 Scan now", type="primary"):
    if not tfs:
        st.warning("අවම වශයෙන් timeframe එකක් තෝරන්න.")
    else:
        bar = st.progress(0.0, text="Scanning ...")
        try:
            st.session_state["rsi_scan"] = run_scan(exch, tuple(tfs), cfg, min_vol, top_n,
                                                    lambda x: bar.progress(min(x, 1.0), text=f"{x:.0%}"))
        except Exception as e:
            st.error(f"Scan failed: {e}. වෙනත් Data source එකක් තෝරන්න.")
        bar.empty()

res = st.session_state.get("rsi_scan")
if not res:
    st.info("**Scan now** එබුවාම ප්‍රතිඵල එනවා. Timeframes වැඩි වෙන තරමට වැඩි වෙලාවක් යනවා (coins × timeframes ගණනට requests යනවා).")
else:
    st.caption(f"{res['n']} coins | TF: {', '.join(res['tfs'])} | {res['time']} UTC | errors: {len(res['errors'])}")
    if res["errors"]:
        with st.expander("Errors (පළමු 10)"):
            st.write("\n".join(res["errors"][:10]))

    tab1, tab2 = st.tabs(["📉 Divergence", "🌡 RSI overbought / oversold"])

    with tab1:
        df = res["div"]
        if df.empty:
            st.info("මේ scan එකේ divergence එකක් හම්බවුණේ නැහැ. සැකසුම් ලිහිල් කරන්න: "
                    "\"overbought/oversold වෙන්න ඕන\" ටික් එක අයින් කරන්න, pivot සංවේදීතාව අඩු කරන්න, හෝ උපරිම පරණකම වැඩි කරන්න.")
        else:
            f1, f2, f3 = st.columns(3)
            dirs = f1.multiselect("Direction", ["LONG", "SHORT"], default=["LONG", "SHORT"])
            tf_sel = f2.multiselect("Timeframe", res["tfs"], default=res["tfs"])
            min_cnt = f3.slider("අවම timeframes ගණන (එකම coin, එකම දිශාව)", 1, max(1, len(res["tfs"])), 1)
            d = df[df["dir"].isin(dirs) & df["tf"].isin(tf_sel) & (df["tf_count"] >= min_cnt)]
            fmt = lambda x: f"{x:.6g}"
            if d.empty:
                st.info("Filters වලට ගැළපෙන දෙයක් නැහැ.")
            else:
                table = pd.DataFrame({
                    "Coin": d["coin"], "TF": d["tf"],
                    "Dir": d["dir"].map({"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}),
                    "Type": d["side"] + " " + d["kind"],
                    "Price pivots": [f"{fmt(a)} → {fmt(b)}" for a, b in zip(d["p1"], d["p2"])],
                    "RSI pivots": [f"{a:.0f} → {b:.0f}" for a, b in zip(d["r1"], d["r2"])],
                    "Pivot ago (candles)": d["ago"], "Gap (candles)": d["gap"],
                    "Same-dir TFs": d["tf_count"], "Price": d["price"].map(fmt), "Vol 24h (M$)": d["vol_m"].round(0),
                })
                st.success(f"{len(table)} signals ({d['coin'].nunique()} coins)")
                st.dataframe(table, hide_index=True)
                st.download_button("⬇️ Download CSV", table.to_csv(index=False), "rsi_divergence.csv", "text/csv")
        st.markdown("""
**Bearish regular divergence**: මිල ඉහළට ගිහින් අලුත් **higher high** එකක් හැදුවා, ඒත් RSI **lower high** එකක් (දුර්වල වෙනවා). ඒක ඉහළ යන ගමන දුර්වල වෙන බවට සලකුණක්, SHORT පැත්තට.
**Bullish regular divergence**: මිල **lower low** එකක් හැදුවා, ඒත් RSI **higher low** එකක්. ඒක LONG පැත්තට සලකුණක්.
**Hidden**: Bearish hidden = මිල lower high, RSI higher high (පහළ trend එක දිගටම යන්න ඉඩ). Bullish hidden = මිල higher low, RSI lower low.
**Same-dir TFs**: එකම coin එකට එකම දිශාවට divergence timeframes කීයකද තියෙන්නේ. වැඩි නම් සංඥාව ශක්තිමත්ය කියලා සැලකිය හැකියි, ඒත් සහතිකයක් නෙවෙයි.
Divergence එක අන්තිම **closed** candles වලින් විතරයි ගණනය කරන්නේ (හැදෙමින් තියෙන candle එක ගණන් ගන්නේ නැහැ).
""")

    with tab2:
        rsi_map = res["rsi"]
        rows = []
        for coin, m in rsi_map.items():
            if not m:
                continue
            ext = [tf for tf, v in m.items() if v >= ob or v <= os_]
            rows.append((coin, m, len(ext)))
        only_ext = st.checkbox("Overbought / oversold තියෙන coins විතරක් පෙන්නන්න", True)
        if only_ext:
            rows = [r for r in rows if r[2] > 0]
        rows.sort(key=lambda r: -r[2])
        if not rows:
            st.info("පෙන්නන්න coins නැහැ.")
        else:
            def cell(v):
                if v is None:
                    return ""
                return f"🔴 {v:.0f}" if v >= ob else f"🟢 {v:.0f}" if v <= os_ else f"{v:.0f}"
            mat = pd.DataFrame([{"Coin": c, **{tf: cell(m.get(tf)) for tf in res["tfs"]}, "Extreme TFs": n}
                                for c, m, n in rows])
            st.dataframe(mat, hide_index=True)
            st.caption(f"🔴 = RSI ≥ {ob} (overbought), 🟢 = RSI ≤ {os_} (oversold). RSI = Wilder 14, අන්තිම closed candle එකේ.")

st.caption("Candidates only, not trade advice. Divergence එකක් trend එක හැරෙනවා කියලා සහතික නැහැ, ඒක කාලයක් trend එක දිගටම යන්න පුළුවන්. Chart එකේ තහවුරු කරන්න.")
