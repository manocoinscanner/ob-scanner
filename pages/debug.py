import traceback

import streamlit as st

from core.data import SOURCES, find_symbol, make_ex
from core.rank import book_features, flow_features, get_book, get_trades

st.title("🛠 Debug: order book + trades")
name = st.selectbox("Data source", list(SOURCES))
coin = st.text_input("Coin", "SOL").strip().upper()

if st.button("Test"):
    ex = make_ex(SOURCES[name])
    ex.load_markets()
    sym = find_symbol(ex, coin)
    st.write("Symbol:", sym)
    if sym:
        st.subheader("1. Order book")
        try:
            bk = get_book(ex, sym)
            st.write("bids:", len(bk["bids"]), "asks:", len(bk["asks"]))
            st.write("best bid/ask:", bk["bids"][0], bk["asks"][0])
            st.write("deepest bid/ask:", bk["bids"][-1], bk["asks"][-1])
            price = (float(bk["bids"][0][0]) + float(bk["asks"][0][0])) / 2
            f = book_features(bk, price, "LONG", price * 1.02)
            st.write("book_features:", f if f else "None (book too thin or too shallow)")
        except Exception:
            st.code(traceback.format_exc())

        st.subheader("2. Trades")
        try:
            tr = get_trades(ex, sym)
            st.write("trades:", len(tr))
            if tr:
                st.write("window (min):", (max(t[0] for t in tr) - min(t[0] for t in tr)) / 60000)
                st.write("first 3:", tr[:3])
            f = flow_features(tr, "LONG")
            st.write("flow_features:", f if f else "None (<50 trades or window < 1 min)")
        except Exception:
            st.code(traceback.format_exc())
