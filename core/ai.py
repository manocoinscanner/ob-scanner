"""Optional AI reviewer. Uses free OpenAI-compatible APIs (Gemini, Groq, OpenRouter, ...) and
falls back to the next provider automatically when one hits its limit or fails."""
from __future__ import annotations

import json
import re
import time

import requests

COOLDOWN: dict = {}  # provider name -> unix time until which it is skipped

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

RULES = """Scanner rules (exact):
- Direction: LONG if 4H EMA20 > EMA50, SHORT if below (tie: sign of 24h change).
- Score parts (see 'why'): 4H trend up to 25, 1H trend 10, Clean trend (efficiency ratio) up to 15, Volume (RVOL) up to 15,
  Strength vs BTC up to 10, Setup 25, Derivs bonus up to 15 (Score = sum).
- Setup 'Breakout': 4H close beyond the previous 20-candle high (LONG) / low (SHORT).
  Setup 'Pullback': price within 1 ATR of 1H EMA20 and 1H RSI 38-55 (LONG) / 45-62 (SHORT). '-' = none.
- Derivs: +10 if open interest 24h change > 0 and price move agrees with the direction; +5 if funding <= 0.005% (LONG) or >= 0.01% (SHORT).
- Trade plan: entry = price, stop = 1.5 x ATR(1H) against the direction, TP1 = 1.5R, TP2 = 3R."""

SYSTEM = (
    "You are a strict crypto trading-signal auditor. You get the scanner output for ONE coin and the scanner's exact rules. "
    "Use ONLY the data given. Never invent prices, news, indicators or chart patterns. Check whether the direction, setup, "
    "score parts and trade plan are consistent with the data and the rules, and point out weaknesses (e.g. RSI stretched, "
    "volume weak, funding crowded, price far from entry, low R:R). You cannot see the chart, so say when something cannot be verified. "
    "Reply with ONLY one JSON object, no markdown: "
    '{"verdict":"AGREE|CAUTION|REJECT","confidence":0-100,"direction_ok":true|false,'
    '"checks":[{"rule":"short name","ok":true|false,"note":"short"}],"risks":["short"],'
    '"summary":"max 3 sentences, in Sinhala"}\n\n' + RULES
)


def providers(get):
    """Build the list of usable providers from secrets/env. `get(name)` returns a string or None."""
    spec = [
        ("Gemini", "GEMINI_API_KEY", GEMINI_URL, get("GEMINI_MODEL") or "gemini-2.5-flash"),
        ("Groq-70B", "GROQ_API_KEY", GROQ_URL, get("GROQ_MODEL") or "llama-3.3-70b-versatile"),
        ("Groq-8B", "GROQ_API_KEY", GROQ_URL, get("GROQ_MODEL2") or "llama-3.1-8b-instant"),
        ("OpenRouter", "OPENROUTER_API_KEY", OPENROUTER_URL,
         get("OPENROUTER_MODEL") or "meta-llama/llama-3.3-70b-instruct:free"),
        ("Extra", "EXTRA_API_KEY", get("EXTRA_API_URL"), get("EXTRA_MODEL")),
    ]
    return [{"name": n, "key": get(k), "url": u, "model": m} for n, k, u, m in spec if get(k) and u and m]


def ask(messages, provs, max_tokens=900):
    """Try each provider in order. Returns (text, provider name, list of error strings)."""
    errors = []
    for p in provs:
        if COOLDOWN.get(p["name"], 0) > time.time():
            errors.append(f"{p['name']}: cooling down")
            continue
        try:
            r = requests.post(p["url"], timeout=60,
                              headers={"Authorization": f"Bearer {p['key']}", "Content-Type": "application/json"},
                              json={"model": p["model"], "messages": messages, "temperature": 0.2,
                                    "max_tokens": max_tokens})
            if r.status_code == 429:
                COOLDOWN[p["name"]] = time.time() + 120
                errors.append(f"{p['name']}: limit reached (429)")
                continue
            if r.status_code != 200:
                COOLDOWN[p["name"]] = time.time() + 30
                errors.append(f"{p['name']}: HTTP {r.status_code} {r.text[:120]}")
                continue
            text = r.json()["choices"][0]["message"]["content"]
            if text and text.strip():
                return text, p["name"], errors
            errors.append(f"{p['name']}: empty answer")
        except Exception as e:
            COOLDOWN[p["name"]] = time.time() + 30
            errors.append(f"{p['name']}: {str(e)[:120]}")
    return None, None, errors


def clean(row):
    """Row -> JSON-safe dict (numpy types and NaN handled)."""
    keys = ["coin", "dir", "score", "setup", "price", "chg24", "vs_btc", "rvol", "rsi", "er", "atr_pct",
            "oi24", "oi4", "funding", "bonus", "entry", "stop", "tp1", "tp2", "why"]
    out = {}
    for k in keys:
        v = row.get(k)
        try:
            f = float(v)
            out[k] = None if f != f else round(f, 6)
        except (TypeError, ValueError):
            out[k] = v
    return out


def sanity(row):
    """Pure-code checks (no AI): is the trade plan arithmetic consistent? Returns [(name, ok)]."""
    e, s, t1, t2 = (float(row[k]) for k in ("entry", "stop", "tp1", "tp2"))
    long_ = row["dir"] == "LONG"
    checks = [("Stop on the correct side", s < e if long_ else s > e),
              ("TP1 and TP2 on the correct side", (e < t1 < t2) if long_ else (e > t1 > t2))]
    risk = abs(e - s)
    checks.append(("TP1 = 1.5R", risk > 0 and abs(abs(t1 - e) / risk - 1.5) < 0.05))
    checks.append(("Setup exists (Breakout / Pullback)", row.get("setup") in ("Breakout", "Pullback")))
    rsi = float(row["rsi"])
    checks.append(("RSI not stretched against the trade", rsi < 75 if long_ else rsi > 25))
    return checks


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {"verdict": "CAUTION", "confidence": 0, "summary": text.strip()[:400], "checks": [], "risks": []}


def review_coin(row, provs):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(clean(row), ensure_ascii=False)}]
    text, name, errors = ask(msgs, provs)
    base = {"coin": row.get("coin"), "dir": row.get("dir"), "sanity": sanity(row), "errors": errors}
    if text is None:
        return {**base, "verdict": "ERROR", "confidence": 0, "provider": "-", "checks": [], "risks": [],
                "summary": "සියලු AI providers limit වෙලා හෝ error: " + "; ".join(errors)}
    return {**base, **parse(text), "provider": name}
