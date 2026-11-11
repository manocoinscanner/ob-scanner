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
    '"summary":"max 2 sentences, in Sinhala"}. Keep it SHORT: at most 4 checks (notes max 10 words), at most 3 risks. '
    "Put verdict, confidence and direction_ok FIRST.\n\n" + RULES
)


def providers(get):
    """Build the list of usable providers from secrets/env. `get(name)` returns a string or None."""
    spec = [
        ("Gemini", "GEMINI_API_KEY", GEMINI_URL, get("GEMINI_MODEL") or "gemini-3.5-flash"),
        ("Groq-70B", "GROQ_API_KEY", GROQ_URL, get("GROQ_MODEL") or "openai/gpt-oss-120b"),
        ("Groq-8B", "GROQ_API_KEY", GROQ_URL, get("GROQ_MODEL2") or "openai/gpt-oss-20b"),
        ("OpenRouter", "OPENROUTER_API_KEY", OPENROUTER_URL,
         get("OPENROUTER_MODEL") or "meta-llama/llama-3.3-70b-instruct:free"),
        ("Extra", "EXTRA_API_KEY", get("EXTRA_API_URL"), get("EXTRA_MODEL")),
    ]
    return [{"name": n, "key": get(k), "url": u, "model": m} for n, k, u, m in spec if get(k) and u and m]


def _post(p, messages, max_tokens, extras):
    body = {"model": p["model"], "messages": messages, "temperature": 0.2, "max_tokens": max_tokens, **extras}
    return requests.post(p["url"], timeout=60, json=body,
                         headers={"Authorization": f"Bearer {p['key']}", "Content-Type": "application/json"})


def ask(messages, provs, max_tokens=2000):
    """Try each provider in order. Returns (text, provider name, list of error strings)."""
    errors = []
    for p in provs:
        if COOLDOWN.get(p["name"], 0) > time.time():
            errors.append(f"{p['name']}: cooling down")
            continue
        try:
            extras = {"reasoning_effort": "low"} if p["name"].startswith("Groq") and "gpt-oss" in p["model"] else {}
            r = _post(p, messages, max_tokens, extras)
            if r.status_code == 400 and extras:  # provider does not know reasoning_effort: retry without it
                r = _post(p, messages, max_tokens, {})
            if r.status_code in (500, 502, 503, 504):  # temporary server problem: wait a little, retry once
                time.sleep(3)
                r = _post(p, messages, max_tokens, extras if r.status_code != 400 else {})
            if r.status_code in (429, 413):
                COOLDOWN[p["name"]] = time.time() + 120
                errors.append(f"{p['name']}: limit reached ({r.status_code})")
                continue
            if r.status_code != 200:
                COOLDOWN[p["name"]] = time.time() + (15 if r.status_code >= 500 else 60)
                errors.append(f"{p['name']}: HTTP {r.status_code} {r.text[:100]}")
                continue
            text = r.json()["choices"][0]["message"]["content"]
            if text and text.strip():
                return text, p["name"], errors
            errors.append(f"{p['name']}: empty answer")
        except Exception as e:
            COOLDOWN[p["name"]] = time.time() + 30
            errors.append(f"{p['name']}: {str(e)[:100]}")
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
    """Strict JSON first. If the answer is cut off or slightly broken, salvage the key fields."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            out = json.loads(m.group(0))
            if isinstance(out, dict) and "verdict" in out:
                return out
        except Exception:
            pass
    v = re.search(r'"verdict"\s*:\s*"(AGREE|CAUTION|REJECT)"', text, re.I)
    c = re.search(r'"confidence"\s*:\s*(\d+)', text)
    d = re.search(r'"direction_ok"\s*:\s*(true|false)', text, re.I)
    sm = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)', text, re.S)
    if v:
        return {"verdict": v.group(1).upper(), "confidence": int(c.group(1)) if c else 0,
                "direction_ok": (d.group(1).lower() == "true") if d else None, "checks": [], "risks": [],
                "summary": sm.group(1) if sm and sm.group(1).strip() else "(AI උත්තරය කැපිලා, summary නැහැ)"}
    return {"verdict": "CAUTION", "confidence": 0, "checks": [], "risks": [],
            "summary": "AI උත්තරය තේරුම් ගන්න බැරි වුණා. ආයෙත් run කරන්න."}


def review_coin(row, provs):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(clean(row), ensure_ascii=False)}]
    text, name, errors = ask(msgs, provs)
    base = {"coin": row.get("coin"), "dir": row.get("dir"), "sanity": sanity(row), "errors": errors,
            "row": clean(row)}
    if text is None:
        return {**base, "verdict": "ERROR", "confidence": 0, "provider": "-", "checks": [], "risks": [],
                "summary": "සියලු AI providers limit වෙලා හෝ error: " + "; ".join(errors)}
    return {**base, **parse(text), "provider": name}
