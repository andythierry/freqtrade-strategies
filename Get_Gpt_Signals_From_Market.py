#!/usr/bin/env python3
import os, json, time, math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import ccxt

from openai import OpenAI

# ---------- CONFIG ----------
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "kraken")   # "kraken" ou "binance"
SYMBOLS = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT").split(",")
TIMEFRAME = os.getenv("TIMEFRAME", "5m")           # 5m conseillé pour scalping
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "3"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")       # export OPENAI_API_KEY=...
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "signals.json")
# ----------------------------

assert OPENAI_API_KEY, "Set OPENAI_API_KEY in environment."

# --- utils indicateurs (pas de TA-Lib nécessaire) ---
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def zscore(series: pd.Series, window: int = 50) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)

def pct_change(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change(periods=period) * 100.0

def load_ohlcv(exchange, symbol, timeframe, since_ms, limit=2000):
    """
    Récupère OHLCV paginé jusqu’à now (~3 jours en 5m ≈ 864 bougies).
    """
    all_rows = []
    fetch_since = since_ms
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=1000)
        if not batch:
            break
        all_rows += batch
        # Avance le curseur (dernière bougie + 1ms)
        fetch_since = batch[-1][0] + 1
        # stop guard
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000.0)
    return all_rows

def build_features(df: pd.DataFrame) -> dict:
    # colonnes: time, open, high, low, close, volume
    df["ema9"]  = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["rsi14"] = rsi(df["close"], 14)

    # volatilité: ATR-like simple ou true range %
    hl_range = (df["high"] - df["low"]) / df["close"] * 100.0
    df["volatility_pct"] = hl_range.rolling(20).mean()

    # volume z-score (sur 50 bougies)
    df["vol_z"] = zscore(df["volume"], 50)

    # retours récents
    df["ret_1"] = pct_change(df["close"], 1)
    df["ret_12"] = pct_change(df["close"], 12)   # ~1h en 5m
    df["ret_72"] = pct_change(df["close"], 72)   # ~6h
    df["ret_288"] = pct_change(df["close"], 288) # ~24h

    last = df.iloc[-1]
    features = {
        "ts": int(last["time"]),
        "close": round(float(last["close"]), 8),
        "ema9": round(float(last["ema9"]), 8),
        "ema21": round(float(last["ema21"]), 8),
        "rsi14": round(float(last["rsi14"]), 3),
        "volatility_pct": round(float(last["volatility_pct"]), 4) if not math.isnan(last["volatility_pct"]) else None,
        "vol_z": round(float(last["vol_z"]), 3) if not math.isnan(last["vol_z"]) else None,
        "ret_1": round(float(last["ret_1"]), 4) if not math.isnan(last["ret_1"]) else None,
        "ret_12": round(float(last["ret_12"]), 4) if not math.isnan(last["ret_12"]) else None,
        "ret_72": round(float(last["ret_72"]), 4) if not math.isnan(last["ret_72"]) else None,
        "ret_288": round(float(last["ret_288"]), 4) if not math.isnan(last["ret_288"]) else None,
        "ema_cross_up": bool(last["ema9"] > last["ema21"]),
    }
    return features

def main():
    # --------- Exchange ----------
    ex_class = getattr(ccxt, EXCHANGE_ID)
    exchange = ex_class({"enableRateLimit": True})
    # Optionnel: clés si tu veux endpoints privés, mais ici pas utile
    # exchange.apiKey = os.getenv("API_KEY")
    # exchange.secret = os.getenv("API_SECRET")

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    symbols_features = {}
    for sym in SYMBOLS:
        try:
            rows = load_ohlcv(exchange, sym, TIMEFRAME, since_ms)
            if len(rows) < 100:
                print(f"[WARN] Trop peu de données pour {sym} ({len(rows)})")
                continue
            df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
            features = build_features(df)
            symbols_features[sym] = features
        except Exception as e:
            print(f"[ERR] {sym}: {e}")

    if not symbols_features:
        raise RuntimeError("Aucune donnée récupérée.")

    # --------- Prompt compact et JSON-strict ----------
    system = (
        "Tu es un assistant de trading. "
        "Décide BUY / SELL / HOLD par paire à l’horizon intraday (timeframe 5m). "
        "Prends en compte: ema_cross_up, RSI (surachat >70, survente <30), "
        "retours récents, volatilité et volume_zscore (breakouts si vol_z>2). "
        "Réponds UNIQUEMENT en JSON strict valide, format:\n"
        "{\n"
        '  "decisions": {\n'
        '    "SYMBOL": {"decision": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "reason": "<=30 mots"}\n'
        "  }\n"
        "}\n"
        "Pas de texte hors JSON."
    )

    user = {
        "timeframe": TIMEFRAME,
        "lookback_days": LOOKBACK_DAYS,
        "features": symbols_features
    }

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, separators=(",", ":"))}
        ],
    )

    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
        assert "decisions" in parsed and isinstance(parsed["decisions"], dict)
    except Exception as e:
        # fallback strict si le modèle dévie (rare avec response_format)
        raise RuntimeError(f"Réponse non-JSON ou invalide: {e}\n{raw}")

    # On ne garde que la clé -> "BUY|SELL|HOLD" pour Freqtrade
    signals = {pair: v.get("decision","HOLD") for pair, v in parsed["decisions"].items()}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(signals, f, indent=2, sort_keys=True)

    # Optionnel: garder la version détaillée pour audit
    with open("signals_detailed.json", "w") as f:
        json.dump(parsed, f, indent=2)

    print(f"OK → {OUTPUT_FILE}")
    print(json.dumps(signals, indent=2))

if __name__ == "__main__":
    main()
