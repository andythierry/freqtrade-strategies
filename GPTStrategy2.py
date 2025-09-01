import os, json, time, logging, requests
from datetime import datetime, timezone
from freqtrade.strategy.interface import IStrategy

class GPTStrategy2(IStrategy):
    timeframe = "5m"
    process_only_new_candles = True
    can_short = False
    startup_candle_count = 50

    # --- Réglages GPT ---
    gpt_refresh_sec = 300  # rafraîchit les signaux toutes 5 minutes
    gpt_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    gpt_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    gpt_timeout = 45

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gpt_cache = {}          # { "ETH/USDT": {"action":"BUY","confidence":0.72,"comment":"..."} }
        self._gpt_last_fetch = 0.0
        self._log = logging.getLogger(__name__)

    # ---------- GPT: requête groupée ----------
    def _fetch_gpt_bulk_signals(self, pairs):
        """Interroge GPT sur N paires en 1 appel et met à jour _gpt_cache."""
        if not pairs:
            return

        # Anti-spam : utilise le cache récent
        now = time.time()
        if now - self._gpt_last_fetch < self.gpt_refresh_sec and all(p in self._gpt_cache for p in pairs):
            return

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._log.error("GPT_BULK: OPENAI_API_KEY manquant - tous les signaux = HOLD")
            for p in pairs:
                self._gpt_cache[p] = {"action": "HOLD", "confidence": 0.0, "comment": "no_api_key"}
            self._gpt_last_fetch = now
            return

        # Prépare le prompt (strict JSON)
        pairs_str = ", ".join(pairs)
        system_msg = (
            "You are a trading assistant. Return STRICT JSON only.\n"
            "For each pair, choose action in {BUY, SELL, HOLD} and a confidence in [0,1].\n"
            "Keep it short; no explanations, only a concise comment.\n"
        )
        user_msg = (
            "Evaluate these pairs for a short-term trade decision (intraday):\n"
            f"pairs=[{pairs_str}]\n"
            "Return a JSON object like: {\"results\":[{\"pair\":\"ETH/USDT\",\"action\":\"BUY\",\"confidence\":0.72,\"comment\":\"...\"}, ...]}"
        )

        payload = {
            "model": self.gpt_model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            # On demande du JSON "propre"
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        url = f"{self.gpt_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            r = requests.post(url, headers=headers, json=payload, timeout=self.gpt_timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            # Tolère soit {"results":[...]} soit directement [...]
            obj = json.loads(content)
            data = obj.get("results", obj) if isinstance(obj, dict) else obj
            # Normalise et met en cache
            updated = 0
            for item in data:
                pair = item.get("pair")
                act = (item.get("action") or "HOLD").upper()
                conf = float(item.get("confidence") or 0.0)
                cmt = item.get("comment") or ""
                if pair in pairs:
                    self._gpt_cache[pair] = {
                        "action": "BUY" if act == "BUY" else "SELL" if act == "SELL" else "HOLD",
                        "confidence": max(0.0, min(1.0, conf)),
                        "comment": cmt[:180],
                    }
                    updated += 1
            # Fallback pour les paires manquantes
            for p in pairs:
                if p not in self._gpt_cache:
                    self._gpt_cache[p] = {"action": "HOLD", "confidence": 0.0, "comment": "missing_from_gpt"}
            self._gpt_last_fetch = now
            self._log.info("GPT_BULK: updated %s/%s pairs", updated, len(pairs))
        except Exception as e:
            self._log.exception("GPT_BULK: API error -> HOLD for all (%s)", e)
            for p in pairs:
                self._gpt_cache[p] = {"action": "HOLD", "confidence": 0.0, "comment": "api_error"}
            self._gpt_last_fetch = now

    # ---------- Utilitaire: renvoie le signal pour 1 paire ----------
    def _gpt_signal_for(self, pair):
        # Rafraîchit au besoin avec toute la whitelist
        wl = []
        try:
            wl = self.dp.current_whitelist()
        except Exception:
            pass
        if wl:
            self._fetch_gpt_bulk_signals(wl)
        return self._gpt_cache.get(pair, {"action": "HOLD", "confidence": 0.0, "comment": "no_cache"})

    # ---------- Entry / Exit : applique la décision GPT sur la dernière bougie ----------
    def populate_entry_trend(self, dataframe, metadata: dict):
        pair = metadata["pair"]
        sig = self._gpt_signal_for(pair)
        dataframe["enter_long"] = 0
        dataframe["enter_tag"] = None
        # On n'agit que sur la dernière bougie
        if len(dataframe) > 0 and sig["action"] == "BUY":
            last_idx = dataframe.index[-1]
            dataframe.loc[last_idx, "enter_long"] = 1
            dataframe.loc[last_idx, "enter_tag"] = f"GPT_BUY_{sig['confidence']:.2f}"
        return dataframe

    def populate_exit_trend(self, dataframe, metadata: dict):
        pair = metadata["pair"]
        sig = self._gpt_signal_for(pair)
        dataframe["exit_long"] = 0
        dataframe["exit_tag"] = None
        if len(dataframe) > 0 and sig["action"] == "SELL":
            last_idx = dataframe.index[-1]
            dataframe.loc[last_idx, "exit_long"] = 1
            dataframe.loc[last_idx, "exit_tag"] = f"GPT_SELL_{sig['confidence']:.2f}"
        return dataframe
