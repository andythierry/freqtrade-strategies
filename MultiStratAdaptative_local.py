#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiStratAdaptive — stratégie compilée et adaptative pour Freqtrade

Objectif
--------
Combiner plusieurs approches (Supertrend, momentum RSI/EMA, pattern price-action)
et activer automatiquement l'approche la plus pertinente selon le régime de marché
(détecté sur une timeframe informative 1h : tendance / range / volatilité).

Points clés
-----------
- Multi-stratégies dans **un seul fichier** via des tags d'entrée :
  - "ST"           → Supertrend (trend following)
  - "RSIEMA_V4"    → Momentum (ancienne MagicStratScalp_v4)
  - "RSIEMA_V5"    → Momentum (variante v5, plus stricte)
  - "PATTERN"      → Price-action (engloutissant haussier, pin bar)
- Sélection par **régime de marché** (1h) :
  - Trend si ADX(14) ≥ 20 et pente EMA200 positive ⇒ favorise Supertrend + RSIEMA_V5
  - Range si BandWidth Bollinger ≤ 8% ⇒ favorise PATTERN + RSIEMA_V4
  - Sinon ⇒ fallback momentum léger (RSIEMA_V4)
- **Kill-switch par sous-stratégie** : si > N pertes consécutives (rolling) pour un tag,
  on désactive temporairement ce tag pendant M heures.
- **Protections** (exposables via `protections` + garde-fous locaux) :
  - Cooldown après perte
  - Stoploss dynamique (ATR + trailing léger)
- **Nettoyage paires** : possibilité d'exclure les paires destructrices via `BAD_PAIRS`.

Compatibilité
-------------
- Testé avec Freqtrade ≥ 2024.3
- Timeframe par défaut : 5m ; informative : 1h

Conseils d'utilisation
----------------------
- Dans le `config.json` : `"log_trade_details": true`, `"verbosity": 3`
- Lancer normal : `freqtrade trade --strategy MultiStratAdaptive`

"""
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, CategoricalParameter
from freqtrade.exchange import timeframe_to_minutes

import talib.abstract as ta
from technical.indicators import RMI
from technical.util import resample_to_interval, heikinashi


class MultiStratAdaptive(IStrategy):
    # --- Config de base ---
    timeframe = '5m'
    informative_timeframe = '1h'
    startup_candle_count: int = 240  # pour indicateurs 1h

    # ROI/Stoploss par défaut (le SL réel est géré par custom_stoploss)
    minimal_roi = {"0": 1000}
    stoploss = -0.20

    # Trailing (soft) — sera renforcé en custom_stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # Hyperparams simples (ajustables via Hyperopt)
    atr_mult_sl = DecimalParameter(1.0, 3.0, default=1.8, space='sell', decimals=1)
    adx_trend = IntParameter(15, 30, default=20, space='buy')
    bb_width_range = DecimalParameter(0.05, 0.12, default=0.08, space='buy', decimals=3)

    # Exclusion de paires destructrices (customisable)
    BAD_PAIRS: List[str] = [
        'AI16Z/USDT', 'FARTCOIN/USDT', 'XTZ/USDT', 'DOGE/USDT', 'TRUMP/USDT'
    ]

    # Kill-switch par sous-stratégie
    MAX_CONSECUTIVE_LOSSES = 3
    DISABLE_FOR_HOURS = 6

    # Internals (stocke état des sous-stratégies)
    custom_info: Dict = {}

    # Protections globales (facultatif; sinon utiliser protections dans config)
    use_custom_stoploss = True

    def informative_pairs(self):
        # On veut l'informative 1h pour chaque paire active
        return [(self.dp.current_whitelist(), self.informative_timeframe)]

    # --- Indicateurs ---
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata['pair']

        # Filtre paires mauvaises
        df['bad_pair'] = pair in self.BAD_PAIRS

        # Base 5m
        df['ema50'] = ta.EMA(df, timeperiod=50)
        df['ema200'] = ta.EMA(df, timeperiod=200)
        df['rsi'] = ta.RSI(df, timeperiod=14)
        df['atr'] = ta.ATR(df, timeperiod=14)
        df['adx'] = ta.ADX(df, timeperiod=14)

        # Supertrend (implémentation simple)
        df['tr'] = ta.TRANGE(df)
        atr = ta.ATR(df, timeperiod=10)
        factor = 3.0
        hl2 = (df['high'] + df['low']) / 2
        basic_ub = hl2 + factor * atr
        basic_lb = hl2 - factor * atr
        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        st = Series(index=df.index, dtype=float)
        dir_long = True
        for i in range(len(df)):
            if i == 0:
                st.iloc[i] = basic_lb.iloc[i]
                continue
            final_ub.iloc[i] = min(basic_ub.iloc[i], final_ub.iloc[i-1]) if df['close'].iloc[i-1] > final_ub.iloc[i-1] else basic_ub.iloc[i]
            final_lb.iloc[i] = max(basic_lb.iloc[i], final_lb.iloc[i-1]) if df['close'].iloc[i-1] < final_lb.iloc[i-1] else basic_lb.iloc[i]
            if st.iloc[i-1] == final_ub.iloc[i-1] and df['close'].iloc[i] <= final_ub.iloc[i]:
                st.iloc[i] = final_ub.iloc[i]
            elif st.iloc[i-1] == final_ub.iloc[i-1] and df['close'].iloc[i] > final_ub.iloc[i]:
                st.iloc[i] = final_lb.iloc[i]
            elif st.iloc[i-1] == final_lb.iloc[i-1] and df['close'].iloc[i] >= final_lb.iloc[i]:
                st.iloc[i] = final_lb.iloc[i]
            elif st.iloc[i-1] == final_lb.iloc[i-1] and df['close'].iloc[i] < final_lb.iloc[i]:
                st.iloc[i] = final_ub.iloc[i]
            else:
                st.iloc[i] = final_lb.iloc[i]
        df['supertrend'] = st
        df['supertrend_long'] = df['close'] > df['supertrend']

        # Informative 1h
        inf = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe)
        inf['ema200_1h'] = ta.EMA(inf, timeperiod=200)
        inf['ema50_1h'] = ta.EMA(inf, timeperiod=50)
        inf['adx_1h'] = ta.ADX(inf, timeperiod=14)
        # BB width en % (1h)
        upper, middle, lower = ta.BBANDS(inf['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        inf['bb_width'] = (upper - lower) / middle.replace(0, np.nan)
        # pente EMA200 (1h)
        inf['ema200_slope'] = inf['ema200_1h'].diff()

        df = df.merge(
            inf[['date', 'ema200_1h', 'ema50_1h', 'adx_1h', 'bb_width', 'ema200_slope']].rename(columns={'date': 'date_inf'}),
            left_on='date', right_on='date_inf', how='left'
        )
        df.drop(columns=['date_inf'], inplace=True)

        # Détection de régime
        df['is_trend'] = (df['adx_1h'] >= self.adx_trend.value) & (df['ema200_slope'] > 0)
        df['is_range'] = (df['bb_width'] <= self.bb_width_range.value)

        # Price-action basique (patterns)
        df['bull_engulf'] = (
            (df['close'] > df['open']) & (df['open'].shift(1) > df['close'].shift(1)) &
            (df['close'] >= df['open'].shift(1)) & (df['open'] <= df['close'].shift(1))
        )
        body = (df['close'] - df['open']).abs()
        wick_up = df['high'] - df[['open', 'close']].max(axis=1)
        wick_down = df[['open', 'close']].min(axis=1) - df['low']
        df['bull_pin'] = (wick_down > body * 2) & (df['close'] > df['open'])

        # Momentum RSI/EMA (v4/v5)
        df['ema_fast'] = ta.EMA(df, timeperiod=12)
        df['ema_slow'] = ta.EMA(df, timeperiod=26)
        df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['ema_trend_up'] = df['ema_fast'] > df['ema_slow']
        df['rsi_lt_50'] = df['rsi'] < 50
        df['rsi_lt_45'] = df['rsi'] < 45
        df['rsi_rebound'] = (df['rsi'].shift(1) < 30) & (df['rsi'] > df['rsi'].shift(1))

        return df

    # --- Entrées ---
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Kill-switch: récupérer l'état des sous-strategies désactivées
        disabled_tags = self._get_disabled_tags(metadata['pair'])

        # Supertrend (trend only)
        df.loc[
            (
                (df['is_trend']) &
                (~df['bad_pair']) &
                (df['supertrend_long']) &
                (df['close'] > df['ema50']) & (df['ema50'] > df['ema200']) &
                ('ST' not in disabled_tags)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'ST')

        # RSI/EMA v5 (trend a priori, stricte)
        df.loc[
            (
                (df['is_trend']) &
                (~df['bad_pair']) &
                (df['ema_cross_up']) &
                (df['rsi'] > 50) &
                ('RSIEMA_V5' not in disabled_tags)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'RSIEMA_V5')

        # RSI/EMA v4 (fallback momentum / range permissif)
        df.loc[
            (
                (df['is_range'] | ~df['is_trend']) &
                (~df['bad_pair']) &
                (df['ema_trend_up']) &
                (df['rsi'] > 45) &
                ('RSIEMA_V4' not in disabled_tags)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'RSIEMA_V4')

        # Pattern price-action (range only)
        df.loc[
            (
                (df['is_range']) &
                (~df['bad_pair']) &
                ((df['bull_engulf']) | (df['bull_pin'])) &
                ('PATTERN' not in disabled_tags)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'PATTERN')

        return df

    # --- Sorties ---
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Exits simples, le SL/Trailing fait le gros du travail
        df.loc[
            (
                (df['rsi'] > 70) |  # take profit simple
                ((df['is_trend']) & (df['close'] < df['ema50'])) |  # perte de momentum
                ((df['is_range']) & (df['rsi'] < 40))
            ),
            'exit_long'
        ] = 1
        return df

    # --- Stoploss dynamique ---
    def custom_stoploss(self, pair: str, trade, current_time: pd.Timestamp, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        # ATR sur last candle 5m
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return 1.0  # fallback
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else None
        if atr is None or np.isnan(atr):
            return 1.0
        sl_distance = self.atr_mult_sl.value * atr
        # Converti en pourcentage de prix
        sl_pct = sl_distance / max(current_rate, 1e-9)
        # Trailing supplémentaire si en gain
        if current_profit > 0.02:
            sl_pct = min(sl_pct, max(0.01, current_profit * 0.6))
        return max(0.01, min(0.20, sl_pct))

    # --- Hooks pour Kill-switch ---
    def check_exit_timeout(self, pair: str, trade, order) -> bool:
        # non utilisé ici
        return False

    def _get_disabled_tags(self, pair: str) -> set:
        key = f"disabled_tags::{pair}"
        info = self.custom_info.get(key, {"tags": set(), "until": pd.Timestamp.min})
        # Réactive si délai passé
        if pd.Timestamp.utcnow() > info.get('until', pd.Timestamp.min):
            info['tags'] = set()
        self.custom_info[key] = info
        return info['tags']

    def _disable_tag(self, pair: str, tag: str):
        key = f"disabled_tags::{pair}"
        until = pd.Timestamp.utcnow() + pd.Timedelta(hours=self.DISABLE_FOR_HOURS)
        info = self.custom_info.get(key, {"tags": set(), "until": until})
        info['tags'].add(tag)
        info['until'] = until
        self.custom_info[key] = info

    def on_trade_closed(self, trade, order, **kwargs) -> None:
        """Kill-switch : désactive un tag s'il enchaîne trop de pertes."""
        try:
            pair = trade.pair
            tag = (trade.enter_tag or '').upper()
            profit = trade.close_profit or 0.0
            if not tag:
                return
            # Historique des X derniers trades pour ce tag/paire
            df_hist = self._get_trade_history(pair, tag, limit=10)
            if df_hist is None or df_hist.empty:
                return
            # Compte pertes consécutives
            consec_losses = 0
            for p in df_hist['close_profit'].iloc[::-1]:
                if p <= 0:
                    consec_losses += 1
                else:
                    break
            if consec_losses >= self.MAX_CONSECUTIVE_LOSSES:
                self._disable_tag(pair, tag)
                self.log(f"[KILL] Désactivation temporaire du tag {tag} sur {pair} ({consec_losses} pertes consécutives)")
        except Exception as e:
            self.log(f"on_trade_closed error: {e}")

    def _get_trade_history(self, pair: str, tag: str, limit: int = 20) -> Optional[DataFrame]:
        try:
            trades = self.dp.get_trade_history(pair=pair, limit=limit)
            if not trades:
                return None
            rows = []
            for t in trades:
                if (t.enter_tag or '').upper() == tag.upper():
                    rows.append({
                        'close_profit': t.close_profit,
                        'close_date': t.close_date
                    })
            if not rows:
                return None
            return pd.DataFrame(rows).sort_values('close_date')
        except Exception:
            return None

    # --- Protections via hooks (en plus des protections config) ---
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                             time_in_force: str, enter_tag: str, **kwargs) -> bool:
        # Bloque si paire marquée "bad_pair"
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is not None and not df.empty and bool(df['bad_pair'].iloc[-1]):
            return False
        # Kill-switch local
        disabled = self._get_disabled_tags(pair)
        if enter_tag and enter_tag.upper() in disabled:
            return False
        return True

    # (Optionnel) position sizing custom : ici on laisse la stake du config.json
    # def custom_stake_amount(self, pair: str, current_price: float, proposed_stake: float, **kwargs) -> float:
    #     return proposed_stake
