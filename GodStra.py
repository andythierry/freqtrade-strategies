# GodStra Strategy
# Author: @Mablue (Masoud Azizi)
# github: https://github.com/mablue/
# IMPORTANT:Add to your pairlists inside config.json (Under StaticPairList):
#   {
#       "method": "AgeFilter",
#       "min_days_listed": 30
#   },
# IMPORTANT: INSTALL TA BEFOUR RUN(pip install ta)
# IMPORTANT: Use Smallest "max_open_trades" for getting best results inside config.json

# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from functools import reduce
from freqtrade.strategy import merge_informative_pair, informative
import qtpylib
from freqtrade.persistence import Trade
from technical.indicators import bollinger_bands
from pandas_ta import bbands

class GodStra(IStrategy):
    INTERFACE_VERSION: int = 3

    buy_params = {
        "buy-cross-0": "volatility_kcc",
        "buy-indicator-0": "trend_ichimoku_base",
        "buy-int-0": 42,
        "buy-oper-0": "<R",
        "buy-real-0": 0.06295,
    }

    sell_params = {
        "sell-cross-0": "volume_mfi",
        "sell-indicator-0": "trend_kst_diff",
        "sell-int-0": 98,
        "sell-oper-0": "=R",
        "sell-real-0": 0.8779,
    }

    minimal_roi = {"0": 0.3556, "4818": 0.21275, "6395": 0.09024, "22372": 0}
    stoploss = -0.34549
    trailing_stop = True
    trailing_stop_positive = 0.22673
    trailing_stop_positive_offset = 0.2684
    trailing_only_offset_is_reached = True
    timeframe = "12h"

    def dna_size(self, dct: dict):
        def int_from_str(st: str):
            str_int = "".join([d for d in st if d.isdigit()])
            return int(str_int) if str_int else -1
        return len({int_from_str(digit) for digit in dct.keys()})

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = dataframe.dropna()
        
        # TA-Lib global features
        dataframe = add_all_ta_features(
            dataframe,
            open="open", high="high", low="low", close="close", volume="volume",
            fillna=True
        )

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=14)

        # EMA
        dataframe["ema_20"] = ta.EMA(dataframe["close"], timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe["close"], timeperiod=50)

        # Bollinger Bands
        bb = ta.BBANDS(dataframe["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        dataframe["bb_upperband"] = bb["upperband"]
        dataframe["bb_middleband"] = bb["middleband"]
        dataframe["bb_lowerband"] = bb["lowerband"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        for i in range(self.dna_size(self.buy_params)):
            OPR = self.buy_params[f"buy-oper-{i}"]
            IND = self.buy_params[f"buy-indicator-{i}"]
            CRS = self.buy_params[f"buy-cross-{i}"]
            INT = self.buy_params[f"buy-int-{i}"]
            REAL = self.buy_params[f"buy-real-{i}"]
            DFIND = dataframe[IND]
            DFCRS = dataframe[CRS]

            if OPR == ">":
                conditions.append(DFIND > DFCRS)
            elif OPR == "=":
                conditions.append(np.isclose(DFIND, DFCRS))
            elif OPR == "<":
                conditions.append(DFIND < DFCRS)
            elif OPR == "CA":
                conditions.append(qtpylib.crossed_above(DFIND, DFCRS))
            elif OPR == "CB":
                conditions.append(qtpylib.crossed_below(DFIND, DFCRS))
            elif OPR == ">I":
                conditions.append(DFIND > INT)
            elif OPR == "=I":
                conditions.append(DFIND == INT)
            elif OPR == "<I":
                conditions.append(DFIND < INT)
            elif OPR == ">R":
                conditions.append(DFIND > REAL)
            elif OPR == "=R":
                conditions.append(np.isclose(DFIND, REAL))
            elif OPR == "<R":
                conditions.append(DFIND < REAL)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        for i in range(self.dna_size(self.sell_params)):
            OPR = self.sell_params[f"sell-oper-{i}"]
            IND = self.sell_params[f"sell-indicator-{i}"]
            CRS = self.sell_params[f"sell-cross-{i}"]
            INT = self.sell_params[f"sell-int-{i}"]
            REAL = self.sell_params[f"sell-real-{i}"]
            DFIND = dataframe[IND]
            DFCRS = dataframe[CRS]

            if OPR == ">":
                conditions.append(DFIND > DFCRS)
            elif OPR == "=":
                conditions.append(np.isclose(DFIND, DFCRS))
            elif OPR == "<":
                conditions.append(DFIND < DFCRS)
            elif OPR == "CA":
                conditions.append(qtpylib.crossed_above(DFIND, DFCRS))
            elif OPR == "CB":
                conditions.append(qtpylib.crossed_below(DFIND, DFCRS))
            elif OPR == ">I":
                conditions.append(DFIND > INT)
            elif OPR == "=I":
                conditions.append(DFIND == INT)
            elif OPR == "<I":
                conditions.append(DFIND < INT)
            elif OPR == ">R":
                conditions.append(DFIND > REAL)
            elif OPR == "=R":
                conditions.append(np.isclose(DFIND, REAL))
            elif OPR == "<R":
                conditions.append(DFIND < REAL)

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1

        return dataframe
