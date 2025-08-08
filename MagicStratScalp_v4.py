
from freqtrade.strategy import IStrategy, merge_informative_pair, DecimalParameter
from freqtrade.persistence import Trade
import pandas as pd
import talib.abstract as ta
import numpy as np
from typing import Optional

class MagicStratScalp_v4(IStrategy):
    timeframe = '5m'
    informative_timeframe = '1h'
    startup_candle_count: int = 50

    # Paramètres optimisables pour l'entrée
    volume_multiplier = DecimalParameter(0.1, 2.0, default=0.1, decimals=2, space="buy")
    rsi_threshold = DecimalParameter(10, 40, default=30, decimals=0, space="buy")
    ema_gap_ratio_buy = DecimalParameter(0.9, 0.94, default=1, decimals=2, space="buy")

    # Paramètres optimisables pour la sortie
    sell_rsi_threshold = DecimalParameter(60, 90, default=89, decimals=0, space="sell")
    ema_gap_ratio_sell = DecimalParameter(0.9, 1.2, default=1.14, decimals=2, space="sell")

    minimal_roi = {"0": 0.02}
    stoploss = -0.01
    trailing_stop = False
    process_only_new_candles = True
    use_custom_stoploss = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True
    can_short = False

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=5)
        boll_upper, boll_middle, boll_lower = ta.BBANDS(dataframe['close'], timeperiod=20)
        dataframe['bb_upperband'] = boll_upper
        dataframe['bb_lowerband'] = boll_lower
        dataframe['donchian_upper'] = dataframe['high'].rolling(window=20).max()
        dataframe['donchian_lower'] = dataframe['low'].rolling(window=20).min()
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=30).mean()
        dataframe['date'] = pd.to_datetime(dataframe['date'])

        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        informative['ema9_1h'] = ta.EMA(informative, timeperiod=9)
        informative['ema21_1h'] = ta.EMA(informative, timeperiod=21)
        informative['volume_1h'] = informative['volume']
        informative['volume_mean_1h'] = informative['volume'].rolling(window=24).mean()

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe, ffill=True)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if dataframe.empty or dataframe.shape[0] == 0:
            df = pd.DataFrame(index=pd.RangeIndex(0))
            df['enter_long'] = []
            df['enter_tag'] = []
            return df

        if 'date' in dataframe.columns:
            dataframe = dataframe.set_index('date', drop=False)
        else:
            dataframe.index = pd.RangeIndex(len(dataframe))

        dataframe['enter_long'] = 0

        required_cols = ['ema9_1h', 'ema21_1h', 'volume_1h', 'volume_mean_1h']
        for col in required_cols:
            if col not in dataframe.columns:
                dataframe[col] = np.nan
        dataframe['enter_long'] = 0
        dataframe.loc[
            (
                (dataframe['ema9'] > dataframe['ema21'] * self.ema_gap_ratio_buy.value) &
                (dataframe['rsi'] < self.rsi_threshold.value)&
                (dataframe['volume'] > dataframe['volume_mean_slow'] * self.volume_multiplier.value)&
                (dataframe['rsi'] > dataframe['rsi'].shift(1))

                
            ),
            ['enter_long', 'enter_tag']
        ] = [1, 'RSI+EMA_Entry']
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_tag'] = ''
        dataframe.loc[
            (
                (dataframe['rsi'] > self.sell_rsi_threshold.value) &
                (dataframe['ema9'] < dataframe['ema21']*self.ema_gap_ratio_sell.value )&
                (dataframe['rsi'] < dataframe['rsi'].shift(1))
            ),
            ['exit_long', 'exit_tag']
        ] = [1, 'RSI_or_EMA_Exit']
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: pd.Timestamp,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        return None
 #
               # (dataframe['ema9'] > dataframe['ema21'] ) &
                # #&
                #(dataframe['close'] < dataframe['bb_lowerband']) &
                #(dataframe['close'] <= dataframe['donchian_lower']) &
                #(dataframe['volume'] > dataframe['volume_mean_slow'] * self.volume_multiplier.value) &
               # (dataframe.index.hour >= 8) & (dataframe.index.hour < 20)
