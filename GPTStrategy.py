import json
from freqtrade.strategy import IStrategy

class GPTStrategy(IStrategy):
    timeframe = "5m"

    def load_signals(self):
        try:
            with open("signals.json", "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[:, 'enter_long'] = 0
        signals = self.load_signals()
        pair = metadata['pair']
        if signals.get(pair) == "BUY":
            dataframe.loc[:, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[:, 'exit_long'] = 0
        signals = self.load_signals()
        pair = metadata['pair']
        if signals.get(pair) == "SELL":
            dataframe.loc[:, 'exit_long'] = 1
        return dataframe
