import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCHEDULER_DIR = PROJECT_ROOT / "scheduler_alerts"

for path in (BACKEND_DIR, SCHEDULER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from strategy import SupertrendStrategy  # noqa: E402
from run_strategy_alerts import collect_latest_events  # noqa: E402


class LastBarEntryStrategy(SupertrendStrategy):
    def calculate_indicators(self, df):
        df = super().calculate_indicators(df)
        df["ema200"] = 10.0
        df["supertrend"] = 9.0
        df["direction"] = -1
        df["adx"] = 30.0
        df.loc[df.index[-1], "supertrend"] = 11.0
        df.loc[df.index[-1], "direction"] = 1
        return df


class StrategyAlertTests(unittest.TestCase):
    def test_latest_entry_signal_is_not_overwritten_by_end_of_data_close(self):
        dates = pd.date_range("2026-01-01", periods=8, freq="D")
        data = pd.DataFrame({
            "Date": dates,
            "Open": [11.0] * len(dates),
            "High": [12.0] * len(dates),
            "Low": [9.0] * len(dates),
            "Close": [11.0] * len(dates),
            "Volume": [1000] * len(dates),
        })

        strategy = LastBarEntryStrategy(
            atr_length=2,
            factor=3,
            ema_length=2,
            swing_lookback=2,
            exit_mode="trend",
            entry_mode="adx_anytime",
            adx_threshold=25,
            long_only=True,
        )

        results, trades = strategy.backtest(data)

        self.assertEqual(results.iloc[-1]["signal"], 1)
        self.assertEqual(trades[-1]["exit_reason"], "End of Data")

        events = collect_latest_events(
            "TEST",
            {
                "mode": "adx_trend",
                "atr_length": 2,
                "factor": 3,
                "ema_length": 2,
                "swing_lookback": 2,
                "max_trades": 1,
                "alert_on": ["entry_signal", "exit_signal"],
            },
            results,
            trades,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["title"], "TEST LONG entry signal")


if __name__ == "__main__":
    unittest.main()
