import pandas as pd
import numpy as np


WEEKLY_CLOSE_RULES = {
    'friday': 'W-FRI',
    'saturday': 'W-SAT',
    'sunday': 'W-SUN',
}


def resolve_weekly_close_day(value):
    day = str(value or 'friday').strip().lower()
    return day if day in WEEKLY_CLOSE_RULES else 'friday'


class SimpleMovingAverageStrategy:
    """
    A simple SMA crossover strategy:
    - Buy signal: when fast SMA crosses above slow SMA
    - Sell signal: when fast SMA crosses below slow SMA
    """
    
    def __init__(self, fast_period=20, slow_period=50, initial_balance=100000):
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("SMA periods must be positive numbers.")
        if fast_period >= slow_period:
            raise ValueError("Fast SMA period must be smaller than slow SMA period.")
        if initial_balance <= 0:
            raise ValueError("Initial balance must be greater than zero.")

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.initial_balance = float(initial_balance)
        self.trades = []
        
    def calculate_indicators(self, df):
        """Calculate SMA indicators"""
        df['sma_fast'] = df['Close'].rolling(window=self.fast_period).mean()
        df['sma_slow'] = df['Close'].rolling(window=self.slow_period).mean()
        return df
    
    def generate_signals(self, df):
        """Generate trading signals based on SMA crossover"""
        df['position'] = 0
        df['signal'] = 0
        
        # Detect crossovers directly
        for i in range(1, len(df)):
            fast_curr = df['sma_fast'].iloc[i]
            slow_curr = df['sma_slow'].iloc[i]
            fast_prev = df['sma_fast'].iloc[i-1]
            slow_prev = df['sma_slow'].iloc[i-1]
            
            # Skip if any value is NaN
            if pd.isna(fast_curr) or pd.isna(slow_curr) or pd.isna(fast_prev) or pd.isna(slow_prev):
                continue
            
            # Buy signal: fast SMA crosses above slow SMA
            if fast_prev <= slow_prev and fast_curr > slow_curr:
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'position'] = 1
            # Sell signal: fast SMA crosses below slow SMA
            elif fast_prev >= slow_prev and fast_curr < slow_curr:
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'position'] = 0
            else:
                # No signal, maintain previous position
                df.loc[df.index[i], 'position'] = df['position'].iloc[i-1]
        
        return df
    
    def backtest(self, df, initial_capital=None, trade_start_index=0):
        """Run backtest with the strategy"""
        if len(df) < self.slow_period:
            raise ValueError(
                f"Not enough data to calculate a {self.slow_period}-day SMA. "
                f"Received {len(df)} rows."
            )

        df = self.calculate_indicators(df)
        df = self.generate_signals(df)
        capital = float(initial_capital if initial_capital is not None else self.initial_balance)
        df['asset'] = capital
        
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        qty = 0
        trade_capital = capital
        
        # Start from index where both SMAs have values
        start_idx = max(self.fast_period, self.slow_period, int(trade_start_index or 0))
        
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            
            # Check if SMAs have valid values
            sma_fast_val = row['sma_fast']
            sma_slow_val = row['sma_slow']
            
            # Use pd.isna() on scalar values properly
            if pd.isna(sma_fast_val) or pd.isna(sma_slow_val):
                continue
            
            signal_val = row['signal']
            
            # Buy signal
            if not pd.isna(signal_val) and signal_val == 1 and position == 0:
                position = 1
                entry_price = row['Close']
                entry_date = row['Date']
                trade_capital = capital
                qty = capital / entry_price if entry_price else 0
                
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': None,
                    'exit_price': None,
                    'type': 'buy',
                    'qty': qty,
                    'pnl': None,
                    'return': None
                })
            
            # Sell signal
            elif not pd.isna(signal_val) and signal_val == -1 and position == 1:
                position = 0
                exit_price = row['Close']
                exit_date = row['Date']
                
                if len(trades) > 0 and trades[-1]['exit_date'] is None:
                    pnl = (exit_price - entry_price) * qty
                    return_pct = (pnl / trade_capital) * 100 if trade_capital else 0
                    capital += pnl
                    
                    trades[-1]['exit_date'] = exit_date
                    trades[-1]['exit_price'] = exit_price
                    trades[-1]['pnl'] = pnl
                    trades[-1]['return'] = return_pct
                    trades[-1]['asset_after'] = capital

            if position == 1:
                df.loc[df.index[idx], 'asset'] = capital + ((row['Close'] - entry_price) * qty)
            else:
                df.loc[df.index[idx], 'asset'] = capital

        if position == 1 and trades and trades[-1]['exit_date'] is None:
            final_row = df.iloc[-1]
            exit_price = final_row['Close']
            pnl = (exit_price - entry_price) * qty
            return_pct = (pnl / trade_capital) * 100 if trade_capital else 0
            capital += pnl
            trades[-1]['exit_date'] = final_row['Date']
            trades[-1]['exit_price'] = exit_price
            trades[-1]['exit_reason'] = 'End of Data'
            trades[-1]['pnl'] = pnl
            trades[-1]['return'] = return_pct
            trades[-1]['asset_after'] = capital
            df.loc[df.index[-1], 'asset'] = capital
        
        self.trades = trades
        return df, trades


class SupertrendStrategy:
    """
    Supertrend + EMA strategy for manual backtests.
    Long when Supertrend flips green above the EMA.
    Short when Supertrend flips red below the EMA.
    """

    def __init__(
        self,
        atr_length=14,
        factor=3,
        ema_length=200,
        swing_lookback=12,
        tp_multiplier=2.0,
        max_trades=1,
        leverage=1,
        initial_balance=10000,
        exit_mode='trend',
        entry_mode='flip',
        weekly_close_day='friday',
        adx_threshold=25,
        adx_trend_lookback=3,
    ):
        if atr_length <= 0 or ema_length <= 0 or swing_lookback <= 0:
            raise ValueError("ATR, EMA, and swing lookback must be positive numbers.")
        if factor <= 0 or tp_multiplier <= 0 or max_trades <= 0 or leverage <= 0:
            raise ValueError("Supertrend factor, TP multiplier, max trades, and leverage must be positive.")
        if adx_threshold <= 0:
            raise ValueError("ADX threshold must be greater than zero.")
        if adx_trend_lookback <= 0:
            raise ValueError("ADX trend lookback must be greater than zero.")

        self.atr_length = int(atr_length)
        self.factor = float(factor)
        self.ema_length = int(ema_length)
        self.swing_lookback = int(swing_lookback)
        self.tp_multiplier = float(tp_multiplier)
        self.max_trades = int(max_trades)
        self.leverage = float(leverage)
        self.initial_balance = float(initial_balance)
        self.exit_mode = exit_mode if exit_mode in {'trend', 'tp', 'sl'} else 'trend'
        self.entry_mode = entry_mode if entry_mode in {'flip', 'cross', 'weekly_long', 'weekly_bull_ema', 'adx_anytime', 'adx_uptrend'} else 'flip'
        self.weekly_close_day = resolve_weekly_close_day(weekly_close_day)
        self.adx_threshold = float(adx_threshold)
        self.adx_trend_lookback = int(adx_trend_lookback)
        self.trades = []

    def _rma(self, series, length):
        values = series.to_numpy(dtype=float)
        out = np.empty(len(values))
        if len(values) == 0:
            return pd.Series(out, index=series.index)

        out[0] = values[0]
        alpha = 1.0 / length
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return pd.Series(out, index=series.index)

    def _calculate_supertrend(self, high, low, close):
        true_range = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = self._rma(true_range, self.atr_length)

        hl2 = (high + low) / 2
        upperband = (hl2 + self.factor * atr).to_numpy()
        lowerband = (hl2 - self.factor * atr).to_numpy()
        close_vals = close.to_numpy()

        final_upper = np.full(len(close), np.nan, dtype=float)
        final_lower = np.full(len(close), np.nan, dtype=float)
        supertrend = np.full(len(close), np.nan, dtype=float)
        direction = np.full(len(close), -1, dtype=int)

        if len(close) == 0:
            return supertrend, direction

        start = 0
        final_upper[start] = upperband[start]
        final_lower[start] = lowerband[start]
        supertrend[start] = final_upper[start]
        direction[start] = -1

        for i in range(start + 1, len(close)):
            if np.isnan(upperband[i]) or np.isnan(lowerband[i]):
                continue

            prev_upper = final_upper[i - 1]
            prev_lower = final_lower[i - 1]
            prev_supertrend = supertrend[i - 1]

            final_upper[i] = upperband[i] if (upperband[i] < prev_upper or close_vals[i - 1] > prev_upper) else prev_upper
            final_lower[i] = lowerband[i] if (lowerband[i] > prev_lower or close_vals[i - 1] < prev_lower) else prev_lower

            if prev_supertrend == prev_upper:
                direction[i] = 1 if close_vals[i] > final_upper[i] else -1
            else:
                direction[i] = -1 if close_vals[i] < final_lower[i] else 1

            supertrend[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

        return supertrend, direction

    def _calculate_adx(self, high, low, close):
        high_vals = high.to_numpy(dtype=float)
        low_vals = low.to_numpy(dtype=float)

        up_move = pd.Series(high_vals - np.roll(high_vals, 1), index=high.index)
        down_move = pd.Series(np.roll(low_vals, 1) - low_vals, index=low.index)
        up_move.iloc[0] = 0
        down_move.iloc[0] = 0

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        true_range = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr = self._rma(true_range, self.atr_length)
        plus_di = 100 * (self._rma(plus_dm, self.atr_length) / atr.replace(0, np.nan))
        minus_di = 100 * (self._rma(minus_dm, self.atr_length) / atr.replace(0, np.nan))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        return self._rma(dx.fillna(0), self.atr_length)

    def _calculate_weekly_supertrend_filter(self, df):
        if 'Date' not in df.columns or df.empty:
            return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index, dtype=int)

        weekly = (
            df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            .copy()
            .sort_values('Date')
            .set_index('Date')
            .resample(WEEKLY_CLOSE_RULES[self.weekly_close_day], label='right', closed='right')
            .agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
            })
            .dropna(subset=['Open', 'High', 'Low', 'Close'])
            .reset_index()
        )

        if weekly.empty:
            return pd.Series(np.nan, index=df.index), pd.Series(0, index=df.index, dtype=int)

        weekly_supertrend, weekly_direction = self._calculate_supertrend(
            weekly['High'],
            weekly['Low'],
            weekly['Close'],
        )
        weekly_signals = pd.DataFrame({
            'Date': weekly['Date'],
            'weekly_supertrend': weekly_supertrend,
            'weekly_direction': weekly_direction,
        }).sort_values('Date')

        aligned = pd.merge_asof(
            df[['Date']].copy().sort_values('Date'),
            weekly_signals,
            on='Date',
            direction='backward',
        ).set_index(df.sort_values('Date').index)

        aligned = aligned.reindex(df.index)
        return aligned['weekly_supertrend'], aligned['weekly_direction'].fillna(0).astype(int)

    def calculate_indicators(self, df):
        df['ema200'] = df['Close'].ewm(span=self.ema_length, adjust=False).mean()

        high = df['High']
        low = df['Low']
        close = df['Close']
        supertrend, direction = self._calculate_supertrend(high, low, close)

        df['supertrend'] = supertrend
        df['direction'] = direction
        df['adx'] = self._calculate_adx(high, low, close)
        if self.entry_mode in {'weekly_long', 'weekly_bull_ema'}:
            df['weekly_supertrend'], df['weekly_direction'] = self._calculate_weekly_supertrend_filter(df)
        else:
            df['weekly_supertrend'] = np.nan
            df['weekly_direction'] = 0
        df['sma_fast'] = np.nan
        df['sma_slow'] = np.nan
        df['signal'] = 0
        df['asset'] = self.initial_balance
        return df

    def backtest(self, df, trade_start_index=0):
        minimum_rows = max(self.ema_length, self.swing_lookback, self.atr_length) + 3
        if len(df) < minimum_rows:
            raise ValueError(
                f"Not enough data for Supertrend strategy. Received {len(df)} rows; "
                f"need at least {minimum_rows}."
            )

        df = self.calculate_indicators(df)

        position = 0
        entry_price = 0
        qty = 0
        equity = self.initial_balance
        trade_equity = equity
        current_trade = None
        trades = []
        pending_entry = None
        trade_count = 0
        awaiting_cross_entry = self.entry_mode == 'cross'
        swing_low = df['Low'].shift(1).rolling(self.swing_lookback).min().to_numpy()
        swing_high = df['High'].shift(1).rolling(self.swing_lookback).max().to_numpy()

        close_vals = df['Close'].to_numpy(dtype=float)
        open_vals = df['Open'].to_numpy(dtype=float)
        ema_vals = df['ema200'].to_numpy(dtype=float)
        supertrend_vals = df['supertrend'].to_numpy(dtype=float)
        direction = df['direction'].to_numpy(dtype=int)
        prev_close = np.roll(close_vals, 1)
        prev_close[0] = close_vals[0]
        prev_ema = np.roll(ema_vals, 1)
        prev_ema[0] = ema_vals[0]
        crossed_above = (close_vals > ema_vals) & (prev_close <= prev_ema)
        crossed_below = (close_vals < ema_vals) & (prev_close >= prev_ema)
        crossed_above[0] = False
        crossed_below[0] = False
        ema_breakthru = crossed_above | crossed_below
        prev_supertrend = np.roll(supertrend_vals, 1)
        prev_supertrend[0] = supertrend_vals[0]
        supertrend_crossed_above_ema = (
            (supertrend_vals > ema_vals)
            & (prev_supertrend <= prev_ema)
        )
        supertrend_crossed_below_ema = (
            (supertrend_vals < ema_vals)
            & (prev_supertrend >= prev_ema)
        )

        # Cross is only valid when trend direction is unchanged before/after cross.
        # This rejects band-switch artifacts where direction flips on the cross bar.
        same_trend_across_cross = direction == np.roll(direction, 1)
        same_trend_across_cross[0] = False
        supertrend_crossed_above_ema[0] = False
        supertrend_crossed_below_ema[0] = False

        prev_direction = np.roll(direction, 1)
        prev_direction[0] = direction[0]
        stable_direction = direction == prev_direction

        # Ignore synthetic Supertrend/EMA crosses caused by band switching on
        # trend-flip bars (upper band becomes lower band and vice versa). Also
        # ignore crosses on the first bar after a flip, where the band jump can
        # still create a delayed crossover artifact.
        trend_changed = direction != prev_direction
        trend_changed[0] = False
        prev_trend_changed = np.roll(trend_changed, 1)
        prev_trend_changed[0] = False
        recent_trend_change = trend_changed | prev_trend_changed

        valid_cross_bar = stable_direction & (~recent_trend_change)
        supertrend_crossed_above_ema &= valid_cross_bar
        supertrend_crossed_below_ema &= valid_cross_bar
        supertrend_ema_cross = supertrend_crossed_above_ema | supertrend_crossed_below_ema

        if self.entry_mode in {'adx_anytime', 'adx_uptrend'}:
            start_idx = max(int(trade_start_index or 0) + self.swing_lookback, 1)
        else:
            start_idx = max(self.swing_lookback + 1, self.atr_length + 1, int(trade_start_index or 0), 1)
        if self.entry_mode == 'adx_uptrend':
            start_idx = max(start_idx, int(trade_start_index or 0) + self.adx_trend_lookback)
        required_cross_side = None
        if self.entry_mode == 'cross':
            ref_idx = max(start_idx - 1, 0)
            if pd.notna(supertrend_vals[ref_idx]) and pd.notna(ema_vals[ref_idx]):
                if supertrend_vals[ref_idx] < ema_vals[ref_idx]:
                    required_cross_side = 'long'
                elif supertrend_vals[ref_idx] > ema_vals[ref_idx]:
                    required_cross_side = 'short'
                else:
                    awaiting_cross_entry = False
            else:
                awaiting_cross_entry = False

        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            just_turned_green = row['direction'] == 1 and prev['direction'] == -1
            just_turned_red = row['direction'] == -1 and prev['direction'] == 1

            reset_breakthru = supertrend_ema_cross[i] if self.entry_mode == 'cross' else ema_breakthru[i]
            can_reset_trade_window = position == 0 and pending_entry is None
            if reset_breakthru and can_reset_trade_window:
                trade_count = 0
                pending_entry = None
                awaiting_cross_entry = self.entry_mode == 'cross'
                if self.entry_mode == 'cross':
                    if supertrend_crossed_above_ema[i]:
                        required_cross_side = 'long'
                    elif supertrend_crossed_below_ema[i]:
                        required_cross_side = 'short'

            if (
                self.entry_mode == 'cross'
                and pending_entry is not None
                and pending_entry.get('is_cross_entry')
                and (just_turned_green or just_turned_red)
            ):
                # Do not execute cross entries on Supertrend flip bars.
                pending_entry = None

            if pending_entry is not None and position == 0:
                fill_row = df.iloc[i]
                sig_i = pending_entry['signal_bar']
                trade_equity = equity
                fill_price = float(fill_row['Open'])
                fill_qty = max((equity * self.leverage) / fill_price, 0)

                if fill_qty > 0:
                    if pending_entry['side'] == 'long':
                        risk = fill_price - swing_low[sig_i]
                        if self.exit_mode == 'trend' or (pd.notna(risk) and risk > 0):
                            position = 1
                            entry_price = fill_price
                            qty = fill_qty
                            tp_price = fill_price + (self.tp_multiplier * risk) if self.exit_mode == 'tp' else None
                            sl_price = float(swing_low[sig_i]) if self.exit_mode in {'tp', 'sl'} else None
                            current_trade = {
                                'entry_date': fill_row['Date'],
                                'entry_reason': 'cross_entry' if pending_entry.get('is_cross_entry') else 'trend_follow_on',
                                'entry_price': entry_price,
                                'exit_date': None,
                                'exit_price': None,
                                'type': 'long',
                                'qty': qty,
                                'tp': tp_price,
                                'sl': sl_price,
                                'exit_reason': None,
                                'pnl': None,
                                'return': None,
                            }
                            if pending_entry.get('count_toward_max', True):
                                trade_count += 1
                            if pending_entry.get('is_cross_entry'):
                                awaiting_cross_entry = False
                            df.loc[df.index[i], 'signal'] = 1
                    else:
                        risk = swing_high[sig_i] - fill_price
                        if self.exit_mode == 'trend' or (pd.notna(risk) and risk > 0):
                            position = -1
                            entry_price = fill_price
                            qty = fill_qty
                            tp_price = fill_price - (self.tp_multiplier * risk) if self.exit_mode == 'tp' else None
                            sl_price = float(swing_high[sig_i]) if self.exit_mode in {'tp', 'sl'} else None
                            current_trade = {
                                'entry_date': fill_row['Date'],
                                'entry_reason': 'cross_entry' if pending_entry.get('is_cross_entry') else 'trend_follow_on',
                                'entry_price': entry_price,
                                'exit_date': None,
                                'exit_price': None,
                                'type': 'short',
                                'qty': qty,
                                'tp': tp_price,
                                'sl': sl_price,
                                'exit_reason': None,
                                'pnl': None,
                                'return': None,
                            }
                            if pending_entry.get('count_toward_max', True):
                                trade_count += 1
                            if pending_entry.get('is_cross_entry'):
                                awaiting_cross_entry = False
                            df.loc[df.index[i], 'signal'] = -1
                pending_entry = None

            if position == 1:
                sl_hit = (
                    self.exit_mode in {'tp', 'sl'}
                    and current_trade is not None
                    and current_trade.get('sl') is not None
                    and float(row['Low']) <= float(current_trade['sl'])
                )
                tp_hit = (
                    self.exit_mode == 'tp'
                    and current_trade is not None
                    and current_trade.get('tp') is not None
                    and float(row['High']) >= float(current_trade['tp'])
                )
                if sl_hit:
                    exit_price = float(current_trade['sl'])
                    pnl = (exit_price - entry_price) * qty
                    equity += pnl
                    current_trade.update({
                        'exit_date': row['Date'],
                        'exit_price': exit_price,
                        'exit_reason': 'SL',
                        'pnl': pnl,
                        'return': (pnl / trade_equity) * 100 if trade_equity else 0,
                        'asset_after': equity,
                    })
                    trades.append(current_trade)
                    df.loc[df.index[i], 'signal'] = -2
                    position = 0
                elif tp_hit:
                    exit_price = float(current_trade['tp'])
                    pnl = (exit_price - entry_price) * qty
                    equity += pnl
                    current_trade.update({
                        'exit_date': row['Date'],
                        'exit_price': exit_price,
                        'exit_reason': 'TP',
                        'pnl': pnl,
                        'return': (pnl / trade_equity) * 100 if trade_equity else 0,
                        'asset_after': equity,
                    })
                    trades.append(current_trade)
                    df.loc[df.index[i], 'signal'] = 2
                    position = 0
                elif self.exit_mode == 'trend' and just_turned_red:
                    exit_price = row['Close']
                    pnl = (exit_price - entry_price) * qty
                    equity += pnl
                    current_trade.update({
                        'exit_date': row['Date'],
                        'exit_price': exit_price,
                        'exit_reason': 'Trend Change',
                        'pnl': pnl,
                        'return': (pnl / trade_equity) * 100 if trade_equity else 0,
                        'asset_after': equity,
                    })
                    trades.append(current_trade)
                    df.loc[df.index[i], 'signal'] = -2
                    position = 0

            elif position == -1:
                sl_hit = (
                    self.exit_mode in {'tp', 'sl'}
                    and current_trade is not None
                    and current_trade.get('sl') is not None
                    and float(row['High']) >= float(current_trade['sl'])
                )
                tp_hit = (
                    self.exit_mode == 'tp'
                    and current_trade is not None
                    and current_trade.get('tp') is not None
                    and float(row['Low']) <= float(current_trade['tp'])
                )
                if sl_hit:
                    exit_price = float(current_trade['sl'])
                    pnl = (entry_price - exit_price) * qty
                    equity += pnl
                    current_trade.update({
                        'exit_date': row['Date'],
                        'exit_price': exit_price,
                        'exit_reason': 'SL',
                        'pnl': pnl,
                        'return': (pnl / trade_equity) * 100 if trade_equity else 0,
                        'asset_after': equity,
                    })
                    trades.append(current_trade)
                    df.loc[df.index[i], 'signal'] = 2
                    position = 0
                elif tp_hit:
                    exit_price = float(current_trade['tp'])
                    pnl = (entry_price - exit_price) * qty
                    equity += pnl
                    current_trade.update({
                        'exit_date': row['Date'],
                        'exit_price': exit_price,
                        'exit_reason': 'TP',
                        'pnl': pnl,
                        'return': (pnl / trade_equity) * 100 if trade_equity else 0,
                        'asset_after': equity,
                    })
                    trades.append(current_trade)
                    df.loc[df.index[i], 'signal'] = 2
                    position = 0
                elif self.exit_mode == 'trend' and just_turned_green:
                    exit_price = row['Close']
                    pnl = (entry_price - exit_price) * qty
                    equity += pnl
                    current_trade.update({
                        'exit_date': row['Date'],
                        'exit_price': exit_price,
                        'exit_reason': 'Trend Change',
                        'pnl': pnl,
                        'return': (pnl / trade_equity) * 100 if trade_equity else 0,
                        'asset_after': equity,
                    })
                    trades.append(current_trade)
                    df.loc[df.index[i], 'signal'] = 2
                    position = 0

            base_ok = (
                position == 0
                and pending_entry is None
                and (self.exit_mode == 'trend' or not np.isnan(swing_low[i]))
                and (self.exit_mode == 'trend' or not np.isnan(swing_high[i]))
            )
            limited_base_ok = base_ok and trade_count < self.max_trades

            if self.entry_mode == 'cross':
                cross_long_cond = (
                    supertrend_crossed_above_ema[i]
                    and not just_turned_green
                    and not just_turned_red
                    and pd.notna(row['supertrend'])
                    and pd.notna(row['ema200'])
                    and (self.exit_mode == 'trend' or (row['Open'] - swing_low[i]) > 0)
                    and base_ok
                )
                cross_short_cond = (
                    supertrend_crossed_below_ema[i]
                    and not just_turned_green
                    and not just_turned_red
                    and pd.notna(row['supertrend'])
                    and pd.notna(row['ema200'])
                    and (self.exit_mode == 'trend' or (swing_high[i] - row['Open']) > 0)
                    and base_ok
                )
                # In cross mode, only Supertrend/EMA cross entries are allowed.
                trend_long_cond = (
                    not awaiting_cross_entry
                    and required_cross_side == 'long'
                    and row['direction'] == 1
                    and row['Close'] > row['ema200']
                    and row['Open'] > row['ema200']
                    and (self.exit_mode == 'trend' or (row['Close'] - swing_low[i]) > 0)
                    and limited_base_ok
                )
                trend_short_cond = (
                    not awaiting_cross_entry
                    and required_cross_side == 'short'
                    and row['direction'] == -1
                    and row['Close'] < row['ema200']
                    and row['Open'] < row['ema200']
                    and (self.exit_mode == 'trend' or (swing_high[i] - row['Close']) > 0)
                    and limited_base_ok
                )
            elif self.entry_mode == 'weekly_long':
                long_cond = (
                    just_turned_green
                    and row['Close'] > row['ema200']
                    and row['Open'] > row['ema200']
                    and row['weekly_direction'] == 1
                    and limited_base_ok
                )
                short_cond = (
                    just_turned_red
                    and row['Close'] < row['ema200']
                    and row['Open'] < row['ema200']
                    and row['weekly_direction'] == -1
                    and limited_base_ok
                )
            elif self.entry_mode == 'weekly_bull_ema':
                long_cond = (
                    row['direction'] == 1
                    and row['Close'] > row['ema200']
                    and row['weekly_direction'] == 1
                    and limited_base_ok
                )
                short_cond = (
                    row['direction'] == -1
                    and row['Close'] < row['ema200']
                    and row['weekly_direction'] == -1
                    and limited_base_ok
                )
            elif self.entry_mode in {'adx_anytime', 'adx_uptrend'}:
                adx_uptrend_ok = (
                    self.entry_mode != 'adx_uptrend'
                    or (
                        i >= self.adx_trend_lookback
                        and pd.notna(row['adx'])
                        and pd.notna(df.iloc[i - self.adx_trend_lookback]['adx'])
                        and (row['adx'] - df.iloc[i - self.adx_trend_lookback]['adx']) > 0
                    )
                )
                long_cond = (
                    row['direction'] == 1
                    and row['Close'] > row['ema200']
                    and row['adx'] >= self.adx_threshold
                    and adx_uptrend_ok
                    and limited_base_ok
                )
                short_cond = (
                    row['direction'] == -1
                    and row['Close'] < row['ema200']
                    and row['adx'] >= self.adx_threshold
                    and adx_uptrend_ok
                    and limited_base_ok
                )
            else:
                long_cond = (
                    just_turned_green
                    and row['Close'] > row['ema200']
                    and row['Open'] > row['ema200']
                    and (self.exit_mode == 'trend' or (row['Close'] - swing_low[i]) > 0)
                    and limited_base_ok
                )
                short_cond = (
                    just_turned_red
                    and row['Close'] < row['ema200']
                    and row['Open'] < row['ema200']
                    and (self.exit_mode == 'trend' or (swing_high[i] - row['Close']) > 0)
                    and limited_base_ok
                )

            if self.entry_mode == 'cross':
                if cross_long_cond and i + 1 < len(df):
                    pending_entry = {
                        'side': 'long',
                        'signal_bar': i,
                        'count_toward_max': False,
                        'is_cross_entry': True,
                    }
                    df.loc[df.index[i], 'signal'] = 1
                elif cross_short_cond and i + 1 < len(df):
                    pending_entry = {
                        'side': 'short',
                        'signal_bar': i,
                        'count_toward_max': False,
                        'is_cross_entry': True,
                    }
                    df.loc[df.index[i], 'signal'] = -1
                elif trend_long_cond and i + 1 < len(df):
                    pending_entry = {
                        'side': 'long',
                        'signal_bar': i,
                        'count_toward_max': True,
                        'is_cross_entry': False,
                    }
                    df.loc[df.index[i], 'signal'] = 1
                elif trend_short_cond and i + 1 < len(df):
                    pending_entry = {
                        'side': 'short',
                        'signal_bar': i,
                        'count_toward_max': True,
                        'is_cross_entry': False,
                    }
                    df.loc[df.index[i], 'signal'] = -1
            elif self.entry_mode in {'weekly_bull_ema', 'adx_anytime', 'adx_uptrend'}:
                if long_cond:
                    trade_equity = equity
                    entry_price = float(row['Close'])
                    risk = entry_price - swing_low[i]
                    qty = max((equity * self.leverage) / entry_price, 0)
                    can_enter = self.exit_mode == 'trend' or (pd.notna(risk) and risk > 0)
                    if qty > 0 and can_enter:
                        position = 1
                        tp_price = entry_price + (self.tp_multiplier * risk) if self.exit_mode == 'tp' else None
                        sl_price = float(swing_low[i]) if self.exit_mode in {'tp', 'sl'} else None
                        current_trade = {
                            'entry_date': row['Date'],
                            'entry_reason': (
                                'daily_supertrend_ema_adx_uptrend'
                                if self.entry_mode == 'adx_uptrend'
                                else 'daily_supertrend_ema_adx'
                                if self.entry_mode == 'adx_anytime'
                                else 'daily_weekly_bull_ema'
                            ),
                            'entry_price': entry_price,
                            'exit_date': None,
                            'exit_price': None,
                            'type': 'long',
                            'qty': qty,
                            'tp': tp_price,
                            'sl': sl_price,
                            'exit_reason': None,
                            'pnl': None,
                            'return': None,
                        }
                        trade_count += 1
                        df.loc[df.index[i], 'signal'] = 1
                elif short_cond:
                    trade_equity = equity
                    entry_price = float(row['Close'])
                    risk = swing_high[i] - entry_price
                    qty = max((equity * self.leverage) / entry_price, 0)
                    can_enter = self.exit_mode == 'trend' or (pd.notna(risk) and risk > 0)
                    if qty > 0 and can_enter:
                        position = -1
                        tp_price = entry_price - (self.tp_multiplier * risk) if self.exit_mode == 'tp' else None
                        sl_price = float(swing_high[i]) if self.exit_mode in {'tp', 'sl'} else None
                        current_trade = {
                            'entry_date': row['Date'],
                            'entry_reason': (
                                'daily_supertrend_ema_adx_uptrend_short'
                                if self.entry_mode == 'adx_uptrend'
                                else 'daily_supertrend_ema_adx_short'
                                if self.entry_mode == 'adx_anytime'
                                else 'daily_weekly_bear_ema'
                            ),
                            'entry_price': entry_price,
                            'exit_date': None,
                            'exit_price': None,
                            'type': 'short',
                            'qty': qty,
                            'tp': tp_price,
                            'sl': sl_price,
                            'exit_reason': None,
                            'pnl': None,
                            'return': None,
                        }
                        trade_count += 1
                        df.loc[df.index[i], 'signal'] = -1
            else:
                if long_cond and i + 1 < len(df):
                    pending_entry = {'side': 'long', 'signal_bar': i, 'count_toward_max': True}
                    df.loc[df.index[i], 'signal'] = 1
                elif short_cond and i + 1 < len(df):
                    pending_entry = {'side': 'short', 'signal_bar': i, 'count_toward_max': True}
                    df.loc[df.index[i], 'signal'] = -1

            if position == 1:
                df.loc[df.index[i], 'asset'] = equity + ((float(row['Close']) - entry_price) * qty)
            elif position == -1:
                df.loc[df.index[i], 'asset'] = equity + ((entry_price - float(row['Close'])) * qty)
            else:
                df.loc[df.index[i], 'asset'] = equity

        if position != 0 and current_trade is not None and current_trade.get('exit_date') is None:
            final_row = df.iloc[-1]
            exit_price = float(final_row['Close'])
            if position == 1:
                pnl = (exit_price - entry_price) * qty
            else:
                pnl = (entry_price - exit_price) * qty
            return_pct = (pnl / trade_equity) * 100 if trade_equity else 0
            equity += pnl

            current_trade.update({
                'exit_date': final_row['Date'],
                'exit_price': exit_price,
                'exit_reason': 'End of Data',
                'pnl': pnl,
                'return': return_pct,
                'asset_after': equity,
            })
            trades.append(current_trade)
            df.loc[df.index[-1], 'signal'] = 2
            df.loc[df.index[-1], 'asset'] = equity

        self.trades = trades
        return df, trades


class SupertrendEmaGridSearchStrategy:
    """
    Grid search runner for a Supertrend + EMA strategy.

    The defaults mirror the demo script's preset parameter grid:
    ATR length 10-17, Supertrend factor 3.0-3.6, swing lookback 10-20,
    take-profit multiplier 1.0-2.0, and max trades 1-3.
    """

    DEFAULT_PARAM_GRID = {
        'atr_length': list(range(10, 18)),
        'factor': np.arange(3.0, 3.7, 0.1).round(1).tolist(),
        'swing_lookback': list(range(10, 21)),
        'tp_multiplier': np.arange(1.0, 2.1, 0.1).round(1).tolist(),
        'max_trades': list(range(1, 4)),
    }

    def __init__(
        self,
        ema_length=200,
        leverage=1,
        initial_equity=10000,
        min_trades=5,
        sort_by='composite',
        exit_mode='tp',
        evaluation_start_index=0,
        weekly_close_day='friday',
        adx_threshold=25,
        adx_trend_lookback=3,
    ):
        if ema_length <= 0:
            raise ValueError('EMA length must be greater than zero.')
        if leverage <= 0:
            raise ValueError('Leverage must be greater than zero.')
        if initial_equity <= 0:
            raise ValueError('Initial equity must be greater than zero.')
        if adx_threshold <= 0:
            raise ValueError('ADX threshold must be greater than zero.')
        if adx_trend_lookback <= 0:
            raise ValueError('ADX trend lookback must be greater than zero.')

        self.ema_length = int(ema_length)
        self.leverage = float(leverage)
        self.initial_equity = float(initial_equity)
        self.min_trades = int(min_trades)
        self.sort_by = sort_by if sort_by in {
            'composite', 'sharpe', 'profit_factor', 'net_return', 'win_rate'
        } else 'composite'
        requested_mode = str(exit_mode or 'tp').strip().lower()
        if requested_mode not in {
            'tp', 'trend', 'cross_trend', 'weekly_trend', 'weekly_bull_ema',
            'adx_trend', 'adx_tp', 'adx_uptrend', 'adx_uptrend_tp'
        }:
            requested_mode = 'tp'
        self.strategy_mode = requested_mode
        self.exit_mode = 'tp' if requested_mode in {'tp', 'adx_tp', 'adx_uptrend_tp'} else 'trend'
        self.entry_mode = (
            'cross' if requested_mode == 'cross_trend'
            else 'weekly_bull_ema' if requested_mode == 'weekly_bull_ema'
            else 'weekly_long' if requested_mode == 'weekly_trend'
            else 'adx_uptrend' if requested_mode in {'adx_uptrend', 'adx_uptrend_tp'}
            else 'adx_anytime' if requested_mode in {'adx_trend', 'adx_tp'}
            else 'flip'
        )
        self.evaluation_start_index = max(int(evaluation_start_index or 0), 0)
        self.weekly_close_day = resolve_weekly_close_day(weekly_close_day)
        self.adx_threshold = float(adx_threshold)
        self.adx_trend_lookback = int(adx_trend_lookback)

    @staticmethod
    def _rma(arr, length):
        out = np.empty(len(arr))
        if len(arr) == 0:
            return out

        out[0] = arr[0]
        alpha = 1.0 / length
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _ema(arr, length):
        alpha = 2.0 / (length + 1)
        out = np.empty(len(arr))
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
        return out

    def _build_supertrend(self, high_arr, low_arr, close_arr, atr, factor):
        n = len(close_arr)
        hl2 = (high_arr + low_arr) / 2.0
        raw_upper = hl2 + factor * atr
        raw_lower = hl2 - factor * atr

        fu = np.full(n, np.nan, dtype=float)
        fl = np.full(n, np.nan, dtype=float)
        st = np.full(n, np.nan, dtype=float)
        dire = np.full(n, -1, dtype=int)

        valid = np.where(~np.isnan(atr))[0]
        if len(valid) == 0:
            turned_green = np.zeros(n, dtype=bool)
            turned_red = np.zeros(n, dtype=bool)
            return turned_green, turned_red, dire, st

        start = valid[0]
        fu[start] = raw_upper[start]
        fl[start] = raw_lower[start]
        st[start] = fu[start]
        dire[start] = -1

        for i in range(start + 1, n):
            if np.isnan(raw_upper[i]) or np.isnan(raw_lower[i]):
                continue

            fu[i] = raw_upper[i] if (raw_upper[i] < fu[i - 1] or close_arr[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = raw_lower[i] if (raw_lower[i] > fl[i - 1] or close_arr[i - 1] < fl[i - 1]) else fl[i - 1]

            if st[i - 1] == fu[i - 1]:
                dire[i] = 1 if close_arr[i] > fu[i] else -1
            else:
                dire[i] = -1 if close_arr[i] < fl[i] else 1

            st[i] = fl[i] if dire[i] == 1 else fu[i]

        turned_green = (dire == 1) & (np.roll(dire, 1) == -1)
        turned_red = (dire == -1) & (np.roll(dire, 1) == 1)
        turned_green[0] = False
        turned_red[0] = False
        return turned_green, turned_red, dire, st

    def _build_adx(self, high_arr, low_arr, close_arr, atr, atr_length):
        prev_high = np.empty(len(high_arr))
        prev_low = np.empty(len(low_arr))
        prev_high[0] = high_arr[0]
        prev_low[0] = low_arr[0]
        prev_high[1:] = high_arr[:-1]
        prev_low[1:] = low_arr[:-1]

        up_move = high_arr - prev_high
        down_move = prev_low - low_arr
        up_move[0] = 0.0
        down_move[0] = 0.0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        safe_atr = np.where(atr == 0, np.nan, atr)
        plus_di = 100 * (self._rma(plus_dm, atr_length) / safe_atr)
        minus_di = 100 * (self._rma(minus_dm, atr_length) / safe_atr)
        di_sum = plus_di + minus_di
        dx = 100 * (np.abs(plus_di - minus_di) / np.where(di_sum == 0, np.nan, di_sum))
        return self._rma(np.nan_to_num(dx, nan=0.0), atr_length)

    def _build_weekly_direction_filter(self, date_arr, open_arr, high_arr, low_arr, close_arr, atr_length, factor):
        rows = pd.DataFrame({
            'Date': pd.to_datetime(date_arr),
            'Open': open_arr,
            'High': high_arr,
            'Low': low_arr,
            'Close': close_arr,
            '_row': np.arange(len(close_arr)),
        }).sort_values('Date')

        weekly = (
            rows.set_index('Date')
            .resample(WEEKLY_CLOSE_RULES[self.weekly_close_day], label='right', closed='right')
            .agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
            })
            .dropna(subset=['Open', 'High', 'Low', 'Close'])
            .reset_index()
        )
        if weekly.empty:
            return np.zeros(len(close_arr), dtype=int)

        weekly_close = weekly['Close'].to_numpy(dtype=float)
        weekly_high = weekly['High'].to_numpy(dtype=float)
        weekly_low = weekly['Low'].to_numpy(dtype=float)
        prev_close = np.empty(len(weekly_close))
        prev_close[0] = weekly_close[0]
        prev_close[1:] = weekly_close[:-1]
        true_range = np.maximum(
            weekly_high - weekly_low,
            np.maximum(np.abs(weekly_high - prev_close), np.abs(weekly_low - prev_close))
        )
        weekly_atr = self._rma(true_range, atr_length)
        _, _, weekly_direction, _ = self._build_supertrend(
            weekly_high,
            weekly_low,
            weekly_close,
            weekly_atr,
            factor,
        )

        weekly_signals = pd.DataFrame({
            'Date': weekly['Date'],
            'weekly_direction': weekly_direction,
        }).sort_values('Date')
        aligned = pd.merge_asof(
            rows[['Date', '_row']].sort_values('Date'),
            weekly_signals,
            on='Date',
            direction='backward',
        )
        out = np.zeros(len(close_arr), dtype=int)
        out[aligned['_row'].to_numpy(dtype=int)] = aligned['weekly_direction'].fillna(0).to_numpy(dtype=int)
        return out

    def _run_single_backtest(self, params, arrays, caches):
        close_arr = arrays['close']
        high_arr = arrays['high']
        low_arr = arrays['low']
        open_arr = arrays['open']
        date_arr = arrays['date']
        ema200 = arrays['ema200']
        ema_breakthru = arrays['ema_breakthru']
        n = len(close_arr)

        supertrend_key = (params['atr_length'], params['factor'])
        turned_green, turned_red, direction, supertrend_vals = caches['supertrend'][supertrend_key]
        weekly_direction = caches.get('weekly_direction', {}).get(supertrend_key)
        if weekly_direction is None:
            weekly_direction = np.zeros(n, dtype=int)
        adx_vals = caches.get('adx', {}).get(params['atr_length'])
        if adx_vals is None:
            adx_vals = np.zeros(n, dtype=float)
        sw_low = caches['sw_low'][params['swing_lookback']]
        sw_high = caches['sw_high'][params['swing_lookback']]

        equity = float(self.initial_equity)
        trade_count = 0
        position = None
        pending_entry = None
        awaiting_cross_entry = self.entry_mode == 'cross'
        pnl_list = []
        equity_vals = []
        trades = []

        if self.entry_mode in {'adx_anytime', 'adx_uptrend'}:
            start_index = max(self.ema_length, self.evaluation_start_index + params['swing_lookback'])
        else:
            start_index = max(self.ema_length, self.evaluation_start_index)
        if self.entry_mode == 'adx_uptrend':
            start_index = max(start_index, self.evaluation_start_index + self.adx_trend_lookback)
        prev_ema200 = np.roll(ema200, 1)
        prev_ema200[0] = ema200[0]
        supertrend_crossed_above_ema = np.zeros(n, dtype=bool)
        supertrend_crossed_below_ema = np.zeros(n, dtype=bool)
        supertrend_ema_cross = np.zeros(n, dtype=bool)
        required_cross_side = None

        if self.entry_mode == 'cross':
            prev_supertrend = np.roll(supertrend_vals, 1)
            prev_supertrend[0] = supertrend_vals[0]
            supertrend_crossed_above_ema = (
                (supertrend_vals > ema200)
                & (prev_supertrend <= prev_ema200)
            )
            supertrend_crossed_below_ema = (
                (supertrend_vals < ema200)
                & (prev_supertrend >= prev_ema200)
            )
            supertrend_crossed_above_ema[0] = False
            supertrend_crossed_below_ema[0] = False

            prev_direction = np.roll(direction, 1)
            prev_direction[0] = direction[0]
            stable_direction = direction == prev_direction
            trend_changed = direction != prev_direction
            trend_changed[0] = False
            prev_trend_changed = np.roll(trend_changed, 1)
            prev_trend_changed[0] = False
            recent_trend_change = trend_changed | prev_trend_changed
            valid_cross_bar = stable_direction & (~recent_trend_change)

            supertrend_crossed_above_ema &= valid_cross_bar
            supertrend_crossed_below_ema &= valid_cross_bar
            supertrend_ema_cross = supertrend_crossed_above_ema | supertrend_crossed_below_ema

            ref_idx = max(start_index - 1, 0)
            if np.isnan(supertrend_vals[ref_idx]) or np.isnan(ema200[ref_idx]):
                awaiting_cross_entry = False
            elif supertrend_vals[ref_idx] < ema200[ref_idx]:
                required_cross_side = 'long'
            elif supertrend_vals[ref_idx] > ema200[ref_idx]:
                required_cross_side = 'short'
            else:
                awaiting_cross_entry = False

        for i in range(start_index, n):
            c = close_arr[i]
            o = open_arr[i]
            bh = high_arr[i]
            bl = low_arr[i]
            e2 = ema200[i]
            just_turned_green = turned_green[i]
            just_turned_red = turned_red[i]

            reset_breakthru = supertrend_ema_cross[i] if self.entry_mode == 'cross' else ema_breakthru[i]
            if reset_breakthru and position is None and pending_entry is None:
                trade_count = 0
                pending_entry = None
                awaiting_cross_entry = self.entry_mode == 'cross'
                if self.entry_mode == 'cross':
                    if supertrend_crossed_above_ema[i]:
                        required_cross_side = 'long'
                    elif supertrend_crossed_below_ema[i]:
                        required_cross_side = 'short'

            if (
                self.entry_mode == 'cross'
                and pending_entry is not None
                and pending_entry.get('is_cross_entry')
                and (just_turned_green or just_turned_red)
            ):
                pending_entry = None

            if pending_entry is not None and position is None:
                fill = o
                sig_i = pending_entry['signal_bar']
                trade_equity = equity
                qty = max((equity * self.leverage) / fill, 0)

                if qty > 0:
                    if pending_entry['side'] == 'long':
                        risk = fill - sw_low[sig_i]
                        if self.exit_mode == 'trend' or risk > 0:
                            position = {
                                'side': 'long', 'entry': fill, 'qty': qty,
                                'tp': fill + params['tp_multiplier'] * risk if self.exit_mode == 'tp' else None,
                                'sl': sw_low[sig_i] if self.exit_mode == 'tp' else None,
                                'trade_equity': trade_equity,
                                'entry_index': i,
                                'entry_reason': 'cross_entry' if pending_entry.get('is_cross_entry') else 'trend_follow_on',
                            }
                            if pending_entry.get('count_toward_max', True):
                                trade_count += 1
                            if pending_entry.get('is_cross_entry'):
                                awaiting_cross_entry = False
                    else:
                        risk = sw_high[sig_i] - fill
                        if self.exit_mode == 'trend' or risk > 0:
                            position = {
                                'side': 'short', 'entry': fill, 'qty': qty,
                                'tp': fill - params['tp_multiplier'] * risk if self.exit_mode == 'tp' else None,
                                'sl': sw_high[sig_i] if self.exit_mode == 'tp' else None,
                                'trade_equity': trade_equity,
                                'entry_index': i,
                                'entry_reason': 'cross_entry' if pending_entry.get('is_cross_entry') else 'trend_follow_on',
                            }
                            if pending_entry.get('count_toward_max', True):
                                trade_count += 1
                            if pending_entry.get('is_cross_entry'):
                                awaiting_cross_entry = False
                pending_entry = None

            if position is not None:
                exited = False
                exit_px = c
                exit_index = i

                if position['side'] == 'long':
                    if self.exit_mode == 'tp' and bl <= position['sl']:
                        exit_px = position['sl']
                        exited = True
                    elif self.exit_mode == 'tp' and bh >= position['tp']:
                        exit_px = position['tp']
                        exited = True
                    elif self.exit_mode == 'trend' and turned_red[i]:
                        exit_px = open_arr[i + 1] if i + 1 < n else c
                        exit_index = min(i + 1, n - 1)
                        exited = True
                else:
                    if self.exit_mode == 'tp' and bh >= position['sl']:
                        exit_px = position['sl']
                        exited = True
                    elif self.exit_mode == 'tp' and bl <= position['tp']:
                        exit_px = position['tp']
                        exited = True
                    elif self.exit_mode == 'trend' and turned_green[i]:
                        exit_px = open_arr[i + 1] if i + 1 < n else c
                        exit_index = min(i + 1, n - 1)
                        exited = True

                if exited:
                    pnl = ((exit_px - position['entry']) if position['side'] == 'long'
                           else (position['entry'] - exit_px)) * position['qty']
                    equity += pnl
                    pnl_list.append(pnl)
                    asset_after = equity
                    if self.exit_mode == 'tp' and exit_px == position['sl']:
                        exit_reason = 'SL'
                    elif self.exit_mode == 'tp' and exit_px == position['tp']:
                        exit_reason = 'TP'
                    elif turned_red[i] or turned_green[i]:
                        exit_reason = 'Trend Change'
                    else:
                        exit_reason = 'End of Data'
                    trades.append({
                        'entry_index': position['entry_index'],
                        'entry_price': position['entry'],
                        'exit_index': exit_index,
                        'exit_price': exit_px,
                        'side': position['side'],
                        'entry_date': date_arr[position['entry_index']],
                        'exit_date': date_arr[exit_index],
                        'type': position['side'],
                        'entry_reason': position.get('entry_reason'),
                        'tp': position['tp'],
                        'sl': position['sl'],
                        'exit_reason': exit_reason,
                        'asset_after': asset_after,
                        'pnl': pnl,
                        'return': (pnl / position['trade_equity']) * 100 if position.get('trade_equity') else 0,
                    })
                    position = None

            base_ok = (
                position is None
                and pending_entry is None
                and (self.exit_mode == 'trend' or not np.isnan(sw_low[i]))
                and (self.exit_mode == 'trend' or not np.isnan(sw_high[i]))
            )
            limited_base_ok = base_ok and trade_count < params['max_trades']

            if self.entry_mode == 'cross':
                cross_long_sig = (
                    supertrend_crossed_above_ema[i]
                    and not just_turned_green
                    and not just_turned_red
                    and not np.isnan(supertrend_vals[i])
                    and not np.isnan(e2)
                    and (self.exit_mode == 'trend' or (o - sw_low[i]) > 0)
                    and base_ok
                )
                cross_short_sig = (
                    supertrend_crossed_below_ema[i]
                    and not just_turned_green
                    and not just_turned_red
                    and not np.isnan(supertrend_vals[i])
                    and not np.isnan(e2)
                    and (self.exit_mode == 'trend' or (sw_high[i] - o) > 0)
                    and base_ok
                )
                trend_long_sig = (
                    not awaiting_cross_entry
                    and required_cross_side == 'long'
                    and direction[i] == 1
                    and c > e2 and o > e2
                    and (self.exit_mode == 'trend' or (c - sw_low[i]) > 0)
                    and limited_base_ok
                )
                trend_short_sig = (
                    not awaiting_cross_entry
                    and required_cross_side == 'short'
                    and direction[i] == -1
                    and c < e2 and o < e2
                    and (self.exit_mode == 'trend' or (sw_high[i] - c) > 0)
                    and limited_base_ok
                )

                if cross_long_sig and i + 1 < n:
                    pending_entry = {
                        'side': 'long',
                        'signal_bar': i,
                        'count_toward_max': False,
                        'is_cross_entry': True,
                    }
                elif cross_short_sig and i + 1 < n:
                    pending_entry = {
                        'side': 'short',
                        'signal_bar': i,
                        'count_toward_max': False,
                        'is_cross_entry': True,
                    }
                elif trend_long_sig and i + 1 < n:
                    pending_entry = {
                        'side': 'long',
                        'signal_bar': i,
                        'count_toward_max': True,
                        'is_cross_entry': False,
                    }
                elif trend_short_sig and i + 1 < n:
                    pending_entry = {
                        'side': 'short',
                        'signal_bar': i,
                        'count_toward_max': True,
                        'is_cross_entry': False,
                    }
            elif self.entry_mode == 'weekly_long':
                long_sig = (
                    just_turned_green
                    and c > e2 and o > e2
                    and weekly_direction[i] == 1
                    and limited_base_ok
                )
                short_sig = (
                    just_turned_red
                    and c < e2 and o < e2
                    and weekly_direction[i] == -1
                    and limited_base_ok
                )
                if long_sig and i + 1 < n:
                    pending_entry = {'side': 'long', 'signal_bar': i, 'count_toward_max': True}
                elif short_sig and i + 1 < n:
                    pending_entry = {'side': 'short', 'signal_bar': i, 'count_toward_max': True}
            elif self.entry_mode == 'weekly_bull_ema':
                long_sig = (
                    direction[i] == 1
                    and c > e2
                    and weekly_direction[i] == 1
                    and limited_base_ok
                )
                short_sig = (
                    direction[i] == -1
                    and c < e2
                    and weekly_direction[i] == -1
                    and limited_base_ok
                )

                if long_sig:
                    qty = max((equity * self.leverage) / c, 0)
                    if qty > 0:
                        position = {
                            'side': 'long', 'entry': c, 'qty': qty,
                            'tp': None, 'sl': None,
                            'trade_equity': equity,
                            'entry_index': i,
                            'entry_reason': 'daily_weekly_bull_ema',
                        }
                        trade_count += 1
                elif short_sig:
                    qty = max((equity * self.leverage) / c, 0)
                    if qty > 0:
                        position = {
                            'side': 'short', 'entry': c, 'qty': qty,
                            'tp': None, 'sl': None,
                            'trade_equity': equity,
                            'entry_index': i,
                            'entry_reason': 'daily_weekly_bear_ema',
                        }
                        trade_count += 1
            elif self.entry_mode in {'adx_anytime', 'adx_uptrend'}:
                adx_uptrend_ok = (
                    self.entry_mode != 'adx_uptrend'
                    or (
                        i >= self.adx_trend_lookback
                        and not np.isnan(adx_vals[i])
                        and not np.isnan(adx_vals[i - self.adx_trend_lookback])
                        and (adx_vals[i] - adx_vals[i - self.adx_trend_lookback]) > 0
                    )
                )
                long_sig = (
                    direction[i] == 1
                    and c > e2
                    and adx_vals[i] >= self.adx_threshold
                    and adx_uptrend_ok
                    and limited_base_ok
                )
                short_sig = (
                    direction[i] == -1
                    and c < e2
                    and adx_vals[i] >= self.adx_threshold
                    and adx_uptrend_ok
                    and limited_base_ok
                )

                if long_sig:
                    risk = c - sw_low[i]
                    qty = max((equity * self.leverage) / c, 0)
                    if qty > 0 and (self.exit_mode == 'trend' or risk > 0):
                        position = {
                            'side': 'long', 'entry': c, 'qty': qty,
                            'tp': c + params['tp_multiplier'] * risk if self.exit_mode == 'tp' else None,
                            'sl': sw_low[i] if self.exit_mode == 'tp' else None,
                            'trade_equity': equity,
                            'entry_index': i,
                            'entry_reason': (
                                'daily_supertrend_ema_adx_uptrend'
                                if self.entry_mode == 'adx_uptrend'
                                else 'daily_supertrend_ema_adx'
                            ),
                        }
                        trade_count += 1
                elif short_sig:
                    risk = sw_high[i] - c
                    qty = max((equity * self.leverage) / c, 0)
                    if qty > 0 and (self.exit_mode == 'trend' or risk > 0):
                        position = {
                            'side': 'short', 'entry': c, 'qty': qty,
                            'tp': c - params['tp_multiplier'] * risk if self.exit_mode == 'tp' else None,
                            'sl': sw_high[i] if self.exit_mode == 'tp' else None,
                            'trade_equity': equity,
                            'entry_index': i,
                            'entry_reason': (
                                'daily_supertrend_ema_adx_uptrend_short'
                                if self.entry_mode == 'adx_uptrend'
                                else 'daily_supertrend_ema_adx_short'
                            ),
                        }
                        trade_count += 1
            else:
                long_sig = (
                    just_turned_green
                    and c > e2 and o > e2
                    and (self.exit_mode == 'trend' or (c - sw_low[i]) > 0)
                    and limited_base_ok
                )
                short_sig = (
                    just_turned_red
                    and c < e2 and o < e2
                    and (self.exit_mode == 'trend' or (sw_high[i] - c) > 0)
                    and limited_base_ok
                )

                if long_sig and i + 1 < n:
                    pending_entry = {'side': 'long', 'signal_bar': i, 'count_toward_max': True}
                elif short_sig and i + 1 < n:
                    pending_entry = {'side': 'short', 'signal_bar': i, 'count_toward_max': True}

            equity_vals.append(max(equity, 0))

        if position is not None:
            lc = close_arr[-1]
            pnl = ((lc - position['entry']) if position['side'] == 'long'
                   else (position['entry'] - lc)) * position['qty']
            equity += pnl
            pnl_list.append(pnl)
            exit_reason = 'End of Data'
            asset_after = equity
            trades.append({
                'entry_index': position['entry_index'],
                'entry_price': position['entry'],
                'exit_index': n - 1,
                'exit_price': lc,
                'side': position['side'],
                'entry_date': date_arr[position['entry_index']],
                'exit_date': date_arr[n - 1],
                'type': position['side'],
                'entry_reason': position.get('entry_reason'),
                'tp': position.get('tp'),
                'sl': position.get('sl'),
                'exit_reason': exit_reason,
                'asset_after': asset_after,
                'pnl': pnl,
                'return': (pnl / position['trade_equity']) * 100 if position.get('trade_equity') else 0,
            })

        total = len(pnl_list)
        if total < self.min_trades:
            return None, trades

        pnl_arr = np.array(pnl_list)
        wins = pnl_arr[pnl_arr > 0]
        losses = pnl_arr[pnl_arr <= 0]
        win_rate = len(wins) / total
        gross_profit = wins.sum() if len(wins) else 0.0
        gross_loss = abs(losses.sum()) if len(losses) else 1e-9
        profit_factor = gross_profit / gross_loss
        net_return = (equity / self.initial_equity - 1) * 100

        eq_s = pd.Series(equity_vals)
        daily_r = eq_s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        std = daily_r.std()
        sharpe = daily_r.mean() / std * np.sqrt(252) if std and std > 0 else 0.0
        max_dd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100 if len(eq_s) else 0.0

        composite = (
            0.4 * np.clip(sharpe, -3, 5)
            + 0.4 * np.clip(profit_factor, 0, 10)
            + 0.2 * (win_rate * 10)
        )

        return {
            **params,
            'exit_mode': self.strategy_mode,
            'entry_mode': self.entry_mode,
            'total_trades': total,
            'win_rate': round(win_rate * 100, 1),
            'profit_factor': round(float(profit_factor), 3),
            'net_return': round(float(net_return), 2),
            'sharpe': round(float(sharpe), 3),
            'max_drawdown': round(float(max_dd), 2),
            'composite': round(float(composite), 4),
        }, trades

    def _dedupe_ranked_results(self, ranked_results):
        if self.exit_mode == 'tp':
            return ranked_results

        unique_results = []
        seen = set()
        for row in ranked_results:
            key = (
                row.get('atr_length'),
                row.get('factor'),
                row.get('swing_lookback'),
                row.get('max_trades'),
                row.get('entry_mode'),
                row.get('exit_mode'),
            )
            if key in seen:
                continue
            seen.add(key)
            unique_results.append(row)
        return unique_results

    def run(self, df, top_n=20):
        if len(df) < self.ema_length + 10:
            raise ValueError(
                f'Not enough data for Supertrend + EMA grid search. '
                f'Received {len(df)} rows; need at least {self.ema_length + 10}.'
            )

        work = df.copy().dropna(subset=['Open', 'High', 'Low', 'Close'])
        close_arr = work['Close'].to_numpy(dtype=float)
        high_arr = work['High'].to_numpy(dtype=float)
        low_arr = work['Low'].to_numpy(dtype=float)
        open_arr = work['Open'].to_numpy(dtype=float)
        date_arr = work['Date'].to_numpy()
        n = len(work)

        prev_c = np.empty(n)
        prev_c[0] = close_arr[0]
        prev_c[1:] = close_arr[:-1]
        tr = np.maximum(
            high_arr - low_arr,
            np.maximum(np.abs(high_arr - prev_c), np.abs(low_arr - prev_c))
        )

        ema200 = self._ema(close_arr, self.ema_length)
        prev_close_s = np.roll(close_arr, 1)
        prev_close_s[0] = close_arr[0]
        prev_ema200 = np.roll(ema200, 1)
        prev_ema200[0] = ema200[0]
        crossed_above = (close_arr > ema200) & (prev_close_s <= prev_ema200)
        crossed_below = (close_arr < ema200) & (prev_close_s >= prev_ema200)
        crossed_above[0] = False
        crossed_below[0] = False
        ema_breakthru = crossed_above | crossed_below

        grid = self.DEFAULT_PARAM_GRID
        atr_cache = {length: self._rma(tr, length) for length in set(grid['atr_length'])}
        adx_cache = {
            length: self._build_adx(high_arr, low_arr, close_arr, atr_cache[length], length)
            for length in set(grid['atr_length'])
        }
        sw_low_cache = {
            length: pd.Series(low_arr).shift(1).rolling(length).min().values
            for length in set(grid['swing_lookback'])
        }
        sw_high_cache = {
            length: pd.Series(high_arr).shift(1).rolling(length).max().values
            for length in set(grid['swing_lookback'])
        }
        supertrend_cache = {}
        for atr_length in set(grid['atr_length']):
            for factor in set(grid['factor']):
                supertrend_cache[(atr_length, factor)] = self._build_supertrend(
                    high_arr, low_arr, close_arr, atr_cache[atr_length], factor
                )
        weekly_direction_cache = {}
        if self.entry_mode in {'weekly_long', 'weekly_bull_ema'}:
            for atr_length in set(grid['atr_length']):
                for factor in set(grid['factor']):
                    weekly_direction_cache[(atr_length, factor)] = self._build_weekly_direction_filter(
                        date_arr,
                        open_arr,
                        high_arr,
                        low_arr,
                        close_arr,
                        atr_length,
                        factor,
                    )

        arrays = {
            'close': close_arr, 'high': high_arr, 'low': low_arr, 'open': open_arr,
            'date': date_arr,
            'ema200': ema200, 'ema_breakthru': ema_breakthru,
        }
        caches = {
            'sw_low': sw_low_cache,
            'sw_high': sw_high_cache,
            'supertrend': supertrend_cache,
            'weekly_direction': weekly_direction_cache,
            'adx': adx_cache,
        }

        results = []
        best_trades = []
        best_score = None
        combos_run = 0

        for atr_length in grid['atr_length']:
            for factor in grid['factor']:
                for swing_lookback in grid['swing_lookback']:
                    for tp_multiplier in grid['tp_multiplier']:
                        for max_trades in grid['max_trades']:
                            params = {
                                'atr_length': atr_length,
                                'factor': factor,
                                'swing_lookback': swing_lookback,
                                'tp_multiplier': tp_multiplier,
                                'max_trades': max_trades,
                            }
                            combos_run += 1
                            row, trades = self._run_single_backtest(params, arrays, caches)
                            if row is None:
                                continue
                            results.append(row)
                            score = row[self.sort_by]
                            if best_score is None or score > best_score:
                                best_score = score
                                best_trades = trades

        if not results:
            return {
                'rows': work,
                'results': [],
                'top_results': [],
                'best': None,
                'best_trades': [],
                'combos_run': combos_run,
                'valid_results': 0,
                'skipped_results': combos_run,
                'sort_by': self.sort_by,
                'exit_mode': self.strategy_mode,
                'entry_mode': self.entry_mode,
                'ema': ema200,
                'best_supertrend': None,
            }

        raw_result_count = len(results)
        results = sorted(results, key=lambda item: item[self.sort_by], reverse=True)
        results = self._dedupe_ranked_results(results)
        best = results[0]
        best_key = (best['atr_length'], best['factor'])
        best_supertrend = supertrend_cache[best_key][3]

        return {
            'rows': work,
            'results': results,
            'top_results': results[:top_n],
            'best': best,
            'best_trades': best_trades,
            'combos_run': combos_run,
            'valid_results': len(results),
            'raw_valid_results': raw_result_count,
            'deduped_results': raw_result_count - len(results),
            'skipped_results': combos_run - raw_result_count,
            'sort_by': self.sort_by,
            'exit_mode': self.strategy_mode,
            'entry_mode': self.entry_mode,
            'ema': ema200,
            'best_supertrend': best_supertrend,
        }
