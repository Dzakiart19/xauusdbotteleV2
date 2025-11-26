import pandas as pd
import numpy as np
from typing import Dict, Optional, Union
from numpy.typing import NDArray


class IndicatorError(Exception):
    """Base exception for indicator calculation errors"""
    pass

def validate_series(series: pd.Series, min_length: int = 1, fill_value: float = 0.0) -> pd.Series:
    """
    Validate and sanitize a pandas Series before mathematical operations.
    
    Args:
        series: The pandas Series to validate
        min_length: Minimum required length
        fill_value: Value to use for filling NaN/None values
    
    Returns:
        Validated Series with NaN values filled
    
    Raises:
        ValueError: If series is None or not a pandas Series
    """
    if series is None:
        raise ValueError("Series cannot be None")
    
    if not isinstance(series, pd.Series):
        raise ValueError(f"Expected pandas Series, got {type(series)}")
    
    if len(series) < min_length:
        return pd.Series([fill_value] * min_length)
    
    return series.fillna(fill_value)


def _ensure_series(data: Union[pd.Series, pd.DataFrame, np.ndarray, float, int], 
                   index: Optional[pd.Index] = None) -> pd.Series:
    """
    Convert various data types to pandas Series.
    
    Args:
        data: Input data (Series, DataFrame, ndarray, float, or int)
        index: Optional index for the resulting Series
    
    Returns:
        pandas Series
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        if len(data.columns) == 1:
            return pd.Series(data.iloc[:, 0], index=data.index)
        return pd.Series(data.iloc[:, 0], index=data.index)
    elif isinstance(data, np.ndarray):
        return pd.Series(data, index=index)
    elif isinstance(data, (float, int)):
        if index is not None:
            return pd.Series([data] * len(index), index=index)
        return pd.Series([data])
    else:
        return pd.Series([data])


def safe_divide(numerator: Union[pd.Series, pd.DataFrame, np.ndarray, float], 
                denominator: Union[pd.Series, pd.DataFrame, np.ndarray, float], 
                fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two Series, handling division by zero and NaN values.
    
    Args:
        numerator: Numerator (Series, DataFrame, ndarray, or float)
        denominator: Denominator (Series, DataFrame, ndarray, or float)
        fill_value: Value to use when division is undefined
    
    Returns:
        Result Series with safe division
    """
    num_series = _ensure_series(numerator)
    denom_series = _ensure_series(denominator, index=num_series.index if hasattr(num_series, 'index') else None)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num_series / denom_series
        result = result.replace([np.inf, -np.inf], fill_value)
        result = result.fillna(fill_value)
    return pd.Series(result)


def safe_series_operation(series: pd.Series, operation: str = 'value', 
                          index: int = -1, default: float = 0.0) -> float:
    """
    Safely extract a value from a Series with null checking.
    
    Args:
        series: The pandas Series
        operation: Type of operation ('value', 'mean', 'sum', 'min', 'max')
        index: Index position for 'value' operation
        default: Default value if operation fails
    
    Returns:
        The extracted value or default
    """
    try:
        if series is None or len(series) == 0:
            return default
        
        if operation == 'value':
            if abs(index) > len(series):
                return default
            val = series.iloc[index]
            return default if bool(pd.isna(val)) else float(val)
        elif operation == 'mean':
            val = series.mean()
            return default if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
        elif operation == 'sum':
            val = series.sum()
            return default if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
        elif operation == 'min':
            val = series.min()
            return default if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
        elif operation == 'max':
            val = series.max()
            return default if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)
        else:
            return default
    except (IndexError, KeyError, TypeError):
        return default


class IndicatorEngine:
    def __init__(self, config):
        self.config = config
        self.ema_periods = config.EMA_PERIODS
        self.rsi_period = config.RSI_PERIOD
        self.stoch_k_period = config.STOCH_K_PERIOD
        self.stoch_d_period = config.STOCH_D_PERIOD
        self.stoch_smooth_k = config.STOCH_SMOOTH_K
        self.atr_period = config.ATR_PERIOD
        self.macd_fast = config.MACD_FAST
        self.macd_slow = config.MACD_SLOW
        self.macd_signal = config.MACD_SIGNAL
    
    def _validate_dataframe(self, df: pd.DataFrame, required_columns: Optional[list] = None) -> bool:
        """
        Validate DataFrame has required columns and sufficient data.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
        
        Returns:
            True if valid, False otherwise
        """
        if df is None or not isinstance(df, pd.DataFrame):
            return False
        
        if len(df) == 0:
            return False
        
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in df.columns:
                return False
        
        return True
    
    def _get_column_series(self, df: pd.DataFrame, column: str, fill_value: float = 0.0) -> pd.Series:
        """
        Safely get a column from DataFrame with null handling.
        
        Args:
            df: Source DataFrame
            column: Column name
            fill_value: Value to fill NaN with
        
        Returns:
            Validated Series
        """
        if column not in df.columns:
            return pd.Series([fill_value] * len(df), index=df.index)
        
        return pd.Series(df[column].fillna(fill_value))
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Exponential Moving Average with null handling."""
        if not self._validate_dataframe(df, ['close']):
            return pd.Series([0.0])
        
        close = self._get_column_series(df, 'close')
        if len(close) < period:
            return pd.Series([0.0] * len(close), index=df.index)
        
        result = close.ewm(span=period, adjust=False).mean()
        return pd.Series(result.fillna(0.0))
    
    def calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate RSI with proper null handling for division operations."""
        if not self._validate_dataframe(df, ['close']):
            return pd.Series([50.0])
        
        close = self._get_column_series(df, 'close')
        if len(close) < period + 1:
            return pd.Series([50.0] * len(close), index=df.index)
        
        delta = close.diff()
        delta = delta.fillna(0.0)
        
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        avg_gain = pd.Series(avg_gain).fillna(0.0)
        avg_loss = pd.Series(avg_loss).fillna(0.0)
        
        rs = safe_divide(avg_gain, avg_loss, fill_value=0.0)
        
        rsi = 100 - (100 / (1 + rs))
        rsi = pd.Series(rsi).fillna(50.0)
        rsi = rsi.clip(0, 100)
        
        return pd.Series(rsi)
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int, d_period: int, smooth_k: int) -> tuple:
        """Calculate Stochastic oscillator with null handling."""
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            empty_series = pd.Series([50.0])
            return empty_series, empty_series
        
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        close = self._get_column_series(df, 'close')
        
        if len(df) < k_period:
            empty_series = pd.Series([50.0] * len(df), index=df.index)
            return empty_series, empty_series
        
        low_min = low.rolling(window=k_period, min_periods=1).min()
        high_max = high.rolling(window=k_period, min_periods=1).max()
        
        low_min = low_min.fillna(low)
        high_max = high_max.fillna(high)
        
        range_diff = high_max - low_min
        range_diff = range_diff.replace(0, 1e-10)
        
        stoch_k = 100 * safe_divide(close - low_min, range_diff, fill_value=50.0)
        stoch_k = stoch_k.fillna(50.0).clip(0, 100)
        
        stoch_k = stoch_k.rolling(window=smooth_k, min_periods=1).mean()
        stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
        
        stoch_k = stoch_k.fillna(50.0)
        stoch_d = stoch_d.fillna(50.0)
        
        return stoch_k, stoch_d
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range with null handling."""
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            return pd.Series([0.0])
        
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        close = self._get_column_series(df, 'close')
        
        if len(df) < 2:
            return pd.Series([0.0] * len(df), index=df.index)
        
        prev_close = close.shift(1).fillna(close)
        
        high_low = (high - low).abs()
        high_close = (high - prev_close).abs()
        low_close = (low - prev_close).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr = pd.Series(tr).fillna(0.0)
        
        atr = tr.rolling(window=period, min_periods=1).mean()
        atr = pd.Series(atr).fillna(0.0)
        
        return pd.Series(atr)
    
    def calculate_volume_average(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate volume average with null handling."""
        if not self._validate_dataframe(df, ['volume']):
            return pd.Series([0.0])
        
        volume = self._get_column_series(df, 'volume')
        
        if len(volume) < period:
            result = volume.rolling(window=len(volume), min_periods=1).mean()
            return pd.Series(result).fillna(0.0)
        
        result = volume.rolling(window=period, min_periods=1).mean()
        return pd.Series(result).fillna(0.0)
    
    def calculate_twin_range_filter(self, df: pd.DataFrame, period: int = 27, multiplier: float = 2.0) -> tuple:
        """
        Twin Range Filter - Indikator untuk filter trend menggunakan smooth range
        
        Args:
            df: DataFrame dengan kolom OHLC
            period: Periode untuk smoothing (default 27)
            multiplier: Multiplier untuk range calculation (default 2.0)
        
        Returns:
            tuple: (upper_filter, lower_filter, trend_direction)
                   trend_direction: 1 untuk bullish, -1 untuk bearish, 0 untuk neutral
        """
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            empty = pd.Series([0.0])
            return empty, empty, pd.Series([0])
        
        close = self._get_column_series(df, 'close')
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        
        if len(df) < period:
            zeros = pd.Series([0.0] * len(df), index=df.index)
            trend = pd.Series([0] * len(df), index=df.index)
            return zeros, zeros, trend
        
        range_val = (high - low).fillna(0.0)
        smooth_range = range_val.ewm(span=period, adjust=False).mean()
        smooth_range = smooth_range.fillna(0.0)
        
        range_filter = smooth_range * multiplier
        
        upper_filter = close + range_filter
        lower_filter = close - range_filter
        
        upper_filter = upper_filter.ewm(span=period, adjust=False).mean().fillna(close)
        lower_filter = lower_filter.ewm(span=period, adjust=False).mean().fillna(close)
        
        trend = pd.Series(0, index=df.index)
        for i in range(1, len(df)):
            close_val = close.iloc[i] if not pd.isna(close.iloc[i]) else 0.0
            upper_prev = upper_filter.iloc[i-1] if not pd.isna(upper_filter.iloc[i-1]) else close_val
            lower_prev = lower_filter.iloc[i-1] if not pd.isna(lower_filter.iloc[i-1]) else close_val
            
            if close_val > upper_prev:
                trend.iloc[i] = 1
            elif close_val < lower_prev:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1] if i > 0 else 0
        
        return upper_filter, lower_filter, trend
    
    def calculate_market_bias_cerebr(self, df: pd.DataFrame, period: int = 60, smoothing: int = 10) -> tuple:
        """
        Market Bias (CEREBR) - Indikator untuk deteksi bias pasar
        
        Args:
            df: DataFrame dengan kolom OHLC
            period: Periode untuk CEREBR calculation (default 60)
            smoothing: Periode smoothing (default 10)
        
        Returns:
            tuple: (cerebr_value, cerebr_signal, bias_direction)
                   cerebr_value: Nilai CEREBR
                   cerebr_signal: Signal line (smoothed)
                   bias_direction: 1 untuk bullish bias, -1 untuk bearish bias, 0 untuk neutral
        """
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            empty = pd.Series([50.0])
            return empty, empty, pd.Series([0])
        
        close = self._get_column_series(df, 'close')
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        
        if len(df) < period:
            default_val = pd.Series([50.0] * len(df), index=df.index)
            trend = pd.Series([0] * len(df), index=df.index)
            return default_val, default_val, trend
        
        high_period = high.rolling(window=period, min_periods=1).max()
        low_period = low.rolling(window=period, min_periods=1).min()
        
        high_period = high_period.fillna(high)
        low_period = low_period.fillna(low)
        
        range_period = high_period - low_period
        range_period = range_period.replace(0, 1e-10)
        range_period = range_period.fillna(1e-10)
        
        cerebr_raw = safe_divide(close - low_period, range_period, fill_value=0.5) * 100
        cerebr_raw = cerebr_raw.fillna(50.0)
        
        cerebr_value = cerebr_raw.ewm(span=smoothing, adjust=False).mean()
        cerebr_signal = cerebr_value.ewm(span=smoothing, adjust=False).mean()
        
        cerebr_value = cerebr_value.fillna(50.0)
        cerebr_signal = cerebr_signal.fillna(50.0)
        
        bias_direction = pd.Series(0, index=df.index)
        for i in range(len(df)):
            val = cerebr_value.iloc[i] if not pd.isna(cerebr_value.iloc[i]) else 50.0
            if val > 60:
                bias_direction.iloc[i] = 1
            elif val < 40:
                bias_direction.iloc[i] = -1
            else:
                bias_direction.iloc[i] = 0
        
        return cerebr_value, cerebr_signal, bias_direction
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD with null handling."""
        if not self._validate_dataframe(df, ['close']):
            empty = pd.Series([0.0])
            return empty, empty, empty
        
        close = self._get_column_series(df, 'close')
        
        if len(close) < slow:
            zeros = pd.Series([0.0] * len(df), index=df.index)
            return zeros, zeros, zeros
        
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        ema_fast = ema_fast.fillna(close)
        ema_slow = ema_slow.fillna(close)
        
        macd_line = ema_fast - ema_slow
        macd_line = macd_line.fillna(0.0)
        
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        macd_signal = macd_signal.fillna(0.0)
        
        macd_histogram = macd_line - macd_signal
        macd_histogram = macd_histogram.fillna(0.0)
        
        return macd_line, macd_signal, macd_histogram
    
    def get_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate all indicators with comprehensive null handling.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            Dictionary of indicator values or None if insufficient data
        """
        if not self._validate_dataframe(df, ['open', 'high', 'low', 'close']):
            return None
        
        min_required = max(30, max(self.ema_periods + [self.rsi_period, self.stoch_k_period, self.atr_period]) + 10)
        
        if len(df) < min_required:
            return None
        
        indicators = {}
        
        for period in self.ema_periods:
            ema_series = self.calculate_ema(df, period)
            indicators[f'ema_{period}'] = safe_series_operation(ema_series, 'value', -1, 0.0)
        
        rsi_series = self.calculate_rsi(df, self.rsi_period)
        indicators['rsi'] = safe_series_operation(rsi_series, 'value', -1, 50.0)
        indicators['rsi_prev'] = safe_series_operation(rsi_series, 'value', -2, 50.0)
        
        try:
            rsi_tail = rsi_series.tail(20).fillna(50.0).tolist()
            indicators['rsi_history'] = [float(v) if not pd.isna(v) else 50.0 for v in rsi_tail]
        except (IndicatorError, Exception):
            indicators['rsi_history'] = [50.0] * 20
        
        stoch_k, stoch_d = self.calculate_stochastic(
            df, self.stoch_k_period, self.stoch_d_period, self.stoch_smooth_k
        )
        indicators['stoch_k'] = safe_series_operation(stoch_k, 'value', -1, 50.0)
        indicators['stoch_d'] = safe_series_operation(stoch_d, 'value', -1, 50.0)
        indicators['stoch_k_prev'] = safe_series_operation(stoch_k, 'value', -2, 50.0)
        indicators['stoch_d_prev'] = safe_series_operation(stoch_d, 'value', -2, 50.0)
        
        atr_series = self.calculate_atr(df, self.atr_period)
        indicators['atr'] = safe_series_operation(atr_series, 'value', -1, 0.0)
        
        macd_line, macd_signal, macd_histogram = self.calculate_macd(
            df, self.macd_fast, self.macd_slow, self.macd_signal
        )
        indicators['macd'] = safe_series_operation(macd_line, 'value', -1, 0.0)
        indicators['macd_signal'] = safe_series_operation(macd_signal, 'value', -1, 0.0)
        indicators['macd_histogram'] = safe_series_operation(macd_histogram, 'value', -1, 0.0)
        indicators['macd_prev'] = safe_series_operation(macd_line, 'value', -2, 0.0)
        indicators['macd_signal_prev'] = safe_series_operation(macd_signal, 'value', -2, 0.0)
        
        volume_series = self._get_column_series(df, 'volume')
        indicators['volume'] = safe_series_operation(volume_series, 'value', -1, 0.0)
        vol_avg_series = self.calculate_volume_average(df)
        indicators['volume_avg'] = safe_series_operation(vol_avg_series, 'value', -1, 0.0)
        
        trf_upper, trf_lower, trf_trend = self.calculate_twin_range_filter(df, period=27, multiplier=2.0)
        indicators['trf_upper'] = safe_series_operation(trf_upper, 'value', -1, 0.0)
        indicators['trf_lower'] = safe_series_operation(trf_lower, 'value', -1, 0.0)
        indicators['trf_trend'] = safe_series_operation(trf_trend, 'value', -1, 0)
        indicators['trf_trend_prev'] = safe_series_operation(trf_trend, 'value', -2, 0)
        
        cerebr_value, cerebr_signal, cerebr_bias = self.calculate_market_bias_cerebr(df, period=60, smoothing=10)
        indicators['cerebr_value'] = safe_series_operation(cerebr_value, 'value', -1, 50.0)
        indicators['cerebr_signal'] = safe_series_operation(cerebr_signal, 'value', -1, 50.0)
        indicators['cerebr_bias'] = safe_series_operation(cerebr_bias, 'value', -1, 0)
        indicators['cerebr_bias_prev'] = safe_series_operation(cerebr_bias, 'value', -2, 0)
        
        close_series = self._get_column_series(df, 'close')
        high_series = self._get_column_series(df, 'high')
        low_series = self._get_column_series(df, 'low')
        
        indicators['close'] = safe_series_operation(close_series, 'value', -1, 0.0)
        indicators['high'] = safe_series_operation(high_series, 'value', -1, 0.0)
        indicators['low'] = safe_series_operation(low_series, 'value', -1, 0.0)
        
        return indicators
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        VWAP = cumsum(typical_price * volume) / cumsum(volume)
        typical_price = (high + low + close) / 3
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            pd.Series: VWAP values
        """
        if not self._validate_dataframe(df, ['high', 'low', 'close', 'volume']):
            return pd.Series([0.0])
        
        high = self._get_column_series(df, 'high')
        low = self._get_column_series(df, 'low')
        close = self._get_column_series(df, 'close')
        volume = self._get_column_series(df, 'volume')
        
        if len(df) < 1:
            return pd.Series([0.0])
        
        typical_price = (high + low + close) / 3
        typical_price = typical_price.fillna(close)
        
        tp_volume = typical_price * volume
        tp_volume = tp_volume.fillna(0.0)
        
        has_date_index = False
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                has_date_index = True
            elif 'time' in df.columns:
                df_time = pd.to_datetime(df['time'], errors='coerce')
                if not df_time.isna().all():
                    has_date_index = True
        except (IndicatorError, Exception):
            has_date_index = False
        
        if has_date_index:
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    date_series = df.index.date
                else:
                    date_series = pd.to_datetime(df['time']).dt.date
                
                vwap = pd.Series(index=df.index, dtype=float)
                
                for date in pd.Series(date_series).unique():
                    mask = date_series == date
                    if isinstance(mask, np.ndarray):
                        mask = pd.Series(mask, index=df.index)
                    
                    cum_tp_vol = tp_volume[mask].cumsum()
                    cum_vol = volume[mask].cumsum()
                    
                    daily_vwap = safe_divide(cum_tp_vol, cum_vol, fill_value=typical_price[mask].iloc[-1] if len(typical_price[mask]) > 0 else 0.0)
                    vwap.loc[mask] = daily_vwap.values
                
                vwap = vwap.fillna(typical_price)
                return pd.Series(vwap)
            except (IndicatorError, Exception):
                pass
        
        rolling_period = 20
        if len(df) < rolling_period:
            rolling_period = len(df)
        
        cum_tp_vol = tp_volume.rolling(window=rolling_period, min_periods=1).sum()
        cum_vol = volume.rolling(window=rolling_period, min_periods=1).sum()
        
        vwap = safe_divide(cum_tp_vol, cum_vol, fill_value=0.0)
        vwap = vwap.fillna(typical_price)
        
        return pd.Series(vwap)
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect candlestick patterns from the last candle.
        
        Patterns detected:
        - Pinbar (bullish/bearish): small body (<30% range), long wick (>60% range)
        - Hammer/Inverted Hammer: small body, lower/upper wick > 2x body
        - Engulfing (bullish/bearish): current candle engulfs previous
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            Dict with pattern detection results
        """
        default_result = {
            'bullish_pinbar': False,
            'bearish_pinbar': False,
            'hammer': False,
            'inverted_hammer': False,
            'bullish_engulfing': False,
            'bearish_engulfing': False
        }
        
        if not self._validate_dataframe(df, ['open', 'high', 'low', 'close']):
            return default_result
        
        if len(df) < 2:
            return default_result
        
        try:
            open_curr = float(df['open'].iloc[-1]) if not pd.isna(df['open'].iloc[-1]) else 0.0
            high_curr = float(df['high'].iloc[-1]) if not pd.isna(df['high'].iloc[-1]) else 0.0
            low_curr = float(df['low'].iloc[-1]) if not pd.isna(df['low'].iloc[-1]) else 0.0
            close_curr = float(df['close'].iloc[-1]) if not pd.isna(df['close'].iloc[-1]) else 0.0
            
            open_prev = float(df['open'].iloc[-2]) if not pd.isna(df['open'].iloc[-2]) else 0.0
            high_prev = float(df['high'].iloc[-2]) if not pd.isna(df['high'].iloc[-2]) else 0.0
            low_prev = float(df['low'].iloc[-2]) if not pd.isna(df['low'].iloc[-2]) else 0.0
            close_prev = float(df['close'].iloc[-2]) if not pd.isna(df['close'].iloc[-2]) else 0.0
        except (IndexError, KeyError, TypeError):
            return default_result
        
        total_range = high_curr - low_curr
        if total_range <= 0:
            total_range = 1e-10
        
        body = abs(close_curr - open_curr)
        body_pct = body / total_range
        
        is_bullish = close_curr > open_curr
        body_top = max(open_curr, close_curr)
        body_bottom = min(open_curr, close_curr)
        upper_wick = high_curr - body_top
        lower_wick = body_bottom - low_curr
        
        result = default_result.copy()
        
        if body_pct < 0.30:
            upper_wick_pct = upper_wick / total_range
            lower_wick_pct = lower_wick / total_range
            
            if lower_wick_pct > 0.60:
                result['bullish_pinbar'] = True
            if upper_wick_pct > 0.60:
                result['bearish_pinbar'] = True
        
        body_size = body if body > 0 else 1e-10
        
        if body_pct < 0.35 and lower_wick > 2 * body_size and upper_wick < body_size:
            result['hammer'] = True
        
        if body_pct < 0.35 and upper_wick > 2 * body_size and lower_wick < body_size:
            result['inverted_hammer'] = True
        
        prev_is_bearish = close_prev < open_prev
        prev_is_bullish = close_prev > open_prev
        
        if is_bullish and prev_is_bearish:
            if close_curr > open_prev and open_curr < close_prev:
                result['bullish_engulfing'] = True
        
        if not is_bullish and prev_is_bullish:
            if close_curr < open_prev and open_curr > close_prev:
                result['bearish_engulfing'] = True
        
        return result
    
    def calculate_micro_support_resistance(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        """
        Calculate micro support and resistance levels from swing highs/lows.
        
        Args:
            df: DataFrame with OHLC data
            lookback: Number of candles to look back for swing detection
        
        Returns:
            Dict with support/resistance levels:
            - nearest_support: Closest support below current price
            - nearest_resistance: Closest resistance above current price
            - support_levels: List of support levels
            - resistance_levels: List of resistance levels
        """
        default_result = {
            'nearest_support': 0.0,
            'nearest_resistance': 0.0,
            'support_levels': [],
            'resistance_levels': []
        }
        
        if not self._validate_dataframe(df, ['high', 'low', 'close']):
            return default_result
        
        if len(df) < 5:
            return default_result
        
        try:
            high = self._get_column_series(df, 'high')
            low = self._get_column_series(df, 'low')
            close = self._get_column_series(df, 'close')
            
            current_price = float(close.iloc[-1]) if not pd.isna(close.iloc[-1]) else 0.0
            
            actual_lookback = min(lookback, len(df))
            df_subset = df.tail(actual_lookback)
            high_subset = high.tail(actual_lookback)
            low_subset = low.tail(actual_lookback)
            
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(df_subset) - 2):
                high_val = float(high_subset.iloc[i]) if not pd.isna(high_subset.iloc[i]) else 0.0
                high_prev1 = float(high_subset.iloc[i-1]) if not pd.isna(high_subset.iloc[i-1]) else 0.0
                high_prev2 = float(high_subset.iloc[i-2]) if not pd.isna(high_subset.iloc[i-2]) else 0.0
                high_next1 = float(high_subset.iloc[i+1]) if not pd.isna(high_subset.iloc[i+1]) else 0.0
                high_next2 = float(high_subset.iloc[i+2]) if not pd.isna(high_subset.iloc[i+2]) else 0.0
                
                if high_val > high_prev1 and high_val > high_prev2 and high_val > high_next1 and high_val > high_next2:
                    swing_highs.append(high_val)
                
                low_val = float(low_subset.iloc[i]) if not pd.isna(low_subset.iloc[i]) else 0.0
                low_prev1 = float(low_subset.iloc[i-1]) if not pd.isna(low_subset.iloc[i-1]) else 0.0
                low_prev2 = float(low_subset.iloc[i-2]) if not pd.isna(low_subset.iloc[i-2]) else 0.0
                low_next1 = float(low_subset.iloc[i+1]) if not pd.isna(low_subset.iloc[i+1]) else 0.0
                low_next2 = float(low_subset.iloc[i+2]) if not pd.isna(low_subset.iloc[i+2]) else 0.0
                
                if low_val < low_prev1 and low_val < low_prev2 and low_val < low_next1 and low_val < low_next2:
                    swing_lows.append(low_val)
            
            if not swing_lows:
                swing_lows.append(float(low_subset.min()))
            if not swing_highs:
                swing_highs.append(float(high_subset.max()))
            
            swing_highs = sorted(list(set(swing_highs)))
            swing_lows = sorted(list(set(swing_lows)))
            
            resistance_levels = [h for h in swing_highs if h > current_price]
            support_levels = [l for l in swing_lows if l < current_price]
            
            nearest_resistance = min(resistance_levels) if resistance_levels else (max(swing_highs) if swing_highs else current_price * 1.01)
            nearest_support = max(support_levels) if support_levels else (min(swing_lows) if swing_lows else current_price * 0.99)
            
            return {
                'nearest_support': float(nearest_support),
                'nearest_resistance': float(nearest_resistance),
                'support_levels': [float(s) for s in sorted(support_levels, reverse=True)[:5]],
                'resistance_levels': [float(r) for r in sorted(resistance_levels)[:5]]
            }
            
        except (IndicatorError, Exception):
            return default_result
    
    def calculate_volume_confirmation(self, df: pd.DataFrame, period: int = 10) -> Dict:
        """
        Calculate volume confirmation indicators.
        
        Args:
            df: DataFrame with volume data
            period: Period for volume average calculation (default: 10)
        
        Returns:
            Dict with volume analysis:
            - volume_current: Current volume
            - volume_avg: Average volume over period
            - is_volume_strong: True if current volume > average
            - volume_ratio: Current volume / Average volume
        """
        default_result = {
            'volume_current': 0.0,
            'volume_avg': 0.0,
            'is_volume_strong': False,
            'volume_ratio': 1.0
        }
        
        if not self._validate_dataframe(df, ['volume']):
            return default_result
        
        if len(df) < 1:
            return default_result
        
        try:
            volume = self._get_column_series(df, 'volume')
            
            current_volume = float(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else 0.0
            
            actual_period = min(period, len(df))
            if actual_period < 1:
                actual_period = 1
            
            volume_avg_series = volume.rolling(window=actual_period, min_periods=1).mean()
            volume_avg = float(volume_avg_series.iloc[-1]) if not pd.isna(volume_avg_series.iloc[-1]) else 0.0
            
            if volume_avg > 0:
                volume_ratio = current_volume / volume_avg
            else:
                volume_ratio = 1.0
            
            is_volume_strong = current_volume > volume_avg
            
            return {
                'volume_current': float(current_volume),
                'volume_avg': float(volume_avg),
                'is_volume_strong': bool(is_volume_strong),
                'volume_ratio': float(volume_ratio)
            }
            
        except (IndicatorError, Exception):
            return default_result
