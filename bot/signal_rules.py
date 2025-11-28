"""
Aggressive Signal Rules untuk Bot Trading XAUUSD.

Modul ini berisi 4 aturan sinyal untuk aggressive scalping:
1. M1 Quick Scalp Signal (20-50+ signals/hari)
2. M5 Swing Entry (5-15 signals/hari)
3. S/R Mean-Reversion (3-8 signals/hari)
4. Breakout Confirmation (2-5 signals/hari)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
from enum import Enum

from bot.logger import setup_logger
from bot.strategy import safe_float, is_valid_number, safe_divide
from bot.indicators import IndicatorEngine, safe_series_operation

logger = setup_logger('SignalRules')


class SignalRuleError(Exception):
    """Base exception for signal rule errors"""
    pass


class RuleType(str, Enum):
    """Enum untuk tipe signal rule"""
    M1_SCALP = 'M1_SCALP'
    M5_SWING = 'M5_SWING'
    SR_REVERSION = 'SR_REVERSION'
    BREAKOUT = 'BREAKOUT'


class SignalType(str, Enum):
    """Enum untuk tipe signal"""
    BUY = 'BUY'
    SELL = 'SELL'
    NONE = 'NONE'


@dataclass
class SignalResult:
    """Dataclass untuk hasil signal dari setiap rule"""
    signal_type: str = 'NONE'
    rule_name: str = ''
    confidence: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    reason: str = ''
    confluence_count: int = 0
    confluence_details: List[str] = field(default_factory=list)
    entry_price: float = 0.0
    
    def is_valid(self) -> bool:
        """Check if signal is valid (not NONE)"""
        return self.signal_type in ('BUY', 'SELL') and self.confidence > 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SignalResult to dictionary"""
        return {
            'signal_type': self.signal_type,
            'rule_name': self.rule_name,
            'confidence': self.confidence,
            'sl_pips': self.sl_pips,
            'tp_pips': self.tp_pips,
            'reason': self.reason,
            'confluence_count': self.confluence_count,
            'confluence_details': self.confluence_details,
            'entry_price': self.entry_price
        }


class AggressiveSignalRules:
    """
    Class untuk menghitung 4 aggressive signal rules untuk XAUUSD scalping.
    
    Rules:
    1. M1 Quick Scalp (20-50+ signals/day) - Ultra short-term scalping
    2. M5 Swing Entry (5-15 signals/day) - Short-term swing trades
    3. S/R Mean-Reversion (3-8 signals/day) - Range-bound market reversal
    4. Breakout Confirmation (2-5 signals/day) - Breakout continuation
    """
    
    PIP_VALUE_XAUUSD = 0.1
    
    M1_SL_MIN = 8.0
    M1_SL_MAX = 12.0
    M1_TP_MIN = 15.0
    M1_TP_MAX = 30.0
    M1_MIN_CONFLUENCE = 2
    
    M5_SL_MIN = 15.0
    M5_SL_MAX = 20.0
    M5_TP_MIN = 30.0
    M5_TP_MAX = 50.0
    M5_MIN_CONFLUENCE = 3
    M5_ADX_THRESHOLD = 20
    
    SR_SL_MIN = 10.0
    SR_SL_MAX = 15.0
    SR_TP_MIN = 20.0
    SR_TP_MAX = 40.0
    SR_MIN_CONFLUENCE = 2
    SR_ADX_MAX = 20
    SR_ATR_MAX_RATIO = 1.5
    SR_PROXIMITY_PIPS = 5.0
    
    BO_SL_MIN = 12.0
    BO_SL_MAX = 18.0
    BO_TP_MIN = 40.0
    BO_TP_MAX = 80.0
    BO_MIN_CONFLUENCE = 3
    BO_ADX_TARGET = 25
    BO_VOLUME_THRESHOLD = 1.5
    BO_SR_PROXIMITY_PIPS = 5.0
    
    def __init__(self, config, indicator_engine: Optional[IndicatorEngine] = None):
        """
        Inisialisasi AggressiveSignalRules.
        
        Args:
            config: Objek konfigurasi bot
            indicator_engine: Instance IndicatorEngine (opsional)
        """
        self.config = config
        self.indicator_engine = indicator_engine or IndicatorEngine(config)
        logger.info("AggressiveSignalRules initialized")
    
    def _safe_get_value(self, data: Any, index: int = -1, default: float = 0.0) -> float:
        """Safely get value from series/dataframe with NaN/Inf handling"""
        if data is None:
            return default
        if isinstance(data, pd.DataFrame):
            if len(data.columns) > 0:
                data = data.iloc[:, 0]
            else:
                return default
        if not isinstance(data, pd.Series):
            try:
                data = pd.Series(data)
            except (TypeError, ValueError):
                return default
        return safe_series_operation(data, 'value', index, default)
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                                    std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with OHLC data
            period: Period for moving average (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if df is None or len(df) < period:
            empty = pd.Series([0.0])
            return empty, empty, empty
        
        try:
            close = df['close'] if 'close' in df.columns else None
            if close is None:
                empty = pd.Series([0.0])
                return empty, empty, empty
            
            middle_band = pd.Series(close.rolling(window=period, min_periods=1).mean())
            std = pd.Series(close.rolling(window=period, min_periods=1).std())
            
            upper_band = pd.Series(middle_band + (std * std_dev))
            lower_band = pd.Series(middle_band - (std * std_dev))
            
            middle_band = pd.Series(middle_band.replace([np.inf, -np.inf], np.nan).fillna(close))
            upper_band = pd.Series(upper_band.replace([np.inf, -np.inf], np.nan).fillna(close))
            lower_band = pd.Series(lower_band.replace([np.inf, -np.inf], np.nan).fillna(close))
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logger.warning(f"Gagal menghitung Bollinger Bands: {str(e)}")
            empty = pd.Series([0.0])
            return empty, empty, empty
    
    def _get_candle_wick_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get candle wick data for current candle.
        
        Returns:
            Dict with upper_wick, lower_wick, body_size, wick_ratio
        """
        result = {
            'upper_wick': 0.0,
            'lower_wick': 0.0,
            'body_size': 0.0,
            'wick_ratio': 0.0,
            'is_bullish': False
        }
        
        if df is None or len(df) < 1:
            return result
        
        try:
            last_candle = df.iloc[-1]
            open_price = safe_float(last_candle.get('open', 0), 0.0)
            high_price = safe_float(last_candle.get('high', 0), 0.0)
            low_price = safe_float(last_candle.get('low', 0), 0.0)
            close_price = safe_float(last_candle.get('close', 0), 0.0)
            
            if open_price <= 0 or close_price <= 0:
                return result
            
            result['is_bullish'] = close_price > open_price
            body_top = max(open_price, close_price)
            body_bottom = min(open_price, close_price)
            
            result['upper_wick'] = high_price - body_top
            result['lower_wick'] = body_bottom - low_price
            result['body_size'] = body_top - body_bottom
            
            total_range = high_price - low_price
            if total_range > 0:
                result['wick_ratio'] = (result['upper_wick'] + result['lower_wick']) / total_range
            
            return result
            
        except Exception as e:
            logger.warning(f"Error getting wick data: {str(e)}")
            return result
    
    def _calculate_sl_tp(self, base_sl: float, base_tp: float, 
                         sl_min: float, sl_max: float,
                         tp_min: float, tp_max: float,
                         volatility_multiplier: float = 1.0) -> Tuple[float, float]:
        """
        Calculate SL and TP with volatility adjustment.
        
        Args:
            base_sl: Base stop loss in pips
            base_tp: Base take profit in pips
            sl_min, sl_max: SL range limits
            tp_min, tp_max: TP range limits
            volatility_multiplier: Multiplier based on current volatility
        
        Returns:
            Tuple of (sl_pips, tp_pips)
        """
        adjusted_sl = base_sl * volatility_multiplier
        adjusted_tp = base_tp * volatility_multiplier
        
        sl_pips = max(sl_min, min(sl_max, adjusted_sl))
        tp_pips = max(tp_min, min(tp_max, adjusted_tp))
        
        return sl_pips, tp_pips
    
    def check_m1_scalp_signal(self, df_m1: pd.DataFrame, 
                               df_m5: Optional[pd.DataFrame] = None,
                               df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 1: M1 Quick Scalp Signal (20-50+ signals/hari)
        
        Trigger Conditions:
        - M1 candle close above EMA5 (BUY) or below EMA5 (SELL)
        - RSI(14) > 50 AND RSI < 75 (BUY) or RSI < 50 AND RSI > 25 (SELL)
        - Volume > avg * 1.1
        - MACD histogram positive & increasing (BUY) or negative & decreasing (SELL)
        
        Confirmation:
        - M5 EMA5 > EMA20 (uptrend) for BUY, EMA5 < EMA20 for SELL
        - H1 not at resistance (BUY) or not at support (SELL)
        
        Execute IF: 2+ confluence from {RSI direction, volume spike, MACD histogram, MA alignment}
        
        Args:
            df_m1: DataFrame M1 timeframe
            df_m5: DataFrame M5 timeframe (optional)
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            SignalResult
        """
        result = SignalResult(rule_name=RuleType.M1_SCALP.value)
        
        if df_m1 is None or len(df_m1) < 30:
            result.reason = "Insufficient M1 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m1)
            if indicators is None:
                result.reason = "Failed to calculate M1 indicators"
                return result
            
            close = safe_float(indicators.get('close', 0), 0.0)
            ema_5 = safe_float(indicators.get('ema_5', 0), 0.0)
            ema_20 = safe_float(indicators.get('ema_20', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            rsi_prev = safe_float(indicators.get('rsi_prev', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            macd_histogram = safe_float(indicators.get('macd_histogram', 0), 0.0)
            
            macd_line, macd_signal, macd_hist_series = self.indicator_engine.calculate_macd(df_m1)
            macd_hist_current = self._safe_get_value(macd_hist_series, -1, 0.0)
            macd_hist_prev = self._safe_get_value(macd_hist_series, -2, 0.0)
            
            if close <= 0 or ema_5 <= 0:
                result.reason = "Invalid price data"
                return result
            
            buy_confluences = []
            sell_confluences = []
            
            if close > ema_5:
                buy_confluences.append("M1 close > EMA5")
            elif close < ema_5:
                sell_confluences.append("M1 close < EMA5")
            
            if 50 < rsi < 75:
                if rsi > rsi_prev:
                    buy_confluences.append(f"RSI bullish momentum ({rsi:.1f})")
            elif 25 < rsi < 50:
                if rsi < rsi_prev:
                    sell_confluences.append(f"RSI bearish momentum ({rsi:.1f})")
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            if volume_ratio > 1.1:
                if close > ema_5:
                    buy_confluences.append(f"Volume spike ({volume_ratio:.2f}x)")
                else:
                    sell_confluences.append(f"Volume spike ({volume_ratio:.2f}x)")
            
            if macd_hist_current > 0 and macd_hist_current > macd_hist_prev:
                buy_confluences.append("MACD histogram positive & increasing")
            elif macd_hist_current < 0 and macd_hist_current < macd_hist_prev:
                sell_confluences.append("MACD histogram negative & decreasing")
            
            if ema_5 > ema_20:
                buy_confluences.append("EMA5 > EMA20 alignment")
            elif ema_5 < ema_20:
                sell_confluences.append("EMA5 < EMA20 alignment")
            
            m5_confirmation_buy = True
            m5_confirmation_sell = True
            if df_m5 is not None and len(df_m5) >= 30:
                m5_indicators = self.indicator_engine.get_indicators(df_m5)
                if m5_indicators:
                    m5_ema5 = safe_float(m5_indicators.get('ema_5', 0), 0.0)
                    m5_ema20 = safe_float(m5_indicators.get('ema_20', 0), 0.0)
                    if m5_ema5 > m5_ema20:
                        buy_confluences.append("M5 EMA5 > EMA20 confirmation")
                        m5_confirmation_sell = False
                    elif m5_ema5 < m5_ema20:
                        sell_confluences.append("M5 EMA5 < EMA20 confirmation")
                        m5_confirmation_buy = False
            
            h1_at_resistance = False
            h1_at_support = False
            if df_h1 is not None and len(df_h1) >= 30:
                sr_data = self.indicator_engine.calculate_micro_support_resistance(df_h1, lookback=50)
                h1_close = self._safe_get_value(df_h1['close'], -1, 0.0) if 'close' in df_h1.columns else 0.0
                resistance = sr_data.get('nearest_resistance', 0)
                support = sr_data.get('nearest_support', 0)
                
                if h1_close > 0:
                    dist_to_resistance = abs(resistance - h1_close) / self.PIP_VALUE_XAUUSD if resistance > 0 else 999
                    dist_to_support = abs(h1_close - support) / self.PIP_VALUE_XAUUSD if support > 0 else 999
                    
                    h1_at_resistance = dist_to_resistance < 10
                    h1_at_support = dist_to_support < 10
                    
                    if not h1_at_resistance:
                        buy_confluences.append("H1 not at resistance")
                    if not h1_at_support:
                        sell_confluences.append("H1 not at support")
            
            buy_count = len(buy_confluences)
            sell_count = len(sell_confluences)
            
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            avg_atr = atr
            volatility_mult = min(1.5, max(0.7, atr / avg_atr)) if avg_atr > 0 else 1.0
            
            if buy_count >= self.M1_MIN_CONFLUENCE and buy_count > sell_count and m5_confirmation_buy and not h1_at_resistance:
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_confluences
                result.confidence = min(1.0, 0.4 + (buy_count * 0.15))
                
                sl, tp = self._calculate_sl_tp(
                    10.0, 22.0,
                    self.M1_SL_MIN, self.M1_SL_MAX,
                    self.M1_TP_MIN, self.M1_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M1 Quick Scalp BUY: {buy_count} confluences - " + ", ".join(buy_confluences[:3])
                
            elif sell_count >= self.M1_MIN_CONFLUENCE and sell_count > buy_count and m5_confirmation_sell and not h1_at_support:
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_confluences
                result.confidence = min(1.0, 0.4 + (sell_count * 0.15))
                
                sl, tp = self._calculate_sl_tp(
                    10.0, 22.0,
                    self.M1_SL_MIN, self.M1_SL_MAX,
                    self.M1_TP_MIN, self.M1_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M1 Quick Scalp SELL: {sell_count} confluences - " + ", ".join(sell_confluences[:3])
            else:
                result.reason = f"M1 Scalp: Buy={buy_count}, Sell={sell_count} (min={self.M1_MIN_CONFLUENCE})"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in M1 Scalp signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_m5_swing_signal(self, df_m5: pd.DataFrame,
                               df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 2: M5 Swing Entry (5-15 signals/hari)
        
        Trigger Conditions:
        - M5 candle close
        - Price break above EMA20 (BUY) or below EMA20 (SELL)
        - Volume spike (> average)
        - RSI direction match (rising for BUY, falling for SELL)
        - MACD histogram match (positive for BUY, negative for SELL)
        
        Confirmation:
        - H1 above MA50 aligned (BUY) or below MA50 (SELL)
        - Last 5 H1 candles consistency
        
        Execute IF: 3+ confluence AND (ADX > 20 OR Breakout detected)
        
        Args:
            df_m5: DataFrame M5 timeframe
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            SignalResult
        """
        result = SignalResult(rule_name=RuleType.M5_SWING.value)
        
        if df_m5 is None or len(df_m5) < 50:
            result.reason = "Insufficient M5 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m5)
            if indicators is None:
                result.reason = "Failed to calculate M5 indicators"
                return result
            
            close = safe_float(indicators.get('close', 0), 0.0)
            close_prev = self._safe_get_value(df_m5['close'], -2, 0.0) if 'close' in df_m5.columns else 0.0
            ema_20 = safe_float(indicators.get('ema_20', 0), 0.0)
            ema_50 = safe_float(indicators.get('ema_50', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            rsi_prev = safe_float(indicators.get('rsi_prev', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            macd_histogram = safe_float(indicators.get('macd_histogram', 0), 0.0)
            adx = safe_float(indicators.get('adx', 0), 0.0)
            adx_prev = safe_float(indicators.get('adx_prev', 0), 0.0)
            
            ema_20_series = self.indicator_engine.calculate_ema(df_m5, 20)
            ema_20_prev = self._safe_get_value(ema_20_series, -2, 0.0)
            
            if close <= 0 or ema_20 <= 0:
                result.reason = "Invalid price data"
                return result
            
            buy_confluences = []
            sell_confluences = []
            
            price_break_above = close > ema_20 and close_prev <= ema_20_prev
            price_break_below = close < ema_20 and close_prev >= ema_20_prev
            
            if price_break_above:
                buy_confluences.append("Price break above EMA20")
            elif price_break_below:
                sell_confluences.append("Price break below EMA20")
            elif close > ema_20:
                buy_confluences.append("Price above EMA20")
            elif close < ema_20:
                sell_confluences.append("Price below EMA20")
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            if volume_ratio > 1.2:
                if close > ema_20:
                    buy_confluences.append(f"Volume spike ({volume_ratio:.2f}x)")
                else:
                    sell_confluences.append(f"Volume spike ({volume_ratio:.2f}x)")
            
            if rsi > rsi_prev and 45 < rsi < 70:
                buy_confluences.append(f"RSI rising ({rsi:.1f})")
            elif rsi < rsi_prev and 30 < rsi < 55:
                sell_confluences.append(f"RSI falling ({rsi:.1f})")
            
            if macd_histogram > 0:
                buy_confluences.append("MACD histogram positive")
            elif macd_histogram < 0:
                sell_confluences.append("MACD histogram negative")
            
            if ema_20 > ema_50:
                buy_confluences.append("EMA20 > EMA50 trend")
            elif ema_20 < ema_50:
                sell_confluences.append("EMA20 < EMA50 trend")
            
            h1_confirmation_buy = True
            h1_confirmation_sell = True
            h1_candle_consistency = 0
            
            if df_h1 is not None and len(df_h1) >= 30:
                h1_indicators = self.indicator_engine.get_indicators(df_h1)
                if h1_indicators:
                    h1_close = safe_float(h1_indicators.get('close', 0), 0.0)
                    h1_ema50 = safe_float(h1_indicators.get('ema_50', 0), 0.0)
                    
                    if h1_close > h1_ema50:
                        buy_confluences.append("H1 above MA50")
                        h1_confirmation_sell = False
                    elif h1_close < h1_ema50:
                        sell_confluences.append("H1 below MA50")
                        h1_confirmation_buy = False
                
                if 'close' in df_h1.columns and 'open' in df_h1.columns:
                    last_5_h1 = df_h1.tail(5)
                    bullish_count = sum(1 for _, c in last_5_h1.iterrows() 
                                       if safe_float(c.get('close', 0), 0) > safe_float(c.get('open', 0), 0))
                    bearish_count = 5 - bullish_count
                    
                    if bullish_count >= 4:
                        buy_confluences.append("H1 bullish consistency (4/5+)")
                        h1_candle_consistency = 1
                    elif bearish_count >= 4:
                        sell_confluences.append("H1 bearish consistency (4/5+)")
                        h1_candle_consistency = -1
            
            has_adx_condition = adx > self.M5_ADX_THRESHOLD
            is_breakout = price_break_above or price_break_below
            has_momentum = has_adx_condition or is_breakout
            
            if has_adx_condition:
                if adx > adx_prev:
                    if close > ema_20:
                        buy_confluences.append(f"ADX rising ({adx:.1f})")
                    else:
                        sell_confluences.append(f"ADX rising ({adx:.1f})")
            
            buy_count = len(buy_confluences)
            sell_count = len(sell_confluences)
            
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            volatility_mult = min(1.3, max(0.8, 1.0))
            
            if (buy_count >= self.M5_MIN_CONFLUENCE and 
                buy_count > sell_count and 
                h1_confirmation_buy and 
                has_momentum):
                
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_confluences
                result.confidence = min(1.0, 0.45 + (buy_count * 0.12))
                
                if is_breakout:
                    result.confidence += 0.1
                
                sl, tp = self._calculate_sl_tp(
                    17.0, 40.0,
                    self.M5_SL_MIN, self.M5_SL_MAX,
                    self.M5_TP_MIN, self.M5_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M5 Swing BUY: {buy_count} confluences, ADX={adx:.1f} - " + ", ".join(buy_confluences[:3])
                
            elif (sell_count >= self.M5_MIN_CONFLUENCE and 
                  sell_count > buy_count and 
                  h1_confirmation_sell and 
                  has_momentum):
                
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_confluences
                result.confidence = min(1.0, 0.45 + (sell_count * 0.12))
                
                if is_breakout:
                    result.confidence += 0.1
                
                sl, tp = self._calculate_sl_tp(
                    17.0, 40.0,
                    self.M5_SL_MIN, self.M5_SL_MAX,
                    self.M5_TP_MIN, self.M5_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"M5 Swing SELL: {sell_count} confluences, ADX={adx:.1f} - " + ", ".join(sell_confluences[:3])
            else:
                momentum_status = f"ADX={adx:.1f}" if has_adx_condition else "No momentum"
                result.reason = f"M5 Swing: Buy={buy_count}, Sell={sell_count} (min={self.M5_MIN_CONFLUENCE}, {momentum_status})"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in M5 Swing signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_sr_reversion_signal(self, df_m5: pd.DataFrame,
                                   df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 3: S/R Mean-Reversion (3-8 signals/hari, ranging market)
        
        Trigger Conditions:
        - Price within 5 pips of S/R level
        - Wick test level (rejection candle)
        - RSI extreme (< 30 for BUY, > 70 for SELL)
        - Volume decrease (consolidation)
        
        Confirmation:
        - M5 close within 1 pip of S/R
        - Bollinger mid nearby
        - Stochastic extreme (< 20 for BUY, > 80 for SELL)
        
        Execute IF: ADX < 20 AND ATR < 150% average AND 2+ confluence
        
        Args:
            df_m5: DataFrame M5 timeframe
            df_h1: DataFrame H1 timeframe (optional for S/R levels)
        
        Returns:
            SignalResult
        """
        result = SignalResult(rule_name=RuleType.SR_REVERSION.value)
        
        if df_m5 is None or len(df_m5) < 50:
            result.reason = "Insufficient M5 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m5)
            if indicators is None:
                result.reason = "Failed to calculate M5 indicators"
                return result
            
            close = safe_float(indicators.get('close', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            stoch_k = safe_float(indicators.get('stoch_k', 50), 50.0)
            stoch_d = safe_float(indicators.get('stoch_d', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            adx = safe_float(indicators.get('adx', 25), 25.0)
            atr = safe_float(indicators.get('atr', 1.0), 1.0)
            
            if close <= 0:
                result.reason = "Invalid price data"
                return result
            
            sr_source = df_h1 if df_h1 is not None and len(df_h1) >= 30 else df_m5
            sr_data = self.indicator_engine.calculate_micro_support_resistance(sr_source, lookback=50)
            
            support = sr_data.get('nearest_support', 0)
            resistance = sr_data.get('nearest_resistance', 0)
            support_levels = sr_data.get('support_levels', [])
            resistance_levels = sr_data.get('resistance_levels', [])
            
            dist_to_support_pips = abs(close - support) / self.PIP_VALUE_XAUUSD if support > 0 else 999
            dist_to_resistance_pips = abs(resistance - close) / self.PIP_VALUE_XAUUSD if resistance > 0 else 999
            
            upper_bb, middle_bb, lower_bb = self._calculate_bollinger_bands(df_m5, period=20, std_dev=2.0)
            bb_mid = self._safe_get_value(middle_bb, -1, close)
            dist_to_bb_mid_pips = abs(close - bb_mid) / self.PIP_VALUE_XAUUSD
            
            atr_series = self.indicator_engine.calculate_atr(df_m5, 14)
            atr_avg = self._safe_get_value(atr_series.rolling(window=20, min_periods=1).mean(), -1, atr)
            atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
            
            wick_data = self._get_candle_wick_data(df_m5)
            
            if adx >= self.SR_ADX_MAX:
                result.reason = f"ADX too high ({adx:.1f} >= {self.SR_ADX_MAX}) - Trending market"
                return result
            
            if atr_ratio >= self.SR_ATR_MAX_RATIO:
                result.reason = f"ATR ratio too high ({atr_ratio:.2f} >= {self.SR_ATR_MAX_RATIO}) - High volatility"
                return result
            
            buy_confluences = []
            sell_confluences = []
            
            near_support = dist_to_support_pips <= self.SR_PROXIMITY_PIPS
            near_resistance = dist_to_resistance_pips <= self.SR_PROXIMITY_PIPS
            
            if near_support:
                buy_confluences.append(f"Price near support ({dist_to_support_pips:.1f} pips)")
            if near_resistance:
                sell_confluences.append(f"Price near resistance ({dist_to_resistance_pips:.1f} pips)")
            
            if wick_data['lower_wick'] > wick_data['body_size'] and wick_data['is_bullish']:
                buy_confluences.append("Bullish rejection wick")
            elif wick_data['upper_wick'] > wick_data['body_size'] and not wick_data['is_bullish']:
                sell_confluences.append("Bearish rejection wick")
            
            if rsi < 30:
                buy_confluences.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                sell_confluences.append(f"RSI overbought ({rsi:.1f})")
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            if volume_ratio < 0.9:
                if near_support:
                    buy_confluences.append(f"Volume decreasing ({volume_ratio:.2f}x)")
                elif near_resistance:
                    sell_confluences.append(f"Volume decreasing ({volume_ratio:.2f}x)")
            
            if dist_to_support_pips <= 1.0:
                buy_confluences.append("M5 close at support level")
            elif dist_to_resistance_pips <= 1.0:
                sell_confluences.append("M5 close at resistance level")
            
            if dist_to_bb_mid_pips <= 15:
                if near_support:
                    buy_confluences.append("Bollinger mid nearby")
                elif near_resistance:
                    sell_confluences.append("Bollinger mid nearby")
            
            if stoch_k < 20 and stoch_d < 20:
                buy_confluences.append(f"Stochastic oversold ({stoch_k:.1f})")
            elif stoch_k > 80 and stoch_d > 80:
                sell_confluences.append(f"Stochastic overbought ({stoch_k:.1f})")
            
            buy_confluences.append(f"ADX low ({adx:.1f} - ranging)")
            sell_confluences.append(f"ADX low ({adx:.1f} - ranging)")
            
            buy_count = len(buy_confluences)
            sell_count = len(sell_confluences)
            
            if (buy_count >= self.SR_MIN_CONFLUENCE + 1 and 
                buy_count > sell_count and 
                near_support and
                rsi < 40):
                
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_confluences
                result.confidence = min(1.0, 0.5 + (buy_count * 0.1))
                
                if rsi < 30:
                    result.confidence += 0.1
                if stoch_k < 20:
                    result.confidence += 0.05
                
                volatility_mult = min(1.2, max(0.8, atr_ratio))
                sl, tp = self._calculate_sl_tp(
                    12.0, 30.0,
                    self.SR_SL_MIN, self.SR_SL_MAX,
                    self.SR_TP_MIN, self.SR_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"S/R Reversion BUY: {buy_count} confluences at support - " + ", ".join(buy_confluences[:3])
                
            elif (sell_count >= self.SR_MIN_CONFLUENCE + 1 and 
                  sell_count > buy_count and 
                  near_resistance and
                  rsi > 60):
                
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_confluences
                result.confidence = min(1.0, 0.5 + (sell_count * 0.1))
                
                if rsi > 70:
                    result.confidence += 0.1
                if stoch_k > 80:
                    result.confidence += 0.05
                
                volatility_mult = min(1.2, max(0.8, atr_ratio))
                sl, tp = self._calculate_sl_tp(
                    12.0, 30.0,
                    self.SR_SL_MIN, self.SR_SL_MAX,
                    self.SR_TP_MIN, self.SR_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"S/R Reversion SELL: {sell_count} confluences at resistance - " + ", ".join(sell_confluences[:3])
            else:
                result.reason = f"S/R Reversion: Buy={buy_count}, Sell={sell_count} (Support={dist_to_support_pips:.1f}p, Resistance={dist_to_resistance_pips:.1f}p)"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in S/R Reversion signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_breakout_signal(self, df_m5: pd.DataFrame,
                               df_h1: Optional[pd.DataFrame] = None) -> SignalResult:
        """
        RULE 4: Breakout Confirmation (2-5 signals/hari)
        
        Trigger Conditions:
        - Price break 5/10 candle high (BUY) or low (SELL)
        - Close outside breakout level
        - Volume > 150% average
        - RSI momentum match (above 50 for BUY, below 50 for SELL)
        
        Confirmation:
        - M5 confirm breakout (close outside level)
        - H1 no immediate S/R ahead (room to run)
        
        Execute IF: ADX increasing toward 25+ AND 3+ confluence AND not within 5 pips of H1 S/R
        
        Args:
            df_m5: DataFrame M5 timeframe
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            SignalResult
        """
        result = SignalResult(rule_name=RuleType.BREAKOUT.value)
        
        if df_m5 is None or len(df_m5) < 50:
            result.reason = "Insufficient M5 data"
            return result
        
        try:
            indicators = self.indicator_engine.get_indicators(df_m5)
            if indicators is None:
                result.reason = "Failed to calculate M5 indicators"
                return result
            
            close = safe_float(indicators.get('close', 0), 0.0)
            rsi = safe_float(indicators.get('rsi', 50), 50.0)
            volume = safe_float(indicators.get('volume', 0), 0.0)
            volume_avg = safe_float(indicators.get('volume_avg', 1), 1.0)
            adx = safe_float(indicators.get('adx', 0), 0.0)
            adx_prev = safe_float(indicators.get('adx_prev', 0), 0.0)
            macd_histogram = safe_float(indicators.get('macd_histogram', 0), 0.0)
            
            if close <= 0:
                result.reason = "Invalid price data"
                return result
            
            high_series = df_m5['high'] if 'high' in df_m5.columns else None
            low_series = df_m5['low'] if 'low' in df_m5.columns else None
            
            if high_series is None or low_series is None:
                result.reason = "Missing high/low data"
                return result
            
            lookback_5 = min(5, len(df_m5) - 1)
            lookback_10 = min(10, len(df_m5) - 1)
            
            prev_high_5 = high_series.iloc[-lookback_5-1:-1].max() if lookback_5 > 0 else close
            prev_low_5 = low_series.iloc[-lookback_5-1:-1].min() if lookback_5 > 0 else close
            prev_high_10 = high_series.iloc[-lookback_10-1:-1].max() if lookback_10 > 0 else close
            prev_low_10 = low_series.iloc[-lookback_10-1:-1].min() if lookback_10 > 0 else close
            
            breakout_up_5 = close > prev_high_5
            breakout_down_5 = close < prev_low_5
            breakout_up_10 = close > prev_high_10
            breakout_down_10 = close < prev_low_10
            
            h1_sr_ahead = False
            h1_sr_distance = 999
            if df_h1 is not None and len(df_h1) >= 30:
                sr_data = self.indicator_engine.calculate_micro_support_resistance(df_h1, lookback=50)
                h1_resistance = sr_data.get('nearest_resistance', 0)
                h1_support = sr_data.get('nearest_support', 0)
                
                if close > 0:
                    dist_to_h1_resistance = abs(h1_resistance - close) / self.PIP_VALUE_XAUUSD if h1_resistance > 0 else 999
                    dist_to_h1_support = abs(close - h1_support) / self.PIP_VALUE_XAUUSD if h1_support > 0 else 999
                    
                    if breakout_up_5 or breakout_up_10:
                        h1_sr_distance = dist_to_h1_resistance
                        h1_sr_ahead = dist_to_h1_resistance < self.BO_SR_PROXIMITY_PIPS
                    elif breakout_down_5 or breakout_down_10:
                        h1_sr_distance = dist_to_h1_support
                        h1_sr_ahead = dist_to_h1_support < self.BO_SR_PROXIMITY_PIPS
            
            buy_confluences = []
            sell_confluences = []
            
            if breakout_up_10:
                buy_confluences.append("Break 10-candle high")
            elif breakout_up_5:
                buy_confluences.append("Break 5-candle high")
            
            if breakout_down_10:
                sell_confluences.append("Break 10-candle low")
            elif breakout_down_5:
                sell_confluences.append("Break 5-candle low")
            
            if (breakout_up_5 or breakout_up_10) and close > prev_high_5:
                buy_confluences.append("Close confirmed above breakout level")
            if (breakout_down_5 or breakout_down_10) and close < prev_low_5:
                sell_confluences.append("Close confirmed below breakout level")
            
            volume_ratio = volume / volume_avg if volume_avg > 0 else 1.0
            if volume_ratio >= self.BO_VOLUME_THRESHOLD:
                if breakout_up_5 or breakout_up_10:
                    buy_confluences.append(f"Volume breakout ({volume_ratio:.2f}x)")
                elif breakout_down_5 or breakout_down_10:
                    sell_confluences.append(f"Volume breakout ({volume_ratio:.2f}x)")
            
            if rsi > 50 and (breakout_up_5 or breakout_up_10):
                buy_confluences.append(f"RSI bullish momentum ({rsi:.1f})")
            elif rsi < 50 and (breakout_down_5 or breakout_down_10):
                sell_confluences.append(f"RSI bearish momentum ({rsi:.1f})")
            
            if macd_histogram > 0:
                buy_confluences.append("MACD histogram positive")
            elif macd_histogram < 0:
                sell_confluences.append("MACD histogram negative")
            
            adx_increasing = adx > adx_prev
            adx_toward_target = adx >= 20 and adx_increasing
            
            if adx_toward_target:
                if breakout_up_5 or breakout_up_10:
                    buy_confluences.append(f"ADX increasing ({adx:.1f})")
                elif breakout_down_5 or breakout_down_10:
                    sell_confluences.append(f"ADX increasing ({adx:.1f})")
            
            if not h1_sr_ahead:
                if breakout_up_5 or breakout_up_10:
                    buy_confluences.append(f"Room to run ({h1_sr_distance:.1f} pips)")
                elif breakout_down_5 or breakout_down_10:
                    sell_confluences.append(f"Room to run ({h1_sr_distance:.1f} pips)")
            
            buy_count = len(buy_confluences)
            sell_count = len(sell_confluences)
            
            has_breakout = breakout_up_5 or breakout_up_10 or breakout_down_5 or breakout_down_10
            has_volume = volume_ratio >= self.BO_VOLUME_THRESHOLD
            
            if (buy_count >= self.BO_MIN_CONFLUENCE and 
                buy_count > sell_count and 
                (breakout_up_5 or breakout_up_10) and
                adx_increasing and
                not h1_sr_ahead and
                has_volume):
                
                result.signal_type = SignalType.BUY.value
                result.confluence_count = buy_count
                result.confluence_details = buy_confluences
                result.confidence = min(1.0, 0.5 + (buy_count * 0.1))
                
                if breakout_up_10:
                    result.confidence += 0.1
                if volume_ratio >= 2.0:
                    result.confidence += 0.1
                if adx >= self.BO_ADX_TARGET:
                    result.confidence += 0.1
                
                atr = safe_float(indicators.get('atr', 1.0), 1.0)
                volatility_mult = min(1.5, max(0.8, 1.0))
                
                sl, tp = self._calculate_sl_tp(
                    15.0, 60.0,
                    self.BO_SL_MIN, self.BO_SL_MAX,
                    self.BO_TP_MIN, self.BO_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"Breakout BUY: {buy_count} confluences, ADX={adx:.1f} - " + ", ".join(buy_confluences[:3])
                
            elif (sell_count >= self.BO_MIN_CONFLUENCE and 
                  sell_count > buy_count and 
                  (breakout_down_5 or breakout_down_10) and
                  adx_increasing and
                  not h1_sr_ahead and
                  has_volume):
                
                result.signal_type = SignalType.SELL.value
                result.confluence_count = sell_count
                result.confluence_details = sell_confluences
                result.confidence = min(1.0, 0.5 + (sell_count * 0.1))
                
                if breakout_down_10:
                    result.confidence += 0.1
                if volume_ratio >= 2.0:
                    result.confidence += 0.1
                if adx >= self.BO_ADX_TARGET:
                    result.confidence += 0.1
                
                atr = safe_float(indicators.get('atr', 1.0), 1.0)
                volatility_mult = min(1.5, max(0.8, 1.0))
                
                sl, tp = self._calculate_sl_tp(
                    15.0, 60.0,
                    self.BO_SL_MIN, self.BO_SL_MAX,
                    self.BO_TP_MIN, self.BO_TP_MAX,
                    volatility_mult
                )
                result.sl_pips = sl
                result.tp_pips = tp
                result.entry_price = close
                result.reason = f"Breakout SELL: {sell_count} confluences, ADX={adx:.1f} - " + ", ".join(sell_confluences[:3])
            else:
                bo_status = "Breakout detected" if has_breakout else "No breakout"
                vol_status = f"Vol={volume_ratio:.2f}x" if has_volume else "Low volume"
                adx_status = f"ADX={adx:.1f}" if adx_increasing else f"ADX flat ({adx:.1f})"
                result.reason = f"Breakout: Buy={buy_count}, Sell={sell_count} ({bo_status}, {vol_status}, {adx_status})"
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Breakout signal: {str(e)}")
            result.reason = f"Error: {str(e)}"
            return result
    
    def check_all_signals(self, df_m1: Optional[pd.DataFrame] = None,
                          df_m5: Optional[pd.DataFrame] = None,
                          df_h1: Optional[pd.DataFrame] = None) -> List[SignalResult]:
        """
        Check all 4 signal rules and return list of valid signals.
        
        Args:
            df_m1: DataFrame M1 timeframe (optional)
            df_m5: DataFrame M5 timeframe (optional)
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            List of SignalResult for each rule
        """
        results = []
        
        if df_m1 is not None:
            m1_signal = self.check_m1_scalp_signal(df_m1, df_m5, df_h1)
            results.append(m1_signal)
            if m1_signal.is_valid():
                logger.info(f"M1 Scalp Signal: {m1_signal.signal_type} | Confidence: {m1_signal.confidence:.2f} | {m1_signal.reason}")
        
        if df_m5 is not None:
            m5_signal = self.check_m5_swing_signal(df_m5, df_h1)
            results.append(m5_signal)
            if m5_signal.is_valid():
                logger.info(f"M5 Swing Signal: {m5_signal.signal_type} | Confidence: {m5_signal.confidence:.2f} | {m5_signal.reason}")
            
            sr_signal = self.check_sr_reversion_signal(df_m5, df_h1)
            results.append(sr_signal)
            if sr_signal.is_valid():
                logger.info(f"S/R Reversion Signal: {sr_signal.signal_type} | Confidence: {sr_signal.confidence:.2f} | {sr_signal.reason}")
            
            bo_signal = self.check_breakout_signal(df_m5, df_h1)
            results.append(bo_signal)
            if bo_signal.is_valid():
                logger.info(f"Breakout Signal: {bo_signal.signal_type} | Confidence: {bo_signal.confidence:.2f} | {bo_signal.reason}")
        
        return results
    
    def get_best_signal(self, df_m1: Optional[pd.DataFrame] = None,
                         df_m5: Optional[pd.DataFrame] = None,
                         df_h1: Optional[pd.DataFrame] = None) -> Optional[SignalResult]:
        """
        Get the best signal from all rules based on confidence.
        
        Args:
            df_m1: DataFrame M1 timeframe (optional)
            df_m5: DataFrame M5 timeframe (optional)
            df_h1: DataFrame H1 timeframe (optional)
        
        Returns:
            Best SignalResult or None if no valid signal
        """
        all_signals = self.check_all_signals(df_m1, df_m5, df_h1)
        
        valid_signals = [s for s in all_signals if s.is_valid()]
        
        if not valid_signals:
            return None
        
        best_signal = max(valid_signals, key=lambda s: s.confidence)
        
        logger.info(f"Best Signal Selected: {best_signal.rule_name} {best_signal.signal_type} | Confidence: {best_signal.confidence:.2f}")
        
        return best_signal
