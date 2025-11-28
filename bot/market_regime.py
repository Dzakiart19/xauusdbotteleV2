"""
Market Regime Detector untuk Bot Trading XAUUSD.

Modul ini menyediakan deteksi kondisi pasar untuk optimasi strategi trading:
- Trend Strength Detection (ADX-based)
- Volatility Assessment (ATR-based)
- Price Position Analysis (Support/Resistance)
- Breakout Detection
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum

from bot.logger import setup_logger
from bot.strategy import safe_float, is_valid_number
from bot.indicators import IndicatorEngine, safe_series_operation

logger = setup_logger('MarketRegime')


class MarketRegimeError(Exception):
    """Base exception for market regime detection errors"""
    pass


class RegimeType(str, Enum):
    """Enum untuk tipe market regime"""
    STRONG_TREND = 'strong_trend'
    MODERATE_TREND = 'moderate_trend'
    RANGE_BOUND = 'range_bound'
    BREAKOUT = 'breakout'
    HIGH_VOLATILITY = 'high_volatility'
    WEAK_TREND = 'weak_trend'
    UNKNOWN = 'unknown'


class BiasType(str, Enum):
    """Enum untuk bias arah trading"""
    BUY = 'BUY'
    SELL = 'SELL'
    NEUTRAL = 'NEUTRAL'


@dataclass
class TrendAnalysis:
    """Data hasil analisis trend"""
    adx: float = 0.0
    plus_di: float = 0.0
    minus_di: float = 0.0
    trend_strength: str = 'none'
    trend_direction: str = 'neutral'
    is_strong_trend: bool = False
    is_trending: bool = False


@dataclass
class VolatilityAnalysis:
    """Data hasil analisis volatility"""
    current_atr: float = 0.0
    average_atr: float = 0.0
    atr_ratio: float = 1.0
    volatility_level: str = 'normal'
    is_high_volatility: bool = False
    is_low_volatility: bool = False
    suggested_sl_multiplier: float = 1.0


@dataclass
class PricePositionAnalysis:
    """Data hasil analisis posisi harga"""
    current_price: float = 0.0
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    distance_to_support: float = 0.0
    distance_to_resistance: float = 0.0
    price_position: str = 'midpoint'
    position_bias: str = 'NEUTRAL'
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)


@dataclass
class BreakoutAnalysis:
    """Data hasil analisis breakout"""
    is_tight_range: bool = False
    range_compression_ratio: float = 1.0
    is_breakout_up: bool = False
    is_breakout_down: bool = False
    breakout_candles: int = 10
    volume_increasing: bool = False
    volume_ratio: float = 1.0
    breakout_probability: float = 0.0
    breakout_direction: str = 'none'


@dataclass
class MarketRegime:
    """Dataclass utama untuk hasil analisis market regime"""
    regime_type: str = 'unknown'
    bias: str = 'NEUTRAL'
    strictness_level: float = 1.0
    
    trend_analysis: TrendAnalysis = field(default_factory=TrendAnalysis)
    volatility_analysis: VolatilityAnalysis = field(default_factory=VolatilityAnalysis)
    price_position: PricePositionAnalysis = field(default_factory=PricePositionAnalysis)
    breakout_analysis: BreakoutAnalysis = field(default_factory=BreakoutAnalysis)
    
    confidence: float = 0.0
    analysis_timestamp: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    recommended_strategies: List[str] = field(default_factory=list)
    avoid_strategies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert MarketRegime to dictionary"""
        return {
            'regime_type': self.regime_type,
            'bias': self.bias,
            'strictness_level': self.strictness_level,
            'trend': {
                'adx': self.trend_analysis.adx,
                'plus_di': self.trend_analysis.plus_di,
                'minus_di': self.trend_analysis.minus_di,
                'strength': self.trend_analysis.trend_strength,
                'direction': self.trend_analysis.trend_direction,
                'is_trending': self.trend_analysis.is_trending
            },
            'volatility': {
                'current_atr': self.volatility_analysis.current_atr,
                'average_atr': self.volatility_analysis.average_atr,
                'atr_ratio': self.volatility_analysis.atr_ratio,
                'level': self.volatility_analysis.volatility_level,
                'sl_multiplier': self.volatility_analysis.suggested_sl_multiplier
            },
            'price_position': {
                'current': self.price_position.current_price,
                'support': self.price_position.nearest_support,
                'resistance': self.price_position.nearest_resistance,
                'position': self.price_position.price_position,
                'bias': self.price_position.position_bias
            },
            'breakout': {
                'is_tight_range': self.breakout_analysis.is_tight_range,
                'compression_ratio': self.breakout_analysis.range_compression_ratio,
                'is_breakout_up': self.breakout_analysis.is_breakout_up,
                'is_breakout_down': self.breakout_analysis.is_breakout_down,
                'volume_increasing': self.breakout_analysis.volume_increasing,
                'probability': self.breakout_analysis.breakout_probability
            },
            'confidence': self.confidence,
            'timestamp': self.analysis_timestamp,
            'recommendations': {
                'use': self.recommended_strategies,
                'avoid': self.avoid_strategies
            },
            'warnings': self.warnings
        }


class MarketRegimeDetector:
    """
    Market Regime Detector untuk mengidentifikasi kondisi pasar XAUUSD.
    
    Fitur utama:
    1. Trend Strength Detection (ADX-based)
    2. Volatility Assessment (ATR-based)
    3. Price Position Analysis (Support/Resistance)
    4. Breakout Detection
    """
    
    ADX_STRONG_TREND = 35
    ADX_MODERATE_TREND = 20
    ADX_WEAK_TREND_LOW = 15
    ADX_WEAK_TREND_HIGH = 20
    
    ATR_HIGH_VOLATILITY_THRESHOLD = 1.5
    ATR_NORMAL_VOLATILITY_LOW = 1.0
    ATR_NORMAL_VOLATILITY_HIGH = 1.5
    ATR_LOW_VOLATILITY_THRESHOLD = 1.0
    
    ATR_AVERAGE_PERIOD = 20
    BREAKOUT_CANDLES = 10
    RANGE_TIGHT_THRESHOLD = 0.7
    
    SUPPORT_PROXIMITY_PCT = 0.002
    RESISTANCE_PROXIMITY_PCT = 0.002
    
    def __init__(self, config, indicator_engine: Optional[IndicatorEngine] = None):
        """
        Inisialisasi MarketRegimeDetector.
        
        Args:
            config: Objek konfigurasi bot
            indicator_engine: Instance IndicatorEngine (opsional, akan dibuat jika tidak ada)
        """
        self.config = config
        self.indicator_engine = indicator_engine or IndicatorEngine(config)
        self._last_regime: Optional[MarketRegime] = None
        logger.info("MarketRegimeDetector initialized")
    
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
    
    def _analyze_trend_strength(self, df: pd.DataFrame, adx_period: int = 14) -> TrendAnalysis:
        """
        Analisis kekuatan trend berdasarkan ADX.
        
        ADX Thresholds:
        - ADX > 35: Strong trend (trend-following only)
        - ADX 20-35: Moderate trend (allow both trend & reversal)
        - ADX 15-20: Weak trend (strict confluence required)
        - ADX < 15: No trend/Ranging (mean-reversion strategy)
        
        Args:
            df: DataFrame dengan data OHLC
            adx_period: Periode untuk kalkulasi ADX
            
        Returns:
            TrendAnalysis dengan hasil analisis
        """
        result = TrendAnalysis()
        
        try:
            if df is None or len(df) < adx_period + 1:
                logger.warning(f"Insufficient data for ADX calculation (need {adx_period + 1}, got {len(df) if df is not None else 0})")
                return result
            
            adx_series, plus_di_series, minus_di_series = self.indicator_engine.calculate_adx(df, adx_period)
            
            result.adx = safe_float(self._safe_get_value(adx_series, -1, 0.0), 0.0, "adx")
            result.plus_di = safe_float(self._safe_get_value(plus_di_series, -1, 0.0), 0.0, "plus_di")
            result.minus_di = safe_float(self._safe_get_value(minus_di_series, -1, 0.0), 0.0, "minus_di")
            
            if result.adx > self.ADX_STRONG_TREND:
                result.trend_strength = 'strong'
                result.is_strong_trend = True
                result.is_trending = True
            elif result.adx >= self.ADX_MODERATE_TREND:
                result.trend_strength = 'moderate'
                result.is_strong_trend = False
                result.is_trending = True
            elif result.adx >= self.ADX_WEAK_TREND_LOW:
                result.trend_strength = 'weak'
                result.is_strong_trend = False
                result.is_trending = False
            else:
                result.trend_strength = 'none'
                result.is_strong_trend = False
                result.is_trending = False
            
            if result.plus_di > result.minus_di:
                result.trend_direction = 'bullish'
            elif result.minus_di > result.plus_di:
                result.trend_direction = 'bearish'
            else:
                result.trend_direction = 'neutral'
            
            logger.debug(f"Trend Analysis: ADX={result.adx:.2f}, +DI={result.plus_di:.2f}, -DI={result.minus_di:.2f}, Strength={result.trend_strength}")
            
        except Exception as e:
            logger.error(f"Error in trend strength analysis: {str(e)}")
        
        return result
    
    def _analyze_volatility(self, df: pd.DataFrame, atr_period: int = 14) -> VolatilityAnalysis:
        """
        Analisis volatilitas berdasarkan ATR.
        
        Volatility Thresholds:
        - High volatility (ATR > 150% of 20-SMA ATR): Allow scalping dengan tighter SL
        - Normal volatility (ATR 100-150%): Standard confluence requirement
        - Low volatility (ATR < 100%): Increase confluence requirement
        
        Args:
            df: DataFrame dengan data OHLC
            atr_period: Periode untuk kalkulasi ATR
            
        Returns:
            VolatilityAnalysis dengan hasil analisis
        """
        result = VolatilityAnalysis()
        
        try:
            if df is None or len(df) < max(atr_period, self.ATR_AVERAGE_PERIOD):
                logger.warning("Insufficient data for volatility analysis")
                return result
            
            atr_series = self.indicator_engine.calculate_atr(df, atr_period)
            result.current_atr = safe_float(self._safe_get_value(atr_series, -1, 0.0), 0.0, "current_atr")
            
            if len(atr_series) >= self.ATR_AVERAGE_PERIOD:
                avg_atr = atr_series.rolling(window=self.ATR_AVERAGE_PERIOD, min_periods=1).mean()
                result.average_atr = safe_float(self._safe_get_value(avg_atr, -1, result.current_atr), result.current_atr, "average_atr")
            else:
                result.average_atr = safe_float(atr_series.mean(), result.current_atr, "average_atr")
            
            if result.average_atr > 0:
                result.atr_ratio = safe_float(result.current_atr / result.average_atr, 1.0, "atr_ratio")
            else:
                result.atr_ratio = 1.0
            
            if result.atr_ratio > self.ATR_HIGH_VOLATILITY_THRESHOLD:
                result.volatility_level = 'high'
                result.is_high_volatility = True
                result.is_low_volatility = False
                result.suggested_sl_multiplier = 0.8
            elif result.atr_ratio >= self.ATR_LOW_VOLATILITY_THRESHOLD:
                result.volatility_level = 'normal'
                result.is_high_volatility = False
                result.is_low_volatility = False
                result.suggested_sl_multiplier = 1.0
            else:
                result.volatility_level = 'low'
                result.is_high_volatility = False
                result.is_low_volatility = True
                result.suggested_sl_multiplier = 1.2
            
            logger.debug(f"Volatility Analysis: ATR={result.current_atr:.2f}, Avg={result.average_atr:.2f}, Ratio={result.atr_ratio:.2f}, Level={result.volatility_level}")
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
        
        return result
    
    def _analyze_price_position(self, df: pd.DataFrame, lookback: int = 50) -> PricePositionAnalysis:
        """
        Analisis posisi harga relatif terhadap support/resistance.
        
        Position Analysis:
        - Price near support: Bias BUY signals
        - Price near resistance: Bias SELL signals
        - Price at midpoint: Neutral
        
        Args:
            df: DataFrame dengan data OHLC
            lookback: Periode lookback untuk support/resistance
            
        Returns:
            PricePositionAnalysis dengan hasil analisis
        """
        result = PricePositionAnalysis()
        
        try:
            if df is None or len(df) < 5:
                logger.warning("Insufficient data for price position analysis")
                return result
            
            close = df['close'] if 'close' in df.columns else None
            if close is None or len(close) == 0:
                return result
            
            result.current_price = safe_float(self._safe_get_value(close, -1, 0.0), 0.0, "current_price")
            
            if result.current_price <= 0:
                logger.warning("Invalid current price for position analysis")
                return result
            
            sr_data = self.indicator_engine.calculate_micro_support_resistance(df, lookback)
            
            result.nearest_support = safe_float(sr_data.get('nearest_support', 0.0), 0.0, "nearest_support")
            result.nearest_resistance = safe_float(sr_data.get('nearest_resistance', 0.0), 0.0, "nearest_resistance")
            result.support_levels = sr_data.get('support_levels', [])
            result.resistance_levels = sr_data.get('resistance_levels', [])
            
            if result.current_price > 0:
                result.distance_to_support = safe_float(
                    (result.current_price - result.nearest_support) / result.current_price,
                    0.0, "distance_to_support"
                )
                result.distance_to_resistance = safe_float(
                    (result.nearest_resistance - result.current_price) / result.current_price,
                    0.0, "distance_to_resistance"
                )
            
            if result.distance_to_support <= self.SUPPORT_PROXIMITY_PCT:
                result.price_position = 'near_support'
                result.position_bias = BiasType.BUY.value
            elif result.distance_to_resistance <= self.RESISTANCE_PROXIMITY_PCT:
                result.price_position = 'near_resistance'
                result.position_bias = BiasType.SELL.value
            else:
                total_range = result.distance_to_support + result.distance_to_resistance
                if total_range > 0:
                    support_pct = result.distance_to_support / total_range
                    if support_pct < 0.3:
                        result.price_position = 'lower_zone'
                        result.position_bias = BiasType.BUY.value
                    elif support_pct > 0.7:
                        result.price_position = 'upper_zone'
                        result.position_bias = BiasType.SELL.value
                    else:
                        result.price_position = 'midpoint'
                        result.position_bias = BiasType.NEUTRAL.value
                else:
                    result.price_position = 'midpoint'
                    result.position_bias = BiasType.NEUTRAL.value
            
            logger.debug(f"Price Position: Current={result.current_price:.2f}, Support={result.nearest_support:.2f}, Resistance={result.nearest_resistance:.2f}, Position={result.price_position}")
            
        except Exception as e:
            logger.error(f"Error in price position analysis: {str(e)}")
        
        return result
    
    def _analyze_breakout(self, df: pd.DataFrame, breakout_candles: int = 10) -> BreakoutAnalysis:
        """
        Analisis potensi breakout.
        
        Breakout Detection:
        - Last N candles range tight? Prepare breakout entry
        - Price break previous N candles high/low? Continuation signal
        - Volume increasing? Higher follow-through probability
        
        Args:
            df: DataFrame dengan data OHLC
            breakout_candles: Jumlah candle untuk analisis breakout
            
        Returns:
            BreakoutAnalysis dengan hasil analisis
        """
        result = BreakoutAnalysis()
        result.breakout_candles = breakout_candles
        
        try:
            if df is None or len(df) < breakout_candles + 1:
                logger.warning(f"Insufficient data for breakout analysis (need {breakout_candles + 1})")
                return result
            
            recent_df = df.tail(breakout_candles)
            
            high = recent_df['high'] if 'high' in recent_df.columns else None
            low = recent_df['low'] if 'low' in recent_df.columns else None
            close = df['close'] if 'close' in df.columns else None
            
            if high is None or low is None or close is None:
                return result
            
            period_high = safe_float(high.max(), 0.0, "period_high")
            period_low = safe_float(low.min(), 0.0, "period_low")
            current_close = safe_float(self._safe_get_value(close, -1, 0.0), 0.0, "current_close")
            
            period_range = period_high - period_low
            
            if len(df) > breakout_candles * 2:
                earlier_df = df.iloc[-(breakout_candles * 2):-breakout_candles]
                earlier_high = earlier_df['high'].max() if 'high' in earlier_df.columns else period_high
                earlier_low = earlier_df['low'].min() if 'low' in earlier_df.columns else period_low
                earlier_range = safe_float(earlier_high - earlier_low, period_range, "earlier_range")
                
                if earlier_range > 0:
                    result.range_compression_ratio = safe_float(period_range / earlier_range, 1.0, "compression_ratio")
                else:
                    result.range_compression_ratio = 1.0
            else:
                result.range_compression_ratio = 1.0
            
            result.is_tight_range = result.range_compression_ratio < self.RANGE_TIGHT_THRESHOLD
            
            if len(df) > breakout_candles:
                prev_period_high = safe_float(df.iloc[:-1].tail(breakout_candles)['high'].max(), period_high, "prev_period_high")
                prev_period_low = safe_float(df.iloc[:-1].tail(breakout_candles)['low'].min(), period_low, "prev_period_low")
                
                result.is_breakout_up = current_close > prev_period_high
                result.is_breakout_down = current_close < prev_period_low
            
            vol_data = self.indicator_engine.calculate_volume_confirmation(df, period=breakout_candles)
            result.volume_increasing = vol_data.get('is_volume_strong', False)
            result.volume_ratio = safe_float(vol_data.get('volume_ratio', 1.0), 1.0, "volume_ratio")
            
            probability = 0.0
            
            if result.is_tight_range:
                probability += 0.3
            
            if result.is_breakout_up or result.is_breakout_down:
                probability += 0.4
            
            if result.volume_increasing:
                probability += 0.2
            
            if result.volume_ratio > 1.5:
                probability += 0.1
            
            result.breakout_probability = min(probability, 1.0)
            
            if result.is_breakout_up:
                result.breakout_direction = 'up'
            elif result.is_breakout_down:
                result.breakout_direction = 'down'
            elif result.is_tight_range:
                result.breakout_direction = 'pending'
            else:
                result.breakout_direction = 'none'
            
            logger.debug(f"Breakout Analysis: Tight={result.is_tight_range}, BreakUp={result.is_breakout_up}, BreakDown={result.is_breakout_down}, VolInc={result.volume_increasing}, Prob={result.breakout_probability:.2f}")
            
        except Exception as e:
            logger.error(f"Error in breakout analysis: {str(e)}")
        
        return result
    
    def _determine_regime_type(self, 
                                trend: TrendAnalysis, 
                                volatility: VolatilityAnalysis, 
                                breakout: BreakoutAnalysis) -> str:
        """
        Tentukan tipe market regime berdasarkan hasil analisis.
        
        Priority:
        1. Breakout (jika ada breakout aktif dengan probabilitas tinggi)
        2. High Volatility (jika volatilitas sangat tinggi)
        3. Strong Trend (jika ADX > 35)
        4. Moderate Trend (jika ADX 20-35)
        5. Weak Trend (jika ADX 15-20)
        6. Range Bound (default untuk ADX < 15)
        """
        if (breakout.is_breakout_up or breakout.is_breakout_down) and breakout.breakout_probability > 0.5:
            return RegimeType.BREAKOUT.value
        
        if volatility.is_high_volatility and volatility.atr_ratio > 2.0:
            return RegimeType.HIGH_VOLATILITY.value
        
        if trend.is_strong_trend:
            return RegimeType.STRONG_TREND.value
        
        if trend.is_trending and trend.trend_strength == 'moderate':
            return RegimeType.MODERATE_TREND.value
        
        if trend.trend_strength == 'weak':
            return RegimeType.WEAK_TREND.value
        
        return RegimeType.RANGE_BOUND.value
    
    def _determine_bias(self, 
                        trend: TrendAnalysis, 
                        price_pos: PricePositionAnalysis, 
                        breakout: BreakoutAnalysis) -> str:
        """
        Tentukan bias arah trading berdasarkan hasil analisis.
        
        Priority:
        1. Breakout direction (jika ada breakout aktif)
        2. Strong trend direction (jika ADX tinggi)
        3. Price position bias (support/resistance proximity)
        """
        if breakout.is_breakout_up and breakout.breakout_probability > 0.5:
            return BiasType.BUY.value
        if breakout.is_breakout_down and breakout.breakout_probability > 0.5:
            return BiasType.SELL.value
        
        if trend.is_strong_trend or (trend.is_trending and trend.trend_strength == 'moderate'):
            if trend.trend_direction == 'bullish':
                return BiasType.BUY.value
            elif trend.trend_direction == 'bearish':
                return BiasType.SELL.value
        
        return price_pos.position_bias
    
    def _calculate_strictness_level(self, 
                                    trend: TrendAnalysis, 
                                    volatility: VolatilityAnalysis,
                                    breakout: BreakoutAnalysis) -> float:
        """
        Hitung level strictness untuk confluence requirement.
        
        Strictness Range: 0.5 - 2.0
        - 0.5: Sangat longgar (strong trend, clear breakout)
        - 1.0: Normal (moderate conditions)
        - 2.0: Sangat ketat (weak trend, low volatility)
        """
        strictness = 1.0
        
        if trend.is_strong_trend:
            strictness -= 0.3
        elif trend.trend_strength == 'moderate':
            strictness -= 0.1
        elif trend.trend_strength == 'weak':
            strictness += 0.3
        else:
            strictness += 0.5
        
        if volatility.is_low_volatility:
            strictness += 0.3
        elif volatility.is_high_volatility:
            strictness -= 0.1
        
        if breakout.breakout_probability > 0.7:
            strictness -= 0.2
        elif breakout.is_tight_range:
            strictness += 0.1
        
        return max(0.5, min(2.0, strictness))
    
    def _generate_recommendations(self, regime_type: str, 
                                   trend: TrendAnalysis,
                                   volatility: VolatilityAnalysis) -> Tuple[List[str], List[str]]:
        """Generate strategi yang direkomendasikan dan yang harus dihindari."""
        recommended = []
        avoid = []
        
        if regime_type == RegimeType.STRONG_TREND.value:
            recommended = ['trend_following', 'breakout_continuation', 'pullback_entry']
            avoid = ['counter_trend', 'mean_reversion', 'range_trading']
        
        elif regime_type == RegimeType.MODERATE_TREND.value:
            recommended = ['trend_following', 'pullback_entry', 'swing_trading']
            avoid = ['aggressive_scalping', 'pure_range_trading']
        
        elif regime_type == RegimeType.RANGE_BOUND.value:
            recommended = ['mean_reversion', 'range_trading', 'support_resistance_bounce']
            avoid = ['trend_following', 'breakout_anticipation']
        
        elif regime_type == RegimeType.BREAKOUT.value:
            recommended = ['breakout_entry', 'momentum_trading', 'quick_scalp']
            avoid = ['mean_reversion', 'fade_the_move', 'range_trading']
        
        elif regime_type == RegimeType.HIGH_VOLATILITY.value:
            recommended = ['scalping_tight_sl', 'momentum_trading', 'quick_profit_taking']
            avoid = ['wide_stop_trades', 'overnight_positions', 'aggressive_sizing']
        
        elif regime_type == RegimeType.WEAK_TREND.value:
            recommended = ['high_confluence_only', 'small_position', 'quick_exit']
            avoid = ['aggressive_entry', 'large_positions', 'trend_following']
        
        else:
            recommended = ['conservative', 'wait_for_clarity']
            avoid = ['aggressive_trading']
        
        if volatility.is_high_volatility:
            if 'tight_sl' not in str(recommended):
                recommended.append('reduce_position_size')
            if 'wide_stop_trades' not in avoid:
                avoid.append('wide_stop_trades')
        
        return recommended, avoid
    
    def _calculate_confidence(self, 
                              trend: TrendAnalysis, 
                              volatility: VolatilityAnalysis,
                              price_pos: PricePositionAnalysis,
                              breakout: BreakoutAnalysis) -> float:
        """Hitung confidence level untuk analisis (0.0 - 1.0)."""
        confidence = 0.5
        
        if trend.adx > 10:
            confidence += 0.1
        if trend.adx > 25:
            confidence += 0.1
        
        if volatility.current_atr > 0 and volatility.average_atr > 0:
            confidence += 0.1
        
        if price_pos.nearest_support > 0 and price_pos.nearest_resistance > 0:
            confidence += 0.1
        
        if breakout.volume_ratio > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def get_regime(self, 
                   indicators: Dict[str, Any], 
                   m1_df: Optional[pd.DataFrame] = None, 
                   m5_df: Optional[pd.DataFrame] = None) -> MarketRegime:
        """
        Analisis kondisi pasar dan return MarketRegime.
        
        Args:
            indicators: Dict dengan data indikator (dari IndicatorEngine)
            m1_df: DataFrame M1 untuk analisis detail
            m5_df: DataFrame M5 untuk konfirmasi (opsional)
            
        Returns:
            MarketRegime dengan hasil analisis lengkap
        """
        from datetime import datetime
        
        regime = MarketRegime()
        regime.analysis_timestamp = datetime.now().isoformat()
        
        try:
            df = m1_df
            if df is None or len(df) == 0:
                if m5_df is not None and len(m5_df) > 0:
                    df = m5_df
                    regime.warnings.append("Using M5 data as fallback (M1 not available)")
                else:
                    regime.warnings.append("No price data available for analysis")
                    logger.warning("No DataFrame provided for market regime analysis")
                    return regime
            
            regime.trend_analysis = self._analyze_trend_strength(df)
            regime.volatility_analysis = self._analyze_volatility(df)
            regime.price_position = self._analyze_price_position(df)
            regime.breakout_analysis = self._analyze_breakout(df)
            
            if m5_df is not None and len(m5_df) > 0:
                m5_trend = self._analyze_trend_strength(m5_df)
                
                if m5_trend.trend_strength != regime.trend_analysis.trend_strength:
                    regime.warnings.append(f"M1/M5 trend strength mismatch: M1={regime.trend_analysis.trend_strength}, M5={m5_trend.trend_strength}")
                
                if m5_trend.trend_direction != regime.trend_analysis.trend_direction:
                    regime.warnings.append(f"M1/M5 trend direction mismatch: M1={regime.trend_analysis.trend_direction}, M5={m5_trend.trend_direction}")
            
            if indicators:
                if 'adx' in indicators and is_valid_number(indicators.get('adx')):
                    indicator_adx = safe_float(indicators['adx'], 0.0)
                    if abs(indicator_adx - regime.trend_analysis.adx) > 5:
                        regime.trend_analysis.adx = (regime.trend_analysis.adx + indicator_adx) / 2
                
                if 'atr' in indicators and is_valid_number(indicators.get('atr')):
                    indicator_atr = safe_float(indicators['atr'], 0.0)
                    if indicator_atr > 0:
                        regime.volatility_analysis.current_atr = indicator_atr
            
            regime.regime_type = self._determine_regime_type(
                regime.trend_analysis,
                regime.volatility_analysis,
                regime.breakout_analysis
            )
            
            regime.bias = self._determine_bias(
                regime.trend_analysis,
                regime.price_position,
                regime.breakout_analysis
            )
            
            regime.strictness_level = self._calculate_strictness_level(
                regime.trend_analysis,
                regime.volatility_analysis,
                regime.breakout_analysis
            )
            
            regime.recommended_strategies, regime.avoid_strategies = self._generate_recommendations(
                regime.regime_type,
                regime.trend_analysis,
                regime.volatility_analysis
            )
            
            regime.confidence = self._calculate_confidence(
                regime.trend_analysis,
                regime.volatility_analysis,
                regime.price_position,
                regime.breakout_analysis
            )
            
            self._last_regime = regime
            
            logger.info(f"Market Regime: {regime.regime_type} | Bias: {regime.bias} | Strictness: {regime.strictness_level:.2f} | Confidence: {regime.confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error in get_regime: {str(e)}")
            regime.warnings.append(f"Analysis error: {str(e)}")
        
        return regime
    
    def get_last_regime(self) -> Optional[MarketRegime]:
        """Return hasil analisis regime terakhir."""
        return self._last_regime
    
    def is_favorable_for_entry(self, signal_type: str) -> Tuple[bool, str]:
        """
        Cek apakah kondisi market favorable untuk entry signal tertentu.
        
        Args:
            signal_type: 'BUY' atau 'SELL'
            
        Returns:
            Tuple[bool, str]: (is_favorable, reason)
        """
        if self._last_regime is None:
            return False, "No market regime analysis available"
        
        regime = self._last_regime
        
        if regime.regime_type == RegimeType.UNKNOWN.value:
            return False, "Market regime unknown"
        
        if regime.strictness_level > 1.5:
            return False, f"High strictness required ({regime.strictness_level:.2f}), need more confluence"
        
        if regime.regime_type == RegimeType.RANGE_BOUND.value:
            if signal_type == BiasType.BUY.value and regime.price_position.price_position != 'near_support':
                return False, "Range-bound market: BUY only near support"
            if signal_type == BiasType.SELL.value and regime.price_position.price_position != 'near_resistance':
                return False, "Range-bound market: SELL only near resistance"
        
        if regime.regime_type == RegimeType.STRONG_TREND.value:
            if regime.bias != signal_type and regime.bias != BiasType.NEUTRAL.value:
                return False, f"Strong trend favors {regime.bias}, not {signal_type}"
        
        return True, f"Favorable for {signal_type} in {regime.regime_type} regime"
    
    def get_adjusted_sl_multiplier(self) -> float:
        """Return SL multiplier berdasarkan kondisi volatilitas."""
        if self._last_regime is None:
            return 1.0
        return self._last_regime.volatility_analysis.suggested_sl_multiplier
    
    def get_position_size_factor(self) -> float:
        """
        Return faktor untuk adjustment position size berdasarkan regime.
        
        Returns:
            float: Multiplier untuk position size (0.5 - 1.5)
        """
        if self._last_regime is None:
            return 1.0
        
        regime = self._last_regime
        factor = 1.0
        
        if regime.volatility_analysis.is_high_volatility:
            factor *= 0.7
        
        if regime.strictness_level > 1.3:
            factor *= 0.8
        
        if regime.regime_type == RegimeType.STRONG_TREND.value and regime.confidence > 0.7:
            factor *= 1.1
        
        return max(0.5, min(1.5, factor))
