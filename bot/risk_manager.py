from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import pytz
from sqlalchemy.exc import SQLAlchemyError
from bot.logger import setup_logger

logger = setup_logger('RiskManager')

class RiskManagerError(Exception):
    """Exception untuk error pada risk management"""
    pass

class RiskManager:
    def __init__(self, config, db_manager, alert_system=None):
        self.config = config
        self.db = db_manager
        self.alert_system = alert_system
        self.last_signal_time = {}
        self.daily_stats = {}
        self.risk_warning_sent_today = {}
        
    def can_trade(self, user_id: int, signal_type: str, spread: Optional[float] = None) -> tuple[bool, Optional[str]]:
        utc_now = datetime.now(pytz.UTC)
        jakarta_tz = pytz.timezone('Asia/Jakarta')
        jakarta_time = utc_now.astimezone(jakarta_tz)
        today_str = jakarta_time.strftime('%Y-%m-%d')
        
        time_valid, time_reason = self.check_time_filter()
        if not time_valid:
            logger.info(f"Trade blocked for user {user_id}: {time_reason}")
            return False, time_reason
        
        if spread is not None:
            spread_valid, spread_reason = self.check_spread_filter(spread)
            if not spread_valid:
                logger.info(f"Trade blocked for user {user_id}: {spread_reason}")
                return False, spread_reason
        
        # UNLIMITED - Cooldown dihapus, sinyal tanpa batas
        # Tidak ada pengecekan waktu cooldown antar sinyal
        
        cache_key = f"{user_id}_{today_str}"
        cache_ttl = 60
        
        if cache_key in self.daily_stats:
            cached_data = self.daily_stats[cache_key]
            cache_age = (utc_now - cached_data['timestamp']).total_seconds()
            if cache_age < cache_ttl:
                total_daily_pl = cached_data['total_pl']
                logger.debug(f"Using cached daily P/L for user {user_id}: ${total_daily_pl:.2f} (age: {cache_age:.1f}s)")
            else:
                total_daily_pl = None
        else:
            total_daily_pl = None
        
        if total_daily_pl is None:
            session = self.db.get_session()
            if session is None:
                logger.error("Failed to get database session")
                return False, "Error: Database session tidak tersedia"
            try:
                from bot.database import Trade
                
                today_start = jakarta_time.replace(hour=0, minute=0, second=0, microsecond=0)
                today_start_utc = today_start.astimezone(pytz.UTC)
                
                daily_pl = session.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.signal_time >= today_start_utc,
                    Trade.actual_pl.isnot(None)
                ).with_entities(Trade.actual_pl).all()
                
                total_daily_pl = sum([pl[0] for pl in daily_pl if pl[0] is not None])
                
                self.daily_stats[cache_key] = {
                    'total_pl': total_daily_pl,
                    'timestamp': utc_now
                }
                logger.debug(f"Cached daily P/L for user {user_id}: ${total_daily_pl:.2f}")
                
            except SQLAlchemyError as e:
                logger.error(f"Database error checking trade eligibility: {e}")
                return False, f"Error database: {str(e)}"
            except ValueError as e:
                logger.error(f"Nilai tidak valid saat checking trade eligibility: {e}")
                return False, f"Error nilai: {str(e)}"
            except TypeError as e:
                logger.error(f"Tipe data salah saat checking trade eligibility: {e}")
                return False, f"Error tipe data: {str(e)}"
            except Exception as e:
                logger.error(f"Error tidak terduga checking trade eligibility: {e}")
                raise RiskManagerError(f"Error checking trade eligibility: {e}") from e
            finally:
                session.close()
        
        # UNLIMITED - Batas kerugian harian dihapus
        # Tidak ada pengecekan daily loss limit
        
        return True, None
    
    def check_spread_filter(self, spread: float) -> Tuple[bool, str]:
        try:
            max_spread = self.config.MAX_SPREAD_PIPS
            
            if spread <= 0:
                logger.warning(f"Invalid spread value: {spread}")
                return True, "Spread valid (invalid value ignored)"
            
            if spread > max_spread:
                reason = f"Spread terlalu tinggi: {spread:.1f} pips (max: {max_spread:.1f} pips)"
                logger.info(f"Spread filter blocked: {reason}")
                return False, reason
            
            logger.debug(f"Spread check passed: {spread:.1f} pips <= {max_spread:.1f} pips")
            return True, "Spread dalam batas normal"
            
        except AttributeError as e:
            logger.error(f"Config attribute error in check_spread_filter: {e}")
            return True, f"Spread check error (config): {str(e)}"
        except (ValueError, TypeError) as e:
            logger.error(f"Nilai/tipe error in check_spread_filter: {e}")
            return True, f"Spread check error (nilai): {str(e)}"
        except Exception as e:
            logger.error(f"Error tidak terduga in check_spread_filter: {e}")
            return True, f"Spread check error: {str(e)}"
    
    def check_time_filter(self) -> Tuple[bool, str]:
        try:
            if getattr(self.config, 'UNLIMITED_TRADING_HOURS', False):
                logger.debug("UNLIMITED_TRADING_HOURS aktif - time filter dilewati")
                return True, "Mode unlimited - trading 24/7"
            
            utc_now = datetime.now(pytz.UTC)
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            jakarta_time = utc_now.astimezone(jakarta_tz)
            
            current_hour = jakarta_time.hour
            current_weekday = jakarta_time.weekday()
            
            if current_weekday == 4:
                friday_cutoff = self.config.FRIDAY_CUTOFF_HOUR
                if friday_cutoff < 24 and current_hour >= friday_cutoff:
                    reason = f"Trading dihentikan: Jumat setelah jam {friday_cutoff}:00 WIB"
                    logger.info(f"Friday cutoff: {reason}")
                    return False, reason
            
            if current_weekday == 5:
                return False, "Trading dihentikan: Hari Sabtu"
            
            if current_weekday == 6:
                return False, "Trading dihentikan: Hari Minggu"
            
            trading_start = self.config.TRADING_HOURS_START
            trading_end = self.config.TRADING_HOURS_END
            
            if trading_end >= 24:
                logger.debug(f"Trading 24 jam aktif")
                return True, "Trading 24 jam aktif"
            
            if current_hour < trading_start:
                reason = f"Di luar jam trading: sebelum {trading_start}:00 WIB (sekarang {current_hour}:00)"
                logger.info(f"Time filter blocked: {reason}")
                return False, reason
            
            if current_hour >= trading_end:
                reason = f"Di luar jam trading: setelah {trading_end}:00 WIB (sekarang {current_hour}:00)"
                logger.info(f"Time filter blocked: {reason}")
                return False, reason
            
            logger.debug(f"Time check passed: {current_hour}:00 WIB (trading hours: {trading_start}:00-{trading_end}:00)")
            return True, f"Dalam jam trading ({trading_start}:00-{trading_end}:00 WIB)"
            
        except AttributeError as e:
            logger.error(f"Config attribute error in check_time_filter: {e}")
            return True, f"Time check error (config): {str(e)}"
        except KeyError as e:
            logger.error(f"Key error in check_time_filter: {e}")
            return True, f"Time check error (key): {str(e)}"
        except Exception as e:
            logger.error(f"Error tidak terduga in check_time_filter: {e}")
            return True, f"Time check error: {str(e)}"
    
    def record_signal(self, user_id: int):
        self.last_signal_time[user_id] = datetime.now(pytz.UTC)
        logger.debug(f"Signal recorded for user {user_id}, cooldown timer started")
    
    def calculate_position_size(self, account_balance: float, entry_price: float, 
                               stop_loss: float, signal_type: str) -> float:
        try:
            sl_pips = abs(entry_price - stop_loss) * self.config.XAUUSD_PIP_VALUE
            
            if sl_pips <= 0:
                logger.warning(f"Invalid SL pips: {sl_pips}, using default lot size")
                return self.config.LOT_SIZE
            
            lot_size = self.calculate_lot_from_risk(account_balance, sl_pips)
            
            logger.info(f"Calculated position size: {lot_size:.2f} lots "
                       f"(Balance: {account_balance:.2f}, Risk: {self.config.RISK_PER_TRADE_PERCENT}%, "
                       f"SL: {sl_pips:.1f} pips)")
            
            return lot_size
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid calculating position size: {e}")
            return self.config.LOT_SIZE
        except ZeroDivisionError as e:
            logger.error(f"Division by zero calculating position size: {e}")
            return self.config.LOT_SIZE
        except ArithmeticError as e:
            logger.error(f"Arithmetic error calculating position size: {e}")
            return self.config.LOT_SIZE
        except AttributeError as e:
            logger.error(f"Config attribute error calculating position size: {e}")
            return self.config.LOT_SIZE
        except Exception as e:
            logger.error(f"Error tidak terduga calculating position size: {e}")
            return self.config.LOT_SIZE
    
    def calculate_lot_from_risk(self, account_balance: float, sl_pips: float) -> float:
        try:
            if account_balance <= 0:
                logger.warning(f"Invalid account balance: {account_balance}")
                return self.config.LOT_SIZE
            
            if sl_pips <= 0:
                logger.warning(f"Invalid SL pips: {sl_pips}")
                return self.config.LOT_SIZE
            
            risk_percent = self.config.RISK_PER_TRADE_PERCENT
            pip_value = self.config.XAUUSD_PIP_VALUE
            
            risk_amount = account_balance * risk_percent / 100
            
            lot_size = risk_amount / (sl_pips * pip_value)
            
            lot_size = max(0.01, min(lot_size, 1.0))
            
            lot_size = round(lot_size, 2)
            
            logger.debug(f"Lot calculation: Balance={account_balance:.2f}, "
                        f"Risk%={risk_percent}, SL_pips={sl_pips:.1f}, "
                        f"Risk_amount={risk_amount:.2f}, Lot={lot_size:.2f}")
            
            return lot_size
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except ZeroDivisionError as e:
            logger.error(f"Division by zero in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except ArithmeticError as e:
            logger.error(f"Arithmetic error in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except AttributeError as e:
            logger.error(f"Config attribute error in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
        except Exception as e:
            logger.error(f"Error tidak terduga in calculate_lot_from_risk: {e}")
            return self.config.LOT_SIZE
    
    def calculate_dynamic_sl(self, entry_price: float, signal_type: str, 
                            atr: float, spread: float) -> Tuple[float, float]:
        try:
            fixed_risk = getattr(self.config, 'FIXED_RISK_AMOUNT', 2.0)
            lot_size = getattr(self.config, 'LOT_SIZE', 0.01)
            pip_value = getattr(self.config, 'XAUUSD_PIP_VALUE', 10.0)
            
            dollar_per_pip = pip_value * lot_size
            sl_pips_from_risk = fixed_risk / dollar_per_pip if dollar_per_pip > 0 else 20.0
            sl_from_risk = sl_pips_from_risk / pip_value
            
            sl_from_atr = atr * self.config.SL_ATR_MULTIPLIER
            sl_from_min = self.config.MIN_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            sl_from_spread = spread * self.config.MIN_SL_SPREAD_MULTIPLIER
            
            sl_distance = min(sl_from_risk, max(sl_from_atr, sl_from_min, sl_from_spread))
            
            min_tp_ratio = getattr(self.config, 'TP_RR_RATIO', 1.5)
            max_tp_ratio = getattr(self.config, 'TP_RR_RATIO_MAX', 2.5)
            
            trend_strength = getattr(self, '_last_trend_strength', 0.5)
            dynamic_tp_ratio = min_tp_ratio + (max_tp_ratio - min_tp_ratio) * trend_strength
            
            tp_distance = sl_distance * dynamic_tp_ratio
            
            if signal_type == 'BUY':
                sl_price = entry_price - sl_distance
                tp_price = entry_price + tp_distance
            else:
                sl_price = entry_price + sl_distance
                tp_price = entry_price - tp_distance
            
            sl_pips = sl_distance * self.config.XAUUSD_PIP_VALUE
            tp_pips = tp_distance * self.config.XAUUSD_PIP_VALUE
            expected_loss = sl_pips * dollar_per_pip
            expected_profit = tp_pips * dollar_per_pip
            
            logger.info(f"💰 Fixed-Risk SL/TP for {signal_type}: "
                       f"Entry=${entry_price:.2f}, SL=${sl_price:.2f} ({sl_pips:.1f} pips), "
                       f"TP=${tp_price:.2f} ({tp_pips:.1f} pips)")
            logger.info(f"📊 Expected: Max Loss=${expected_loss:.2f}, Target Profit=${expected_profit:.2f}, R:R=1:{dynamic_tp_ratio:.1f}")
            logger.debug(f"SL components - Risk-based: {sl_from_risk:.4f}, ATR: {sl_from_atr:.4f}, "
                        f"Min: {sl_from_min:.4f}, Spread: {sl_from_spread:.4f} -> Selected: {sl_distance:.4f}")
            
            return sl_price, tp_price
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid calculating dynamic SL/TP: {e}")
            default_sl_distance = self.config.DEFAULT_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            default_tp_distance = self.config.DEFAULT_TP_PIPS / self.config.XAUUSD_PIP_VALUE
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except ZeroDivisionError as e:
            logger.error(f"Division by zero calculating dynamic SL/TP: {e}")
            default_sl_distance = self.config.DEFAULT_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            default_tp_distance = self.config.DEFAULT_TP_PIPS / self.config.XAUUSD_PIP_VALUE
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except ArithmeticError as e:
            logger.error(f"Arithmetic error calculating dynamic SL/TP: {e}")
            default_sl_distance = self.config.DEFAULT_SL_PIPS / self.config.XAUUSD_PIP_VALUE
            default_tp_distance = self.config.DEFAULT_TP_PIPS / self.config.XAUUSD_PIP_VALUE
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except AttributeError as e:
            logger.error(f"Config attribute error calculating dynamic SL/TP: {e}")
            default_sl_distance = 50.0 / 10.0
            default_tp_distance = 100.0 / 10.0
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
        except Exception as e:
            logger.error(f"Error tidak terduga calculating dynamic SL/TP: {e}")
            default_sl_distance = 50.0 / 10.0
            default_tp_distance = 100.0 / 10.0
            if signal_type == 'BUY':
                return entry_price - default_sl_distance, entry_price + default_tp_distance
            else:
                return entry_price + default_sl_distance, entry_price - default_tp_distance
    
    def get_daily_stats(self, user_id: int) -> Dict:
        try:
            utc_now = datetime.now(pytz.UTC)
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            jakarta_time = utc_now.astimezone(jakarta_tz)
            
            session = self.db.get_session()
            if session is None:
                logger.error("Failed to get database session for daily stats")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': 'Database session tidak tersedia',
                    'total_trades': 0,
                    'closed_trades': 0,
                    'open_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'breakeven': 0,
                    'win_rate': 0,
                    'total_pl': 0,
                    'total_profit': 0,
                    'total_loss': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 'N/A',
                    'daily_loss_percent': 0,
                    'loss_limit_used': 0,
                    'can_trade': True
                }
            try:
                from bot.database import Trade
                
                today_start = jakarta_time.replace(hour=0, minute=0, second=0, microsecond=0)
                today_start_utc = today_start.astimezone(pytz.UTC)
                
                trades = session.query(Trade).filter(
                    Trade.user_id == user_id,
                    Trade.signal_time >= today_start_utc
                ).all()
                
                total_trades = len(trades)
                closed_trades = [t for t in trades if t.actual_pl is not None]
                
                wins = sum(1 for t in closed_trades if t.actual_pl > 0)
                losses = sum(1 for t in closed_trades if t.actual_pl < 0)
                breakeven = sum(1 for t in closed_trades if t.actual_pl == 0)
                
                total_pl = sum(t.actual_pl for t in closed_trades if t.actual_pl is not None)
                total_profit = sum(t.actual_pl for t in closed_trades if t.actual_pl and t.actual_pl > 0)
                total_loss = sum(t.actual_pl for t in closed_trades if t.actual_pl and t.actual_pl < 0)
                
                win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
                
                avg_win = (total_profit / wins) if wins > 0 else 0
                avg_loss = (abs(total_loss) / losses) if losses > 0 else 0
                profit_factor = (total_profit / abs(total_loss)) if total_loss != 0 else float('inf')
                
                daily_loss_percent = abs(total_pl) / self.config.ACCOUNT_BALANCE * 100 if total_pl < 0 else 0
                loss_limit_used = daily_loss_percent / self.config.DAILY_LOSS_PERCENT * 100 if self.config.DAILY_LOSS_PERCENT > 0 else 0
                
                stats = {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'total_trades': total_trades,
                    'closed_trades': len(closed_trades),
                    'open_trades': total_trades - len(closed_trades),
                    'wins': wins,
                    'losses': losses,
                    'breakeven': breakeven,
                    'win_rate': round(win_rate, 1),
                    'total_pl': round(total_pl, 2),
                    'total_profit': round(total_profit, 2),
                    'total_loss': round(total_loss, 2),
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2),
                    'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
                    'daily_loss_percent': round(daily_loss_percent, 2),
                    'loss_limit_used': round(loss_limit_used, 1),
                    'can_trade': loss_limit_used < 100
                }
                
                logger.debug(f"Daily stats for user {user_id}: {stats}")
                return stats
                
            except SQLAlchemyError as e:
                logger.error(f"Database error getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Database error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except ValueError as e:
                logger.error(f"Nilai tidak valid getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Nilai error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except TypeError as e:
                logger.error(f"Tipe data salah getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Tipe error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except (ZeroDivisionError, ArithmeticError) as e:
                logger.error(f"Arithmetic error getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': f"Kalkulasi error: {str(e)}",
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            except Exception as e:
                logger.error(f"Error tidak terduga getting daily stats: {e}")
                return {
                    'date': jakarta_time.strftime('%Y-%m-%d'),
                    'error': str(e),
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pl': 0,
                    'can_trade': True
                }
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error tidak terduga in get_daily_stats (outer): {e}")
            return {
                'error': str(e),
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pl': 0,
                'can_trade': True
            }
    
    def calculate_pl(self, entry_price: float, exit_price: float, 
                    signal_type: str, lot_size: Optional[float] = None) -> float:
        try:
            if lot_size is None or lot_size <= 0:
                lot_size = self.config.LOT_SIZE
                logger.debug(f"Using default lot size: {lot_size}")
            
            if signal_type == 'BUY':
                price_diff = exit_price - entry_price
            else:
                price_diff = entry_price - exit_price
            
            pips = price_diff * self.config.XAUUSD_PIP_VALUE
            
            pl = pips * lot_size * 10
            
            logger.debug(f"P/L calculation: {signal_type} entry={entry_price:.2f}, "
                        f"exit={exit_price:.2f}, lot={lot_size:.2f}, "
                        f"pips={pips:.1f}, P/L=${pl:.2f}")
            
            return round(pl, 2)
            
        except ValueError as e:
            logger.error(f"Nilai tidak valid calculating P/L: {e}")
            return 0.0
        except ZeroDivisionError as e:
            logger.error(f"Division by zero calculating P/L: {e}")
            return 0.0
        except ArithmeticError as e:
            logger.error(f"Arithmetic error calculating P/L: {e}")
            return 0.0
        except AttributeError as e:
            logger.error(f"Config attribute error calculating P/L: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Error tidak terduga calculating P/L: {e}")
            return 0.0
    
    def calculate_risk_percentage(self, entry_price: float, stop_loss: float, 
                                  signal_type: str, lot_size: Optional[float] = None) -> float:
        """Calculate risk percentage based on account balance and SL distance
        
        Risk % = (SL_pips * lot_size * pip_value) / account_balance * 100
        """
        try:
            if lot_size is None or lot_size <= 0:
                lot_size = self.config.LOT_SIZE
            
            sl_distance = abs(entry_price - stop_loss)
            sl_pips = sl_distance * self.config.XAUUSD_PIP_VALUE
            
            if sl_pips <= 0:
                return 0.0
            
            pip_value_per_lot = 10.0
            risk_amount = sl_pips * lot_size * pip_value_per_lot
            
            account_balance = self.config.ACCOUNT_BALANCE
            if account_balance <= 0:
                return 0.0
            
            risk_percent = (risk_amount / account_balance) * 100
            
            risk_percent = max(0.0, min(risk_percent, 100.0))
            
            logger.debug(f"Risk calc: SL={sl_pips:.1f} pips, Lot={lot_size:.2f}, "
                        f"Risk=${risk_amount:.2f}, Balance=${account_balance:.2f}, "
                        f"Risk%={risk_percent:.2f}%")
            
            return round(risk_percent, 2)
            
        except Exception as e:
            logger.error(f"Error calculating risk percentage: {e}")
            return 0.0
