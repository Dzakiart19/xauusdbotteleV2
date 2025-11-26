"""
Message Templates untuk Telegram
Format pesan yang konsisten dan rapi
"""
from typing import Optional
from datetime import datetime
import pytz

class MessageFormatter:
    """Helper untuk format pesan Telegram yang rapi"""
    
    @staticmethod
    def escape_markdown(text: str) -> str:
        """Escape karakter special untuk Markdown"""
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        return text
    
    @staticmethod
    def progress_bar(current: float, target: float, total_length: int = 10) -> str:
        """Buat progress bar visual"""
        if target == 0:
            return "▱" * total_length
        
        percentage = min(abs(current / target), 1.0)
        filled = int(percentage * total_length)
        empty = total_length - filled
        
        return "▰" * filled + "▱" * empty
    
    @staticmethod
    def signal_alert(signal: dict, signal_source: str = 'auto', config=None) -> str:
        """Format pesan alert sinyal trading - Format Profesional"""
        signal_type = signal['signal']
        direction_icon = "🟢" if signal_type == 'BUY' else "🔴"
        
        entry = signal['entry_price']
        sl = signal['stop_loss']
        tp = signal['take_profit']
        
        sl_pips = signal.get('sl_pips', 0)
        tp_pips = signal.get('tp_pips', 0)
        rr_ratio = signal.get('rr_ratio', 0)
        
        lot_size = signal.get('lot_size', 0.01)
        risk_percent = signal.get('risk_percent', 1.0)
        risk_amount = signal.get('risk_amount', 0)
        account_balance = signal.get('account_balance', 0)
        
        if config and account_balance == 0:
            account_balance = getattr(config, 'ACCOUNT_BALANCE', 0)
        if config and risk_percent == 1.0:
            risk_percent = getattr(config, 'RISK_PER_TRADE_PERCENT', 1.0)
        if risk_amount == 0 and account_balance > 0:
            risk_amount = account_balance * risk_percent / 100
        
        trend_status = signal.get('trend_status', signal.get('trend_description', 'N/A'))
        momentum_status = signal.get('momentum_status', 'N/A')
        volume_status = signal.get('volume_status', 'N/A')
        vwap_status = signal.get('vwap_status', 'N/A')
        
        timeframe = signal.get('timeframe', 'M1')
        timestamp = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')
        
        msg = (
            f"{direction_icon} *SIGNAL {signal_type} - XAUUSD*\n"
            f"{'━' * 22}\n"
            f"📊 Entry: `${entry:.2f}`\n"
            f"🛡️ Stop Loss: `${sl:.2f}` ({sl_pips:.1f} pips)\n"
            f"🎯 Take Profit: `${tp:.2f}` ({tp_pips:.1f} pips)\n"
            f"📈 Risk:Reward = 1:{rr_ratio:.1f}\n\n"
            f"💰 *Risk Management:*\n"
            f"• Lot Size: {lot_size:.2f}\n"
            f"• Risk: {risk_percent:.1f}% (${risk_amount:.2f})\n"
            f"• Modal: ${account_balance:.2f}\n\n"
            f"📋 *Konfirmasi:*\n"
            f"• Trend: {trend_status}\n"
            f"• Momentum: {momentum_status}\n"
            f"• Volume: {volume_status}\n"
            f"• VWAP: {vwap_status}\n\n"
            f"⏰ Waktu: {timestamp}\n"
            f"📊 Timeframe: {timeframe}\n"
            f"{'━' * 22}"
        )
        
        return msg
    
    @staticmethod
    def position_update(position_data: dict) -> str:
        """Format update posisi real-time"""
        signal_type = position_data['signal_type']
        direction_icon = "🟢" if signal_type == 'BUY' else "🔴"
        
        entry = position_data['entry_price']
        current = position_data['current_price']
        sl = position_data['stop_loss']
        tp = position_data['take_profit']
        pl = position_data['unrealized_pl']
        
        price_change = current - entry
        price_change_pct = (price_change / entry) * 100
        
        if signal_type == 'BUY':
            tp_distance_total = tp - entry
            tp_distance_current = current - entry
            sl_distance_total = entry - sl
            sl_distance_current = entry - current
        else:
            tp_distance_total = entry - tp
            tp_distance_current = entry - current
            sl_distance_total = sl - entry
            sl_distance_current = current - entry
        
        tp_progress = max(0, min(100, (tp_distance_current / tp_distance_total * 100) if tp_distance_total > 0 else 0))
        sl_progress = max(0, min(100, (sl_distance_current / sl_distance_total * 100) if sl_distance_total > 0 else 0))
        
        tp_bar = MessageFormatter.progress_bar(tp_distance_current, tp_distance_total, 10)
        
        pl_icon = "💰" if pl >= 0 else "📉"
        pl_text = f"+${pl:.2f}" if pl >= 0 else f"-${abs(pl):.2f}"
        
        msg = (
            f"{direction_icon} *POSISI {signal_type} AKTIF*\n"
            f"{'━' * 32}\n\n"
            f"📍 *Entry:* `${entry:.2f}`\n"
            f"📊 *Current:* `${current:.2f}` ({price_change_pct:+.3f}%)\n"
            f"{pl_icon} *P/L:* `{pl_text}`\n\n"
            f"🎯 *Progress ke TP:*\n"
            f"{tp_bar} {tp_progress:.1f}%\n"
            f"Target: `${tp:.2f}`\n\n"
            f"🛡️ *Stop Loss:* `${sl:.2f}`\n"
            f"⚠️ Risk: {sl_progress:.1f}%\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
        
        return msg
    
    @staticmethod
    def trade_exit(exit_data: dict, pip_value: float = 10.0) -> str:
        """Format pesan trade exit - Format Profesional"""
        result = exit_data['result']
        signal_type = exit_data['signal_type']
        entry = exit_data['entry_price']
        exit_price = exit_data['exit_price']
        pl = exit_data['actual_pl']
        reason = exit_data.get('reason', 'CLOSED')
        duration = exit_data.get('duration', 'N/A')
        
        result_icon = "✅" if result == 'WIN' else "❌"
        result_text = "TP HIT" if reason == 'TP_HIT' else ("SL HIT" if reason in ['SL_HIT', 'DYNAMIC_SL_HIT'] else result)
        
        price_diff = abs(exit_price - entry)
        pl_pips = price_diff * pip_value
        
        pl_emoji = "💰" if pl >= 0 else "📉"
        pl_text = f"+${pl:.2f}" if pl >= 0 else f"-${abs(pl):.2f}"
        
        if isinstance(duration, (int, float)):
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            if hours > 0:
                duration_str = f"{hours}h {minutes}m"
            else:
                duration_str = f"{minutes}m"
        else:
            duration_str = str(duration) if duration else "N/A"
        
        msg = (
            f"{result_icon} *TRADE CLOSED - {result_text}*\n"
            f"{'━' * 22}\n"
            f"📊 Entry: `${entry:.2f}`\n"
            f"📊 Exit: `${exit_price:.2f}`\n"
            f"💰 P/L: {pl_emoji} {pl_text} ({pl_pips:.1f} pips)\n"
            f"⏱️ Duration: {duration_str}\n"
            f"{'━' * 22}"
        )
        
        return msg
    
    @staticmethod
    def waiting_for_signal(signal_source: str = 'auto') -> str:
        """Format pesan menunggu sinyal"""
        icon = "🤖" if signal_source == 'auto' else "👤"
        mode = "otomatis" if signal_source == 'auto' else "manual"
        
        return (
            f"{icon} *Monitoring Aktif*\n"
            f"{'━' * 32}\n\n"
            f"⏳ Menunggu sinyal {mode}...\n"
            f"📊 Menganalisis market XAUUSD\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
    
    @staticmethod
    def session_blocked(active_source: str, requested_source: str) -> str:
        """Format pesan ketika sinyal diblok karena ada sesi aktif"""
        active_icon = "🤖" if active_source == 'auto' else "👤"
        requested_icon = "🤖" if requested_source == 'auto' else "👤"
        
        return (
            f"⛔ *Sinyal Diblok*\n"
            f"{'━' * 32}\n\n"
            f"{active_icon} Ada sinyal *{active_source.upper()}* yang masih aktif!\n\n"
            f"{requested_icon} Tidak bisa buat sinyal *{requested_source.upper()}* sekarang.\n\n"
            f"⏳ Tunggu sampai posisi selesai dulu\n"
            f"(TP/SL tercapai)\n\n"
            f"💡 Cek status: /status"
        )
    
    @staticmethod
    def no_active_position() -> str:
        """Format pesan tidak ada posisi aktif"""
        return (
            f"ℹ️ *Status Posisi*\n"
            f"{'━' * 32}\n\n"
            f"📭 Tidak ada posisi aktif\n\n"
            f"💡 Gunakan:\n"
            f"  • /monitor - Monitoring otomatis\n"
            f"  • /getsignal - Sinyal manual\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
    
    @staticmethod
    def statistics_summary(stats: dict) -> str:
        """Format statistik trading - untuk /performa"""
        total_trades = stats.get('total_trades', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        total_profit = stats.get('total_profit', 0)
        win_rate = stats.get('win_rate', 0)
        
        win_rate_emoji = "🔥" if win_rate >= 70 else "💪" if win_rate >= 50 else "📊"
        pl_emoji = "💰" if total_profit >= 0 else "📉"
        pl_text = f"+${total_profit:.2f}" if total_profit >= 0 else f"-${abs(total_profit):.2f}"
        
        return (
            f"📊 *Statistik Trading*\n"
            f"{'━' * 22}\n\n"
            f"📈 *Total Trades:* {total_trades}\n"
            f"✅ *Wins:* {wins}\n"
            f"❌ *Losses:* {losses}\n"
            f"{win_rate_emoji} *Win Rate:* {win_rate:.1f}%\n\n"
            f"{pl_emoji} *Total P/L:* `{pl_text}`\n\n"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M WIB')}"
        )
    
    @staticmethod
    def daily_stats(stats: dict) -> str:
        """Format statistik harian - untuk /stats command"""
        total_trades = stats.get('total_trades', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        win_rate = stats.get('win_rate', 0)
        net_pl = stats.get('total_pl', 0)
        profit_factor = stats.get('profit_factor', 'N/A')
        
        avg_win = stats.get('avg_win', 0)
        avg_loss = stats.get('avg_loss', 0)
        avg_rr = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        pl_emoji = "💰" if net_pl >= 0 else "📉"
        pl_text = f"+${net_pl:.2f}" if net_pl >= 0 else f"-${abs(net_pl):.2f}"
        
        profit_factor_str = f"{profit_factor:.2f}" if isinstance(profit_factor, (int, float)) else str(profit_factor)
        avg_rr_str = f"1:{avg_rr:.1f}" if avg_rr > 0 else "N/A"
        
        date_str = stats.get('date', datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d'))
        
        return (
            f"📊 *STATISTIK HARI INI*\n"
            f"{'━' * 22}\n"
            f"📈 Total Trade: {total_trades}\n"
            f"✅ Win: {wins} | ❌ Loss: {losses}\n"
            f"📊 Win Rate: {win_rate:.1f}%\n"
            f"💰 Net P/L: {pl_emoji} {pl_text}\n"
            f"📈 Profit Factor: {profit_factor_str}\n"
            f"🎯 Avg RR: {avg_rr_str}\n"
            f"{'━' * 22}\n\n"
            f"📅 Tanggal: {date_str}"
        )
    
    @staticmethod
    def error_message(error_text: str, context: str = "") -> str:
        """Format pesan error"""
        return (
            f"⚠️ *Error*\n"
            f"{'━' * 32}\n\n"
            f"{error_text}\n\n"
            f"{f'Context: {context}' if context else ''}"
            f"⏰ {datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')}"
        )
