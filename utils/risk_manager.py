"""
Risk Manager untuk AGI Forex Trading System
===========================================

Modul untuk mengelola dan mengontrol risiko trading.
"""

import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Kelas untuk mengelola dan mengontrol risiko trading.
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi Risk Manager.
        
        Args:
            config: Konfigurasi untuk risk manager
        """
        self.config = config or {}
        
        # Default risk parameters
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2% per trade
        self.max_risk_per_day = self.config.get('max_risk_per_day', 0.06)  # 6% per day
        self.max_risk_per_week = self.config.get('max_risk_per_week', 0.15)  # 15% per week
        self.max_open_positions = self.config.get('max_open_positions', 5)
        self.max_correlation = self.config.get('max_correlation', 0.7)
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 0.15)  # 15%
        
        # Tracking variables
        self.open_positions = []
        self.daily_risk_used = 0.0
        self.weekly_risk_used = 0.0
        self.current_drawdown = 0.0
        self.last_reset = datetime.now()
        self.daily_trades = []
        self.weekly_trades = []
        
        logger.info("Risk Manager initialized with config: %s", self.config)
    
    def can_open_position(self, symbol, position_size, account_balance, current_positions=None):
        """
        Memeriksa apakah posisi baru dapat dibuka berdasarkan aturan manajemen risiko.
        
        Args:
            symbol: Simbol trading
            position_size: Ukuran posisi yang diusulkan
            account_balance: Saldo akun saat ini
            current_positions: Daftar posisi yang saat ini terbuka (opsional)
            
        Returns:
            tuple: (bool, str) - Apakah posisi dapat dibuka dan alasannya
        """
        # Update tracking data
        self._update_tracking_data()
        
        # Check if we're already at max open positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, "Maximum number of open positions reached"
        
        # Calculate risk for this trade
        risk_amount = position_size / account_balance
        
        # Check if this trade exceeds per-trade risk limit
        if risk_amount > self.max_risk_per_trade:
            return False, f"Trade risk ({risk_amount:.2%}) exceeds maximum per-trade risk ({self.max_risk_per_trade:.2%})"
        
        # Check if this trade would exceed daily risk limit
        if self.daily_risk_used + risk_amount > self.max_risk_per_day:
            return False, f"Daily risk limit reached ({self.daily_risk_used:.2%} used, {self.max_risk_per_day:.2%} max)"
        
        # Check if this trade would exceed weekly risk limit
        if self.weekly_risk_used + risk_amount > self.max_risk_per_week:
            return False, f"Weekly risk limit reached ({self.weekly_risk_used:.2%} used, {self.max_risk_per_week:.2%} max)"
        
        # Check correlation with existing positions
        if current_positions and len(current_positions) > 0:
            correlation_too_high = self._check_correlation(symbol, current_positions)
            if correlation_too_high:
                return False, f"Correlation with existing positions too high (> {self.max_correlation})"
        
        # Check if we're in a significant drawdown
        if self.current_drawdown > self.max_drawdown_threshold:
            return False, f"Current drawdown ({self.current_drawdown:.2%}) exceeds threshold ({self.max_drawdown_threshold:.2%})"
        
        # All checks passed
        return True, "Position can be opened"
    
    def _check_correlation(self, symbol, current_positions):
        """
        Memeriksa korelasi antara simbol yang diusulkan dan posisi yang ada.
        
        Args:
            symbol: Simbol yang akan diperiksa
            current_positions: Daftar posisi saat ini
            
        Returns:
            bool: True jika korelasi terlalu tinggi
        """
        # In a real system, this would calculate actual price correlations
        # For this example, we'll use a simplified approach
        
        # Check if we're already trading this symbol
        if symbol in [pos['symbol'] for pos in current_positions]:
            return True
        
        # Check for currency pair correlations
        base_currency = symbol[:3]
        quote_currency = symbol[3:]
        
        for position in current_positions:
            pos_base = position['symbol'][:3]
            pos_quote = position['symbol'][3:]
            
            # If trading same base or quote currency, consider it correlated
            if base_currency == pos_base or quote_currency == pos_quote:
                # In a real system, we'd calculate the actual correlation coefficient
                # For now, we'll assume 80% correlation for same currencies
                if 0.8 > self.max_correlation:
                    return True
        
        return False
    
    def _update_tracking_data(self):
        """
        Memperbarui data pelacakan risiko harian dan mingguan.
        """
        now = datetime.now()
        
        # Reset daily risk if it's a new day
        if now.date() > self.last_reset.date():
            logger.info("Resetting daily risk tracking")
            self.daily_risk_used = 0.0
            self.daily_trades = []
        
        # Reset weekly risk if it's a new week
        if now.isocalendar()[1] != self.last_reset.isocalendar()[1]:
            logger.info("Resetting weekly risk tracking")
            self.weekly_risk_used = 0.0
            self.weekly_trades = []
        
        self.last_reset = now
    
    def register_trade(self, trade_data):
        """
        Mendaftarkan trade baru dan memperbarui metrik risiko.
        
        Args:
            trade_data: Data trade yang akan didaftarkan
        """
        # Add to tracking lists
        self.daily_trades.append(trade_data)
        self.weekly_trades.append(trade_data)
        
        # Update risk used
        risk = trade_data.get('risk', 0.0)
        self.daily_risk_used += risk
        self.weekly_risk_used += risk
        
        # If it's an open position, add to open positions list
        if trade_data.get('status') == 'open':
            self.open_positions.append(trade_data)
        
        logger.info(f"Trade registered: {trade_data}")
        logger.info(f"Daily risk used: {self.daily_risk_used:.2%}, Weekly risk used: {self.weekly_risk_used:.2%}")
    
    def update_position(self, position_id, update_data):
        """
        Memperbarui posisi yang ada.
        
        Args:
            position_id: ID posisi yang akan diperbarui
            update_data: Data pembaruan
        """
        for i, position in enumerate(self.open_positions):
            if position.get('id') == position_id:
                # Update position data
                self.open_positions[i].update(update_data)
                
                # If position is now closed, remove from open positions
                if update_data.get('status') == 'closed':
                    closed_position = self.open_positions.pop(i)
                    logger.info(f"Position closed: {closed_position}")
                
                break
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, account_balance):
        """
        Menghitung ukuran posisi yang optimal berdasarkan manajemen risiko.
        
        Args:
            symbol: Simbol trading
            entry_price: Harga masuk
            stop_loss: Harga stop loss
            account_balance: Saldo akun
            
        Returns:
            float: Ukuran posisi yang direkomendasikan
        """
        # Calculate risk per pip
        pip_risk = abs(entry_price - stop_loss)
        
        if pip_risk == 0:
            logger.warning("Stop loss is equal to entry price, cannot calculate position size")
            return 0
        
        # Calculate maximum risk amount in account currency
        max_risk_amount = account_balance * self.max_risk_per_trade
        
        # Calculate position size
        position_size = max_risk_amount / pip_risk
        
        # Apply any additional scaling or constraints
        # For example, ensure position size is within allowed limits
        
        logger.info(f"Calculated position size for {symbol}: {position_size}")
        return position_size
    
    def update_drawdown(self, current_equity, peak_equity):
        """
        Memperbarui pelacakan drawdown saat ini.
        
        Args:
            current_equity: Ekuitas akun saat ini
            peak_equity: Ekuitas puncak historis
        """
        if peak_equity <= 0:
            self.current_drawdown = 0
            return
        
        self.current_drawdown = (peak_equity - current_equity) / peak_equity
        logger.info(f"Current drawdown updated: {self.current_drawdown:.2%}")
    
    def get_risk_report(self):
        """
        Membuat laporan risiko saat ini.
        
        Returns:
            dict: Laporan risiko
        """
        return {
            'open_positions': len(self.open_positions),
            'daily_risk_used': self.daily_risk_used,
            'weekly_risk_used': self.weekly_risk_used,
            'current_drawdown': self.current_drawdown,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_risk_per_day': self.max_risk_per_day,
            'max_risk_per_week': self.max_risk_per_week,
            'max_drawdown_threshold': self.max_drawdown_threshold
        }