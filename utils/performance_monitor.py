"""
Performance Monitor untuk AGI Forex Trading System
=================================================

Modul untuk memantau dan menganalisis kinerja sistem trading.
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """
    Kelas untuk memantau dan menganalisis kinerja sistem trading.
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi Performance Monitor.
        
        Args:
            config: Konfigurasi untuk performance monitor
        """
        self.config = config or {}
        self.trades_history = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_profit': 0.0,
            'average_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'expectancy': 0.0,
        }
        self.start_time = datetime.now()
        logger.info("Performance Monitor initialized")
    
    def add_trade(self, trade_data):
        """
        Menambahkan trade ke history dan memperbarui metrik.
        
        Args:
            trade_data: Data trade yang akan ditambahkan
        """
        self.trades_history.append(trade_data)
        self._update_metrics()
        logger.info(f"Trade added: {trade_data}")
    
    def _update_metrics(self):
        """
        Memperbarui metrik kinerja berdasarkan history trade.
        """
        if not self.trades_history:
            return
        
        # Konversi ke DataFrame untuk analisis yang lebih mudah
        df = pd.DataFrame(self.trades_history)
        
        # Metrik dasar
        self.metrics['total_trades'] = len(df)
        self.metrics['winning_trades'] = len(df[df['profit'] > 0])
        self.metrics['losing_trades'] = len(df[df['profit'] <= 0])
        
        # Win rate
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # Profit metrics
        if self.metrics['winning_trades'] > 0:
            self.metrics['average_profit'] = df[df['profit'] > 0]['profit'].mean()
        
        if self.metrics['losing_trades'] > 0:
            self.metrics['average_loss'] = abs(df[df['profit'] <= 0]['profit'].mean())
        
        # Profit factor
        total_profit = df[df['profit'] > 0]['profit'].sum()
        total_loss = abs(df[df['profit'] <= 0]['profit'].sum())
        
        if total_loss > 0:
            self.metrics['profit_factor'] = total_profit / total_loss
        
        # Expectancy
        if self.metrics['total_trades'] > 0:
            self.metrics['expectancy'] = (
                self.metrics['win_rate'] * self.metrics['average_profit'] - 
                (1 - self.metrics['win_rate']) * self.metrics['average_loss']
            )
        
        # Advanced metrics (simplified calculations)
        # In a real system, these would be more sophisticated
        self.metrics['max_drawdown'] = self._calculate_max_drawdown(df)
        self.metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(df)
        self.metrics['sortino_ratio'] = self._calculate_sortino_ratio(df)
        
        if self.metrics['max_drawdown'] > 0:
            self.metrics['calmar_ratio'] = total_profit / self.metrics['max_drawdown']
        
        logger.debug(f"Metrics updated: {self.metrics}")
    
    def _calculate_max_drawdown(self, df):
        """
        Menghitung maximum drawdown dari history trade.
        
        Args:
            df: DataFrame dengan history trade
            
        Returns:
            float: Maximum drawdown
        """
        if df.empty:
            return 0.0
        
        # Simplified calculation - in a real system this would be more sophisticated
        cumulative = df['profit'].cumsum()
        max_so_far = cumulative.cummax()
        drawdown = max_so_far - cumulative
        return drawdown.max() if not drawdown.empty else 0.0
    
    def _calculate_sharpe_ratio(self, df, risk_free_rate=0.02, periods=252):
        """
        Menghitung Sharpe Ratio.
        
        Args:
            df: DataFrame dengan history trade
            risk_free_rate: Risk-free rate tahunan
            periods: Jumlah periode trading dalam setahun
            
        Returns:
            float: Sharpe Ratio
        """
        if df.empty or len(df) < 2:
            return 0.0
        
        # Simplified calculation
        returns = df['profit'].pct_change().dropna()
        excess_returns = returns - (risk_free_rate / periods)
        return excess_returns.mean() / excess_returns.std() * np.sqrt(periods) if excess_returns.std() > 0 else 0.0
    
    def _calculate_sortino_ratio(self, df, risk_free_rate=0.02, periods=252):
        """
        Menghitung Sortino Ratio.
        
        Args:
            df: DataFrame dengan history trade
            risk_free_rate: Risk-free rate tahunan
            periods: Jumlah periode trading dalam setahun
            
        Returns:
            float: Sortino Ratio
        """
        if df.empty or len(df) < 2:
            return 0.0
        
        # Simplified calculation
        returns = df['profit'].pct_change().dropna()
        excess_returns = returns - (risk_free_rate / periods)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods)
        
        return excess_returns.mean() * periods / downside_deviation if downside_deviation > 0 else 0.0
    
    def get_metrics(self):
        """
        Mendapatkan metrik kinerja terkini.
        
        Returns:
            dict: Metrik kinerja
        """
        return self.metrics
    
    def get_summary_report(self):
        """
        Membuat laporan ringkasan kinerja.
        
        Returns:
            str: Laporan ringkasan kinerja
        """
        runtime = datetime.now() - self.start_time
        
        report = [
            "=== PERFORMANCE SUMMARY ===",
            f"Runtime: {runtime}",
            f"Total Trades: {self.metrics['total_trades']}",
            f"Win Rate: {self.metrics['win_rate']:.2%}",
            f"Profit Factor: {self.metrics['profit_factor']:.2f}",
            f"Expectancy: {self.metrics['expectancy']:.4f}",
            f"Max Drawdown: {self.metrics['max_drawdown']:.2f}",
            f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}",
            f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}",
            f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}",
            "=========================="
        ]
        
        return "\n".join(report)
    
    def reset(self):
        """
        Reset performance monitor.
        """
        self.trades_history = []
        self.metrics = {key: 0.0 for key in self.metrics}
        self.start_time = datetime.now()
        logger.info("Performance Monitor reset")