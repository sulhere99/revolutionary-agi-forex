"""
Database Manager untuk AGI Forex Trading System
==============================================

Modul untuk mengelola koneksi dan operasi database.
"""

import logging
import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, MetaData, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func

logger = logging.getLogger(__name__)
Base = declarative_base()

class Trade(Base):
    """Model untuk tabel trades."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # 'buy' or 'sell'
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)
    position_size = Column(Float, nullable=False)
    profit = Column(Float)
    pips = Column(Float)
    status = Column(String(20), default='open')  # 'open', 'closed', 'cancelled'
    stop_loss = Column(Float)
    take_profit = Column(Float)
    strategy = Column(String(50))
    notes = Column(Text)
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol='{self.symbol}', direction='{self.direction}', status='{self.status}')>"

class Signal(Base):
    """Model untuk tabel signals."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # 'buy' or 'sell'
    price = Column(Float, nullable=False)
    time = Column(DateTime, default=datetime.utcnow)
    strength = Column(Float)  # Signal strength/confidence (0-1)
    strategy = Column(String(50))
    timeframe = Column(String(20))
    indicators = Column(Text)  # JSON string of indicators that generated the signal
    executed = Column(Boolean, default=False)
    trade_id = Column(Integer)  # Reference to the trade if executed
    
    def __repr__(self):
        return f"<Signal(id={self.id}, symbol='{self.symbol}', direction='{self.direction}', executed={self.executed})>"

class MarketData(Base):
    """Model untuk tabel market_data."""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(20), nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Float)
    timestamp = Column(DateTime, nullable=False)
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}')>"

class NewsEvent(Base):
    """Model untuk tabel news_events."""
    __tablename__ = 'news_events'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    content = Column(Text)
    source = Column(String(100))
    impact = Column(String(20))  # 'high', 'medium', 'low'
    related_currencies = Column(String(100))
    event_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<NewsEvent(id={self.id}, title='{self.title}', impact='{self.impact}')>"

class SystemLog(Base):
    """Model untuk tabel system_logs."""
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    level = Column(String(20), nullable=False)
    module = Column(String(50))
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemLog(id={self.id}, level='{self.level}', module='{self.module}')>"

class DatabaseManager:
    """
    Kelas untuk mengelola koneksi dan operasi database.
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi Database Manager.
        
        Args:
            config: Konfigurasi untuk database manager
        """
        self.config = config or {}
        
        # Default to SQLite in-memory database if no config provided
        db_url = self.config.get('db_url', 'sqlite:///revolutionary_agi_forex.db')
        
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        
        logger.info(f"Database initialized with URL: {db_url}")
    
    async def initialize(self):
        """
        Metode inisialisasi asinkron untuk kompatibilitas dengan sistem.
        Karena inisialisasi utama sudah dilakukan di __init__, metode ini
        hanya untuk kompatibilitas dengan sistem yang mengharapkan metode initialize().
        
        Returns:
            bool: True jika berhasil
        """
        logger.info("Database manager initialize method called")
        return True
    
    def add_trade(self, trade_data):
        """
        Menambahkan trade baru ke database.
        
        Args:
            trade_data: Data trade yang akan ditambahkan
            
        Returns:
            Trade: Objek trade yang ditambahkan
        """
        trade = Trade(**trade_data)
        self.session.add(trade)
        self.session.commit()
        logger.info(f"Trade added: {trade}")
        return trade
    
    def update_trade(self, trade_id, update_data):
        """
        Memperbarui trade yang ada.
        
        Args:
            trade_id: ID trade yang akan diperbarui
            update_data: Data pembaruan
            
        Returns:
            Trade: Objek trade yang diperbarui
        """
        trade = self.session.query(Trade).filter_by(id=trade_id).first()
        if trade:
            for key, value in update_data.items():
                setattr(trade, key, value)
            self.session.commit()
            logger.info(f"Trade updated: {trade}")
        return trade
    
    def get_trade(self, trade_id):
        """
        Mendapatkan trade berdasarkan ID.
        
        Args:
            trade_id: ID trade yang akan diambil
            
        Returns:
            Trade: Objek trade
        """
        return self.session.query(Trade).filter_by(id=trade_id).first()
    
    def get_open_trades(self):
        """
        Mendapatkan semua trade yang masih terbuka.
        
        Returns:
            list: Daftar trade yang terbuka
        """
        return self.session.query(Trade).filter_by(status='open').all()
    
    def add_signal(self, signal_data):
        """
        Menambahkan sinyal trading baru.
        
        Args:
            signal_data: Data sinyal yang akan ditambahkan
            
        Returns:
            Signal: Objek sinyal yang ditambahkan
        """
        # Convert indicators dict to JSON string if provided
        if 'indicators' in signal_data and isinstance(signal_data['indicators'], dict):
            signal_data['indicators'] = json.dumps(signal_data['indicators'])
            
        signal = Signal(**signal_data)
        self.session.add(signal)
        self.session.commit()
        logger.info(f"Signal added: {signal}")
        return signal
    
    def get_recent_signals(self, symbol=None, limit=10):
        """
        Mendapatkan sinyal terbaru.
        
        Args:
            symbol: Filter berdasarkan simbol (opsional)
            limit: Jumlah maksimum sinyal yang akan diambil
            
        Returns:
            list: Daftar sinyal
        """
        query = self.session.query(Signal).order_by(Signal.time.desc())
        if symbol:
            query = query.filter_by(symbol=symbol)
        return query.limit(limit).all()
    
    def add_market_data(self, data):
        """
        Menambahkan data pasar baru.
        
        Args:
            data: Data pasar yang akan ditambahkan
            
        Returns:
            MarketData: Objek data pasar yang ditambahkan
        """
        market_data = MarketData(**data)
        self.session.add(market_data)
        self.session.commit()
        return market_data
    
    def get_market_data(self, symbol, timeframe, start_time=None, end_time=None, limit=1000):
        """
        Mendapatkan data pasar.
        
        Args:
            symbol: Simbol trading
            timeframe: Timeframe data
            start_time: Waktu mulai (opsional)
            end_time: Waktu akhir (opsional)
            limit: Jumlah maksimum data yang akan diambil
            
        Returns:
            list: Daftar data pasar
        """
        query = self.session.query(MarketData).filter_by(
            symbol=symbol, timeframe=timeframe
        ).order_by(MarketData.timestamp.desc())
        
        if start_time:
            query = query.filter(MarketData.timestamp >= start_time)
        if end_time:
            query = query.filter(MarketData.timestamp <= end_time)
            
        return query.limit(limit).all()
    
    def add_news_event(self, news_data):
        """
        Menambahkan event berita baru.
        
        Args:
            news_data: Data berita yang akan ditambahkan
            
        Returns:
            NewsEvent: Objek berita yang ditambahkan
        """
        news = NewsEvent(**news_data)
        self.session.add(news)
        self.session.commit()
        logger.info(f"News event added: {news}")
        return news
    
    def get_upcoming_news(self, hours_ahead=24):
        """
        Mendapatkan berita yang akan datang.
        
        Args:
            hours_ahead: Jumlah jam ke depan untuk melihat berita
            
        Returns:
            list: Daftar berita yang akan datang
        """
        now = datetime.utcnow()
        future = now + timedelta(hours=hours_ahead)
        return self.session.query(NewsEvent).filter(
            NewsEvent.event_time >= now,
            NewsEvent.event_time <= future
        ).order_by(NewsEvent.event_time).all()
    
    def log_system_event(self, level, message, module=None):
        """
        Mencatat event sistem ke database.
        
        Args:
            level: Level log ('info', 'warning', 'error', 'critical')
            message: Pesan log
            module: Nama modul (opsional)
            
        Returns:
            SystemLog: Objek log yang ditambahkan
        """
        log = SystemLog(level=level, message=message, module=module)
        self.session.add(log)
        self.session.commit()
        return log
    
    def get_performance_stats(self, days=30):
        """
        Mendapatkan statistik kinerja trading.
        
        Args:
            days: Jumlah hari ke belakang untuk analisis
            
        Returns:
            dict: Statistik kinerja
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get closed trades in the period
        trades = self.session.query(Trade).filter(
            Trade.exit_time >= start_date,
            Trade.status == 'closed'
        ).all()
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'average_profit': 0,
                'average_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0
            }
        
        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.profit > 0])
        losing_trades = len([t for t in trades if t.profit <= 0])
        
        total_profit = sum([t.profit for t in trades if t.profit > 0])
        total_loss = abs(sum([t.profit for t in trades if t.profit < 0]))
        
        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_profit': total_profit - total_loss,
            'average_profit': total_profit / winning_trades if winning_trades > 0 else 0,
            'average_loss': total_loss / losing_trades if losing_trades > 0 else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
            # Simplified max drawdown calculation
            'max_drawdown': self._calculate_max_drawdown(trades)
        }
        
        return stats
    
    def _calculate_max_drawdown(self, trades):
        """
        Menghitung maximum drawdown dari daftar trade.
        
        Args:
            trades: Daftar objek Trade
            
        Returns:
            float: Maximum drawdown
        """
        if not trades:
            return 0
            
        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time or datetime.utcnow())
        
        # Calculate cumulative profit
        cumulative = 0
        peak = 0
        max_drawdown = 0
        
        for trade in sorted_trades:
            if trade.profit is not None:
                cumulative += trade.profit
                peak = max(peak, cumulative)
                drawdown = peak - cumulative
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def close(self):
        """
        Menutup koneksi database.
        """
        self.session.close()
        logger.info("Database connection closed")