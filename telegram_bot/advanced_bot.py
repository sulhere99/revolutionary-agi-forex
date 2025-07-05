"""
Advanced Telegram Bot untuk AGI Forex Trading System
===================================================

Bot Telegram yang sangat canggih dengan fitur:
- Real-time trading signals
- Interactive charts dan analysis
- Portfolio management
- Risk management alerts
- Performance tracking
- Custom notifications
- Voice messages
- Multi-language support
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import io
import base64
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, 
    ReplyKeyboardMarkup, KeyboardButton, ParseMode,
    InputMediaPhoto, InputMediaDocument, CallbackQuery
)
from telegram.ext import (
    Updater, CommandHandler, MessageHandler, 
    CallbackQueryHandler, ConversationHandler, Filters
)

import redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import our AGI components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agi_brain import AGIBrain, MarketSignal
from data.market_data_collector import MarketDataCollector, RealTimeQuote, NewsItem

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Database models
Base = declarative_base()

class TelegramUser(Base):
    """Model untuk Telegram users"""
    __tablename__ = 'telegram_users'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, unique=True, nullable=False)
    username = Column(String(100))
    first_name = Column(String(100))
    last_name = Column(String(100))
    language_code = Column(String(10), default='en')
    is_premium = Column(Boolean, default=False)
    subscription_type = Column(String(20), default='free')  # free, basic, premium, vip
    joined_date = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, default=datetime.now)
    notification_settings = Column(Text)  # JSON string
    risk_tolerance = Column(String(10), default='medium')  # low, medium, high
    preferred_pairs = Column(Text)  # JSON string of preferred currency pairs
    timezone = Column(String(50), default='UTC')

class UserPortfolio(Base):
    """Model untuk user portfolio tracking"""
    __tablename__ = 'user_portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    pair = Column(String(10), nullable=False)
    position_type = Column(String(10), nullable=False)  # BUY, SELL
    entry_price = Column(Float, nullable=False)
    position_size = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    entry_time = Column(DateTime, default=datetime.now)
    exit_time = Column(DateTime)
    exit_price = Column(Float)
    profit_loss = Column(Float, default=0)
    status = Column(String(10), default='open')  # open, closed
    signal_id = Column(String(100))

class SignalPerformance(Base):
    """Model untuk tracking signal performance"""
    __tablename__ = 'signal_performance'
    
    id = Column(Integer, primary_key=True)
    signal_id = Column(String(100), unique=True, nullable=False)
    pair = Column(String(10), nullable=False)
    action = Column(String(10), nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    signal_time = Column(DateTime, nullable=False)
    close_time = Column(DateTime)
    close_price = Column(Float)
    result = Column(String(10))  # win, loss, breakeven
    pips_gained = Column(Float, default=0)
    roi_percentage = Column(Float, default=0)
    confidence_score = Column(Float, default=0)

@dataclass
class NotificationSettings:
    """User notification settings"""
    signals_enabled: bool = True
    news_alerts: bool = True
    performance_updates: bool = True
    risk_alerts: bool = True
    market_analysis: bool = True
    voice_messages: bool = False
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "08:00"
    min_signal_confidence: float = 0.7
    preferred_timeframes: List[str] = None

class AdvancedTelegramBot:
    """Advanced Telegram Bot untuk AGI Trading System"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_token = config['telegram_bot_token']
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 1)
        )
        
        # Database setup
        self.db_engine = create_engine(config.get('database_url', 'sqlite:///telegram_bot.db'))
        Base.metadata.create_all(self.db_engine)
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        
        # AGI Brain integration
        self.agi_brain = None
        self.data_collector = None
        
        # Bot application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Conversation states
        self.SELECTING_PAIRS = 1
        self.SETTING_RISK = 2
        self.CONFIGURING_NOTIFICATIONS = 3
        
        # Supported languages
        self.languages = {
            'en': 'English',
            'id': 'Bahasa Indonesia',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文'
        }
        
        # Load translations
        self.translations = self._load_translations()
        
        # Setup handlers
        self._setup_handlers()
        
        # Active users tracking
        self.active_users = set()
        
        logger.info("Advanced Telegram Bot initialized")
    
    def set_agi_brain(self, agi_brain: AGIBrain):
        """Set AGI Brain instance"""
        self.agi_brain = agi_brain
    
    def set_data_collector(self, data_collector: MarketDataCollector):
        """Set Data Collector instance"""
        self.data_collector = data_collector
        # Add callbacks untuk real-time updates
        data_collector.add_data_callback(self._handle_new_market_data)
        data_collector.add_news_callback(self._handle_new_news)
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load language translations"""
        # Simplified translations - in production, load from files
        return {
            'en': {
                'welcome': "🤖 Welcome to AGI Forex Trading Bot!\n\nI'm your advanced AI trading assistant. I can provide:\n• Real-time trading signals\n• Market analysis\n• Portfolio tracking\n• Risk management\n\nUse /help to see all commands.",
                'help': "📚 Available Commands:\n\n/start - Start the bot\n/signals - Get latest signals\n/portfolio - View your portfolio\n/analysis - Market analysis\n/settings - Configure preferences\n/performance - View performance stats\n/news - Latest forex news\n/chart - Generate charts\n/risk - Risk management tools",
                'signals_title': "🎯 Latest Trading Signals",
                'no_signals': "No signals available at the moment.",
                'portfolio_title': "💼 Your Portfolio",
                'empty_portfolio': "Your portfolio is empty. Start trading to see positions here.",
                'analysis_title': "📊 Market Analysis",
                'settings_title': "⚙️ Settings",
                'performance_title': "📈 Performance Statistics"
            },
            'id': {
                'welcome': "🤖 Selamat datang di AGI Forex Trading Bot!\n\nSaya adalah asisten trading AI canggih Anda. Saya dapat memberikan:\n• Signal trading real-time\n• Analisis pasar\n• Tracking portfolio\n• Manajemen risiko\n\nGunakan /help untuk melihat semua perintah.",
                'help': "📚 Perintah yang Tersedia:\n\n/start - Mulai bot\n/signals - Dapatkan signal terbaru\n/portfolio - Lihat portfolio Anda\n/analysis - Analisis pasar\n/settings - Konfigurasi preferensi\n/performance - Lihat statistik performa\n/news - Berita forex terbaru\n/chart - Generate chart\n/risk - Tools manajemen risiko",
                'signals_title': "🎯 Signal Trading Terbaru",
                'no_signals': "Tidak ada signal tersedia saat ini.",
                'portfolio_title': "💼 Portfolio Anda",
                'empty_portfolio': "Portfolio Anda kosong. Mulai trading untuk melihat posisi di sini.",
                'analysis_title': "📊 Analisis Pasar",
                'settings_title': "⚙️ Pengaturan",
                'performance_title': "📈 Statistik Performa"
            }
        }
    
    def _setup_handlers(self):
        """Setup command dan message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("portfolio", self.portfolio_command))
        self.application.add_handler(CommandHandler("analysis", self.analysis_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        self.application.add_handler(CommandHandler("performance", self.performance_command))
        self.application.add_handler(CommandHandler("news", self.news_command))
        self.application.add_handler(CommandHandler("chart", self.chart_command))
        self.application.add_handler(CommandHandler("risk", self.risk_command))
        self.application.add_handler(CommandHandler("subscribe", self.subscribe_command))
        self.application.add_handler(CommandHandler("language", self.language_command))
        
        # Callback query handler
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Message handlers
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    def _get_user_language(self, user_id: int) -> str:
        """Get user's preferred language"""
        user = self.db_session.query(TelegramUser).filter_by(user_id=user_id).first()
        return user.language_code if user else 'en'
    
    def _translate(self, key: str, lang: str = 'en') -> str:
        """Get translated text"""
        return self.translations.get(lang, {}).get(key, self.translations['en'].get(key, key))
    
    def _get_or_create_user(self, update: Update) -> TelegramUser:
        """Get or create user in database"""
        user_data = update.effective_user
        user = self.db_session.query(TelegramUser).filter_by(user_id=user_data.id).first()
        
        if not user:
            user = TelegramUser(
                user_id=user_data.id,
                username=user_data.username,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                language_code=user_data.language_code or 'en'
            )
            self.db_session.add(user)
            self.db_session.commit()
        else:
            # Update last active
            user.last_active = datetime.now()
            self.db_session.commit()
        
        return user
    
    async def start_command(self, update: Update, context):
        """Handle /start command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        welcome_text = self._translate('welcome', lang)
        
        # Create welcome keyboard
        keyboard = [
            [KeyboardButton("🎯 Signals"), KeyboardButton("💼 Portfolio")],
            [KeyboardButton("📊 Analysis"), KeyboardButton("📈 Performance")],
            [KeyboardButton("📰 News"), KeyboardButton("⚙️ Settings")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Send welcome chart
        await self._send_welcome_chart(update, user)
    
    async def help_command(self, update: Update, context):
        """Handle /help command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        help_text = self._translate('help', lang)
        await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)
    
    async def signals_command(self, update: Update, context):
        """Handle /signals command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        try:
            # Get latest signals from AGI Brain
            if not self.agi_brain:
                await update.message.reply_text("AGI Brain not available. Please try again later.")
                return
            
            # Get user's preferred pairs
            preferred_pairs = json.loads(user.preferred_pairs or '["EURUSD", "GBPUSD", "USDJPY"]')
            
            signals_text = f"🎯 *{self._translate('signals_title', lang)}*\n\n"
            
            # Get market data for analysis
            market_data = {}
            for pair in preferred_pairs:
                if self.data_collector:
                    historical_data = self.data_collector.get_historical_data(pair, 'H1', 7)
                    if not historical_data.empty:
                        market_data[pair] = historical_data
            
            if market_data:
                # Get recent news
                recent_news = []
                if self.data_collector:
                    recent_news = self.data_collector.get_recent_news(hours=6, min_relevance=0.5)
                
                news_texts = [news.title + " " + news.content for news in recent_news[:5]]
                
                # Generate signal
                signal = await self.agi_brain.analyze_market(market_data, news_texts)
                
                # Format signal
                signals_text += self._format_signal(signal, lang)
                
                # Create inline keyboard for actions
                keyboard = [
                    [InlineKeyboardButton("📊 Detailed Analysis", callback_data=f"analysis_{signal.pair}")],
                    [InlineKeyboardButton("📈 Chart", callback_data=f"chart_{signal.pair}")],
                    [InlineKeyboardButton("🔔 Set Alert", callback_data=f"alert_{signal.pair}")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(
                    signals_text,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.MARKDOWN
                )
                
                # Send signal chart
                await self._send_signal_chart(update, signal, market_data[signal.pair])
            else:
                await update.message.reply_text(self._translate('no_signals', lang))
        
        except Exception as e:
            logger.error(f"Error in signals command: {e}")
            await update.message.reply_text("Error generating signals. Please try again later.")
    
    async def portfolio_command(self, update: Update, context):
        """Handle /portfolio command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        # Get user's portfolio
        portfolio = self.db_session.query(UserPortfolio).filter_by(
            user_id=user.user_id,
            status='open'
        ).all()
        
        if not portfolio:
            await update.message.reply_text(self._translate('empty_portfolio', lang))
            return
        
        portfolio_text = f"💼 *{self._translate('portfolio_title', lang)}*\n\n"
        
        total_pnl = 0
        for position in portfolio:
            # Calculate current P&L
            current_price = self._get_current_price(position.pair)
            if current_price:
                if position.position_type == 'BUY':
                    pnl = (current_price - position.entry_price) * position.position_size
                else:
                    pnl = (position.entry_price - current_price) * position.position_size
                
                total_pnl += pnl
                
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                
                portfolio_text += f"{pnl_emoji} *{position.pair}*\n"
                portfolio_text += f"   Position: {position.position_type} {position.position_size}\n"
                portfolio_text += f"   Entry: {position.entry_price:.5f}\n"
                portfolio_text += f"   Current: {current_price:.5f}\n"
                portfolio_text += f"   P&L: {pnl:+.2f} USD\n\n"
        
        total_emoji = "🟢" if total_pnl >= 0 else "🔴"
        portfolio_text += f"{total_emoji} *Total P&L: {total_pnl:+.2f} USD*"
        
        # Create portfolio management keyboard
        keyboard = [
            [InlineKeyboardButton("📊 Portfolio Chart", callback_data="portfolio_chart")],
            [InlineKeyboardButton("⚠️ Risk Analysis", callback_data="portfolio_risk")],
            [InlineKeyboardButton("🔄 Refresh", callback_data="portfolio_refresh")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            portfolio_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def analysis_command(self, update: Update, context):
        """Handle /analysis command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        analysis_text = f"📊 *{self._translate('analysis_title', lang)}*\n\n"
        
        try:
            # Get market overview
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD']
            
            analysis_text += "🌍 *Market Overview*\n"
            
            for pair in major_pairs:
                if self.data_collector:
                    quote = self.data_collector.get_latest_quote(pair)
                    if quote:
                        # Simple trend analysis
                        historical = self.data_collector.get_historical_data(pair, 'H1', 1)
                        if not historical.empty:
                            change = ((quote.bid - historical['close'].iloc[0]) / historical['close'].iloc[0]) * 100
                            trend_emoji = "📈" if change > 0 else "📉"
                            analysis_text += f"{trend_emoji} {pair}: {quote.bid:.5f} ({change:+.2f}%)\n"
            
            # Market sentiment
            analysis_text += "\n💭 *Market Sentiment*\n"
            if self.data_collector:
                recent_news = self.data_collector.get_recent_news(hours=12, min_relevance=0.4)
                if recent_news:
                    avg_sentiment = np.mean([news.sentiment for news in recent_news])
                    sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😟"
                    analysis_text += f"{sentiment_emoji} Overall Sentiment: {avg_sentiment:.2f}\n"
                    analysis_text += f"📰 News Items Analyzed: {len(recent_news)}\n"
            
            # Volatility analysis
            analysis_text += "\n📊 *Volatility Analysis*\n"
            for pair in major_pairs[:3]:  # Top 3 pairs
                if self.data_collector:
                    historical = self.data_collector.get_historical_data(pair, 'H1', 7)
                    if not historical.empty:
                        volatility = historical['close'].pct_change().std() * 100
                        vol_level = "High" if volatility > 1.0 else "Medium" if volatility > 0.5 else "Low"
                        analysis_text += f"📊 {pair}: {vol_level} ({volatility:.2f}%)\n"
            
            # Create analysis keyboard
            keyboard = [
                [InlineKeyboardButton("📈 Detailed Charts", callback_data="detailed_charts")],
                [InlineKeyboardButton("🔍 Pair Analysis", callback_data="pair_analysis")],
                [InlineKeyboardButton("📊 Technical Indicators", callback_data="technical_indicators")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                analysis_text,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        
        except Exception as e:
            logger.error(f"Error in analysis command: {e}")
            await update.message.reply_text("Error generating analysis. Please try again later.")
    
    async def performance_command(self, update: Update, context):
        """Handle /performance command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        # Get performance statistics
        closed_positions = self.db_session.query(UserPortfolio).filter_by(
            user_id=user.user_id,
            status='closed'
        ).all()
        
        if not closed_positions:
            await update.message.reply_text("No trading history available yet.")
            return
        
        # Calculate statistics
        total_trades = len(closed_positions)
        winning_trades = len([p for p in closed_positions if p.profit_loss > 0])
        losing_trades = len([p for p in closed_positions if p.profit_loss < 0])
        total_profit = sum(p.profit_loss for p in closed_positions)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Average profit/loss
        avg_win = np.mean([p.profit_loss for p in closed_positions if p.profit_loss > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([p.profit_loss for p in closed_positions if p.profit_loss < 0]) if losing_trades > 0 else 0
        
        performance_text = f"📈 *{self._translate('performance_title', lang)}*\n\n"
        performance_text += f"📊 *Trading Statistics*\n"
        performance_text += f"Total Trades: {total_trades}\n"
        performance_text += f"Winning Trades: {winning_trades} ({win_rate:.1f}%)\n"
        performance_text += f"Losing Trades: {losing_trades}\n\n"
        
        profit_emoji = "🟢" if total_profit >= 0 else "🔴"
        performance_text += f"{profit_emoji} *Total P&L: {total_profit:+.2f} USD*\n\n"
        
        performance_text += f"💰 *Average Results*\n"
        performance_text += f"Avg Win: +{avg_win:.2f} USD\n"
        performance_text += f"Avg Loss: {avg_loss:.2f} USD\n"
        
        if avg_loss != 0:
            risk_reward = abs(avg_win / avg_loss)
            performance_text += f"Risk/Reward: 1:{risk_reward:.2f}\n"
        
        # Create performance keyboard
        keyboard = [
            [InlineKeyboardButton("📊 Performance Chart", callback_data="performance_chart")],
            [InlineKeyboardButton("📈 Equity Curve", callback_data="equity_curve")],
            [InlineKeyboardButton("📋 Detailed Report", callback_data="detailed_report")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            performance_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def news_command(self, update: Update, context):
        """Handle /news command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        if not self.data_collector:
            await update.message.reply_text("News service not available.")
            return
        
        # Get recent news
        recent_news = self.data_collector.get_recent_news(hours=6, min_relevance=0.3)
        
        if not recent_news:
            await update.message.reply_text("No recent news available.")
            return
        
        news_text = "📰 *Latest Forex News*\n\n"
        
        for i, news in enumerate(recent_news[:5]):  # Top 5 news
            sentiment_emoji = "🟢" if news.sentiment > 0.1 else "🔴" if news.sentiment < -0.1 else "🟡"
            relevance_stars = "⭐" * min(int(news.relevance * 5), 5)
            
            news_text += f"{sentiment_emoji} *{news.title[:80]}...*\n"
            news_text += f"📅 {news.timestamp.strftime('%H:%M')} | 📊 {relevance_stars}\n"
            news_text += f"📰 {news.source}\n\n"
        
        # Create news keyboard
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh News", callback_data="refresh_news")],
            [InlineKeyboardButton("📊 Sentiment Analysis", callback_data="news_sentiment")],
            [InlineKeyboardButton("🎯 Impact Analysis", callback_data="news_impact")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            news_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def chart_command(self, update: Update, context):
        """Handle /chart command"""
        user = self._get_or_create_user(update)
        
        # Get pair from command arguments
        pair = 'EURUSD'  # Default
        if context.args:
            pair = context.args[0].upper()
        
        await self._send_advanced_chart(update, pair)
    
    async def settings_command(self, update: Update, context):
        """Handle /settings command"""
        user = self._get_or_create_user(update)
        lang = user.language_code
        
        settings_text = f"⚙️ *{self._translate('settings_title', lang)}*\n\n"
        
        # Current settings
        notification_settings = json.loads(user.notification_settings or '{}')
        preferred_pairs = json.loads(user.preferred_pairs or '["EURUSD", "GBPUSD", "USDJPY"]')
        
        settings_text += f"🔔 Notifications: {'✅' if notification_settings.get('signals_enabled', True) else '❌'}\n"
        settings_text += f"📰 News Alerts: {'✅' if notification_settings.get('news_alerts', True) else '❌'}\n"
        settings_text += f"⚠️ Risk Alerts: {'✅' if notification_settings.get('risk_alerts', True) else '❌'}\n"
        settings_text += f"🎯 Risk Tolerance: {user.risk_tolerance.title()}\n"
        settings_text += f"💱 Preferred Pairs: {', '.join(preferred_pairs[:3])}...\n"
        settings_text += f"🌍 Language: {self.languages.get(lang, 'English')}\n"
        settings_text += f"💎 Subscription: {user.subscription_type.title()}\n"
        
        # Create settings keyboard
        keyboard = [
            [InlineKeyboardButton("🔔 Notifications", callback_data="settings_notifications")],
            [InlineKeyboardButton("💱 Currency Pairs", callback_data="settings_pairs")],
            [InlineKeyboardButton("⚠️ Risk Settings", callback_data="settings_risk")],
            [InlineKeyboardButton("🌍 Language", callback_data="settings_language")],
            [InlineKeyboardButton("💎 Upgrade", callback_data="settings_upgrade")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            settings_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def button_callback(self, update: Update, context):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        user = self._get_or_create_user(update)
        data = query.data
        
        if data.startswith("analysis_"):
            pair = data.split("_")[1]
            await self._send_detailed_analysis(query, pair)
        
        elif data.startswith("chart_"):
            pair = data.split("_")[1]
            await self._send_advanced_chart(query, pair)
        
        elif data.startswith("alert_"):
            pair = data.split("_")[1]
            await self._setup_price_alert(query, pair)
        
        elif data == "portfolio_chart":
            await self._send_portfolio_chart(query, user)
        
        elif data == "portfolio_risk":
            await self._send_risk_analysis(query, user)
        
        elif data == "performance_chart":
            await self._send_performance_chart(query, user)
        
        elif data.startswith("settings_"):
            setting_type = data.split("_")[1]
            await self._handle_settings_callback(query, user, setting_type)
        
        # Add more callback handlers as needed
    
    async def handle_message(self, update: Update, context):
        """Handle text messages"""
        user = self._get_or_create_user(update)
        message_text = update.message.text.lower()
        
        # Handle button presses
        if "signals" in message_text:
            await self.signals_command(update, context)
        elif "portfolio" in message_text:
            await self.portfolio_command(update, context)
        elif "analysis" in message_text:
            await self.analysis_command(update, context)
        elif "performance" in message_text:
            await self.performance_command(update, context)
        elif "news" in message_text:
            await self.news_command(update, context)
        elif "settings" in message_text:
            await self.settings_command(update, context)
        else:
            # AI-powered message handling
            await self._handle_ai_message(update, user, message_text)
    
    async def handle_voice(self, update: Update, context):
        """Handle voice messages"""
        user = self._get_or_create_user(update)
        
        # Download and process voice message
        voice_file = await update.message.voice.get_file()
        
        # Simplified voice processing - in production, use speech-to-text
        await update.message.reply_text(
            "🎤 Voice message received! Voice commands are coming soon in the next update.",
            parse_mode=ParseMode.MARKDOWN
        )
    
    async def error_handler(self, update: Update, context):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again later."
            )
    
    def _format_signal(self, signal: MarketSignal, lang: str = 'en') -> str:
        """Format trading signal untuk display"""
        action_emoji = "🟢" if signal.action == "BUY" else "🔴" if signal.action == "SELL" else "🟡"
        confidence_stars = "⭐" * min(int(signal.confidence * 5), 5)
        
        signal_text = f"{action_emoji} *{signal.pair}* - {signal.action}\n"
        signal_text += f"📊 Confidence: {confidence_stars} ({signal.confidence:.1%})\n"
        signal_text += f"💰 Entry: {signal.entry_price:.5f}\n"
        signal_text += f"🛑 Stop Loss: {signal.stop_loss:.5f}\n"
        signal_text += f"🎯 Take Profit: {signal.take_profit:.5f}\n"
        signal_text += f"📈 R/R Ratio: 1:{signal.risk_reward_ratio:.2f}\n"
        signal_text += f"⏰ Time: {signal.timestamp.strftime('%H:%M:%S')}\n"
        signal_text += f"🏛️ Regime: {signal.market_regime.title()}\n\n"
        signal_text += f"💭 *Reasoning:*\n{signal.reasoning[:200]}...\n"
        
        return signal_text
    
    def _get_current_price(self, pair: str) -> Optional[float]:
        """Get current price untuk currency pair"""
        if self.data_collector:
            quote = self.data_collector.get_latest_quote(pair)
            return quote.bid if quote else None
        return None
    
    async def _send_signal_chart(self, update: Update, signal: MarketSignal, data: pd.DataFrame):
        """Send chart untuk trading signal"""
        try:
            # Create chart using matplotlib
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
            
            # Price chart
            ax1.plot(data.index, data['close'], label='Price', linewidth=2)
            ax1.axhline(y=signal.entry_price, color='blue', linestyle='--', label='Entry')
            ax1.axhline(y=signal.stop_loss, color='red', linestyle='--', label='Stop Loss')
            ax1.axhline(y=signal.take_profit, color='green', linestyle='--', label='Take Profit')
            
            ax1.set_title(f'{signal.pair} - {signal.action} Signal', fontsize=16, fontweight='bold')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            if 'volume' in data.columns:
                ax2.bar(data.index, data['volume'], alpha=0.7, color='gray')
                ax2.set_ylabel('Volume')
                ax2.set_xlabel('Time')
            
            plt.tight_layout()
            
            # Save chart to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Send chart
            await update.message.reply_photo(
                photo=img_buffer,
                caption=f"📊 Chart for {signal.pair} {signal.action} signal"
            )
        
        except Exception as e:
            logger.error(f"Error sending signal chart: {e}")
    
    async def _send_advanced_chart(self, update_or_query, pair: str):
        """Send advanced chart dengan technical indicators"""
        try:
            if not self.data_collector:
                await update_or_query.message.reply_text("Chart service not available.")
                return
            
            # Get historical data
            data = self.data_collector.get_historical_data(pair, 'H1', 30)
            if data.empty:
                await update_or_query.message.reply_text(f"No data available for {pair}")
                return
            
            # Create advanced chart using plotly
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{pair} Price', 'Volume', 'RSI'),
                row_width=[0.2, 0.1, 0.1]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Moving averages
            data['MA20'] = data['close'].rolling(20).mean()
            data['MA50'] = data['close'].rolling(50).mean()
            
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='orange')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='red')),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=data.index, y=data['volume'], name='Volume', marker_color='gray'),
                row=2, col=1
            )
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            fig.update_layout(
                title=f'{pair} Advanced Chart',
                xaxis_rangeslider_visible=False,
                height=800
            )
            
            # Convert to image
            img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            img_buffer = io.BytesIO(img_bytes)
            
            # Send chart
            if hasattr(update_or_query, 'message'):
                await update_or_query.message.reply_photo(
                    photo=img_buffer,
                    caption=f"📊 Advanced chart for {pair}"
                )
            else:
                await update_or_query.edit_message_caption(
                    caption=f"📊 Advanced chart for {pair}"
                )
                await update_or_query.message.reply_photo(photo=img_buffer)
        
        except Exception as e:
            logger.error(f"Error sending advanced chart: {e}")
    
    async def _handle_new_market_data(self, quote: RealTimeQuote):
        """Handle new market data untuk real-time notifications"""
        try:
            # Check for price alerts
            await self._check_price_alerts(quote)
            
            # Check for significant moves
            await self._check_significant_moves(quote)
        
        except Exception as e:
            logger.error(f"Error handling new market data: {e}")
    
    async def _handle_new_news(self, news_item: NewsItem):
        """Handle new news untuk notifications"""
        try:
            # Send high-impact news to subscribed users
            if news_item.relevance > 0.7:
                await self._broadcast_news_alert(news_item)
        
        except Exception as e:
            logger.error(f"Error handling new news: {e}")
    
    async def _broadcast_signal(self, signal: MarketSignal):
        """Broadcast trading signal ke all subscribed users"""
        try:
            # Get all users dengan signal notifications enabled
            users = self.db_session.query(TelegramUser).filter_by(
                subscription_type='premium'  # Only premium users get real-time signals
            ).all()
            
            for user in users:
                try:
                    notification_settings = json.loads(user.notification_settings or '{}')
                    if notification_settings.get('signals_enabled', True):
                        # Check confidence threshold
                        min_confidence = notification_settings.get('min_signal_confidence', 0.7)
                        if signal.confidence >= min_confidence:
                            signal_text = f"🚨 *New Signal Alert*\n\n"
                            signal_text += self._format_signal(signal, user.language_code)
                            
                            await self.application.bot.send_message(
                                chat_id=user.user_id,
                                text=signal_text,
                                parse_mode=ParseMode.MARKDOWN
                            )
                
                except Exception as e:
                    logger.error(f"Error sending signal to user {user.user_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error broadcasting signal: {e}")
    
    async def start_bot(self):
        """Start the Telegram bot"""
        logger.info("Starting Telegram bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()
        
        # Keep the bot running
        while True:
            await asyncio.sleep(1)
    
    async def stop_bot(self):
        """Stop the Telegram bot"""
        logger.info("Stopping Telegram bot...")
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

# Example usage
if __name__ == "__main__":
    config = {
        'telegram_bot_token': 'YOUR_BOT_TOKEN',
        'database_url': 'sqlite:///telegram_bot.db',
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    bot = AdvancedTelegramBot(config)
    
    # Run bot
    asyncio.run(bot.start_bot())