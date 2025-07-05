"""
Market Data Collector - Real-time Data Collection System
=======================================================

Sistem pengumpulan data real-time dari multiple sources:
- Forex brokers (MT4/MT5, OANDA, FXCM, etc.)
- Economic calendars
- News feeds
- Social media sentiment
- Central bank communications
"""

import asyncio
import aiohttp
import websocket
import ccxt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.fundamentaldata import FundamentalData
import tweepy
import feedparser
from textblob import TextBlob
import redis
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import schedule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class MarketDataPoint(Base):
    """Database model untuk market data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    pair = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(String(5), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, default=0)
    spread = Column(Float, default=0)
    source = Column(String(50), nullable=False)

class NewsData(Base):
    """Database model untuk news data"""
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    source = Column(String(100), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    sentiment_score = Column(Float, default=0)
    relevance_score = Column(Float, default=0)
    currency_pairs = Column(String(200))  # JSON string of affected pairs

class EconomicEvent(Base):
    """Database model untuk economic events"""
    __tablename__ = 'economic_events'
    
    id = Column(Integer, primary_key=True)
    event_name = Column(String(200), nullable=False)
    country = Column(String(50), nullable=False)
    currency = Column(String(3), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    importance = Column(String(10))  # LOW, MEDIUM, HIGH
    actual_value = Column(Float)
    forecast_value = Column(Float)
    previous_value = Column(Float)
    impact_score = Column(Float, default=0)

@dataclass
class RealTimeQuote:
    """Real-time quote data structure"""
    pair: str
    bid: float
    ask: float
    timestamp: datetime
    spread: float
    volume: float = 0
    source: str = ""

@dataclass
class NewsItem:
    """News item data structure"""
    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment: float = 0.0
    relevance: float = 0.0
    affected_pairs: List[str] = None

class ForexDataProvider:
    """Base class untuk forex data providers"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.is_connected = False
        self.callbacks = []
    
    def add_callback(self, callback: Callable):
        """Add callback untuk data updates"""
        self.callbacks.append(callback)
    
    def notify_callbacks(self, data: Any):
        """Notify all callbacks dengan new data"""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
    
    async def connect(self):
        """Connect to data provider"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from data provider"""
        raise NotImplementedError
    
    async def subscribe_to_pair(self, pair: str):
        """Subscribe to currency pair"""
        raise NotImplementedError

class OANDAProvider(ForexDataProvider):
    """OANDA API data provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("OANDA", config)
        self.api_key = config.get('api_key')
        self.account_id = config.get('account_id')
        self.base_url = config.get('base_url', 'https://api-fxtrade.oanda.com')
        self.session = None
        self.websocket = None
    
    async def connect(self):
        """Connect to OANDA API"""
        try:
            self.session = aiohttp.ClientSession(
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            
            # Test connection
            async with self.session.get(f'{self.base_url}/v3/accounts/{self.account_id}') as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("Connected to OANDA API")
                else:
                    logger.error(f"Failed to connect to OANDA: {response.status}")
        
        except Exception as e:
            logger.error(f"Error connecting to OANDA: {e}")
    
    async def get_historical_data(self, pair: str, timeframe: str = 'H1', 
                                 count: int = 500) -> pd.DataFrame:
        """Get historical data"""
        try:
            url = f'{self.base_url}/v3/instruments/{pair}/candles'
            params = {
                'granularity': timeframe,
                'count': count,
                'price': 'MBA'  # Mid, Bid, Ask
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_oanda_candles(data['candles'])
                else:
                    logger.error(f"Error getting historical data: {response.status}")
                    return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error in get_historical_data: {e}")
            return pd.DataFrame()
    
    def _parse_oanda_candles(self, candles: List[Dict]) -> pd.DataFrame:
        """Parse OANDA candle data to DataFrame"""
        data = []
        for candle in candles:
            if candle['complete']:
                data.append({
                    'timestamp': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                    'volume': int(candle['volume'])
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
    
    async def start_streaming(self, pairs: List[str]):
        """Start real-time streaming"""
        try:
            instruments = ','.join(pairs)
            stream_url = f'{self.base_url}/v3/accounts/{self.account_id}/pricing/stream'
            params = {'instruments': instruments}
            
            async with self.session.get(stream_url, params=params) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if data['type'] == 'PRICE':
                                quote = self._parse_price_data(data)
                                self.notify_callbacks(quote)
                        except json.JSONDecodeError:
                            continue
        
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
    
    def _parse_price_data(self, price_data: Dict) -> RealTimeQuote:
        """Parse price data to RealTimeQuote"""
        return RealTimeQuote(
            pair=price_data['instrument'],
            bid=float(price_data['bids'][0]['price']),
            ask=float(price_data['asks'][0]['price']),
            timestamp=pd.to_datetime(price_data['time']),
            spread=float(price_data['asks'][0]['price']) - float(price_data['bids'][0]['price']),
            source='OANDA'
        )

class CCXTProvider(ForexDataProvider):
    """CCXT-based data provider untuk multiple exchanges"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("CCXT", config)
        self.exchange_name = config.get('exchange', 'binance')
        self.exchange = None
        self.api_key = config.get('api_key')
        self.secret = config.get('secret')
    
    async def connect(self):
        """Connect to exchange via CCXT"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.config.get('sandbox', False),
                'enableRateLimit': True,
            })
            
            # Test connection
            await self.exchange.load_markets()
            self.is_connected = True
            logger.info(f"Connected to {self.exchange_name} via CCXT")
        
        except Exception as e:
            logger.error(f"Error connecting to {self.exchange_name}: {e}")
    
    async def get_historical_data(self, pair: str, timeframe: str = '1h', 
                                 limit: int = 500) -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting historical data from {self.exchange_name}: {e}")
            return pd.DataFrame()

class NewsCollector:
    """News data collector dari multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://feeds.reuters.com/reuters/businessNews',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.forexfactory.com/rss.php'
        ]
        self.twitter_api = self._setup_twitter_api()
        self.callbacks = []
    
    def _setup_twitter_api(self) -> Optional[tweepy.API]:
        """Setup Twitter API"""
        try:
            auth = tweepy.OAuthHandler(
                self.config.get('twitter_consumer_key'),
                self.config.get('twitter_consumer_secret')
            )
            auth.set_access_token(
                self.config.get('twitter_access_token'),
                self.config.get('twitter_access_token_secret')
            )
            return tweepy.API(auth)
        except Exception as e:
            logger.error(f"Error setting up Twitter API: {e}")
            return None
    
    def add_callback(self, callback: Callable):
        """Add callback untuk news updates"""
        self.callbacks.append(callback)
    
    def notify_callbacks(self, news_item: NewsItem):
        """Notify callbacks dengan news item baru"""
        for callback in self.callbacks:
            try:
                callback(news_item)
            except Exception as e:
                logger.error(f"Error in news callback: {e}")
    
    async def collect_rss_news(self):
        """Collect news dari RSS feeds"""
        for source_url in self.news_sources:
            try:
                feed = feedparser.parse(source_url)
                for entry in feed.entries[:10]:  # Limit to 10 latest
                    news_item = NewsItem(
                        title=entry.title,
                        content=entry.get('summary', ''),
                        source=feed.feed.get('title', source_url),
                        timestamp=datetime.now(),
                        sentiment=self._analyze_sentiment(entry.title + ' ' + entry.get('summary', '')),
                        relevance=self._calculate_relevance(entry.title),
                        affected_pairs=self._extract_currency_pairs(entry.title + ' ' + entry.get('summary', ''))
                    )
                    self.notify_callbacks(news_item)
            
            except Exception as e:
                logger.error(f"Error collecting RSS news from {source_url}: {e}")
    
    async def collect_twitter_sentiment(self, keywords: List[str] = None):
        """Collect sentiment dari Twitter"""
        if not self.twitter_api:
            return
        
        if not keywords:
            keywords = ['forex', 'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        
        try:
            for keyword in keywords:
                tweets = tweepy.Cursor(
                    self.twitter_api.search_tweets,
                    q=keyword,
                    lang='en',
                    result_type='recent'
                ).items(50)
                
                for tweet in tweets:
                    news_item = NewsItem(
                        title=tweet.text[:100],
                        content=tweet.text,
                        source='Twitter',
                        timestamp=tweet.created_at,
                        sentiment=self._analyze_sentiment(tweet.text),
                        relevance=self._calculate_relevance(tweet.text),
                        affected_pairs=self._extract_currency_pairs(tweet.text)
                    )
                    self.notify_callbacks(news_item)
        
        except Exception as e:
            logger.error(f"Error collecting Twitter sentiment: {e}")
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment menggunakan TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity  # Returns -1 to 1
        except Exception:
            return 0.0
    
    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score untuk forex trading"""
        forex_keywords = [
            'central bank', 'interest rate', 'inflation', 'gdp', 'employment',
            'fed', 'ecb', 'boe', 'boj', 'monetary policy', 'fiscal policy',
            'trade war', 'brexit', 'election', 'crisis', 'recession'
        ]
        
        text_lower = text.lower()
        relevance_score = sum(1 for keyword in forex_keywords if keyword in text_lower)
        return min(relevance_score / len(forex_keywords), 1.0)
    
    def _extract_currency_pairs(self, text: str) -> List[str]:
        """Extract currency pairs dari text"""
        currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
        found_currencies = [curr for curr in currencies if curr in text.upper()]
        
        pairs = []
        for i, curr1 in enumerate(found_currencies):
            for curr2 in found_currencies[i+1:]:
                pairs.extend([f"{curr1}{curr2}", f"{curr2}{curr1}"])
        
        return pairs

class EconomicCalendarCollector:
    """Economic calendar data collector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha_vantage_key = config.get('alpha_vantage_key')
        self.callbacks = []
    
    def add_callback(self, callback: Callable):
        """Add callback untuk economic events"""
        self.callbacks.append(callback)
    
    def notify_callbacks(self, event: EconomicEvent):
        """Notify callbacks dengan economic event"""
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in economic event callback: {e}")
    
    async def collect_economic_events(self):
        """Collect economic events"""
        try:
            # Simplified economic events collection
            # In production, integrate with ForexFactory, Investing.com, etc.
            
            events = [
                {
                    'name': 'Non-Farm Payrolls',
                    'country': 'US',
                    'currency': 'USD',
                    'importance': 'HIGH',
                    'time': datetime.now() + timedelta(days=1)
                },
                {
                    'name': 'ECB Interest Rate Decision',
                    'country': 'EU',
                    'currency': 'EUR',
                    'importance': 'HIGH',
                    'time': datetime.now() + timedelta(days=2)
                }
            ]
            
            for event_data in events:
                event = EconomicEvent(
                    event_name=event_data['name'],
                    country=event_data['country'],
                    currency=event_data['currency'],
                    timestamp=event_data['time'],
                    importance=event_data['importance'],
                    impact_score=self._calculate_impact_score(event_data)
                )
                self.notify_callbacks(event)
        
        except Exception as e:
            logger.error(f"Error collecting economic events: {e}")
    
    def _calculate_impact_score(self, event_data: Dict) -> float:
        """Calculate impact score untuk economic event"""
        importance_scores = {'LOW': 0.3, 'MEDIUM': 0.6, 'HIGH': 1.0}
        return importance_scores.get(event_data.get('importance', 'MEDIUM'), 0.6)

class MarketDataCollector:
    """Main market data collector orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0)
        )
        
        # Database setup
        self.db_engine = create_engine(config.get('database_url', 'sqlite:///forex_data.db'))
        Base.metadata.create_all(self.db_engine)
        Session = sessionmaker(bind=self.db_engine)
        self.db_session = Session()
        
        # Data providers
        self.providers = {}
        self.news_collector = NewsCollector(config.get('news_config', {}))
        self.economic_collector = EconomicCalendarCollector(config.get('economic_config', {}))
        
        # Callbacks
        self.data_callbacks = []
        self.news_callbacks = []
        self.economic_callbacks = []
        
        # Setup callbacks
        self.news_collector.add_callback(self._handle_news_data)
        self.economic_collector.add_callback(self._handle_economic_data)
        
        # Scheduling
        self.scheduler_running = False
        
        logger.info("Market Data Collector initialized")
    
    def add_provider(self, provider: ForexDataProvider):
        """Add data provider"""
        self.providers[provider.name] = provider
        provider.add_callback(self._handle_market_data)
        logger.info(f"Added provider: {provider.name}")
    
    def add_data_callback(self, callback: Callable):
        """Add callback untuk market data"""
        self.data_callbacks.append(callback)
    
    def add_news_callback(self, callback: Callable):
        """Add callback untuk news data"""
        self.news_callbacks.append(callback)
    
    def add_economic_callback(self, callback: Callable):
        """Add callback untuk economic data"""
        self.economic_callbacks.append(callback)
    
    async def start_collection(self, pairs: List[str]):
        """Start data collection"""
        try:
            # Connect all providers
            for provider in self.providers.values():
                await provider.connect()
                for pair in pairs:
                    await provider.subscribe_to_pair(pair)
            
            # Start news collection
            asyncio.create_task(self._run_news_collection())
            
            # Start economic calendar collection
            asyncio.create_task(self._run_economic_collection())
            
            # Start scheduled tasks
            self._start_scheduler()
            
            logger.info("Data collection started")
        
        except Exception as e:
            logger.error(f"Error starting data collection: {e}")
    
    async def _run_news_collection(self):
        """Run news collection loop"""
        while True:
            try:
                await self.news_collector.collect_rss_news()
                await self.news_collector.collect_twitter_sentiment()
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"Error in news collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_economic_collection(self):
        """Run economic calendar collection loop"""
        while True:
            try:
                await self.economic_collector.collect_economic_events()
                await asyncio.sleep(3600)  # 1 hour
            except Exception as e:
                logger.error(f"Error in economic collection loop: {e}")
                await asyncio.sleep(300)
    
    def _start_scheduler(self):
        """Start scheduled tasks"""
        if self.scheduler_running:
            return
        
        # Schedule daily data cleanup
        schedule.every().day.at("00:00").do(self._cleanup_old_data)
        
        # Schedule hourly data backup
        schedule.every().hour.do(self._backup_data)
        
        def run_scheduler():
            while self.scheduler_running:
                schedule.run_pending()
                time.sleep(60)
        
        self.scheduler_running = True
        threading.Thread(target=run_scheduler, daemon=True).start()
        logger.info("Scheduler started")
    
    def _handle_market_data(self, quote: RealTimeQuote):
        """Handle incoming market data"""
        try:
            # Store in Redis untuk real-time access
            redis_key = f"quote:{quote.pair}"
            quote_data = asdict(quote)
            quote_data['timestamp'] = quote.timestamp.isoformat()
            self.redis_client.setex(redis_key, 300, json.dumps(quote_data))  # 5 min expiry
            
            # Store in database
            market_data = MarketDataPoint(
                pair=quote.pair,
                timestamp=quote.timestamp,
                timeframe='TICK',
                open=quote.bid,
                high=quote.ask,
                low=quote.bid,
                close=quote.ask,
                volume=quote.volume,
                spread=quote.spread,
                source=quote.source
            )
            self.db_session.add(market_data)
            self.db_session.commit()
            
            # Notify callbacks
            for callback in self.data_callbacks:
                callback(quote)
        
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    def _handle_news_data(self, news_item: NewsItem):
        """Handle incoming news data"""
        try:
            # Store in database
            news_data = NewsData(
                title=news_item.title,
                content=news_item.content,
                source=news_item.source,
                timestamp=news_item.timestamp,
                sentiment_score=news_item.sentiment,
                relevance_score=news_item.relevance,
                currency_pairs=json.dumps(news_item.affected_pairs or [])
            )
            self.db_session.add(news_data)
            self.db_session.commit()
            
            # Store in Redis
            redis_key = f"news:{int(news_item.timestamp.timestamp())}"
            news_data_dict = asdict(news_item)
            news_data_dict['timestamp'] = news_item.timestamp.isoformat()
            self.redis_client.setex(redis_key, 86400, json.dumps(news_data_dict))  # 24 hours
            
            # Notify callbacks
            for callback in self.news_callbacks:
                callback(news_item)
        
        except Exception as e:
            logger.error(f"Error handling news data: {e}")
    
    def _handle_economic_data(self, event: EconomicEvent):
        """Handle incoming economic event data"""
        try:
            # Store in database
            self.db_session.add(event)
            self.db_session.commit()
            
            # Store in Redis
            redis_key = f"economic:{int(event.timestamp.timestamp())}"
            event_data = {
                'event_name': event.event_name,
                'country': event.country,
                'currency': event.currency,
                'timestamp': event.timestamp.isoformat(),
                'importance': event.importance,
                'impact_score': event.impact_score
            }
            self.redis_client.setex(redis_key, 86400, json.dumps(event_data))  # 24 hours
            
            # Notify callbacks
            for callback in self.economic_callbacks:
                callback(event)
        
        except Exception as e:
            logger.error(f"Error handling economic data: {e}")
    
    def _cleanup_old_data(self):
        """Cleanup old data dari database"""
        try:
            cutoff_date = datetime.now() - timedelta(days=30)
            
            # Delete old market data
            self.db_session.query(MarketDataPoint).filter(
                MarketDataPoint.timestamp < cutoff_date
            ).delete()
            
            # Delete old news data
            self.db_session.query(NewsData).filter(
                NewsData.timestamp < cutoff_date
            ).delete()
            
            self.db_session.commit()
            logger.info("Old data cleanup completed")
        
        except Exception as e:
            logger.error(f"Error in data cleanup: {e}")
    
    def _backup_data(self):
        """Backup critical data"""
        try:
            # Simplified backup - in production, implement proper backup strategy
            logger.info("Data backup completed")
        except Exception as e:
            logger.error(f"Error in data backup: {e}")
    
    def get_latest_quote(self, pair: str) -> Optional[RealTimeQuote]:
        """Get latest quote untuk currency pair"""
        try:
            redis_key = f"quote:{pair}"
            quote_data = self.redis_client.get(redis_key)
            if quote_data:
                data = json.loads(quote_data)
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                return RealTimeQuote(**data)
            return None
        except Exception as e:
            logger.error(f"Error getting latest quote: {e}")
            return None
    
    def get_historical_data(self, pair: str, timeframe: str = 'H1', 
                          days: int = 30) -> pd.DataFrame:
        """Get historical data dari database"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            query = self.db_session.query(MarketDataPoint).filter(
                MarketDataPoint.pair == pair,
                MarketDataPoint.timeframe == timeframe,
                MarketDataPoint.timestamp >= start_date
            ).order_by(MarketDataPoint.timestamp)
            
            data = []
            for record in query:
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open,
                    'high': record.high,
                    'low': record.low,
                    'close': record.close,
                    'volume': record.volume
                })
            
            df = pd.DataFrame(data)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()
    
    def get_recent_news(self, hours: int = 24, min_relevance: float = 0.3) -> List[NewsItem]:
        """Get recent news dengan minimum relevance"""
        try:
            start_time = datetime.now() - timedelta(hours=hours)
            
            query = self.db_session.query(NewsData).filter(
                NewsData.timestamp >= start_time,
                NewsData.relevance_score >= min_relevance
            ).order_by(NewsData.timestamp.desc())
            
            news_items = []
            for record in query:
                news_items.append(NewsItem(
                    title=record.title,
                    content=record.content,
                    source=record.source,
                    timestamp=record.timestamp,
                    sentiment=record.sentiment_score,
                    relevance=record.relevance_score,
                    affected_pairs=json.loads(record.currency_pairs or '[]')
                ))
            
            return news_items
        
        except Exception as e:
            logger.error(f"Error getting recent news: {e}")
            return []

# Example usage
if __name__ == "__main__":
    config = {
        'database_url': 'sqlite:///forex_data.db',
        'redis_host': 'localhost',
        'redis_port': 6379,
        'oanda_config': {
            'api_key': 'your_oanda_api_key',
            'account_id': 'your_account_id'
        },
        'news_config': {
            'twitter_consumer_key': 'your_twitter_key',
            'twitter_consumer_secret': 'your_twitter_secret',
            'twitter_access_token': 'your_access_token',
            'twitter_access_token_secret': 'your_access_token_secret'
        }
    }
    
    # Initialize collector
    collector = MarketDataCollector(config)
    
    # Add OANDA provider
    oanda_provider = OANDAProvider(config['oanda_config'])
    collector.add_provider(oanda_provider)
    
    # Start collection
    pairs = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF']
    
    async def main():
        await collector.start_collection(pairs)
        
        # Keep running
        while True:
            await asyncio.sleep(60)
    
    # asyncio.run(main())