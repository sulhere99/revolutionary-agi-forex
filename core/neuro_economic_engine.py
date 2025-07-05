"""
Neuro-Economic Sentiment Engine dengan IoT Integration
======================================================

Revolutionary Real-World Economic Sensor Network yang mengintegrasikan IoT devices,
satellite imagery, social media sentiment, dan economic indicators untuk menciptakan
"Economic Nervous System" yang dapat merasakan perubahan ekonomi sebelum tercermin
di market.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
import time
import random
import math
from collections import defaultdict, deque
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModel
import cv2
# import satellite_image_processing  # Hypothetical satellite processing library
from geopy.geocoders import Nominatim
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from textblob import TextBlob
import tweepy
import praw  # Reddit API
from newsapi import NewsApiClient
import feedparser
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import schedule
import websocket
import ssl

logger = logging.getLogger(__name__)

@dataclass
class IoTSensorReading:
    """IoT sensor reading data"""
    sensor_id: str
    sensor_type: str
    location: Tuple[float, float]  # lat, lon
    timestamp: datetime
    value: float
    unit: str
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EconomicIndicator:
    """Economic indicator data"""
    indicator_name: str
    country: str
    value: float
    previous_value: float
    forecast_value: float
    timestamp: datetime
    importance: str  # high, medium, low
    impact: str  # positive, negative, neutral
    source: str

@dataclass
class SatelliteAnalysis:
    """Satellite imagery analysis result"""
    location: Tuple[float, float]
    analysis_type: str  # port_activity, industrial_activity, etc.
    activity_level: float
    change_from_previous: float
    confidence: float
    timestamp: datetime
    image_metadata: Dict[str, Any]

@dataclass
class SocialSentiment:
    """Social media sentiment data"""
    platform: str
    content: str
    sentiment_score: float
    confidence: float
    engagement_metrics: Dict[str, Any]
    location: Optional[str]
    timestamp: datetime
    keywords: List[str]

@dataclass
class EconomicPulse:
    """Real-time economic pulse measurement"""
    timestamp: datetime
    overall_score: float
    components: Dict[str, float]
    confidence: float
    trend_direction: str
    volatility: float
    regional_breakdown: Dict[str, float]
    sector_breakdown: Dict[str, float]

class IoTSensorManager:
    """Manages IoT sensors for economic data collection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sensors = {}
        self.sensor_readings = deque(maxlen=100000)
        self.data_quality_threshold = config.get('data_quality_threshold', 0.7)
        
        # Initialize sensor types
        self._initialize_sensor_types()
        
        # Database for sensor data
        self.db_engine = create_engine(config.get('database_url', 'sqlite:///iot_sensors.db'))
        self._setup_database()
        
        logger.info("🌐 IoT Sensor Manager initialized")
    
    def _initialize_sensor_types(self):
        """Initialize different types of IoT sensors"""
        self.sensor_types = {
            'traffic_flow': {
                'description': 'Traffic flow sensors for economic activity',
                'locations': [
                    (40.7128, -74.0060),  # New York
                    (51.5074, -0.1278),   # London
                    (35.6762, 139.6503),  # Tokyo
                    (52.5200, 13.4050),   # Berlin
                    (48.8566, 2.3522),    # Paris
                ],
                'update_frequency': 300,  # 5 minutes
                'economic_correlation': 0.8
            },
            'energy_consumption': {
                'description': 'Energy consumption monitoring',
                'locations': [
                    (40.7128, -74.0060),  # New York
                    (51.5074, -0.1278),   # London
                    (35.6762, 139.6503),  # Tokyo
                ],
                'update_frequency': 900,  # 15 minutes
                'economic_correlation': 0.9
            },
            'shipping_activity': {
                'description': 'Port and shipping activity sensors',
                'locations': [
                    (22.3193, 114.1694),  # Hong Kong Port
                    (31.2304, 121.4737),  # Shanghai Port
                    (51.9244, 4.4777),    # Rotterdam Port
                    (33.7490, -118.2437), # Los Angeles Port
                ],
                'update_frequency': 1800,  # 30 minutes
                'economic_correlation': 0.85
            },
            'retail_footfall': {
                'description': 'Retail area foot traffic',
                'locations': [
                    (40.7580, -73.9855),  # Times Square
                    (51.5145, -0.1447),   # Oxford Street
                    (35.6586, 139.7454),  # Shibuya
                ],
                'update_frequency': 600,  # 10 minutes
                'economic_correlation': 0.75
            },
            'industrial_emissions': {
                'description': 'Industrial activity via emissions monitoring',
                'locations': [
                    (39.9042, 116.4074),  # Beijing
                    (28.7041, 77.1025),   # Delhi
                    (40.7128, -74.0060),  # New York
                ],
                'update_frequency': 3600,  # 1 hour
                'economic_correlation': 0.8
            }
        }
    
    def _setup_database(self):
        """Setup database for sensor data"""
        Base = declarative_base()
        
        class SensorReading(Base):
            __tablename__ = 'sensor_readings'
            
            id = Column(Integer, primary_key=True)
            sensor_id = Column(String(50))
            sensor_type = Column(String(50))
            latitude = Column(Float)
            longitude = Column(Float)
            timestamp = Column(DateTime)
            value = Column(Float)
            unit = Column(String(20))
            quality_score = Column(Float)
            meta_data = Column(Text)  # Renamed from 'metadata' to avoid SQLAlchemy reserved keyword
        
        Base.metadata.create_all(self.db_engine)
        self.Session = sessionmaker(bind=self.db_engine)
    
    async def start_sensor_monitoring(self):
        """Start monitoring all IoT sensors"""
        tasks = []
        
        for sensor_type, config in self.sensor_types.items():
            for i, location in enumerate(config['locations']):
                sensor_id = f"{sensor_type}_{i:03d}"
                task = asyncio.create_task(
                    self._monitor_sensor(sensor_id, sensor_type, location, config)
                )
                tasks.append(task)
        
        logger.info(f"🚀 Started monitoring {len(tasks)} IoT sensors")
        
        # Run all sensor monitoring tasks
        await asyncio.gather(*tasks)
    
    async def _monitor_sensor(self, sensor_id: str, sensor_type: str, 
                            location: Tuple[float, float], config: Dict[str, Any]):
        """Monitor individual IoT sensor"""
        while True:
            try:
                # Simulate sensor reading
                reading = await self._simulate_sensor_reading(sensor_id, sensor_type, location)
                
                # Store reading
                self.sensor_readings.append(reading)
                self._store_reading_to_db(reading)
                
                # Wait for next reading
                await asyncio.sleep(config['update_frequency'])
                
            except Exception as e:
                logger.error(f"Error monitoring sensor {sensor_id}: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _simulate_sensor_reading(self, sensor_id: str, sensor_type: str, 
                                     location: Tuple[float, float]) -> IoTSensorReading:
        """Simulate IoT sensor reading (in production, this would be real sensor data)"""
        
        # Base patterns for different sensor types
        base_patterns = {
            'traffic_flow': {
                'base_value': 1000,
                'daily_pattern': True,
                'weekly_pattern': True,
                'noise_level': 0.2,
                'unit': 'vehicles/hour'
            },
            'energy_consumption': {
                'base_value': 500,
                'daily_pattern': True,
                'weekly_pattern': True,
                'noise_level': 0.15,
                'unit': 'MWh'
            },
            'shipping_activity': {
                'base_value': 50,
                'daily_pattern': False,
                'weekly_pattern': True,
                'noise_level': 0.3,
                'unit': 'ships/day'
            },
            'retail_footfall': {
                'base_value': 5000,
                'daily_pattern': True,
                'weekly_pattern': True,
                'noise_level': 0.25,
                'unit': 'people/hour'
            },
            'industrial_emissions': {
                'base_value': 100,
                'daily_pattern': True,
                'weekly_pattern': False,
                'noise_level': 0.1,
                'unit': 'ppm'
            }
        }
        
        pattern = base_patterns.get(sensor_type, base_patterns['traffic_flow'])
        
        # Generate realistic sensor value
        now = datetime.now()
        base_value = pattern['base_value']
        
        # Daily pattern (business hours effect)
        if pattern['daily_pattern']:
            hour_factor = 0.5 + 0.5 * math.sin((now.hour - 6) * math.pi / 12)
            base_value *= hour_factor
        
        # Weekly pattern (weekday vs weekend)
        if pattern['weekly_pattern']:
            weekday_factor = 0.8 if now.weekday() >= 5 else 1.0  # Weekend reduction
            base_value *= weekday_factor
        
        # Add economic cycle influence
        economic_cycle = 1 + 0.1 * math.sin(now.timestamp() / (86400 * 30))  # Monthly cycle
        base_value *= economic_cycle
        
        # Add noise
        noise = random.gauss(0, pattern['noise_level'])
        final_value = max(0, base_value * (1 + noise))
        
        # Quality score (simulate sensor reliability)
        quality_score = random.uniform(0.8, 1.0)
        
        # Metadata
        metadata = {
            'weather_condition': random.choice(['clear', 'cloudy', 'rainy']),
            'temperature': random.uniform(-10, 35),
            'local_events': random.choice([None, 'festival', 'construction', 'holiday'])
        }
        
        return IoTSensorReading(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            location=location,
            timestamp=now,
            value=final_value,
            unit=pattern['unit'],
            quality_score=quality_score,
            metadata=metadata
        )
    
    def _store_reading_to_db(self, reading: IoTSensorReading):
        """Store sensor reading to database"""
        session = self.Session()
        
        try:
            # Create database record (using the class defined in _setup_database)
            # This is a simplified version - in production, use proper ORM
            query = """
                INSERT INTO sensor_readings 
                (sensor_id, sensor_type, latitude, longitude, timestamp, value, unit, quality_score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            # Use raw SQL for simplicity
            session.execute(query, (
                reading.sensor_id,
                reading.sensor_type,
                reading.location[0],
                reading.location[1],
                reading.timestamp,
                reading.value,
                reading.unit,
                reading.quality_score,
                json.dumps(reading.metadata)
            ))
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Error storing sensor reading: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_recent_readings(self, sensor_type: str = None, hours: int = 24) -> List[IoTSensorReading]:
        """Get recent sensor readings"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if sensor_type:
            readings = [r for r in self.sensor_readings 
                       if r.sensor_type == sensor_type and r.timestamp >= cutoff_time]
        else:
            readings = [r for r in self.sensor_readings if r.timestamp >= cutoff_time]
        
        return readings
    
    def analyze_sensor_trends(self, sensor_type: str, days: int = 7) -> Dict[str, Any]:
        """Analyze trends in sensor data"""
        readings = self.get_recent_readings(sensor_type, hours=days*24)
        
        if not readings:
            return {'error': 'No data available'}
        
        # Extract values and timestamps
        values = [r.value for r in readings]
        timestamps = [r.timestamp for r in readings]
        
        # Calculate trend
        if len(values) > 1:
            # Simple linear trend
            x = np.arange(len(values))
            trend_slope = np.polyfit(x, values, 1)[0]
            
            # Calculate statistics
            current_value = values[-1]
            avg_value = np.mean(values)
            std_value = np.std(values)
            
            # Trend direction
            if trend_slope > std_value * 0.1:
                trend_direction = 'increasing'
            elif trend_slope < -std_value * 0.1:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            # Volatility
            volatility = std_value / avg_value if avg_value > 0 else 0
            
            return {
                'sensor_type': sensor_type,
                'current_value': current_value,
                'average_value': avg_value,
                'trend_slope': trend_slope,
                'trend_direction': trend_direction,
                'volatility': volatility,
                'data_points': len(values),
                'time_range_hours': days * 24
            }
        
        return {'error': 'Insufficient data for trend analysis'}

class SatelliteImageryAnalyzer:
    """Analyzes satellite imagery for economic activity"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analysis_cache = {}
        self.analysis_history = deque(maxlen=1000)
        
        # Initialize analysis models
        self._initialize_analysis_models()
        
        logger.info("🛰️ Satellite Imagery Analyzer initialized")
    
    def _initialize_analysis_models(self):
        """Initialize ML models for satellite analysis"""
        # In a real implementation, these would be trained models
        self.models = {
            'port_activity': self._create_port_activity_model(),
            'industrial_activity': self._create_industrial_activity_model(),
            'construction_activity': self._create_construction_model(),
            'agricultural_activity': self._create_agricultural_model(),
            'urban_development': self._create_urban_development_model()
        }
    
    def _create_port_activity_model(self):
        """Create model for analyzing port activity"""
        # Simplified model - in production, use trained CNN
        return {
            'type': 'port_activity',
            'features': ['ship_count', 'container_density', 'crane_activity'],
            'baseline_activity': 0.5
        }
    
    def _create_industrial_activity_model(self):
        """Create model for analyzing industrial activity"""
        return {
            'type': 'industrial_activity',
            'features': ['emission_levels', 'heat_signatures', 'vehicle_activity'],
            'baseline_activity': 0.6
        }
    
    def _create_construction_model(self):
        """Create model for analyzing construction activity"""
        return {
            'type': 'construction_activity',
            'features': ['construction_sites', 'equipment_count', 'material_stockpiles'],
            'baseline_activity': 0.3
        }
    
    def _create_agricultural_model(self):
        """Create model for analyzing agricultural activity"""
        return {
            'type': 'agricultural_activity',
            'features': ['crop_health', 'harvest_activity', 'irrigation_patterns'],
            'baseline_activity': 0.4
        }
    
    def _create_urban_development_model(self):
        """Create model for analyzing urban development"""
        return {
            'type': 'urban_development',
            'features': ['new_buildings', 'infrastructure_projects', 'land_use_changes'],
            'baseline_activity': 0.2
        }
    
    async def analyze_economic_activity(self, location: Tuple[float, float], 
                                      analysis_type: str) -> SatelliteAnalysis:
        """Analyze economic activity from satellite imagery"""
        
        # In a real implementation, this would fetch and analyze actual satellite images
        # For now, we'll simulate the analysis
        
        if analysis_type not in self.models:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        model = self.models[analysis_type]
        
        # Simulate satellite image analysis
        analysis_result = await self._simulate_satellite_analysis(location, model)
        
        # Store in history
        self.analysis_history.append(analysis_result)
        
        return analysis_result
    
    async def _simulate_satellite_analysis(self, location: Tuple[float, float], 
                                         model: Dict[str, Any]) -> SatelliteAnalysis:
        """Simulate satellite imagery analysis"""
        
        # Simulate processing time
        await asyncio.sleep(random.uniform(1, 3))
        
        # Generate realistic activity level based on location and time
        base_activity = model['baseline_activity']
        
        # Location-based adjustments
        lat, lon = location
        
        # Major economic centers have higher activity
        major_centers = [
            (40.7128, -74.0060),  # New York
            (51.5074, -0.1278),   # London
            (35.6762, 139.6503),  # Tokyo
            (31.2304, 121.4737),  # Shanghai
        ]
        
        min_distance = min(
            math.sqrt((lat - center[0])**2 + (lon - center[1])**2)
            for center in major_centers
        )
        
        # Closer to major centers = higher activity
        location_factor = max(0.3, 1 - min_distance / 10)
        
        # Time-based patterns
        now = datetime.now()
        time_factor = 0.8 + 0.4 * math.sin((now.hour - 12) * math.pi / 12)
        
        # Economic cycle influence
        economic_cycle = 1 + 0.2 * math.sin(now.timestamp() / (86400 * 90))  # Quarterly cycle
        
        # Calculate activity level
        activity_level = base_activity * location_factor * time_factor * economic_cycle
        activity_level = max(0, min(1, activity_level + random.gauss(0, 0.1)))
        
        # Calculate change from previous (simulate)
        previous_activity = self.analysis_cache.get(f"{location}_{model['type']}", activity_level)
        change_from_previous = activity_level - previous_activity
        
        # Update cache
        self.analysis_cache[f"{location}_{model['type']}"] = activity_level
        
        # Confidence based on image quality and analysis complexity
        confidence = random.uniform(0.7, 0.95)
        
        # Image metadata
        image_metadata = {
            'satellite': random.choice(['Landsat-8', 'Sentinel-2', 'WorldView-3']),
            'resolution': random.choice(['10m', '30m', '50cm']),
            'cloud_cover': random.uniform(0, 30),
            'acquisition_time': now.isoformat(),
            'processing_algorithm': f"{model['type']}_cnn_v2.1"
        }
        
        return SatelliteAnalysis(
            location=location,
            analysis_type=model['type'],
            activity_level=activity_level,
            change_from_previous=change_from_previous,
            confidence=confidence,
            timestamp=now,
            image_metadata=image_metadata
        )
    
    async def batch_analyze_locations(self, locations: List[Tuple[float, float]], 
                                    analysis_type: str) -> List[SatelliteAnalysis]:
        """Analyze multiple locations in batch"""
        
        tasks = []
        for location in locations:
            task = asyncio.create_task(
                self.analyze_economic_activity(location, analysis_type)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_regional_activity_summary(self, region: str, analysis_type: str) -> Dict[str, Any]:
        """Get regional activity summary"""
        
        # Filter analyses by region and type
        relevant_analyses = [
            analysis for analysis in self.analysis_history
            if analysis.analysis_type == analysis_type
            and self._is_in_region(analysis.location, region)
        ]
        
        if not relevant_analyses:
            return {'error': 'No data available for region'}
        
        # Calculate summary statistics
        activity_levels = [a.activity_level for a in relevant_analyses]
        changes = [a.change_from_previous for a in relevant_analyses]
        
        return {
            'region': region,
            'analysis_type': analysis_type,
            'average_activity': np.mean(activity_levels),
            'activity_std': np.std(activity_levels),
            'average_change': np.mean(changes),
            'locations_analyzed': len(relevant_analyses),
            'last_updated': max(a.timestamp for a in relevant_analyses).isoformat(),
            'activity_trend': 'increasing' if np.mean(changes) > 0.01 else 'decreasing' if np.mean(changes) < -0.01 else 'stable'
        }
    
    def _is_in_region(self, location: Tuple[float, float], region: str) -> bool:
        """Check if location is in specified region"""
        lat, lon = location
        
        # Simplified region definitions
        regions = {
            'north_america': (25, 70, -170, -50),
            'europe': (35, 75, -15, 40),
            'asia': (10, 70, 60, 180),
            'oceania': (-50, 0, 110, 180)
        }
        
        if region.lower() in regions:
            min_lat, max_lat, min_lon, max_lon = regions[region.lower()]
            return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon
        
        return False

class SocialSentimentAnalyzer:
    """Analyzes social media sentiment for economic indicators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.sentiment_history = deque(maxlen=10000)
        
        # Initialize social media APIs
        self._initialize_social_apis()
        
        # Economic keywords to track
        self.economic_keywords = [
            'economy', 'inflation', 'recession', 'growth', 'unemployment',
            'interest rates', 'federal reserve', 'central bank', 'GDP',
            'stock market', 'forex', 'currency', 'trade war', 'tariffs',
            'economic policy', 'fiscal policy', 'monetary policy'
        ]
        
        logger.info("📱 Social Sentiment Analyzer initialized")
    
    def _initialize_social_apis(self):
        """Initialize social media API connections"""
        # Twitter API (using tweepy)
        try:
            twitter_config = self.config.get('twitter', {})
            if twitter_config.get('api_key'):
                auth = tweepy.OAuthHandler(
                    twitter_config['api_key'],
                    twitter_config['api_secret']
                )
                auth.set_access_token(
                    twitter_config['access_token'],
                    twitter_config['access_token_secret']
                )
                self.twitter_api = tweepy.API(auth)
            else:
                self.twitter_api = None
        except Exception as e:
            logger.warning(f"Twitter API initialization failed: {e}")
            self.twitter_api = None
        
        # Reddit API (using praw)
        try:
            reddit_config = self.config.get('reddit', {})
            if reddit_config.get('client_id'):
                self.reddit_api = praw.Reddit(
                    client_id=reddit_config['client_id'],
                    client_secret=reddit_config['client_secret'],
                    user_agent='NeuroEconomicEngine/1.0'
                )
            else:
                self.reddit_api = None
        except Exception as e:
            logger.warning(f"Reddit API initialization failed: {e}")
            self.reddit_api = None
    
    async def analyze_social_sentiment(self, platform: str = 'all') -> List[SocialSentiment]:
        """Analyze social media sentiment"""
        
        sentiments = []
        
        if platform in ['twitter', 'all'] and self.twitter_api:
            twitter_sentiments = await self._analyze_twitter_sentiment()
            sentiments.extend(twitter_sentiments)
        
        if platform in ['reddit', 'all'] and self.reddit_api:
            reddit_sentiments = await self._analyze_reddit_sentiment()
            sentiments.extend(reddit_sentiments)
        
        # Store in history
        self.sentiment_history.extend(sentiments)
        
        return sentiments
    
    async def _analyze_twitter_sentiment(self) -> List[SocialSentiment]:
        """Analyze Twitter sentiment"""
        sentiments = []
        
        try:
            # Search for economic-related tweets
            for keyword in self.economic_keywords[:5]:  # Limit to avoid rate limits
                tweets = tweepy.Cursor(
                    self.twitter_api.search_tweets,
                    q=keyword,
                    lang='en',
                    result_type='recent',
                    tweet_mode='extended'
                ).items(20)  # Limit to 20 tweets per keyword
                
                for tweet in tweets:
                    # Analyze sentiment
                    sentiment_result = self.sentiment_pipeline(tweet.full_text)[0]
                    
                    # Convert to numerical score
                    if sentiment_result['label'] == 'POSITIVE':
                        sentiment_score = sentiment_result['score']
                    else:
                        sentiment_score = -sentiment_result['score']
                    
                    # Extract engagement metrics
                    engagement_metrics = {
                        'retweets': tweet.retweet_count,
                        'likes': tweet.favorite_count,
                        'replies': tweet.reply_count if hasattr(tweet, 'reply_count') else 0
                    }
                    
                    sentiment = SocialSentiment(
                        platform='twitter',
                        content=tweet.full_text,
                        sentiment_score=sentiment_score,
                        confidence=sentiment_result['score'],
                        engagement_metrics=engagement_metrics,
                        location=tweet.user.location if tweet.user.location else None,
                        timestamp=tweet.created_at,
                        keywords=[keyword]
                    )
                    
                    sentiments.append(sentiment)
                
                # Rate limiting
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {e}")
        
        return sentiments
    
    async def _analyze_reddit_sentiment(self) -> List[SocialSentiment]:
        """Analyze Reddit sentiment"""
        sentiments = []
        
        try:
            # Economic subreddits
            economic_subreddits = ['economics', 'investing', 'stocks', 'forex', 'economy']
            
            for subreddit_name in economic_subreddits:
                subreddit = self.reddit_api.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=10):
                    # Analyze post title and content
                    content = f"{post.title} {post.selftext}"
                    
                    if len(content.strip()) > 10:  # Skip very short content
                        sentiment_result = self.sentiment_pipeline(content[:512])[0]  # Limit length
                        
                        # Convert to numerical score
                        if sentiment_result['label'] == 'POSITIVE':
                            sentiment_score = sentiment_result['score']
                        else:
                            sentiment_score = -sentiment_result['score']
                        
                        # Extract engagement metrics
                        engagement_metrics = {
                            'upvotes': post.score,
                            'comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio
                        }
                        
                        sentiment = SocialSentiment(
                            platform='reddit',
                            content=content,
                            sentiment_score=sentiment_score,
                            confidence=sentiment_result['score'],
                            engagement_metrics=engagement_metrics,
                            location=None,  # Reddit doesn't provide location
                            timestamp=datetime.fromtimestamp(post.created_utc),
                            keywords=self._extract_keywords(content)
                        )
                        
                        sentiments.append(sentiment)
                
                # Rate limiting
                await asyncio.sleep(2)
        
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment: {e}")
        
        return sentiments
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract economic keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.economic_keywords:
            if keyword.lower() in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def calculate_sentiment_index(self, hours: int = 24) -> Dict[str, Any]:
        """Calculate overall sentiment index"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_sentiments = [
            s for s in self.sentiment_history
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_sentiments:
            return {'error': 'No recent sentiment data'}
        
        # Calculate weighted sentiment score
        total_weight = 0
        weighted_sentiment = 0
        
        for sentiment in recent_sentiments:
            # Weight by engagement and confidence
            engagement_weight = self._calculate_engagement_weight(sentiment.engagement_metrics)
            weight = sentiment.confidence * engagement_weight
            
            weighted_sentiment += sentiment.sentiment_score * weight
            total_weight += weight
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Calculate sentiment by platform
        platform_sentiments = {}
        for platform in ['twitter', 'reddit']:
            platform_data = [s for s in recent_sentiments if s.platform == platform]
            if platform_data:
                platform_sentiments[platform] = np.mean([s.sentiment_score for s in platform_data])
        
        # Calculate sentiment trend
        if len(recent_sentiments) >= 10:
            # Split into two halves and compare
            mid_point = len(recent_sentiments) // 2
            first_half = recent_sentiments[:mid_point]
            second_half = recent_sentiments[mid_point:]
            
            first_avg = np.mean([s.sentiment_score for s in first_half])
            second_avg = np.mean([s.sentiment_score for s in second_half])
            
            sentiment_trend = 'improving' if second_avg > first_avg + 0.1 else 'declining' if second_avg < first_avg - 0.1 else 'stable'
        else:
            sentiment_trend = 'insufficient_data'
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_grade': self._grade_sentiment(overall_sentiment),
            'platform_breakdown': platform_sentiments,
            'sentiment_trend': sentiment_trend,
            'data_points': len(recent_sentiments),
            'time_range_hours': hours,
            'confidence': np.mean([s.confidence for s in recent_sentiments])
        }
    
    def _calculate_engagement_weight(self, engagement_metrics: Dict[str, Any]) -> float:
        """Calculate engagement weight for sentiment"""
        # Normalize engagement metrics
        if 'retweets' in engagement_metrics:  # Twitter
            weight = 1 + math.log(1 + engagement_metrics.get('retweets', 0)) * 0.1
            weight += math.log(1 + engagement_metrics.get('likes', 0)) * 0.05
        elif 'upvotes' in engagement_metrics:  # Reddit
            weight = 1 + math.log(1 + engagement_metrics.get('upvotes', 0)) * 0.1
            weight += math.log(1 + engagement_metrics.get('comments', 0)) * 0.05
        else:
            weight = 1.0
        
        return min(weight, 3.0)  # Cap at 3x weight
    
    def _grade_sentiment(self, sentiment_score: float) -> str:
        """Grade sentiment score"""
        if sentiment_score >= 0.6:
            return 'Very Positive'
        elif sentiment_score >= 0.2:
            return 'Positive'
        elif sentiment_score >= -0.2:
            return 'Neutral'
        elif sentiment_score >= -0.6:
            return 'Negative'
        else:
            return 'Very Negative'

class EconomicDataCollector:
    """Collects economic data from various sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.economic_data = deque(maxlen=10000)
        
        # Initialize data sources
        self._initialize_data_sources()
        
        logger.info("📊 Economic Data Collector initialized")
    
    def _initialize_data_sources(self):
        """Initialize economic data sources"""
        self.data_sources = {
            'fred': {
                'base_url': 'https://api.stlouisfed.org/fred/series/observations',
                'api_key': self.config.get('fred_api_key'),
                'indicators': ['GDP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS']
            },
            'alpha_vantage': {
                'base_url': 'https://www.alphavantage.co/query',
                'api_key': self.config.get('alpha_vantage_api_key'),
                'indicators': ['REAL_GDP', 'INFLATION', 'UNEMPLOYMENT']
            },
            'yahoo_finance': {
                'indicators': ['^VIX', '^TNX', 'DXY', 'GC=F', 'CL=F']  # VIX, 10Y Treasury, Dollar Index, Gold, Oil
            }
        }
    
    async def collect_economic_indicators(self) -> List[EconomicIndicator]:
        """Collect economic indicators from all sources"""
        
        indicators = []
        
        # Collect from Yahoo Finance (most reliable)
        yahoo_indicators = await self._collect_yahoo_finance_data()
        indicators.extend(yahoo_indicators)
        
        # Collect from other sources if APIs are available
        if self.data_sources['fred']['api_key']:
            fred_indicators = await self._collect_fred_data()
            indicators.extend(fred_indicators)
        
        if self.data_sources['alpha_vantage']['api_key']:
            av_indicators = await self._collect_alpha_vantage_data()
            indicators.extend(av_indicators)
        
        # Store in history
        self.economic_data.extend(indicators)
        
        return indicators
    
    async def _collect_yahoo_finance_data(self) -> List[EconomicIndicator]:
        """Collect data from Yahoo Finance"""
        indicators = []
        
        try:
            symbols = self.data_sources['yahoo_finance']['indicators']
            
            for symbol in symbols:
                # Get recent data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current_value = hist['Close'].iloc[-1]
                    previous_value = hist['Close'].iloc[-2] if len(hist) > 1 else current_value
                    
                    # Map symbol to indicator name
                    indicator_names = {
                        '^VIX': 'VIX_Volatility_Index',
                        '^TNX': '10Y_Treasury_Yield',
                        'DXY': 'US_Dollar_Index',
                        'GC=F': 'Gold_Price',
                        'CL=F': 'Oil_Price'
                    }
                    
                    indicator_name = indicator_names.get(symbol, symbol)
                    
                    # Determine impact
                    change_pct = (current_value - previous_value) / previous_value * 100
                    if abs(change_pct) > 2:
                        impact = 'positive' if change_pct > 0 else 'negative'
                        importance = 'high'
                    elif abs(change_pct) > 1:
                        impact = 'positive' if change_pct > 0 else 'negative'
                        importance = 'medium'
                    else:
                        impact = 'neutral'
                        importance = 'low'
                    
                    indicator = EconomicIndicator(
                        indicator_name=indicator_name,
                        country='US',
                        value=current_value,
                        previous_value=previous_value,
                        forecast_value=current_value,  # No forecast from Yahoo
                        timestamp=datetime.now(),
                        importance=importance,
                        impact=impact,
                        source='yahoo_finance'
                    )
                    
                    indicators.append(indicator)
                
                # Rate limiting
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance data: {e}")
        
        return indicators
    
    async def _collect_fred_data(self) -> List[EconomicIndicator]:
        """Collect data from FRED (Federal Reserve Economic Data)"""
        indicators = []
        
        # Simplified implementation - in production, use proper FRED API
        # This would require actual API calls to FRED
        
        return indicators
    
    async def _collect_alpha_vantage_data(self) -> List[EconomicIndicator]:
        """Collect data from Alpha Vantage"""
        indicators = []
        
        # Simplified implementation - in production, use proper Alpha Vantage API
        # This would require actual API calls to Alpha Vantage
        
        return indicators
    
    def get_economic_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get economic indicators summary"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_indicators = [
            ind for ind in self.economic_data
            if ind.timestamp >= cutoff_time
        ]
        
        if not recent_indicators:
            return {'error': 'No recent economic data'}
        
        # Group by importance
        high_importance = [ind for ind in recent_indicators if ind.importance == 'high']
        medium_importance = [ind for ind in recent_indicators if ind.importance == 'medium']
        low_importance = [ind for ind in recent_indicators if ind.importance == 'low']
        
        # Calculate impact scores
        positive_impact = len([ind for ind in recent_indicators if ind.impact == 'positive'])
        negative_impact = len([ind for ind in recent_indicators if ind.impact == 'negative'])
        neutral_impact = len([ind for ind in recent_indicators if ind.impact == 'neutral'])
        
        total_indicators = len(recent_indicators)
        
        # Overall economic sentiment
        if total_indicators > 0:
            economic_sentiment = (positive_impact - negative_impact) / total_indicators
        else:
            economic_sentiment = 0
        
        return {
            'total_indicators': total_indicators,
            'high_importance_count': len(high_importance),
            'medium_importance_count': len(medium_importance),
            'low_importance_count': len(low_importance),
            'positive_impact_count': positive_impact,
            'negative_impact_count': negative_impact,
            'neutral_impact_count': neutral_impact,
            'economic_sentiment': economic_sentiment,
            'sentiment_grade': self._grade_economic_sentiment(economic_sentiment),
            'recent_indicators': [asdict(ind) for ind in recent_indicators[-10:]]  # Last 10
        }
    
    def _grade_economic_sentiment(self, sentiment: float) -> str:
        """Grade economic sentiment"""
        if sentiment >= 0.3:
            return 'Strong Positive'
        elif sentiment >= 0.1:
            return 'Positive'
        elif sentiment >= -0.1:
            return 'Neutral'
        elif sentiment >= -0.3:
            return 'Negative'
        else:
            return 'Strong Negative'

class NeuroEconomicSentimentNN(nn.Module):
    """Neural network for processing economic sentiment"""
    
    def __init__(self, input_size: int = 100, hidden_sizes: List[int] = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layers for different predictions
        self.feature_extractor = nn.Sequential(*layers)
        
        self.economic_sentiment_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.trend_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Up, Down, Sideways
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        sentiment = self.economic_sentiment_head(features)
        volatility = self.volatility_head(features)
        trend = self.trend_head(features)
        
        return {
            'sentiment': sentiment,
            'volatility': volatility,
            'trend': trend,
            'features': features
        }

class EconomicPredictionModel:
    """Model for predicting economic changes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural network
        self.model = NeuroEconomicSentimentNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = StandardScaler()
        
        # Training data buffer
        self.training_data = deque(maxlen=10000)
        
        logger.info("🧠 Economic Prediction Model initialized")
    
    def prepare_features(self, iot_data: List[IoTSensorReading], 
                        satellite_data: List[SatelliteAnalysis],
                        social_sentiment: List[SocialSentiment],
                        economic_indicators: List[EconomicIndicator]) -> np.ndarray:
        """Prepare features for neural network"""
        
        features = []
        
        # IoT sensor features
        if iot_data:
            sensor_features = self._extract_iot_features(iot_data)
            features.extend(sensor_features)
        else:
            features.extend([0] * 20)  # Placeholder
        
        # Satellite imagery features
        if satellite_data:
            satellite_features = self._extract_satellite_features(satellite_data)
            features.extend(satellite_features)
        else:
            features.extend([0] * 15)  # Placeholder
        
        # Social sentiment features
        if social_sentiment:
            sentiment_features = self._extract_sentiment_features(social_sentiment)
            features.extend(sentiment_features)
        else:
            features.extend([0] * 10)  # Placeholder
        
        # Economic indicator features
        if economic_indicators:
            economic_features = self._extract_economic_features(economic_indicators)
            features.extend(economic_features)
        else:
            features.extend([0] * 15)  # Placeholder
        
        # Time-based features
        time_features = self._extract_time_features()
        features.extend(time_features)
        
        # Ensure we have exactly 100 features
        while len(features) < 100:
            features.append(0)
        
        return np.array(features[:100])
    
    def _extract_iot_features(self, iot_data: List[IoTSensorReading]) -> List[float]:
        """Extract features from IoT sensor data"""
        features = []
        
        # Group by sensor type
        sensor_groups = defaultdict(list)
        for reading in iot_data:
            sensor_groups[reading.sensor_type].append(reading.value)
        
        # Extract statistics for each sensor type
        sensor_types = ['traffic_flow', 'energy_consumption', 'shipping_activity', 'retail_footfall']
        
        for sensor_type in sensor_types:
            values = sensor_groups.get(sensor_type, [0])
            
            features.extend([
                np.mean(values),
                np.std(values),
                np.max(values),
                np.min(values),
                len(values)
            ])
        
        return features
    
    def _extract_satellite_features(self, satellite_data: List[SatelliteAnalysis]) -> List[float]:
        """Extract features from satellite analysis"""
        features = []
        
        # Group by analysis type
        analysis_groups = defaultdict(list)
        for analysis in satellite_data:
            analysis_groups[analysis.analysis_type].append(analysis.activity_level)
        
        # Extract statistics for each analysis type
        analysis_types = ['port_activity', 'industrial_activity', 'construction_activity']
        
        for analysis_type in analysis_types:
            values = analysis_groups.get(analysis_type, [0])
            
            features.extend([
                np.mean(values),
                np.std(values),
                np.max(values),
                np.min(values),
                len(values)
            ])
        
        return features
    
    def _extract_sentiment_features(self, social_sentiment: List[SocialSentiment]) -> List[float]:
        """Extract features from social sentiment"""
        features = []
        
        if social_sentiment:
            sentiment_scores = [s.sentiment_score for s in social_sentiment]
            confidence_scores = [s.confidence for s in social_sentiment]
            
            features.extend([
                np.mean(sentiment_scores),
                np.std(sentiment_scores),
                np.max(sentiment_scores),
                np.min(sentiment_scores),
                np.mean(confidence_scores),
                len(sentiment_scores)
            ])
            
            # Platform breakdown
            twitter_sentiments = [s.sentiment_score for s in social_sentiment if s.platform == 'twitter']
            reddit_sentiments = [s.sentiment_score for s in social_sentiment if s.platform == 'reddit']
            
            features.extend([
                np.mean(twitter_sentiments) if twitter_sentiments else 0,
                np.mean(reddit_sentiments) if reddit_sentiments else 0,
                len(twitter_sentiments),
                len(reddit_sentiments)
            ])
        else:
            features.extend([0] * 10)
        
        return features
    
    def _extract_economic_features(self, economic_indicators: List[EconomicIndicator]) -> List[float]:
        """Extract features from economic indicators"""
        features = []
        
        if economic_indicators:
            values = [ind.value for ind in economic_indicators]
            changes = [(ind.value - ind.previous_value) / ind.previous_value 
                      for ind in economic_indicators if ind.previous_value != 0]
            
            features.extend([
                np.mean(values),
                np.std(values),
                np.mean(changes) if changes else 0,
                np.std(changes) if changes else 0,
                len(economic_indicators)
            ])
            
            # Impact breakdown
            positive_count = len([ind for ind in economic_indicators if ind.impact == 'positive'])
            negative_count = len([ind for ind in economic_indicators if ind.impact == 'negative'])
            high_importance_count = len([ind for ind in economic_indicators if ind.importance == 'high'])
            
            features.extend([
                positive_count,
                negative_count,
                high_importance_count,
                positive_count / len(economic_indicators),
                negative_count / len(economic_indicators)
            ])
        else:
            features.extend([0] * 15)
        
        return features
    
    def _extract_time_features(self) -> List[float]:
        """Extract time-based features"""
        now = datetime.now()
        
        features = [
            now.hour / 24,  # Hour of day
            now.weekday() / 7,  # Day of week
            now.month / 12,  # Month of year
            (now.timestamp() % (86400 * 7)) / (86400 * 7),  # Week cycle
            (now.timestamp() % (86400 * 30)) / (86400 * 30),  # Month cycle
            (now.timestamp() % (86400 * 365)) / (86400 * 365),  # Year cycle
            math.sin(2 * math.pi * now.hour / 24),  # Daily sine
            math.cos(2 * math.pi * now.hour / 24),  # Daily cosine
            math.sin(2 * math.pi * now.weekday() / 7),  # Weekly sine
            math.cos(2 * math.pi * now.weekday() / 7),  # Weekly cosine
        ]
        
        return features
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make economic prediction"""
        
        # Normalize features
        features_normalized = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features_normalized).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features_tensor)
        
        # Extract predictions
        sentiment = outputs['sentiment'].cpu().numpy()[0][0]
        volatility = outputs['volatility'].cpu().numpy()[0][0]
        trend_probs = outputs['trend'].cpu().numpy()[0]
        
        # Interpret trend
        trend_labels = ['up', 'down', 'sideways']
        predicted_trend = trend_labels[np.argmax(trend_probs)]
        trend_confidence = np.max(trend_probs)
        
        return {
            'economic_sentiment': float(sentiment),
            'predicted_volatility': float(volatility),
            'predicted_trend': predicted_trend,
            'trend_confidence': float(trend_confidence),
            'trend_probabilities': {
                'up': float(trend_probs[0]),
                'down': float(trend_probs[1]),
                'sideways': float(trend_probs[2])
            }
        }

class NeuroEconomicEngine:
    """Main Neuro-Economic Sentiment Engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.iot_manager = IoTSensorManager(config.get('iot', {}))
        self.satellite_analyzer = SatelliteImageryAnalyzer(config.get('satellite', {}))
        self.sentiment_analyzer = SocialSentimentAnalyzer(config.get('social', {}))
        self.economic_collector = EconomicDataCollector(config.get('economic', {}))
        self.prediction_model = EconomicPredictionModel(config.get('prediction', {}))
        
        # Economic pulse history
        self.economic_pulse_history = deque(maxlen=1000)
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes
        
        logger.info("🧠 Neuro-Economic Sentiment Engine initialized")
    
    async def start_real_time_monitoring(self):
        """Start real-time economic monitoring"""
        
        # Start IoT sensor monitoring
        iot_task = asyncio.create_task(self.iot_manager.start_sensor_monitoring())
        
        # Start periodic analysis
        analysis_task = asyncio.create_task(self._periodic_analysis())
        
        logger.info("🚀 Real-time economic monitoring started")
        
        # Run monitoring tasks
        await asyncio.gather(iot_task, analysis_task)
    
    async def _periodic_analysis(self):
        """Perform periodic economic analysis"""
        while True:
            try:
                # Perform comprehensive analysis
                economic_pulse = await self.analyze_real_world_economic_pulse()
                
                # Store in history
                self.economic_pulse_history.append(economic_pulse)
                
                # Log significant changes
                if len(self.economic_pulse_history) > 1:
                    previous_pulse = self.economic_pulse_history[-2]
                    change = economic_pulse.overall_score - previous_pulse.overall_score
                    
                    if abs(change) > 0.1:  # Significant change threshold
                        logger.info(f"📊 Significant economic pulse change: {change:+.3f}")
                
                # Wait for next analysis
                await asyncio.sleep(self.config.get('analysis_interval', 1800))  # 30 minutes
                
            except Exception as e:
                logger.error(f"Error in periodic analysis: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def analyze_real_world_economic_pulse(self) -> EconomicPulse:
        """Analyze real-world economic pulse"""
        
        # Check cache
        cache_key = 'economic_pulse'
        if cache_key in self.analysis_cache:
            cache_time, cached_result = self.analysis_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cached_result
        
        logger.info("🔍 Analyzing real-world economic pulse...")
        
        # Collect data from all sources
        tasks = [
            self._collect_iot_data(),
            self._collect_satellite_data(),
            self._collect_social_sentiment(),
            self._collect_economic_indicators()
        ]
        
        iot_data, satellite_data, social_sentiment, economic_indicators = await asyncio.gather(*tasks)
        
        # Prepare features for prediction model
        features = self.prediction_model.prepare_features(
            iot_data, satellite_data, social_sentiment, economic_indicators
        )
        
        # Make prediction
        prediction = self.prediction_model.predict(features)
        
        # Calculate component scores
        components = {
            'iot_activity': self._calculate_iot_score(iot_data),
            'satellite_activity': self._calculate_satellite_score(satellite_data),
            'social_sentiment': self._calculate_social_score(social_sentiment),
            'economic_indicators': self._calculate_economic_score(economic_indicators),
            'ai_prediction': prediction['economic_sentiment']
        }
        
        # Calculate overall score
        weights = {
            'iot_activity': 0.25,
            'satellite_activity': 0.20,
            'social_sentiment': 0.15,
            'economic_indicators': 0.25,
            'ai_prediction': 0.15
        }
        
        overall_score = sum(components[key] * weights[key] for key in components.keys())
        
        # Calculate confidence
        confidence = self._calculate_overall_confidence(
            iot_data, satellite_data, social_sentiment, economic_indicators
        )
        
        # Determine trend direction
        if overall_score > 0.1:
            trend_direction = 'positive'
        elif overall_score < -0.1:
            trend_direction = 'negative'
        else:
            trend_direction = 'neutral'
        
        # Calculate volatility
        volatility = prediction['predicted_volatility']
        
        # Regional and sector breakdown
        regional_breakdown = await self._calculate_regional_breakdown(satellite_data)
        sector_breakdown = self._calculate_sector_breakdown(iot_data, economic_indicators)
        
        # Create economic pulse
        economic_pulse = EconomicPulse(
            timestamp=datetime.now(),
            overall_score=overall_score,
            components=components,
            confidence=confidence,
            trend_direction=trend_direction,
            volatility=volatility,
            regional_breakdown=regional_breakdown,
            sector_breakdown=sector_breakdown
        )
        
        # Cache result
        self.analysis_cache[cache_key] = (time.time(), economic_pulse)
        
        logger.info(f"✅ Economic pulse analysis completed: {overall_score:.3f}")
        
        return economic_pulse
    
    async def _collect_iot_data(self) -> List[IoTSensorReading]:
        """Collect recent IoT sensor data"""
        return self.iot_manager.get_recent_readings(hours=1)
    
    async def _collect_satellite_data(self) -> List[SatelliteAnalysis]:
        """Collect recent satellite analysis data"""
        # Get recent analyses from cache
        recent_analyses = [
            analysis for analysis in self.satellite_analyzer.analysis_history
            if (datetime.now() - analysis.timestamp).total_seconds() < 3600  # Last hour
        ]
        
        # If not enough recent data, perform new analysis
        if len(recent_analyses) < 5:
            key_locations = [
                (40.7128, -74.0060),  # New York
                (51.5074, -0.1278),   # London
                (35.6762, 139.6503),  # Tokyo
                (31.2304, 121.4737),  # Shanghai
            ]
            
            new_analyses = await self.satellite_analyzer.batch_analyze_locations(
                key_locations, 'industrial_activity'
            )
            recent_analyses.extend(new_analyses)
        
        return recent_analyses
    
    async def _collect_social_sentiment(self) -> List[SocialSentiment]:
        """Collect recent social sentiment data"""
        return await self.sentiment_analyzer.analyze_social_sentiment()
    
    async def _collect_economic_indicators(self) -> List[EconomicIndicator]:
        """Collect recent economic indicators"""
        return await self.economic_collector.collect_economic_indicators()
    
    def _calculate_iot_score(self, iot_data: List[IoTSensorReading]) -> float:
        """Calculate IoT activity score"""
        if not iot_data:
            return 0.0
        
        # Group by sensor type and calculate normalized scores
        sensor_groups = defaultdict(list)
        for reading in iot_data:
            sensor_groups[reading.sensor_type].append(reading.value)
        
        scores = []
        for sensor_type, values in sensor_groups.items():
            if values:
                # Normalize based on historical data (simplified)
                avg_value = np.mean(values)
                baseline = self.iot_manager.sensor_types[sensor_type]['baseline_activity'] if sensor_type in self.iot_manager.sensor_types else 0.5
                
                # Score relative to baseline
                score = (avg_value / baseline - 1) if baseline > 0 else 0
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_satellite_score(self, satellite_data: List[SatelliteAnalysis]) -> float:
        """Calculate satellite activity score"""
        if not satellite_data:
            return 0.0
        
        # Calculate weighted average of activity levels
        total_weight = 0
        weighted_score = 0
        
        for analysis in satellite_data:
            weight = analysis.confidence
            score = (analysis.activity_level - 0.5) * 2  # Normalize to [-1, 1]
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_social_score(self, social_sentiment: List[SocialSentiment]) -> float:
        """Calculate social sentiment score"""
        if not social_sentiment:
            return 0.0
        
        # Calculate weighted average sentiment
        total_weight = 0
        weighted_sentiment = 0
        
        for sentiment in social_sentiment:
            # Weight by confidence and engagement
            engagement_weight = self.sentiment_analyzer._calculate_engagement_weight(
                sentiment.engagement_metrics
            )
            weight = sentiment.confidence * engagement_weight
            
            weighted_sentiment += sentiment.sentiment_score * weight
            total_weight += weight
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _calculate_economic_score(self, economic_indicators: List[EconomicIndicator]) -> float:
        """Calculate economic indicators score"""
        if not economic_indicators:
            return 0.0
        
        # Weight by importance and calculate impact score
        importance_weights = {'high': 3, 'medium': 2, 'low': 1}
        impact_scores = {'positive': 1, 'negative': -1, 'neutral': 0}
        
        total_weight = 0
        weighted_score = 0
        
        for indicator in economic_indicators:
            weight = importance_weights.get(indicator.importance, 1)
            score = impact_scores.get(indicator.impact, 0)
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_overall_confidence(self, iot_data: List[IoTSensorReading],
                                    satellite_data: List[SatelliteAnalysis],
                                    social_sentiment: List[SocialSentiment],
                                    economic_indicators: List[EconomicIndicator]) -> float:
        """Calculate overall confidence in analysis"""
        
        confidences = []
        
        # IoT data confidence (based on quality scores)
        if iot_data:
            iot_confidence = np.mean([reading.quality_score for reading in iot_data])
            confidences.append(iot_confidence)
        
        # Satellite data confidence
        if satellite_data:
            satellite_confidence = np.mean([analysis.confidence for analysis in satellite_data])
            confidences.append(satellite_confidence)
        
        # Social sentiment confidence
        if social_sentiment:
            social_confidence = np.mean([sentiment.confidence for sentiment in social_sentiment])
            confidences.append(social_confidence)
        
        # Economic indicators confidence (assume high for official data)
        if economic_indicators:
            confidences.append(0.9)
        
        # Data availability confidence
        data_sources = len([data for data in [iot_data, satellite_data, social_sentiment, economic_indicators] if data])
        availability_confidence = data_sources / 4  # 4 total sources
        confidences.append(availability_confidence)
        
        return np.mean(confidences) if confidences else 0.5
    
    async def _calculate_regional_breakdown(self, satellite_data: List[SatelliteAnalysis]) -> Dict[str, float]:
        """Calculate regional economic activity breakdown"""
        
        regions = ['north_america', 'europe', 'asia', 'oceania']
        regional_scores = {}
        
        for region in regions:
            region_data = [
                analysis for analysis in satellite_data
                if self.satellite_analyzer._is_in_region(analysis.location, region)
            ]
            
            if region_data:
                avg_activity = np.mean([analysis.activity_level for analysis in region_data])
                regional_scores[region] = (avg_activity - 0.5) * 2  # Normalize to [-1, 1]
            else:
                regional_scores[region] = 0.0
        
        return regional_scores
    
    def _calculate_sector_breakdown(self, iot_data: List[IoTSensorReading],
                                  economic_indicators: List[EconomicIndicator]) -> Dict[str, float]:
        """Calculate sector-wise economic activity"""
        
        sectors = {
            'transportation': 0.0,
            'energy': 0.0,
            'retail': 0.0,
            'industrial': 0.0,
            'financial': 0.0
        }
        
        # Map IoT sensors to sectors
        sensor_sector_mapping = {
            'traffic_flow': 'transportation',
            'energy_consumption': 'energy',
            'retail_footfall': 'retail',
            'shipping_activity': 'transportation',
            'industrial_emissions': 'industrial'
        }
        
        # Calculate sector scores from IoT data
        for reading in iot_data:
            sector = sensor_sector_mapping.get(reading.sensor_type)
            if sector:
                # Simplified scoring
                baseline = 0.5
                score = (reading.value / 1000 - baseline) if reading.value > 0 else 0
                sectors[sector] = max(sectors[sector], score)
        
        # Incorporate economic indicators
        for indicator in economic_indicators:
            if 'employment' in indicator.indicator_name.lower():
                sectors['industrial'] += 0.1 if indicator.impact == 'positive' else -0.1
            elif 'consumer' in indicator.indicator_name.lower():
                sectors['retail'] += 0.1 if indicator.impact == 'positive' else -0.1
        
        return sectors
    
    def get_economic_pulse_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get economic pulse summary"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_pulses = [
            pulse for pulse in self.economic_pulse_history
            if pulse.timestamp >= cutoff_time
        ]
        
        if not recent_pulses:
            return {'error': 'No recent economic pulse data'}
        
        # Calculate trends
        scores = [pulse.overall_score for pulse in recent_pulses]
        
        current_score = scores[-1] if scores else 0
        avg_score = np.mean(scores)
        score_trend = 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'declining' if len(scores) > 1 and scores[-1] < scores[0] else 'stable'
        
        # Component analysis
        component_averages = {}
        if recent_pulses:
            for component in recent_pulses[0].components.keys():
                component_scores = [pulse.components[component] for pulse in recent_pulses]
                component_averages[component] = np.mean(component_scores)
        
        # Volatility analysis
        volatilities = [pulse.volatility for pulse in recent_pulses]
        avg_volatility = np.mean(volatilities) if volatilities else 0
        
        return {
            'current_economic_score': current_score,
            'average_economic_score': avg_score,
            'score_trend': score_trend,
            'component_breakdown': component_averages,
            'average_volatility': avg_volatility,
            'confidence': np.mean([pulse.confidence for pulse in recent_pulses]),
            'data_points': len(recent_pulses),
            'time_range_hours': hours,
            'last_updated': recent_pulses[-1].timestamp.isoformat() if recent_pulses else None
        }
    
    def predict_economic_changes(self, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict economic changes"""
        
        if not self.economic_pulse_history:
            return {'error': 'No historical data for prediction'}
        
        # Use recent data for prediction
        recent_pulses = list(self.economic_pulse_history)[-10:]  # Last 10 pulses
        
        if len(recent_pulses) < 3:
            return {'error': 'Insufficient data for prediction'}
        
        # Simple trend-based prediction
        scores = [pulse.overall_score for pulse in recent_pulses]
        
        # Calculate trend
        x = np.arange(len(scores))
        trend_slope = np.polyfit(x, scores, 1)[0]
        
        # Predict future score
        future_score = scores[-1] + trend_slope * (hours_ahead / 24)
        
        # Predict volatility
        volatilities = [pulse.volatility for pulse in recent_pulses]
        avg_volatility = np.mean(volatilities)
        
        # Confidence decreases with time
        prediction_confidence = max(0.3, 0.9 - (hours_ahead / 24) * 0.1)
        
        return {
            'predicted_economic_score': future_score,
            'prediction_confidence': prediction_confidence,
            'predicted_volatility': avg_volatility,
            'trend_slope': trend_slope,
            'prediction_horizon_hours': hours_ahead,
            'based_on_data_points': len(recent_pulses)
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Test neuro-economic engine
    config = {
        'iot': {
            'data_quality_threshold': 0.7,
            'database_url': 'sqlite:///test_iot.db'
        },
        'satellite': {},
        'social': {
            'twitter': {
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'access_token': 'test_token',
                'access_token_secret': 'test_token_secret'
            }
        },
        'economic': {
            'alpha_vantage_api_key': 'test_key'
        },
        'prediction': {},
        'analysis_interval': 60,  # 1 minute for testing
        'cache_ttl': 30  # 30 seconds for testing
    }
    
    neuro_engine = NeuroEconomicEngine(config)
    
    async def test_neuro_economic_engine():
        # Test economic pulse analysis
        economic_pulse = await neuro_engine.analyze_real_world_economic_pulse()
        
        print("🧠 Neuro-Economic Analysis Result:")
        print(f"Overall Score: {economic_pulse.overall_score:.3f}")
        print(f"Trend Direction: {economic_pulse.trend_direction}")
        print(f"Confidence: {economic_pulse.confidence:.3f}")
        print(f"Volatility: {economic_pulse.volatility:.3f}")
        
        print(f"\n📊 Component Breakdown:")
        for component, score in economic_pulse.components.items():
            print(f"  {component}: {score:.3f}")
        
        print(f"\n🌍 Regional Breakdown:")
        for region, score in economic_pulse.regional_breakdown.items():
            print(f"  {region}: {score:.3f}")
        
        # Test prediction
        prediction = neuro_engine.predict_economic_changes(hours_ahead=24)
        print(f"\n🔮 24-Hour Prediction:")
        print(f"Predicted Score: {prediction.get('predicted_economic_score', 'N/A')}")
        print(f"Confidence: {prediction.get('prediction_confidence', 'N/A')}")
        
        # Test summary
        summary = neuro_engine.get_economic_pulse_summary(hours=1)
        print(f"\n📈 Economic Pulse Summary:")
        print(f"Current Score: {summary.get('current_economic_score', 'N/A')}")
        print(f"Trend: {summary.get('score_trend', 'N/A')}")
    
    # asyncio.run(test_neuro_economic_engine())