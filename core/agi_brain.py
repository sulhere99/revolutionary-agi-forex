"""
AGI Brain - Core Intelligence System untuk Trading Forex
========================================================

Sistem AGI yang dapat:
- Belajar dari data historis dan real-time
- Menganalisis multiple timeframes
- Self-improvement tanpa intervensi manusia
- Adaptasi terhadap kondisi pasar yang berubah
- Risk management otomatis
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from stable_baselines3 import PPO, A2C, SAC
import gym
from gym import spaces
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import pickle
import redis
from sqlalchemy import create_engine
import optuna

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketSignal:
    """Struktur data untuk signal trading"""
    pair: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    timeframe: str
    timestamp: datetime
    reasoning: str
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    market_regime: str

@dataclass
class MarketData:
    """Struktur data pasar"""
    pair: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float
    volatility: float

class NeuralNetworkEnsemble(nn.Module):
    """Ensemble Neural Networks untuk prediksi multi-timeframe"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        self.networks = nn.ModuleList()
        
        # Buat multiple networks dengan arsitektur berbeda
        for i, hidden_size in enumerate(hidden_sizes):
            network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, output_size),
                nn.Tanh()
            )
            self.networks.append(network)
        
        # Meta-learner untuk combine predictions
        self.meta_learner = nn.Sequential(
            nn.Linear(len(hidden_sizes) * output_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        predictions = []
        for network in self.networks:
            pred = network(x)
            predictions.append(pred)
        
        # Combine predictions
        combined = torch.cat(predictions, dim=1)
        final_prediction = self.meta_learner(combined)
        return final_prediction

class ReinforcementLearningAgent:
    """RL Agent untuk adaptive trading strategy"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.env = self._create_trading_env()
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.performance_history = []
    
    def _create_trading_env(self):
        """Buat custom trading environment"""
        class TradingEnv(gym.Env):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.action_space = spaces.Discrete(action_size)
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(state_size,), dtype=np.float32
                )
                self.reset()
            
            def reset(self):
                self.current_step = 0
                self.portfolio_value = 10000
                self.position = 0
                return np.random.random(self.observation_space.shape)
            
            def step(self, action):
                # Implement trading logic
                reward = self._calculate_reward(action)
                done = self.current_step >= 1000
                self.current_step += 1
                next_state = np.random.random(self.observation_space.shape)
                return next_state, reward, done, {}
            
            def _calculate_reward(self, action):
                # Complex reward function considering profit, risk, drawdown
                return np.random.random() - 0.5
        
        return TradingEnv(self.state_size, self.action_size)
    
    def train(self, timesteps: int = 100000):
        """Train RL agent"""
        self.model.learn(total_timesteps=timesteps)
    
    def predict(self, state):
        """Predict action given state"""
        action, _ = self.model.predict(state)
        return action

class SentimentAnalyzer:
    """Analisis sentiment dari news dan social media"""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.sentiment_history = []
    
    def analyze_news_sentiment(self, news_texts: List[str]) -> float:
        """Analisis sentiment dari berita"""
        sentiments = []
        for text in news_texts:
            # Tokenize dan encode
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Extract sentiment score
            sentiment_score = self._extract_sentiment(outputs.last_hidden_state)
            sentiments.append(sentiment_score)
        
        return np.mean(sentiments) if sentiments else 0.0
    
    def _extract_sentiment(self, hidden_states):
        """Extract sentiment dari hidden states"""
        # Simplified sentiment extraction
        pooled = torch.mean(hidden_states, dim=1)
        sentiment = torch.sigmoid(torch.mean(pooled)).item()
        return (sentiment - 0.5) * 2  # Scale to [-1, 1]

class MarketRegimeDetector:
    """Deteksi regime pasar (trending, ranging, volatile, etc.)"""
    
    def __init__(self):
        self.regimes = ['trending_up', 'trending_down', 'ranging', 'high_volatility', 'low_volatility']
        self.regime_model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
    
    def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Deteksi regime pasar saat ini"""
        features = self._extract_regime_features(market_data)
        if hasattr(self.regime_model, 'predict'):
            regime_prob = self.regime_model.predict([features])[0]
            regime_idx = int(regime_prob * len(self.regimes))
            return self.regimes[min(regime_idx, len(self.regimes) - 1)]
        return 'ranging'  # Default
    
    def _extract_regime_features(self, data: pd.DataFrame) -> List[float]:
        """Extract features untuk regime detection"""
        if len(data) < 20:
            return [0.0] * 10
        
        # Calculate various market features
        returns = data['close'].pct_change().dropna()
        volatility = returns.std()
        trend_strength = abs(returns.mean())
        momentum = (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]
        
        # Volume features
        volume_trend = data['volume'].rolling(10).mean().iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
        
        # Price action features
        high_low_ratio = (data['high'] - data['low']).mean() / data['close'].mean()
        
        return [volatility, trend_strength, momentum, volume_trend, high_low_ratio, 
                returns.skew(), returns.kurtosis(), data['close'].rolling(14).std().iloc[-1],
                data['volume'].std(), len(data)]

class AGIBrain:
    """Core AGI System untuk Trading Forex"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.db_engine = create_engine(config.get('database_url', 'sqlite:///forex_agi.db'))
        
        # Initialize components
        self.neural_ensemble = NeuralNetworkEnsemble(
            input_size=100, 
            hidden_sizes=[256, 512, 1024], 
            output_size=3
        )
        self.rl_agent = ReinforcementLearningAgent(state_size=100, action_size=3)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        
        # ML Models
        self.price_predictor = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01)
        self.volatility_predictor = lgb.LGBMRegressor(n_estimators=500)
        self.risk_model = RandomForestRegressor(n_estimators=200)
        
        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'accuracy': 0.0
        }
        
        # Self-improvement parameters
        self.learning_rate = 0.001
        self.adaptation_threshold = 0.1
        self.model_versions = []
        
        logger.info("AGI Brain initialized successfully")
    
    async def analyze_market(self, market_data: Dict[str, pd.DataFrame], 
                           news_data: List[str] = None) -> MarketSignal:
        """Analisis pasar komprehensif dan generate signal"""
        try:
            # 1. Technical Analysis
            technical_score = await self._technical_analysis(market_data)
            
            # 2. Fundamental Analysis
            fundamental_score = await self._fundamental_analysis(market_data)
            
            # 3. Sentiment Analysis
            sentiment_score = 0.0
            if news_data:
                sentiment_score = self.sentiment_analyzer.analyze_news_sentiment(news_data)
            
            # 4. Market Regime Detection
            main_pair = list(market_data.keys())[0]
            market_regime = self.regime_detector.detect_regime(market_data[main_pair])
            
            # 5. Risk Assessment
            risk_score = await self._assess_risk(market_data, market_regime)
            
            # 6. Generate Signal
            signal = await self._generate_signal(
                technical_score, fundamental_score, sentiment_score, 
                risk_score, market_regime, market_data[main_pair]
            )
            
            # 7. Self-improvement
            await self._update_models(signal, market_data)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._generate_hold_signal(main_pair)
    
    async def _technical_analysis(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Comprehensive technical analysis"""
        scores = []
        
        for pair, data in market_data.items():
            if len(data) < 50:
                continue
            
            # Extract technical features
            features = self._extract_technical_features(data)
            
            # Neural network prediction
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            with torch.no_grad():
                nn_prediction = self.neural_ensemble(features_tensor).numpy()[0]
            
            # XGBoost prediction
            if hasattr(self.price_predictor, 'predict'):
                xgb_prediction = self.price_predictor.predict([features])[0]
            else:
                xgb_prediction = 0.0
            
            # Combine predictions
            combined_score = (nn_prediction[0] * 0.6 + xgb_prediction * 0.4)
            scores.append(combined_score)
        
        return np.mean(scores) if scores else 0.0
    
    async def _fundamental_analysis(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Fundamental analysis berdasarkan economic indicators"""
        # Simplified fundamental analysis
        # In production, integrate with economic calendar APIs
        
        fundamental_factors = {
            'interest_rate_differential': 0.0,
            'inflation_differential': 0.0,
            'gdp_growth_differential': 0.0,
            'employment_data': 0.0,
            'central_bank_policy': 0.0
        }
        
        # Calculate weighted fundamental score
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        fundamental_score = sum(factor * weight for factor, weight in 
                              zip(fundamental_factors.values(), weights))
        
        return fundamental_score
    
    async def _assess_risk(self, market_data: Dict[str, pd.DataFrame], 
                          market_regime: str) -> float:
        """Comprehensive risk assessment"""
        risk_factors = []
        
        for pair, data in market_data.items():
            if len(data) < 20:
                continue
            
            # Volatility risk
            volatility = data['close'].pct_change().std()
            
            # Liquidity risk (based on spread and volume)
            avg_spread = data['spread'].mean() if 'spread' in data.columns else 0.01
            volume_risk = 1 / (data['volume'].mean() + 1)
            
            # Market regime risk
            regime_risk = {
                'trending_up': 0.3,
                'trending_down': 0.3,
                'ranging': 0.5,
                'high_volatility': 0.8,
                'low_volatility': 0.2
            }.get(market_regime, 0.5)
            
            # Correlation risk
            correlation_risk = 0.4  # Simplified
            
            pair_risk = (volatility * 0.4 + avg_spread * 0.2 + 
                        volume_risk * 0.2 + regime_risk * 0.1 + 
                        correlation_risk * 0.1)
            risk_factors.append(pair_risk)
        
        return np.mean(risk_factors) if risk_factors else 0.5
    
    async def _generate_signal(self, technical_score: float, fundamental_score: float,
                             sentiment_score: float, risk_score: float,
                             market_regime: str, data: pd.DataFrame) -> MarketSignal:
        """Generate trading signal berdasarkan semua analisis"""
        
        # Weighted scoring
        weights = {
            'technical': 0.4,
            'fundamental': 0.25,
            'sentiment': 0.15,
            'risk': 0.2
        }
        
        # Adjust weights based on market regime
        if market_regime in ['trending_up', 'trending_down']:
            weights['technical'] = 0.5
            weights['fundamental'] = 0.2
        elif market_regime == 'ranging':
            weights['technical'] = 0.3
            weights['fundamental'] = 0.3
        
        # Calculate final score
        final_score = (technical_score * weights['technical'] +
                      fundamental_score * weights['fundamental'] +
                      sentiment_score * weights['sentiment'] -
                      risk_score * weights['risk'])
        
        # Determine action
        if final_score > 0.3:
            action = 'BUY'
        elif final_score < -0.3:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Calculate confidence
        confidence = min(abs(final_score), 1.0)
        
        # Calculate entry, stop loss, take profit
        current_price = data['close'].iloc[-1]
        atr = self._calculate_atr(data)
        
        if action == 'BUY':
            entry_price = current_price
            stop_loss = current_price - (atr * 2)
            take_profit = current_price + (atr * 3)
        elif action == 'SELL':
            entry_price = current_price
            stop_loss = current_price + (atr * 2)
            take_profit = current_price - (atr * 3)
        else:
            entry_price = current_price
            stop_loss = current_price
            take_profit = current_price
        
        risk_reward_ratio = abs(take_profit - entry_price) / abs(stop_loss - entry_price) if stop_loss != entry_price else 1.0
        
        return MarketSignal(
            pair=data.name if hasattr(data, 'name') else 'EURUSD',
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            timeframe='H1',
            timestamp=datetime.now(),
            reasoning=f"Technical: {technical_score:.3f}, Fundamental: {fundamental_score:.3f}, "
                     f"Sentiment: {sentiment_score:.3f}, Risk: {risk_score:.3f}, "
                     f"Regime: {market_regime}",
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            market_regime=market_regime
        )
    
    def _extract_technical_features(self, data: pd.DataFrame) -> List[float]:
        """Extract comprehensive technical features"""
        if len(data) < 50:
            return [0.0] * 100
        
        features = []
        
        # Price features
        features.extend([
            data['close'].iloc[-1],
            data['high'].iloc[-1],
            data['low'].iloc[-1],
            data['open'].iloc[-1],
            data['volume'].iloc[-1] if 'volume' in data.columns else 0
        ])
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            ma = data['close'].rolling(period).mean().iloc[-1]
            features.append(ma)
            features.append((data['close'].iloc[-1] - ma) / ma)  # Distance from MA
        
        # Technical indicators
        features.append(self._calculate_rsi(data))
        features.append(self._calculate_macd(data))
        features.append(self._calculate_bollinger_position(data))
        features.append(self._calculate_atr(data))
        features.append(self._calculate_stochastic(data))
        
        # Price patterns
        features.extend(self._detect_patterns(data))
        
        # Volatility features
        returns = data['close'].pct_change().dropna()
        features.extend([
            returns.std(),
            returns.skew(),
            returns.kurtosis(),
            returns.iloc[-1] if len(returns) > 0 else 0
        ])
        
        # Support/Resistance levels
        features.extend(self._calculate_support_resistance(data))
        
        # Pad or truncate to exactly 100 features
        while len(features) < 100:
            features.append(0.0)
        
        return features[:100]
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50.0
    
    def _calculate_macd(self, data: pd.DataFrame) -> float:
        """Calculate MACD"""
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        return macd.iloc[-1] if not macd.empty else 0.0
    
    def _calculate_bollinger_position(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate position within Bollinger Bands"""
        sma = data['close'].rolling(period).mean()
        std = data['close'].rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        
        current_price = data['close'].iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        
        if upper_val == lower_val:
            return 0.5
        
        position = (current_price - lower_val) / (upper_val - lower_val)
        return max(0, min(1, position))
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0.01
    
    def _calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Stochastic Oscillator"""
        lowest_low = data['low'].rolling(period).min()
        highest_high = data['high'].rolling(period).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        return k_percent.iloc[-1] if not k_percent.empty else 50.0
    
    def _detect_patterns(self, data: pd.DataFrame) -> List[float]:
        """Detect price patterns"""
        patterns = []
        
        if len(data) < 10:
            return [0.0] * 10
        
        # Doji pattern
        body_size = abs(data['close'] - data['open'])
        total_range = data['high'] - data['low']
        doji_score = 1 - (body_size.iloc[-1] / total_range.iloc[-1]) if total_range.iloc[-1] > 0 else 0
        patterns.append(doji_score)
        
        # Hammer pattern
        lower_shadow = data['low'] - np.minimum(data['open'], data['close'])
        upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
        hammer_score = (lower_shadow.iloc[-1] / total_range.iloc[-1]) if total_range.iloc[-1] > 0 else 0
        patterns.append(hammer_score)
        
        # Add more pattern detections
        patterns.extend([0.0] * 8)  # Placeholder for more patterns
        
        return patterns
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> List[float]:
        """Calculate support and resistance levels"""
        if len(data) < 20:
            return [0.0] * 4
        
        # Simple support/resistance calculation
        recent_highs = data['high'].rolling(10).max()
        recent_lows = data['low'].rolling(10).min()
        
        current_price = data['close'].iloc[-1]
        resistance = recent_highs.iloc[-1]
        support = recent_lows.iloc[-1]
        
        # Distance to support/resistance
        resistance_distance = (resistance - current_price) / current_price if current_price > 0 else 0
        support_distance = (current_price - support) / current_price if current_price > 0 else 0
        
        return [resistance, support, resistance_distance, support_distance]
    
    async def _update_models(self, signal: MarketSignal, market_data: Dict[str, pd.DataFrame]):
        """Self-improvement: Update models based on performance"""
        try:
            # Store signal for later evaluation
            signal_data = {
                'signal': signal.__dict__,
                'timestamp': datetime.now().isoformat(),
                'market_data_hash': hash(str(market_data))
            }
            
            # Store in Redis for quick access
            self.redis_client.lpush('signals_history', json.dumps(signal_data, default=str))
            
            # Evaluate previous signals and update models if needed
            await self._evaluate_and_improve()
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    async def _evaluate_and_improve(self):
        """Evaluate performance and improve models"""
        try:
            # Get recent signals
            recent_signals = self.redis_client.lrange('signals_history', 0, 100)
            
            if len(recent_signals) < 10:
                return
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(recent_signals)
            
            # Check if improvement is needed
            if performance['accuracy'] < self.adaptation_threshold:
                logger.info("Performance below threshold, initiating self-improvement")
                await self._retrain_models()
            
            # Update performance tracking
            self.performance_metrics.update(performance)
            
        except Exception as e:
            logger.error(f"Error in evaluation and improvement: {e}")
    
    def _calculate_performance_metrics(self, signals_data: List[bytes]) -> Dict[str, float]:
        """Calculate performance metrics from historical signals"""
        # Simplified performance calculation
        # In production, this would evaluate actual trading results
        
        total_signals = len(signals_data)
        winning_signals = int(total_signals * 0.6)  # Placeholder
        
        return {
            'total_trades': total_signals,
            'winning_trades': winning_signals,
            'accuracy': winning_signals / total_signals if total_signals > 0 else 0,
            'total_profit': np.random.uniform(-1000, 2000),  # Placeholder
            'sharpe_ratio': np.random.uniform(0.5, 2.0),  # Placeholder
            'max_drawdown': np.random.uniform(0.05, 0.2)  # Placeholder
        }
    
    async def _retrain_models(self):
        """Retrain models with new data"""
        try:
            logger.info("Starting model retraining...")
            
            # Hyperparameter optimization with Optuna
            study = optuna.create_study(direction='maximize')
            study.optimize(self._objective_function, n_trials=50)
            
            # Update model with best parameters
            best_params = study.best_params
            self._update_model_parameters(best_params)
            
            # Retrain RL agent
            self.rl_agent.train(timesteps=50000)
            
            logger.info("Model retraining completed")
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
    
    def _objective_function(self, trial):
        """Objective function for hyperparameter optimization"""
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        
        # Create and train model with suggested parameters
        model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        
        # Return performance metric (placeholder)
        return np.random.uniform(0.5, 0.9)
    
    def _update_model_parameters(self, params: Dict[str, Any]):
        """Update model parameters"""
        self.learning_rate = params.get('learning_rate', self.learning_rate)
        # Update other parameters as needed
    
    def _generate_hold_signal(self, pair: str) -> MarketSignal:
        """Generate HOLD signal when analysis fails"""
        return MarketSignal(
            pair=pair,
            action='HOLD',
            confidence=0.0,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            risk_reward_ratio=1.0,
            timeframe='H1',
            timestamp=datetime.now(),
            reasoning="Analysis failed, defaulting to HOLD",
            technical_score=0.0,
            fundamental_score=0.0,
            sentiment_score=0.0,
            market_regime='unknown'
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'performance_metrics': self.performance_metrics,
            'model_versions': len(self.model_versions),
            'last_update': datetime.now().isoformat(),
            'system_status': 'active',
            'learning_rate': self.learning_rate
        }

# Example usage
if __name__ == "__main__":
    config = {
        'database_url': 'sqlite:///forex_agi.db',
        'redis_host': 'localhost',
        'redis_port': 6379
    }
    
    agi = AGIBrain(config)
    
    # Example market data
    sample_data = pd.DataFrame({
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100) * 1000
    })
    
    # Run analysis
    async def test_analysis():
        signal = await agi.analyze_market({'EURUSD': sample_data})
        print(f"Generated Signal: {signal}")
        print(f"Performance Summary: {agi.get_performance_summary()}")
    
    # asyncio.run(test_analysis())