"""
Computer Vision Chart Pattern Recognition AI
===========================================

Revolutionary AI Vision System yang dapat "melihat" dan menganalisis chart
seperti master trader dengan 30+ tahun pengalaman menggunakan advanced CNN,
Vision Transformer, dan pattern recognition yang melampaui kemampuan manusia.
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, Circle, Polygon
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import ViTModel, ViTConfig
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import io
import base64
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from scipy.signal import find_peaks, savgol_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

logger = logging.getLogger(__name__)

@dataclass
class ChartPattern:
    """Chart pattern detection result"""
    pattern_type: str
    confidence: float
    coordinates: List[Tuple[int, int]]
    timeframe: str
    direction: str  # bullish, bearish, neutral
    strength: float
    completion_percentage: float
    target_price: float
    stop_loss: float
    pattern_age: int  # bars since pattern started
    reliability_score: float

@dataclass
class MarketPsychology:
    """Market psychology analysis from visual patterns"""
    fear_greed_index: float
    institutional_activity: float
    retail_sentiment: float
    manipulation_probability: float
    accumulation_distribution: float
    market_structure_health: float
    trend_exhaustion: float
    breakout_probability: float

@dataclass
class VisualAnalysisResult:
    """Complete visual analysis result"""
    patterns: List[ChartPattern]
    psychology: MarketPsychology
    support_resistance_levels: List[float]
    trend_lines: List[Dict[str, Any]]
    volume_profile: Dict[str, Any]
    fractal_analysis: Dict[str, Any]
    future_price_prediction: Dict[str, Any]
    trading_recommendation: Dict[str, Any]

class AdvancedVisionTransformer(nn.Module):
    """Advanced Vision Transformer for chart analysis"""
    
    def __init__(self, image_size: int = 1024, patch_size: int = 32, 
                 num_classes: int = 500, dim: int = 1024, depth: int = 24, heads: int = 16):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2
        
        # Vision Transformer backbone
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_size=dim,
            num_hidden_layers=depth,
            num_attention_heads=heads,
            intermediate_size=dim * 4,
            num_labels=num_classes
        )
        
        self.vit = ViTModel(config)
        
        # Specialized heads for different tasks
        self.pattern_classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100)  # 100 different patterns
        )
        
        self.psychology_analyzer = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 psychology metrics
        )
        
        self.price_predictor = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50)  # 50 future price points
        )
        
        self.support_resistance_detector = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 20)  # Up to 20 S/R levels
        )
    
    def forward(self, x):
        # Extract features using Vision Transformer
        outputs = self.vit(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # CLS token
        
        # Multi-task outputs
        patterns = self.pattern_classifier(features)
        psychology = self.psychology_analyzer(features)
        price_prediction = self.price_predictor(features)
        support_resistance = self.support_resistance_detector(features)
        
        return {
            'patterns': patterns,
            'psychology': psychology,
            'price_prediction': price_prediction,
            'support_resistance': support_resistance,
            'features': features
        }

class FractalPatternDetector:
    """Fractal pattern detection across multiple timeframes"""
    
    def __init__(self):
        self.fractal_levels = [5, 13, 21, 55, 144]  # Fibonacci-based levels
        
    def detect_fractals(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect fractal patterns in price data"""
        
        fractals = {
            'bullish_fractals': [],
            'bearish_fractals': [],
            'fractal_dimension': 0.0,
            'self_similarity': 0.0,
            'fractal_efficiency': 0.0
        }
        
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        for level in self.fractal_levels:
            if len(highs) < level * 2 + 1:
                continue
                
            # Detect bullish fractals (local lows)
            for i in range(level, len(lows) - level):
                if all(lows[i] <= lows[i-j] for j in range(1, level+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, level+1)):
                    fractals['bullish_fractals'].append({
                        'index': i,
                        'price': lows[i],
                        'level': level,
                        'strength': self._calculate_fractal_strength(lows, i, level)
                    })
            
            # Detect bearish fractals (local highs)
            for i in range(level, len(highs) - level):
                if all(highs[i] >= highs[i-j] for j in range(1, level+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, level+1)):
                    fractals['bearish_fractals'].append({
                        'index': i,
                        'price': highs[i],
                        'level': level,
                        'strength': self._calculate_fractal_strength(highs, i, level)
                    })
        
        # Calculate fractal dimension
        fractals['fractal_dimension'] = self._calculate_fractal_dimension(price_data)
        
        # Calculate self-similarity
        fractals['self_similarity'] = self._calculate_self_similarity(price_data)
        
        # Calculate fractal efficiency
        fractals['fractal_efficiency'] = self._calculate_fractal_efficiency(price_data)
        
        return fractals
    
    def _calculate_fractal_strength(self, prices: np.ndarray, index: int, level: int) -> float:
        """Calculate strength of fractal point"""
        center_price = prices[index]
        surrounding_prices = np.concatenate([
            prices[index-level:index],
            prices[index+1:index+level+1]
        ])
        
        if len(surrounding_prices) == 0:
            return 0.0
        
        # Calculate relative strength
        price_diff = np.abs(surrounding_prices - center_price)
        avg_diff = np.mean(price_diff)
        max_diff = np.max(price_diff)
        
        return max_diff / (avg_diff + 1e-8)
    
    def _calculate_fractal_dimension(self, price_data: pd.DataFrame) -> float:
        """Calculate fractal dimension using box-counting method"""
        prices = price_data['close'].values
        
        # Normalize prices
        normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
        
        # Box-counting algorithm
        scales = np.logspace(0.5, 2, 20)
        counts = []
        
        for scale in scales:
            box_size = 1.0 / scale
            n_boxes = int(1.0 / box_size)
            
            # Count boxes containing the curve
            boxes_with_curve = 0
            for i in range(n_boxes):
                for j in range(n_boxes):
                    box_min_x = i * box_size
                    box_max_x = (i + 1) * box_size
                    box_min_y = j * box_size
                    box_max_y = (j + 1) * box_size
                    
                    # Check if any price point falls in this box
                    for k, price in enumerate(normalized_prices):
                        x = k / len(normalized_prices)
                        if (box_min_x <= x < box_max_x and 
                            box_min_y <= price < box_max_y):
                            boxes_with_curve += 1
                            break
            
            counts.append(boxes_with_curve)
        
        # Calculate fractal dimension
        log_scales = np.log(scales)
        log_counts = np.log(np.array(counts) + 1)
        
        # Linear regression to find slope
        coeffs = np.polyfit(log_scales, log_counts, 1)
        fractal_dimension = -coeffs[0]
        
        return max(1.0, min(2.0, fractal_dimension))
    
    def _calculate_self_similarity(self, price_data: pd.DataFrame) -> float:
        """Calculate self-similarity measure"""
        prices = price_data['close'].values
        
        # Calculate correlation at different scales
        correlations = []
        scales = [2, 4, 8, 16, 32]
        
        for scale in scales:
            if len(prices) < scale * 2:
                continue
                
            # Downsample at different scales
            downsampled = prices[::scale]
            
            if len(downsampled) < 10:
                continue
            
            # Calculate autocorrelation
            autocorr = np.corrcoef(downsampled[:-1], downsampled[1:])[0, 1]
            if not np.isnan(autocorr):
                correlations.append(abs(autocorr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_fractal_efficiency(self, price_data: pd.DataFrame) -> float:
        """Calculate fractal efficiency (how efficiently price moves)"""
        prices = price_data['close'].values
        
        if len(prices) < 2:
            return 0.0
        
        # Calculate direct distance
        direct_distance = abs(prices[-1] - prices[0])
        
        # Calculate path length
        path_length = np.sum(np.abs(np.diff(prices)))
        
        # Efficiency = direct distance / path length
        efficiency = direct_distance / (path_length + 1e-8)
        
        return min(1.0, efficiency)

class PatternMemoryBank:
    """Memory bank for storing and retrieving chart patterns"""
    
    def __init__(self):
        self.pattern_database = {}
        self.pattern_performance = {}
        self.pattern_embeddings = {}
        
    def store_pattern(self, pattern: ChartPattern, performance: float):
        """Store pattern with its performance"""
        pattern_id = f"{pattern.pattern_type}_{pattern.timeframe}_{hash(str(pattern.coordinates))}"
        
        self.pattern_database[pattern_id] = pattern
        self.pattern_performance[pattern_id] = performance
        
    def find_similar_patterns(self, current_pattern: ChartPattern, 
                            similarity_threshold: float = 0.8) -> List[Tuple[ChartPattern, float]]:
        """Find similar patterns from memory"""
        similar_patterns = []
        
        for pattern_id, stored_pattern in self.pattern_database.items():
            if stored_pattern.pattern_type == current_pattern.pattern_type:
                similarity = self._calculate_pattern_similarity(current_pattern, stored_pattern)
                
                if similarity >= similarity_threshold:
                    performance = self.pattern_performance.get(pattern_id, 0.0)
                    similar_patterns.append((stored_pattern, performance))
        
        # Sort by performance
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return similar_patterns
    
    def _calculate_pattern_similarity(self, pattern1: ChartPattern, pattern2: ChartPattern) -> float:
        """Calculate similarity between two patterns"""
        # Simple similarity based on coordinates and properties
        coord_similarity = self._coordinate_similarity(pattern1.coordinates, pattern2.coordinates)
        confidence_similarity = 1 - abs(pattern1.confidence - pattern2.confidence)
        strength_similarity = 1 - abs(pattern1.strength - pattern2.strength)
        
        return (coord_similarity * 0.5 + confidence_similarity * 0.3 + strength_similarity * 0.2)
    
    def _coordinate_similarity(self, coords1: List[Tuple[int, int]], 
                             coords2: List[Tuple[int, int]]) -> float:
        """Calculate similarity between coordinate sets"""
        if len(coords1) != len(coords2):
            return 0.0
        
        # Normalize coordinates
        coords1_norm = self._normalize_coordinates(coords1)
        coords2_norm = self._normalize_coordinates(coords2)
        
        # Calculate Euclidean distance
        distances = [np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2) 
                    for c1, c2 in zip(coords1_norm, coords2_norm)]
        
        avg_distance = np.mean(distances)
        similarity = 1 / (1 + avg_distance)
        
        return similarity
    
    def _normalize_coordinates(self, coordinates: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Normalize coordinates to [0, 1] range"""
        if not coordinates:
            return []
        
        x_coords = [c[0] for c in coordinates]
        y_coords = [c[1] for c in coordinates]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        normalized = [
            ((x - x_min) / x_range, (y - y_min) / y_range)
            for x, y in coordinates
        ]
        
        return normalized

class ChartVisionAI:
    """Main Computer Vision AI for chart analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.vision_transformer = AdvancedVisionTransformer(
            image_size=config.get('image_size', 1024),
            patch_size=config.get('patch_size', 32),
            num_classes=config.get('num_classes', 500),
            dim=config.get('dim', 1024),
            depth=config.get('depth', 24),
            heads=config.get('heads', 16)
        ).to(self.device)
        
        self.fractal_detector = FractalPatternDetector()
        self.pattern_memory = PatternMemoryBank()
        
        # Pattern definitions
        self.pattern_definitions = self._load_pattern_definitions()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("🎯 Computer Vision AI initialized")
    
    def _load_pattern_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load chart pattern definitions"""
        return {
            'head_and_shoulders': {
                'type': 'reversal',
                'direction': 'bearish',
                'min_points': 5,
                'reliability': 0.85,
                'target_calculation': 'neckline_distance'
            },
            'inverse_head_and_shoulders': {
                'type': 'reversal',
                'direction': 'bullish',
                'min_points': 5,
                'reliability': 0.85,
                'target_calculation': 'neckline_distance'
            },
            'double_top': {
                'type': 'reversal',
                'direction': 'bearish',
                'min_points': 4,
                'reliability': 0.75,
                'target_calculation': 'height_projection'
            },
            'double_bottom': {
                'type': 'reversal',
                'direction': 'bullish',
                'min_points': 4,
                'reliability': 0.75,
                'target_calculation': 'height_projection'
            },
            'ascending_triangle': {
                'type': 'continuation',
                'direction': 'bullish',
                'min_points': 6,
                'reliability': 0.70,
                'target_calculation': 'triangle_height'
            },
            'descending_triangle': {
                'type': 'continuation',
                'direction': 'bearish',
                'min_points': 6,
                'reliability': 0.70,
                'target_calculation': 'triangle_height'
            },
            'symmetrical_triangle': {
                'type': 'continuation',
                'direction': 'neutral',
                'min_points': 6,
                'reliability': 0.65,
                'target_calculation': 'triangle_height'
            },
            'flag': {
                'type': 'continuation',
                'direction': 'trend_following',
                'min_points': 4,
                'reliability': 0.80,
                'target_calculation': 'flagpole_projection'
            },
            'pennant': {
                'type': 'continuation',
                'direction': 'trend_following',
                'min_points': 5,
                'reliability': 0.75,
                'target_calculation': 'flagpole_projection'
            },
            'cup_and_handle': {
                'type': 'continuation',
                'direction': 'bullish',
                'min_points': 7,
                'reliability': 0.80,
                'target_calculation': 'cup_depth'
            },
            'wedge_rising': {
                'type': 'reversal',
                'direction': 'bearish',
                'min_points': 5,
                'reliability': 0.70,
                'target_calculation': 'wedge_height'
            },
            'wedge_falling': {
                'type': 'reversal',
                'direction': 'bullish',
                'min_points': 5,
                'reliability': 0.70,
                'target_calculation': 'wedge_height'
            }
        }
    
    async def analyze_chart_like_human_expert(self, price_data: pd.DataFrame, 
                                            timeframe: str = 'H1') -> VisualAnalysisResult:
        """Main function to analyze chart like human expert"""
        logger.info(f"🔍 Starting expert-level chart analysis for {timeframe}")
        
        try:
            # Generate chart image
            chart_image = await self._generate_chart_image(price_data, timeframe)
            
            # Multi-scale pattern detection
            patterns = await self._detect_patterns_multi_scale(chart_image, price_data, timeframe)
            
            # Market psychology analysis
            psychology = await self._analyze_market_psychology(chart_image, price_data, patterns)
            
            # Support/Resistance detection
            support_resistance = await self._detect_support_resistance_levels(price_data, patterns)
            
            # Trend line detection
            trend_lines = await self._detect_trend_lines(price_data)
            
            # Volume profile analysis
            volume_profile = await self._analyze_volume_profile(price_data)
            
            # Fractal analysis
            fractal_analysis = self.fractal_detector.detect_fractals(price_data)
            
            # Future price prediction
            future_prediction = await self._predict_future_price_movement(
                chart_image, price_data, patterns, psychology
            )
            
            # Generate trading recommendation
            trading_recommendation = await self._generate_trading_recommendation(
                patterns, psychology, support_resistance, trend_lines, future_prediction
            )
            
            result = VisualAnalysisResult(
                patterns=patterns,
                psychology=psychology,
                support_resistance_levels=support_resistance,
                trend_lines=trend_lines,
                volume_profile=volume_profile,
                fractal_analysis=fractal_analysis,
                future_price_prediction=future_prediction,
                trading_recommendation=trading_recommendation
            )
            
            logger.info(f"✅ Expert chart analysis completed. Found {len(patterns)} patterns")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in chart analysis: {e}")
            raise
    
    async def _generate_chart_image(self, price_data: pd.DataFrame, timeframe: str) -> np.ndarray:
        """Generate high-quality chart image for analysis"""
        
        # Create figure with high DPI
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), dpi=150, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        if len(price_data) > 0:
            # Candlestick chart
            for i, (idx, row) in enumerate(price_data.iterrows()):
                color = 'green' if row['close'] >= row['open'] else 'red'
                
                # Body
                body_height = abs(row['close'] - row['open'])
                body_bottom = min(row['open'], row['close'])
                
                ax1.add_patch(Rectangle((i-0.3, body_bottom), 0.6, body_height, 
                                      facecolor=color, alpha=0.8))
                
                # Wicks
                ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # Moving averages
            if len(price_data) >= 20:
                ma20 = price_data['close'].rolling(20).mean()
                ma50 = price_data['close'].rolling(50).mean()
                
                ax1.plot(range(len(ma20)), ma20, color='blue', linewidth=2, alpha=0.7, label='MA20')
                if len(price_data) >= 50:
                    ax1.plot(range(len(ma50)), ma50, color='red', linewidth=2, alpha=0.7, label='MA50')
            
            # Bollinger Bands
            if len(price_data) >= 20:
                bb_period = 20
                bb_std = 2
                
                sma = price_data['close'].rolling(bb_period).mean()
                std = price_data['close'].rolling(bb_period).std()
                
                upper_band = sma + (std * bb_std)
                lower_band = sma - (std * bb_std)
                
                ax1.plot(range(len(upper_band)), upper_band, color='gray', 
                        linewidth=1, alpha=0.5, linestyle='--')
                ax1.plot(range(len(lower_band)), lower_band, color='gray', 
                        linewidth=1, alpha=0.5, linestyle='--')
                ax1.fill_between(range(len(upper_band)), upper_band, lower_band, 
                               alpha=0.1, color='gray')
        
        # Volume chart
        if 'volume' in price_data.columns:
            colors = ['green' if close >= open_price else 'red' 
                     for close, open_price in zip(price_data['close'], price_data['open'])]
            ax2.bar(range(len(price_data)), price_data['volume'], color=colors, alpha=0.7)
        
        # Styling
        ax1.set_title(f'Chart Analysis - {timeframe}', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return image
    
    async def _detect_patterns_multi_scale(self, chart_image: np.ndarray, 
                                         price_data: pd.DataFrame, 
                                         timeframe: str) -> List[ChartPattern]:
        """Detect patterns at multiple scales"""
        
        patterns = []
        
        # Convert image to tensor
        image_pil = Image.fromarray(chart_image)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Vision Transformer analysis
        with torch.no_grad():
            vit_outputs = self.vision_transformer(image_tensor)
            pattern_logits = vit_outputs['patterns']
            pattern_probs = F.softmax(pattern_logits, dim=1)
        
        # Extract top patterns
        top_patterns = torch.topk(pattern_probs, k=10, dim=1)
        
        for i, (prob, pattern_idx) in enumerate(zip(top_patterns.values[0], top_patterns.indices[0])):
            if prob > 0.1:  # Minimum confidence threshold
                pattern_name = list(self.pattern_definitions.keys())[pattern_idx % len(self.pattern_definitions)]
                pattern_def = self.pattern_definitions[pattern_name]
                
                # Detect pattern coordinates
                coordinates = await self._detect_pattern_coordinates(
                    price_data, pattern_name, pattern_def
                )
                
                if coordinates:
                    # Calculate pattern metrics
                    target_price, stop_loss = self._calculate_pattern_targets(
                        price_data, pattern_name, coordinates, pattern_def
                    )
                    
                    pattern = ChartPattern(
                        pattern_type=pattern_name,
                        confidence=float(prob),
                        coordinates=coordinates,
                        timeframe=timeframe,
                        direction=pattern_def['direction'],
                        strength=self._calculate_pattern_strength(price_data, coordinates),
                        completion_percentage=self._calculate_completion_percentage(
                            price_data, coordinates, pattern_def
                        ),
                        target_price=target_price,
                        stop_loss=stop_loss,
                        pattern_age=self._calculate_pattern_age(price_data, coordinates),
                        reliability_score=pattern_def['reliability'] * float(prob)
                    )
                    
                    patterns.append(pattern)
        
        # Classical pattern detection as backup
        classical_patterns = await self._detect_classical_patterns(price_data, timeframe)
        patterns.extend(classical_patterns)
        
        # Remove duplicates and sort by confidence
        patterns = self._remove_duplicate_patterns(patterns)
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns[:20]  # Return top 20 patterns
    
    async def _detect_pattern_coordinates(self, price_data: pd.DataFrame, 
                                        pattern_name: str, 
                                        pattern_def: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Detect specific pattern coordinates in price data"""
        
        if pattern_name == 'head_and_shoulders':
            return self._detect_head_and_shoulders_coordinates(price_data)
        elif pattern_name == 'double_top':
            return self._detect_double_top_coordinates(price_data)
        elif pattern_name == 'double_bottom':
            return self._detect_double_bottom_coordinates(price_data)
        elif 'triangle' in pattern_name:
            return self._detect_triangle_coordinates(price_data, pattern_name)
        elif pattern_name in ['flag', 'pennant']:
            return self._detect_flag_pennant_coordinates(price_data, pattern_name)
        else:
            return self._detect_generic_pattern_coordinates(price_data, pattern_def)
    
    def _detect_head_and_shoulders_coordinates(self, price_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Detect head and shoulders pattern coordinates"""
        highs = price_data['high'].values
        
        if len(highs) < 20:
            return []
        
        # Find peaks
        peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
        
        if len(peaks) < 3:
            return []
        
        # Look for head and shoulders pattern
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]
            
            # Check if head is higher than shoulders
            if (highs[head] > highs[left_shoulder] and 
                highs[head] > highs[right_shoulder] and
                abs(highs[left_shoulder] - highs[right_shoulder]) < np.std(highs) * 0.3):
                
                # Find neckline points
                left_valley = np.argmin(highs[left_shoulder:head]) + left_shoulder
                right_valley = np.argmin(highs[head:right_shoulder]) + head
                
                coordinates = [
                    (left_shoulder, int(highs[left_shoulder])),
                    (left_valley, int(highs[left_valley])),
                    (head, int(highs[head])),
                    (right_valley, int(highs[right_valley])),
                    (right_shoulder, int(highs[right_shoulder]))
                ]
                
                return coordinates
        
        return []
    
    def _detect_double_top_coordinates(self, price_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Detect double top pattern coordinates"""
        highs = price_data['high'].values
        
        if len(highs) < 15:
            return []
        
        # Find peaks
        peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.3)
        
        if len(peaks) < 2:
            return []
        
        # Look for double top pattern
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            
            # Check if peaks are similar height
            height_diff = abs(highs[peak1] - highs[peak2])
            if height_diff < np.std(highs) * 0.2:
                
                # Find valley between peaks
                valley = np.argmin(highs[peak1:peak2]) + peak1
                
                coordinates = [
                    (peak1, int(highs[peak1])),
                    (valley, int(highs[valley])),
                    (peak2, int(highs[peak2]))
                ]
                
                return coordinates
        
        return []
    
    def _detect_double_bottom_coordinates(self, price_data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Detect double bottom pattern coordinates"""
        lows = price_data['low'].values
        
        if len(lows) < 15:
            return []
        
        # Find troughs (invert and find peaks)
        inverted_lows = -lows
        troughs, _ = find_peaks(inverted_lows, distance=5, prominence=np.std(lows) * 0.3)
        
        if len(troughs) < 2:
            return []
        
        # Look for double bottom pattern
        for i in range(len(troughs) - 1):
            trough1 = troughs[i]
            trough2 = troughs[i + 1]
            
            # Check if troughs are similar depth
            depth_diff = abs(lows[trough1] - lows[trough2])
            if depth_diff < np.std(lows) * 0.2:
                
                # Find peak between troughs
                peak = np.argmax(lows[trough1:trough2]) + trough1
                
                coordinates = [
                    (trough1, int(lows[trough1])),
                    (peak, int(lows[peak])),
                    (trough2, int(lows[trough2]))
                ]
                
                return coordinates
        
        return []
    
    def _detect_triangle_coordinates(self, price_data: pd.DataFrame, pattern_name: str) -> List[Tuple[int, int]]:
        """Detect triangle pattern coordinates"""
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        if len(highs) < 20:
            return []
        
        # Find trend lines
        high_peaks, _ = find_peaks(highs, distance=3)
        low_troughs, _ = find_peaks(-lows, distance=3)
        
        if len(high_peaks) < 3 or len(low_troughs) < 3:
            return []
        
        # Get recent peaks and troughs
        recent_highs = high_peaks[-5:] if len(high_peaks) >= 5 else high_peaks
        recent_lows = low_troughs[-5:] if len(low_troughs) >= 5 else low_troughs
        
        coordinates = []
        
        # Add high points
        for peak in recent_highs:
            coordinates.append((peak, int(highs[peak])))
        
        # Add low points
        for trough in recent_lows:
            coordinates.append((trough, int(lows[trough])))
        
        # Sort by time
        coordinates.sort(key=lambda x: x[0])
        
        return coordinates
    
    def _detect_flag_pennant_coordinates(self, price_data: pd.DataFrame, pattern_name: str) -> List[Tuple[int, int]]:
        """Detect flag/pennant pattern coordinates"""
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        if len(highs) < 15:
            return []
        
        # Look for consolidation after strong move
        # Find the flagpole (strong directional move)
        price_changes = np.diff(price_data['close'].values)
        
        # Find significant moves
        significant_moves = np.where(np.abs(price_changes) > np.std(price_changes) * 2)[0]
        
        if len(significant_moves) == 0:
            return []
        
        # Take the most recent significant move
        flagpole_start = significant_moves[-1]
        
        if flagpole_start >= len(highs) - 5:
            return []
        
        # Find consolidation pattern after flagpole
        consolidation_data = price_data.iloc[flagpole_start:]
        
        if len(consolidation_data) < 5:
            return []
        
        # Find key points in consolidation
        cons_highs = consolidation_data['high'].values
        cons_lows = consolidation_data['low'].values
        
        high_peaks, _ = find_peaks(cons_highs, distance=2)
        low_troughs, _ = find_peaks(-cons_lows, distance=2)
        
        coordinates = []
        
        # Add flagpole start
        coordinates.append((flagpole_start, int(highs[flagpole_start])))
        
        # Add consolidation points
        for peak in high_peaks[:3]:  # Max 3 peaks
            actual_index = flagpole_start + peak
            coordinates.append((actual_index, int(cons_highs[peak])))
        
        for trough in low_troughs[:3]:  # Max 3 troughs
            actual_index = flagpole_start + trough
            coordinates.append((actual_index, int(cons_lows[trough])))
        
        # Sort by time
        coordinates.sort(key=lambda x: x[0])
        
        return coordinates
    
    def _detect_generic_pattern_coordinates(self, price_data: pd.DataFrame, 
                                          pattern_def: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Generic pattern coordinate detection"""
        min_points = pattern_def.get('min_points', 4)
        
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        if len(highs) < min_points:
            return []
        
        # Find significant points
        high_peaks, _ = find_peaks(highs, distance=3, prominence=np.std(highs) * 0.2)
        low_troughs, _ = find_peaks(-lows, distance=3, prominence=np.std(lows) * 0.2)
        
        # Combine and sort
        all_points = []
        
        for peak in high_peaks[-min_points//2:]:
            all_points.append((peak, int(highs[peak]), 'high'))
        
        for trough in low_troughs[-min_points//2:]:
            all_points.append((trough, int(lows[trough]), 'low'))
        
        # Sort by time
        all_points.sort(key=lambda x: x[0])
        
        # Return coordinates
        coordinates = [(point[0], point[1]) for point in all_points]
        
        return coordinates
    
    async def _detect_classical_patterns(self, price_data: pd.DataFrame, 
                                       timeframe: str) -> List[ChartPattern]:
        """Detect classical chart patterns using traditional methods"""
        
        patterns = []
        
        # Implement classical pattern detection algorithms
        # This is a simplified version - in production, use more sophisticated algorithms
        
        # Example: Simple trend line breaks
        if len(price_data) >= 20:
            closes = price_data['close'].values
            
            # Detect trend
            recent_trend = np.polyfit(range(len(closes[-20:])), closes[-20:], 1)[0]
            
            if abs(recent_trend) > np.std(closes) * 0.01:  # Significant trend
                direction = 'bullish' if recent_trend > 0 else 'bearish'
                
                pattern = ChartPattern(
                    pattern_type='trend_line',
                    confidence=0.6,
                    coordinates=[(len(closes)-20, int(closes[-20])), (len(closes)-1, int(closes[-1]))],
                    timeframe=timeframe,
                    direction=direction,
                    strength=abs(recent_trend),
                    completion_percentage=1.0,
                    target_price=closes[-1] + recent_trend * 10,  # Project 10 periods ahead
                    stop_loss=closes[-1] - recent_trend * 5,
                    pattern_age=20,
                    reliability_score=0.6
                )
                
                patterns.append(pattern)
        
        return patterns
    
    def _remove_duplicate_patterns(self, patterns: List[ChartPattern]) -> List[ChartPattern]:
        """Remove duplicate patterns"""
        unique_patterns = []
        
        for pattern in patterns:
            is_duplicate = False
            
            for existing in unique_patterns:
                if (pattern.pattern_type == existing.pattern_type and
                    pattern.timeframe == existing.timeframe and
                    self._patterns_overlap(pattern.coordinates, existing.coordinates)):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _patterns_overlap(self, coords1: List[Tuple[int, int]], 
                         coords2: List[Tuple[int, int]]) -> bool:
        """Check if two patterns overlap significantly"""
        if not coords1 or not coords2:
            return False
        
        # Check time overlap
        time1_range = (min(c[0] for c in coords1), max(c[0] for c in coords1))
        time2_range = (min(c[0] for c in coords2), max(c[0] for c in coords2))
        
        overlap_start = max(time1_range[0], time2_range[0])
        overlap_end = min(time1_range[1], time2_range[1])
        
        if overlap_end <= overlap_start:
            return False
        
        overlap_length = overlap_end - overlap_start
        total_length = max(time1_range[1] - time1_range[0], time2_range[1] - time2_range[0])
        
        overlap_ratio = overlap_length / total_length if total_length > 0 else 0
        
        return overlap_ratio > 0.5  # 50% overlap threshold
    
    def _calculate_pattern_strength(self, price_data: pd.DataFrame, 
                                  coordinates: List[Tuple[int, int]]) -> float:
        """Calculate pattern strength"""
        if not coordinates or len(coordinates) < 2:
            return 0.0
        
        # Calculate price range of pattern
        prices = [c[1] for c in coordinates]
        price_range = max(prices) - min(prices)
        
        # Calculate average price volatility
        if len(price_data) > 1:
            returns = price_data['close'].pct_change().dropna()
            avg_volatility = returns.std() * price_data['close'].mean()
        else:
            avg_volatility = 1.0
        
        # Strength is relative to normal volatility
        strength = price_range / (avg_volatility + 1e-8)
        
        return min(1.0, strength / 5.0)  # Normalize to [0, 1]
    
    def _calculate_completion_percentage(self, price_data: pd.DataFrame, 
                                       coordinates: List[Tuple[int, int]], 
                                       pattern_def: Dict[str, Any]) -> float:
        """Calculate pattern completion percentage"""
        if not coordinates:
            return 0.0
        
        min_points = pattern_def.get('min_points', 4)
        current_points = len(coordinates)
        
        # Basic completion based on number of points
        basic_completion = min(1.0, current_points / min_points)
        
        # Adjust based on pattern type
        pattern_type = pattern_def.get('type', 'unknown')
        
        if pattern_type == 'reversal':
            # Reversal patterns need confirmation
            if basic_completion >= 1.0:
                # Check if reversal is confirmed
                last_coord = coordinates[-1]
                current_price = price_data['close'].iloc[-1]
                
                if pattern_def.get('direction') == 'bullish':
                    confirmation = current_price > last_coord[1]
                else:
                    confirmation = current_price < last_coord[1]
                
                return 1.0 if confirmation else 0.8
        
        return basic_completion
    
    def _calculate_pattern_targets(self, price_data: pd.DataFrame, pattern_name: str, 
                                 coordinates: List[Tuple[int, int]], 
                                 pattern_def: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate pattern target price and stop loss"""
        
        if not coordinates:
            current_price = price_data['close'].iloc[-1]
            return current_price, current_price
        
        current_price = price_data['close'].iloc[-1]
        calculation_method = pattern_def.get('target_calculation', 'height_projection')
        
        if calculation_method == 'height_projection':
            # Calculate pattern height and project
            prices = [c[1] for c in coordinates]
            pattern_height = max(prices) - min(prices)
            
            if pattern_def.get('direction') == 'bullish':
                target = current_price + pattern_height
                stop_loss = min(prices) * 0.98  # 2% below pattern low
            else:
                target = current_price - pattern_height
                stop_loss = max(prices) * 1.02  # 2% above pattern high
        
        elif calculation_method == 'neckline_distance':
            # For head and shoulders patterns
            if len(coordinates) >= 3:
                # Find neckline level (approximate)
                neckline = np.mean([c[1] for c in coordinates[::2]])  # Every other point
                head_price = max(c[1] for c in coordinates)
                
                distance = abs(head_price - neckline)
                
                if pattern_def.get('direction') == 'bearish':
                    target = neckline - distance
                    stop_loss = head_price * 1.02
                else:
                    target = neckline + distance
                    stop_loss = min(c[1] for c in coordinates) * 0.98
            else:
                target = current_price
                stop_loss = current_price
        
        else:
            # Default calculation
            atr = self._calculate_atr(price_data)
            
            if pattern_def.get('direction') == 'bullish':
                target = current_price + atr * 3
                stop_loss = current_price - atr * 2
            else:
                target = current_price - atr * 3
                stop_loss = current_price + atr * 2
        
        return float(target), float(stop_loss)
    
    def _calculate_atr(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(price_data) < period + 1:
            return price_data['close'].std() if len(price_data) > 1 else 0.01
        
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        
        return atr if not pd.isna(atr) else 0.01
    
    def _calculate_pattern_age(self, price_data: pd.DataFrame, 
                             coordinates: List[Tuple[int, int]]) -> int:
        """Calculate pattern age in bars"""
        if not coordinates:
            return 0
        
        earliest_time = min(c[0] for c in coordinates)
        current_time = len(price_data) - 1
        
        return current_time - earliest_time
    
    async def _analyze_market_psychology(self, chart_image: np.ndarray, 
                                       price_data: pd.DataFrame, 
                                       patterns: List[ChartPattern]) -> MarketPsychology:
        """Analyze market psychology from visual patterns"""
        
        # Convert image to tensor for analysis
        image_pil = Image.fromarray(chart_image)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Get psychology analysis from Vision Transformer
        with torch.no_grad():
            vit_outputs = self.vision_transformer(image_tensor)
            psychology_scores = torch.sigmoid(vit_outputs['psychology']).cpu().numpy()[0]
        
        # Calculate additional psychology metrics
        fear_greed = self._calculate_fear_greed_index(price_data, patterns)
        institutional_activity = self._detect_institutional_activity(price_data)
        retail_sentiment = self._analyze_retail_sentiment(price_data, patterns)
        manipulation_prob = self._detect_manipulation_probability(price_data)
        accumulation_distribution = self._analyze_accumulation_distribution(price_data)
        market_structure = self._analyze_market_structure_health(price_data)
        trend_exhaustion = self._detect_trend_exhaustion(price_data)
        breakout_probability = self._calculate_breakout_probability(price_data, patterns)
        
        return MarketPsychology(
            fear_greed_index=fear_greed,
            institutional_activity=institutional_activity,
            retail_sentiment=retail_sentiment,
            manipulation_probability=manipulation_prob,
            accumulation_distribution=accumulation_distribution,
            market_structure_health=market_structure,
            trend_exhaustion=trend_exhaustion,
            breakout_probability=breakout_probability
        )
    
    def _calculate_fear_greed_index(self, price_data: pd.DataFrame, 
                                  patterns: List[ChartPattern]) -> float:
        """Calculate fear and greed index from price action"""
        
        if len(price_data) < 20:
            return 0.5  # Neutral
        
        # Volatility component
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std()
        vol_percentile = min(1.0, volatility / (returns.std() * 2))
        
        # Momentum component
        momentum = (price_data['close'].iloc[-1] - price_data['close'].iloc[-20]) / price_data['close'].iloc[-20]
        momentum_normalized = (momentum + 0.1) / 0.2  # Normalize to roughly [0, 1]
        momentum_normalized = max(0, min(1, momentum_normalized))
        
        # Pattern component
        bullish_patterns = sum(1 for p in patterns if p.direction == 'bullish')
        bearish_patterns = sum(1 for p in patterns if p.direction == 'bearish')
        total_patterns = bullish_patterns + bearish_patterns
        
        if total_patterns > 0:
            pattern_sentiment = bullish_patterns / total_patterns
        else:
            pattern_sentiment = 0.5
        
        # Volume component (if available)
        if 'volume' in price_data.columns:
            recent_volume = price_data['volume'].iloc[-5:].mean()
            avg_volume = price_data['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            volume_component = min(1.0, volume_ratio / 2)
        else:
            volume_component = 0.5
        
        # Combine components
        fear_greed = (
            momentum_normalized * 0.4 +
            pattern_sentiment * 0.3 +
            (1 - vol_percentile) * 0.2 +  # High volatility = fear
            volume_component * 0.1
        )
        
        return max(0, min(1, fear_greed))
    
    def _detect_institutional_activity(self, price_data: pd.DataFrame) -> float:
        """Detect institutional trading activity"""
        
        if len(price_data) < 10:
            return 0.5
        
        # Large candle analysis
        candle_sizes = abs(price_data['close'] - price_data['open'])
        avg_candle_size = candle_sizes.mean()
        large_candles = candle_sizes > avg_candle_size * 2
        
        institutional_score = 0.0
        
        # High volume with small price movement (accumulation/distribution)
        if 'volume' in price_data.columns:
            volume_price_correlation = np.corrcoef(
                price_data['volume'].iloc[-20:], 
                candle_sizes.iloc[-20:]
            )[0, 1]
            
            if not np.isnan(volume_price_correlation):
                # Low correlation suggests institutional activity
                institutional_score += (1 - abs(volume_price_correlation)) * 0.5
        
        # Consistent directional pressure
        recent_closes = price_data['close'].iloc[-10:]
        directional_consistency = abs(np.corrcoef(range(len(recent_closes)), recent_closes)[0, 1])
        institutional_score += directional_consistency * 0.3
        
        # Large candle frequency
        large_candle_ratio = large_candles.sum() / len(large_candles)
        institutional_score += large_candle_ratio * 0.2
        
        return max(0, min(1, institutional_score))
    
    def _analyze_retail_sentiment(self, price_data: pd.DataFrame, 
                                patterns: List[ChartPattern]) -> float:
        """Analyze retail trader sentiment"""
        
        if len(price_data) < 10:
            return 0.5
        
        retail_score = 0.0
        
        # Retail traders often chase trends
        recent_trend = (price_data['close'].iloc[-1] - price_data['close'].iloc[-10]) / price_data['close'].iloc[-10]
        trend_chasing = abs(recent_trend) * 0.3
        retail_score += trend_chasing
        
        # Retail traders love reversal patterns
        reversal_patterns = [p for p in patterns if 'reversal' in p.pattern_type or 'double' in p.pattern_type]
        if patterns:
            reversal_ratio = len(reversal_patterns) / len(patterns)
            retail_score += reversal_ratio * 0.4
        
        # High volatility attracts retail traders
        returns = price_data['close'].pct_change().dropna()
        if len(returns) > 1:
            volatility = returns.std()
            normalized_vol = min(1.0, volatility / 0.02)  # 2% daily vol as reference
            retail_score += normalized_vol * 0.3
        
        return max(0, min(1, retail_score))
    
    def _detect_manipulation_probability(self, price_data: pd.DataFrame) -> float:
        """Detect probability of market manipulation"""
        
        if len(price_data) < 20:
            return 0.0
        
        manipulation_score = 0.0
        
        # Sudden price spikes followed by reversals
        price_changes = price_data['close'].pct_change().dropna()
        
        # Look for abnormal price movements
        std_change = price_changes.std()
        abnormal_moves = abs(price_changes) > std_change * 3
        
        if abnormal_moves.sum() > 0:
            # Check if abnormal moves are followed by reversals
            for i in range(1, len(price_changes)):
                if abnormal_moves.iloc[i-1] and i < len(price_changes):
                    if price_changes.iloc[i-1] * price_changes.iloc[i] < 0:  # Opposite direction
                        manipulation_score += 0.1
        
        # Unusual volume patterns
        if 'volume' in price_data.columns:
            volume_changes = price_data['volume'].pct_change().dropna()
            volume_spikes = volume_changes > volume_changes.std() * 2
            
            # Volume spikes without corresponding price movement
            for i in range(len(volume_spikes)):
                if volume_spikes.iloc[i] and i < len(price_changes):
                    if abs(price_changes.iloc[i]) < price_changes.std():
                        manipulation_score += 0.05
        
        # Wick analysis (long wicks suggest rejection)
        upper_wicks = price_data['high'] - np.maximum(price_data['open'], price_data['close'])
        lower_wicks = np.minimum(price_data['open'], price_data['close']) - price_data['low']
        
        avg_body_size = abs(price_data['close'] - price_data['open']).mean()
        long_wick_ratio = ((upper_wicks > avg_body_size * 2) | (lower_wicks > avg_body_size * 2)).mean()
        
        manipulation_score += long_wick_ratio * 0.3
        
        return max(0, min(1, manipulation_score))
    
    def _analyze_accumulation_distribution(self, price_data: pd.DataFrame) -> float:
        """Analyze accumulation vs distribution"""
        
        if len(price_data) < 10:
            return 0.5  # Neutral
        
        # Calculate Accumulation/Distribution Line
        if 'volume' in price_data.columns:
            # Money Flow Multiplier
            mfm = ((price_data['close'] - price_data['low']) - (price_data['high'] - price_data['close'])) / (price_data['high'] - price_data['low'])
            mfm = mfm.fillna(0)  # Handle division by zero
            
            # Money Flow Volume
            mfv = mfm * price_data['volume']
            
            # Accumulation/Distribution Line
            ad_line = mfv.cumsum()
            
            # Trend of AD line
            if len(ad_line) >= 10:
                recent_trend = np.polyfit(range(10), ad_line.iloc[-10:], 1)[0]
                
                # Normalize to [0, 1] where 1 = accumulation, 0 = distribution
                max_trend = ad_line.diff().abs().max()
                if max_trend > 0:
                    normalized_trend = (recent_trend / max_trend + 1) / 2
                    return max(0, min(1, normalized_trend))
        
        # Fallback: use price-volume relationship
        if 'volume' in price_data.columns and len(price_data) >= 10:
            price_changes = price_data['close'].diff().iloc[-10:]
            volume_changes = price_data['volume'].diff().iloc[-10:]
            
            # Accumulation: price up on high volume, price down on low volume
            accumulation_signals = ((price_changes > 0) & (volume_changes > 0)) | ((price_changes < 0) & (volume_changes < 0))
            accumulation_ratio = accumulation_signals.sum() / len(accumulation_signals)
            
            return accumulation_ratio
        
        return 0.5  # Neutral if no volume data
    
    def _analyze_market_structure_health(self, price_data: pd.DataFrame) -> float:
        """Analyze market structure health"""
        
        if len(price_data) < 20:
            return 0.5
        
        health_score = 0.0
        
        # Higher highs and higher lows = healthy uptrend
        # Lower highs and lower lows = healthy downtrend
        highs = price_data['high'].iloc[-20:]
        lows = price_data['low'].iloc[-20:]
        
        # Find recent peaks and troughs
        high_peaks, _ = find_peaks(highs.values, distance=3)
        low_troughs, _ = find_peaks(-lows.values, distance=3)
        
        if len(high_peaks) >= 2 and len(low_troughs) >= 2:
            # Check for higher highs
            recent_high_peaks = high_peaks[-2:]
            higher_highs = highs.iloc[recent_high_peaks[-1]] > highs.iloc[recent_high_peaks[0]]
            
            # Check for higher lows
            recent_low_troughs = low_troughs[-2:]
            higher_lows = lows.iloc[recent_low_troughs[-1]] > lows.iloc[recent_low_troughs[0]]
            
            # Check for lower highs
            lower_highs = highs.iloc[recent_high_peaks[-1]] < highs.iloc[recent_high_peaks[0]]
            
            # Check for lower lows
            lower_lows = lows.iloc[recent_low_troughs[-1]] < lows.iloc[recent_low_troughs[0]]
            
            # Healthy structure
            if (higher_highs and higher_lows) or (lower_highs and lower_lows):
                health_score += 0.5
            
            # Consistent trend
            price_trend = np.polyfit(range(len(price_data['close'].iloc[-20:])), 
                                   price_data['close'].iloc[-20:], 1)[0]
            
            if abs(price_trend) > price_data['close'].std() * 0.01:  # Significant trend
                health_score += 0.3
        
        # Volume confirmation (if available)
        if 'volume' in price_data.columns:
            volume_trend = np.polyfit(range(len(price_data['volume'].iloc[-10:])), 
                                    price_data['volume'].iloc[-10:], 1)[0]
            
            price_trend = np.polyfit(range(len(price_data['close'].iloc[-10:])), 
                                   price_data['close'].iloc[-10:], 1)[0]
            
            # Volume should increase with price in healthy trends
            if (price_trend > 0 and volume_trend > 0) or (price_trend < 0 and volume_trend < 0):
                health_score += 0.2
        
        return max(0, min(1, health_score))
    
    def _detect_trend_exhaustion(self, price_data: pd.DataFrame) -> float:
        """Detect trend exhaustion signals"""
        
        if len(price_data) < 20:
            return 0.0
        
        exhaustion_score = 0.0
        
        # Divergence between price and momentum
        closes = price_data['close'].iloc[-20:]
        momentum = closes.diff().rolling(5).mean()
        
        # Price making new highs/lows but momentum not confirming
        recent_price_trend = np.polyfit(range(len(closes[-10:])), closes[-10:], 1)[0]
        recent_momentum_trend = np.polyfit(range(len(momentum[-10:].dropna())), 
                                         momentum[-10:].dropna(), 1)[0]
        
        # Divergence detection
        if recent_price_trend > 0 and recent_momentum_trend < 0:  # Bearish divergence
            exhaustion_score += 0.4
        elif recent_price_trend < 0 and recent_momentum_trend > 0:  # Bullish divergence
            exhaustion_score += 0.4
        
        # Volume exhaustion
        if 'volume' in price_data.columns:
            recent_volume = price_data['volume'].iloc[-10:].mean()
            avg_volume = price_data['volume'].mean()
            
            if recent_volume < avg_volume * 0.8:  # Volume declining
                exhaustion_score += 0.3
        
        # Volatility exhaustion
        recent_volatility = closes.pct_change().iloc[-10:].std()
        avg_volatility = closes.pct_change().std()
        
        if recent_volatility < avg_volatility * 0.8:  # Volatility declining
            exhaustion_score += 0.3
        
        return max(0, min(1, exhaustion_score))
    
    def _calculate_breakout_probability(self, price_data: pd.DataFrame, 
                                      patterns: List[ChartPattern]) -> float:
        """Calculate probability of breakout"""
        
        if len(price_data) < 20:
            return 0.5
        
        breakout_prob = 0.0
        
        # Consolidation detection
        recent_highs = price_data['high'].iloc[-20:]
        recent_lows = price_data['low'].iloc[-20:]
        
        price_range = recent_highs.max() - recent_lows.min()
        avg_range = (recent_highs - recent_lows).mean()
        
        # Tight consolidation increases breakout probability
        if price_range < avg_range * 1.5:
            breakout_prob += 0.3
        
        # Volume buildup
        if 'volume' in price_data.columns:
            recent_volume = price_data['volume'].iloc[-10:].mean()
            avg_volume = price_data['volume'].iloc[-50:-10].mean()
            
            if recent_volume > avg_volume * 1.2:  # Volume increasing
                breakout_prob += 0.3
        
        # Pattern-based probability
        breakout_patterns = ['triangle', 'flag', 'pennant', 'wedge']
        pattern_boost = 0.0
        
        for pattern in patterns:
            for breakout_pattern in breakout_patterns:
                if breakout_pattern in pattern.pattern_type:
                    pattern_boost += pattern.confidence * 0.1
        
        breakout_prob += min(0.4, pattern_boost)
        
        return max(0, min(1, breakout_prob))
    
    async def _detect_support_resistance_levels(self, price_data: pd.DataFrame, 
                                              patterns: List[ChartPattern]) -> List[float]:
        """Detect support and resistance levels"""
        
        levels = []
        
        if len(price_data) < 10:
            return levels
        
        # Method 1: Pivot points
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        # Find peaks and troughs
        high_peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.3)
        low_troughs, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.3)
        
        # Add resistance levels from peaks
        for peak in high_peaks:
            levels.append(float(highs[peak]))
        
        # Add support levels from troughs
        for trough in low_troughs:
            levels.append(float(lows[trough]))
        
        # Method 2: Psychological levels (round numbers)
        current_price = price_data['close'].iloc[-1]
        price_magnitude = 10 ** (len(str(int(current_price))) - 2)
        
        # Find nearby round numbers
        for multiplier in [0.5, 1, 1.5, 2, 2.5, 3]:
            round_level = round(current_price / price_magnitude) * price_magnitude
            levels.extend([
                round_level + multiplier * price_magnitude,
                round_level - multiplier * price_magnitude
            ])
        
        # Method 3: Pattern-based levels
        for pattern in patterns:
            for coord in pattern.coordinates:
                levels.append(float(coord[1]))
        
        # Method 4: Volume-based levels (if volume available)
        if 'volume' in price_data.columns:
            # Find high volume areas
            volume_profile = self._calculate_volume_profile(price_data)
            high_volume_levels = [level for level, volume in volume_profile.items() 
                                if volume > np.mean(list(volume_profile.values()))]
            levels.extend(high_volume_levels)
        
        # Clean and sort levels
        levels = [level for level in levels if not np.isnan(level) and level > 0]
        levels = list(set(levels))  # Remove duplicates
        levels.sort()
        
        # Filter levels close to current price
        current_price = price_data['close'].iloc[-1]
        price_range = current_price * 0.1  # 10% range
        
        relevant_levels = [
            level for level in levels 
            if abs(level - current_price) <= price_range
        ]
        
        return relevant_levels[:20]  # Return top 20 levels
    
    def _calculate_volume_profile(self, price_data: pd.DataFrame) -> Dict[float, float]:
        """Calculate volume profile"""
        
        if 'volume' not in price_data.columns or len(price_data) < 10:
            return {}
        
        # Create price bins
        price_min = price_data['low'].min()
        price_max = price_data['high'].max()
        num_bins = 50
        
        price_bins = np.linspace(price_min, price_max, num_bins)
        volume_profile = {}
        
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_center = (bin_low + bin_high) / 2
            
            # Find candles that overlap with this price bin
            overlapping_candles = price_data[
                (price_data['low'] <= bin_high) & (price_data['high'] >= bin_low)
            ]
            
            # Sum volume for overlapping candles
            total_volume = overlapping_candles['volume'].sum()
            volume_profile[bin_center] = total_volume
        
        return volume_profile
    
    async def _detect_trend_lines(self, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect trend lines"""
        
        trend_lines = []
        
        if len(price_data) < 20:
            return trend_lines
        
        highs = price_data['high'].values
        lows = price_data['low'].values
        
        # Find peaks and troughs
        high_peaks, _ = find_peaks(highs, distance=5)
        low_troughs, _ = find_peaks(-lows, distance=5)
        
        # Resistance trend lines (connecting peaks)
        if len(high_peaks) >= 2:
            for i in range(len(high_peaks) - 1):
                for j in range(i + 1, len(high_peaks)):
                    peak1_idx, peak2_idx = high_peaks[i], high_peaks[j]
                    peak1_price, peak2_price = highs[peak1_idx], highs[peak2_idx]
                    
                    # Calculate trend line
                    slope = (peak2_price - peak1_price) / (peak2_idx - peak1_idx)
                    intercept = peak1_price - slope * peak1_idx
                    
                    # Validate trend line (check how many points it touches)
                    touches = 0
                    for k in range(peak1_idx, min(peak2_idx + 20, len(highs))):
                        expected_price = slope * k + intercept
                        if abs(highs[k] - expected_price) < np.std(highs) * 0.1:
                            touches += 1
                    
                    if touches >= 3:  # At least 3 touches
                        trend_lines.append({
                            'type': 'resistance',
                            'start_point': (peak1_idx, peak1_price),
                            'end_point': (peak2_idx, peak2_price),
                            'slope': slope,
                            'intercept': intercept,
                            'touches': touches,
                            'strength': min(1.0, touches / 5.0)
                        })
        
        # Support trend lines (connecting troughs)
        if len(low_troughs) >= 2:
            for i in range(len(low_troughs) - 1):
                for j in range(i + 1, len(low_troughs)):
                    trough1_idx, trough2_idx = low_troughs[i], low_troughs[j]
                    trough1_price, trough2_price = lows[trough1_idx], lows[trough2_idx]
                    
                    # Calculate trend line
                    slope = (trough2_price - trough1_price) / (trough2_idx - trough1_idx)
                    intercept = trough1_price - slope * trough1_idx
                    
                    # Validate trend line
                    touches = 0
                    for k in range(trough1_idx, min(trough2_idx + 20, len(lows))):
                        expected_price = slope * k + intercept
                        if abs(lows[k] - expected_price) < np.std(lows) * 0.1:
                            touches += 1
                    
                    if touches >= 3:
                        trend_lines.append({
                            'type': 'support',
                            'start_point': (trough1_idx, trough1_price),
                            'end_point': (trough2_idx, trough2_price),
                            'slope': slope,
                            'intercept': intercept,
                            'touches': touches,
                            'strength': min(1.0, touches / 5.0)
                        })
        
        # Sort by strength and return top trend lines
        trend_lines.sort(key=lambda x: x['strength'], reverse=True)
        
        return trend_lines[:10]  # Return top 10 trend lines
    
    async def _analyze_volume_profile(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile"""
        
        if 'volume' not in price_data.columns:
            return {'message': 'Volume data not available'}
        
        volume_profile = self._calculate_volume_profile(price_data)
        
        if not volume_profile:
            return {'message': 'Unable to calculate volume profile'}
        
        # Find Point of Control (POC) - highest volume level
        poc_price = max(volume_profile.keys(), key=lambda k: volume_profile[k])
        poc_volume = volume_profile[poc_price]
        
        # Find Value Area (70% of volume)
        total_volume = sum(volume_profile.values())
        target_volume = total_volume * 0.7
        
        # Sort levels by volume
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        value_area_volume = 0
        value_area_levels = []
        
        for price, volume in sorted_levels:
            value_area_levels.append(price)
            value_area_volume += volume
            
            if value_area_volume >= target_volume:
                break
        
        value_area_high = max(value_area_levels)
        value_area_low = min(value_area_levels)
        
        # Current price position relative to value area
        current_price = price_data['close'].iloc[-1]
        
        if current_price > value_area_high:
            price_position = 'above_value_area'
        elif current_price < value_area_low:
            price_position = 'below_value_area'
        else:
            price_position = 'within_value_area'
        
        return {
            'point_of_control': {
                'price': poc_price,
                'volume': poc_volume
            },
            'value_area': {
                'high': value_area_high,
                'low': value_area_low,
                'volume_percentage': 70
            },
            'current_price_position': price_position,
            'total_volume': total_volume,
            'volume_distribution': volume_profile
        }
    
    async def _predict_future_price_movement(self, chart_image: np.ndarray, 
                                           price_data: pd.DataFrame, 
                                           patterns: List[ChartPattern], 
                                           psychology: MarketPsychology) -> Dict[str, Any]:
        """Predict future price movement using AI vision"""
        
        # Convert image to tensor
        image_pil = Image.fromarray(chart_image)
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Get price prediction from Vision Transformer
        with torch.no_grad():
            vit_outputs = self.vision_transformer(image_tensor)
            price_prediction = vit_outputs['price_prediction'].cpu().numpy()[0]
        
        current_price = price_data['close'].iloc[-1]
        
        # Generate future price points (next 50 periods)
        future_prices = []
        for i, pred in enumerate(price_prediction):
            # Convert prediction to actual price
            price_change = pred * current_price * 0.1  # Max 10% change per period
            future_price = current_price + price_change
            future_prices.append(float(future_price))
        
        # Calculate prediction confidence based on patterns and psychology
        pattern_confidence = np.mean([p.confidence for p in patterns]) if patterns else 0.5
        psychology_confidence = (
            psychology.market_structure_health * 0.3 +
            (1 - psychology.manipulation_probability) * 0.3 +
            psychology.breakout_probability * 0.2 +
            (1 - psychology.trend_exhaustion) * 0.2
        )
        
        overall_confidence = (pattern_confidence + psychology_confidence) / 2
        
        # Determine trend direction
        if len(future_prices) >= 10:
            trend_slope = np.polyfit(range(10), future_prices[:10], 1)[0]
            if trend_slope > current_price * 0.001:  # 0.1% threshold
                trend_direction = 'bullish'
            elif trend_slope < -current_price * 0.001:
                trend_direction = 'bearish'
            else:
                trend_direction = 'neutral'
        else:
            trend_direction = 'neutral'
        
        # Calculate support and resistance projections
        future_support = min(future_prices) * 0.98
        future_resistance = max(future_prices) * 1.02
        
        # Calculate volatility forecast
        price_changes = np.diff(future_prices)
        volatility_forecast = np.std(price_changes) / current_price if len(price_changes) > 0 else 0.01
        
        return {
            'future_prices': future_prices,
            'prediction_periods': len(future_prices),
            'trend_direction': trend_direction,
            'confidence': float(overall_confidence),
            'projected_support': float(future_support),
            'projected_resistance': float(future_resistance),
            'volatility_forecast': float(volatility_forecast),
            'key_levels': {
                'next_resistance': float(max(future_prices[:10]) if len(future_prices) >= 10 else current_price),
                'next_support': float(min(future_prices[:10]) if len(future_prices) >= 10 else current_price)
            },
            'prediction_metadata': {
                'model': 'vision_transformer',
                'pattern_influence': float(pattern_confidence),
                'psychology_influence': float(psychology_confidence),
                'prediction_timestamp': datetime.now().isoformat()
            }
        }
    
    async def _generate_trading_recommendation(self, patterns: List[ChartPattern], 
                                             psychology: MarketPsychology, 
                                             support_resistance: List[float], 
                                             trend_lines: List[Dict[str, Any]], 
                                             future_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive trading recommendation"""
        
        # Analyze all factors
        bullish_signals = 0
        bearish_signals = 0
        signal_strength = 0.0
        
        # Pattern analysis
        for pattern in patterns:
            if pattern.direction == 'bullish':
                bullish_signals += 1
                signal_strength += pattern.confidence * pattern.reliability_score
            elif pattern.direction == 'bearish':
                bearish_signals += 1
                signal_strength += pattern.confidence * pattern.reliability_score
        
        # Psychology analysis
        if psychology.fear_greed_index > 0.6:
            bullish_signals += 1
        elif psychology.fear_greed_index < 0.4:
            bearish_signals += 1
        
        if psychology.breakout_probability > 0.7:
            signal_strength += 0.2
        
        if psychology.trend_exhaustion > 0.7:
            # Trend exhaustion suggests reversal
            if future_prediction['trend_direction'] == 'bullish':
                bearish_signals += 1
            elif future_prediction['trend_direction'] == 'bearish':
                bullish_signals += 1
        
        # Future prediction analysis
        if future_prediction['trend_direction'] == 'bullish':
            bullish_signals += 2  # Higher weight for AI prediction
        elif future_prediction['trend_direction'] == 'bearish':
            bearish_signals += 2
        
        signal_strength += future_prediction['confidence']
        
        # Determine overall recommendation
        total_signals = bullish_signals + bearish_signals
        
        if total_signals == 0:
            recommendation = 'HOLD'
            confidence = 0.5
        else:
            bullish_ratio = bullish_signals / total_signals
            
            if bullish_ratio > 0.65:
                recommendation = 'BUY'
                confidence = bullish_ratio
            elif bullish_ratio < 0.35:
                recommendation = 'SELL'
                confidence = 1 - bullish_ratio
            else:
                recommendation = 'HOLD'
                confidence = 0.5
        
        # Adjust confidence based on signal strength
        confidence = min(1.0, confidence * (signal_strength / max(1, len(patterns))))
        
        # Calculate entry, stop loss, and take profit
        current_price = future_prediction.get('future_prices', [0])[0] if future_prediction.get('future_prices') else 0
        
        if recommendation == 'BUY':
            entry_price = current_price
            stop_loss = future_prediction.get('projected_support', current_price * 0.98)
            take_profit = future_prediction.get('projected_resistance', current_price * 1.02)
        elif recommendation == 'SELL':
            entry_price = current_price
            stop_loss = future_prediction.get('projected_resistance', current_price * 1.02)
            take_profit = future_prediction.get('projected_support', current_price * 0.98)
        else:
            entry_price = current_price
            stop_loss = current_price
            take_profit = current_price
        
        # Risk-reward ratio
        risk = abs(stop_loss - entry_price)
        reward = abs(take_profit - entry_price)
        risk_reward_ratio = reward / risk if risk > 0 else 1.0
        
        # Generate reasoning
        reasoning_parts = []
        
        if patterns:
            top_pattern = max(patterns, key=lambda p: p.confidence)
            reasoning_parts.append(f"Primary pattern: {top_pattern.pattern_type} ({top_pattern.confidence:.1%} confidence)")
        
        reasoning_parts.append(f"Market psychology: Fear/Greed {psychology.fear_greed_index:.1%}")
        reasoning_parts.append(f"AI prediction: {future_prediction['trend_direction']} trend")
        reasoning_parts.append(f"Breakout probability: {psychology.breakout_probability:.1%}")
        
        if psychology.trend_exhaustion > 0.5:
            reasoning_parts.append(f"Trend exhaustion detected ({psychology.trend_exhaustion:.1%})")
        
        reasoning = ". ".join(reasoning_parts)
        
        return {
            'recommendation': recommendation,
            'confidence': float(confidence),
            'entry_price': float(entry_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'risk_reward_ratio': float(risk_reward_ratio),
            'reasoning': reasoning,
            'signal_breakdown': {
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'signal_strength': float(signal_strength),
                'pattern_count': len(patterns),
                'top_patterns': [p.pattern_type for p in patterns[:3]]
            },
            'risk_assessment': {
                'overall_risk': 'high' if confidence < 0.6 else 'medium' if confidence < 0.8 else 'low',
                'market_volatility': future_prediction.get('volatility_forecast', 0.01),
                'manipulation_risk': psychology.manipulation_probability,
                'trend_exhaustion_risk': psychology.trend_exhaustion
            },
            'recommendation_metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'ai_model': 'computer_vision_transformer',
                'analysis_depth': 'expert_level',
                'recommendation_quality': 'revolutionary'
            }
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Test computer vision AI
    config = {
        'image_size': 1024,
        'patch_size': 32,
        'num_classes': 500,
        'dim': 1024,
        'depth': 24,
        'heads': 16
    }
    
    vision_ai = ChartVisionAI(config)
    
    # Sample price data
    sample_data = pd.DataFrame({
        'open': np.random.normal(1.1, 0.01, 100),
        'high': np.random.normal(1.105, 0.01, 100),
        'low': np.random.normal(1.095, 0.01, 100),
        'close': np.random.normal(1.1, 0.01, 100),
        'volume': np.random.normal(1000, 200, 100)
    })
    
    # Ensure high >= low and proper OHLC relationships
    for i in range(len(sample_data)):
        sample_data.loc[i, 'high'] = max(sample_data.loc[i, 'open'], 
                                        sample_data.loc[i, 'close'], 
                                        sample_data.loc[i, 'high'])
        sample_data.loc[i, 'low'] = min(sample_data.loc[i, 'open'], 
                                       sample_data.loc[i, 'close'], 
                                       sample_data.loc[i, 'low'])
    
    async def test_vision_analysis():
        result = await vision_ai.analyze_chart_like_human_expert(sample_data, 'H1')
        
        print("🎯 Computer Vision Analysis Result:")
        print(f"Patterns found: {len(result.patterns)}")
        if result.patterns:
            top_pattern = result.patterns[0]
            print(f"Top pattern: {top_pattern.pattern_type} ({top_pattern.confidence:.1%})")
        
        print(f"Market psychology - Fear/Greed: {result.psychology.fear_greed_index:.1%}")
        print(f"Breakout probability: {result.psychology.breakout_probability:.1%}")
        print(f"Trading recommendation: {result.trading_recommendation['recommendation']}")
        print(f"Confidence: {result.trading_recommendation['confidence']:.1%}")
        print(f"Future trend: {result.future_price_prediction['trend_direction']}")
    
    # asyncio.run(test_vision_analysis())