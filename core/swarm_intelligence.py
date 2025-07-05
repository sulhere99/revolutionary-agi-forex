"""
Swarm Intelligence Trading Network
==================================

Revolutionary Multi-Agent Swarm System dimana 1000+ AI agents berkolaborasi
seperti koloni semut atau lebah, masing-masing dengan specialization berbeda,
berkomunikasi dan berkoordinasi untuk menghasilkan collective intelligence
yang superior.
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
import random
import math
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import heapq
import time
from enum import Enum

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Types of agents in the swarm"""
    SCOUT = "scout"
    ANALYST = "analyst"
    RISK_MANAGER = "risk_manager"
    EXECUTOR = "executor"
    LEARNER = "learner"
    COORDINATOR = "coordinator"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"

class SwarmMessage:
    """Message structure for swarm communication"""
    def __init__(self, sender_id: str, receiver_id: str, message_type: str, 
                 content: Dict[str, Any], priority: int = 1, timestamp: datetime = None):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.priority = priority
        self.timestamp = timestamp or datetime.now()
        self.message_id = f"{sender_id}_{int(time.time() * 1000000)}"

@dataclass
class AgentState:
    """State of an individual agent"""
    agent_id: str
    agent_type: AgentType
    position: Tuple[float, float]  # Position in solution space
    fitness: float
    energy: float
    experience: int
    specialization_score: float
    collaboration_score: float
    last_active: datetime
    current_task: Optional[str] = None
    memory: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)

@dataclass
class SwarmDecision:
    """Collective decision made by the swarm"""
    decision_type: str
    action: str
    confidence: float
    consensus_level: float
    participating_agents: List[str]
    reasoning: Dict[str, Any]
    expected_outcome: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    timestamp: datetime
    execution_priority: int

class SwarmCommunicationProtocol:
    """Advanced communication protocol for swarm agents"""
    
    def __init__(self):
        self.message_queue = defaultdict(deque)
        self.broadcast_channels = defaultdict(list)
        self.message_history = deque(maxlen=10000)
        self.communication_graph = nx.Graph()
        self.message_lock = threading.Lock()
        
    def send_message(self, message: SwarmMessage):
        """Send message to specific agent"""
        with self.message_lock:
            self.message_queue[message.receiver_id].append(message)
            self.message_history.append(message)
            
            # Update communication graph
            self.communication_graph.add_edge(message.sender_id, message.receiver_id)
    
    def broadcast_message(self, sender_id: str, channel: str, content: Dict[str, Any]):
        """Broadcast message to all agents in channel"""
        message = SwarmMessage(
            sender_id=sender_id,
            receiver_id="broadcast",
            message_type="broadcast",
            content={"channel": channel, "data": content}
        )
        
        with self.message_lock:
            for agent_id in self.broadcast_channels[channel]:
                self.message_queue[agent_id].append(message)
            self.message_history.append(message)
    
    def receive_messages(self, agent_id: str) -> List[SwarmMessage]:
        """Receive all pending messages for agent"""
        with self.message_lock:
            messages = list(self.message_queue[agent_id])
            self.message_queue[agent_id].clear()
            return messages
    
    def subscribe_to_channel(self, agent_id: str, channel: str):
        """Subscribe agent to broadcast channel"""
        if agent_id not in self.broadcast_channels[channel]:
            self.broadcast_channels[channel].append(agent_id)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication network statistics"""
        return {
            'total_messages': len(self.message_history),
            'active_agents': len(self.message_queue),
            'network_density': nx.density(self.communication_graph),
            'average_clustering': nx.average_clustering(self.communication_graph),
            'network_diameter': nx.diameter(self.communication_graph) if nx.is_connected(self.communication_graph) else -1
        }

class CollectiveMemoryBank:
    """Shared memory system for the swarm"""
    
    def __init__(self):
        self.experiences = defaultdict(list)
        self.patterns = defaultdict(dict)
        self.successful_strategies = []
        self.failed_strategies = []
        self.market_conditions_memory = {}
        self.agent_performance_history = defaultdict(list)
        self.collective_knowledge = {}
        self.memory_lock = threading.Lock()
    
    def store_experience(self, agent_id: str, experience: Dict[str, Any]):
        """Store agent experience in collective memory"""
        with self.memory_lock:
            experience['timestamp'] = datetime.now()
            experience['agent_id'] = agent_id
            self.experiences[experience['type']].append(experience)
            
            # Update agent performance
            if 'performance' in experience:
                self.agent_performance_history[agent_id].append(experience['performance'])
    
    def store_pattern(self, pattern_type: str, pattern_data: Dict[str, Any]):
        """Store discovered pattern"""
        with self.memory_lock:
            pattern_id = f"{pattern_type}_{int(time.time())}"
            self.patterns[pattern_type][pattern_id] = {
                'data': pattern_data,
                'discovered_at': datetime.now(),
                'usage_count': 0,
                'success_rate': 0.0
            }
    
    def retrieve_similar_experiences(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve similar experiences from memory"""
        experience_type = query.get('type', 'general')
        experiences = self.experiences.get(experience_type, [])
        
        # Simple similarity matching (can be enhanced with ML)
        similar_experiences = []
        for exp in experiences[-100:]:  # Check recent experiences
            similarity = self._calculate_similarity(query, exp)
            if similarity > 0.5:  # Threshold
                similar_experiences.append((exp, similarity))
        
        # Sort by similarity and return top results
        similar_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp[0] for exp in similar_experiences[:limit]]
    
    def _calculate_similarity(self, query: Dict[str, Any], experience: Dict[str, Any]) -> float:
        """Calculate similarity between query and experience"""
        # Simple similarity based on common keys
        common_keys = set(query.keys()) & set(experience.keys())
        if not common_keys:
            return 0.0
        
        similarity_scores = []
        for key in common_keys:
            if isinstance(query[key], (int, float)) and isinstance(experience[key], (int, float)):
                # Numerical similarity
                max_val = max(abs(query[key]), abs(experience[key]), 1)
                similarity = 1 - abs(query[key] - experience[key]) / max_val
                similarity_scores.append(similarity)
            elif query[key] == experience[key]:
                # Exact match
                similarity_scores.append(1.0)
            else:
                # No match
                similarity_scores.append(0.0)
        
        return np.mean(similarity_scores)
    
    def get_collective_insights(self) -> Dict[str, Any]:
        """Get insights from collective memory"""
        with self.memory_lock:
            total_experiences = sum(len(exp_list) for exp_list in self.experiences.values())
            
            # Calculate success rates by strategy type
            strategy_performance = {}
            for strategy in self.successful_strategies + self.failed_strategies:
                strategy_type = strategy.get('type', 'unknown')
                if strategy_type not in strategy_performance:
                    strategy_performance[strategy_type] = {'success': 0, 'total': 0}
                
                strategy_performance[strategy_type]['total'] += 1
                if strategy in self.successful_strategies:
                    strategy_performance[strategy_type]['success'] += 1
            
            # Calculate success rates
            for strategy_type in strategy_performance:
                total = strategy_performance[strategy_type]['total']
                success = strategy_performance[strategy_type]['success']
                strategy_performance[strategy_type]['success_rate'] = success / total if total > 0 else 0
            
            return {
                'total_experiences': total_experiences,
                'pattern_count': sum(len(patterns) for patterns in self.patterns.values()),
                'strategy_performance': strategy_performance,
                'top_performing_agents': self._get_top_performing_agents(),
                'memory_efficiency': min(1.0, total_experiences / 10000)  # Efficiency metric
            }
    
    def _get_top_performing_agents(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top performing agents"""
        agent_scores = {}
        
        for agent_id, performances in self.agent_performance_history.items():
            if performances:
                avg_performance = np.mean(performances)
                consistency = 1 - np.std(performances) if len(performances) > 1 else 1
                agent_scores[agent_id] = {
                    'average_performance': avg_performance,
                    'consistency': consistency,
                    'total_actions': len(performances),
                    'composite_score': avg_performance * consistency
                }
        
        # Sort by composite score
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1]['composite_score'], reverse=True)
        
        return [{'agent_id': agent_id, **scores} for agent_id, scores in sorted_agents[:top_n]]

class BaseAgent(ABC):
    """Base class for all swarm agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType, swarm_communication: SwarmCommunicationProtocol,
                 collective_memory: CollectiveMemoryBank):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.swarm_communication = swarm_communication
        self.collective_memory = collective_memory
        
        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            position=(random.random(), random.random()),
            fitness=0.5,
            energy=1.0,
            experience=0,
            specialization_score=0.5,
            collaboration_score=0.5,
            last_active=datetime.now()
        )
        
        self.local_memory = {}
        self.task_queue = deque()
        self.is_active = True
        
        # Subscribe to relevant channels
        self.swarm_communication.subscribe_to_channel(agent_id, f"{agent_type.value}_channel")
        self.swarm_communication.subscribe_to_channel(agent_id, "global_channel")
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned task"""
        pass
    
    @abstractmethod
    def evaluate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate market data according to agent specialization"""
        pass
    
    async def run_agent_loop(self):
        """Main agent execution loop"""
        while self.is_active:
            try:
                # Process incoming messages
                await self._process_messages()
                
                # Execute pending tasks
                await self._execute_pending_tasks()
                
                # Update agent state
                self._update_state()
                
                # Learn from experiences
                await self._learn_from_experiences()
                
                # Collaborate with other agents
                await self._collaborate()
                
                # Rest period
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in agent {self.agent_id}: {e}")
                await asyncio.sleep(1)
    
    async def _process_messages(self):
        """Process incoming messages"""
        messages = self.swarm_communication.receive_messages(self.agent_id)
        
        for message in messages:
            await self._handle_message(message)
    
    async def _handle_message(self, message: SwarmMessage):
        """Handle individual message"""
        if message.message_type == "task_assignment":
            self.task_queue.append(message.content['task'])
        elif message.message_type == "collaboration_request":
            await self._handle_collaboration_request(message)
        elif message.message_type == "knowledge_sharing":
            self._incorporate_shared_knowledge(message.content)
        elif message.message_type == "broadcast":
            await self._handle_broadcast_message(message)
    
    async def _execute_pending_tasks(self):
        """Execute all pending tasks"""
        while self.task_queue:
            task = self.task_queue.popleft()
            try:
                result = await self.execute_task(task)
                
                # Store experience
                experience = {
                    'type': 'task_execution',
                    'task': task,
                    'result': result,
                    'performance': result.get('performance', 0.5)
                }
                self.collective_memory.store_experience(self.agent_id, experience)
                
                # Update fitness based on performance
                performance = result.get('performance', 0.5)
                self.state.fitness = 0.9 * self.state.fitness + 0.1 * performance
                
            except Exception as e:
                logger.error(f"Task execution failed for agent {self.agent_id}: {e}")
    
    def _update_state(self):
        """Update agent state"""
        self.state.last_active = datetime.now()
        self.state.experience += 1
        
        # Energy decay
        self.state.energy = max(0.1, self.state.energy - 0.001)
        
        # Specialization improvement
        if self.state.experience > 0:
            self.state.specialization_score = min(1.0, self.state.specialization_score + 0.001)
    
    async def _learn_from_experiences(self):
        """Learn from past experiences"""
        # Retrieve similar experiences
        query = {
            'type': 'task_execution',
            'agent_type': self.agent_type.value
        }
        
        similar_experiences = self.collective_memory.retrieve_similar_experiences(query, limit=5)
        
        # Learn from successful experiences
        for experience in similar_experiences:
            if experience.get('performance', 0) > 0.7:  # High performance threshold
                self._incorporate_learning(experience)
    
    def _incorporate_learning(self, experience: Dict[str, Any]):
        """Incorporate learning from experience"""
        # Update local memory with successful patterns
        pattern_key = f"successful_{experience['task'].get('type', 'general')}"
        if pattern_key not in self.local_memory:
            self.local_memory[pattern_key] = []
        
        self.local_memory[pattern_key].append(experience)
        
        # Keep only recent successful patterns
        if len(self.local_memory[pattern_key]) > 10:
            self.local_memory[pattern_key] = self.local_memory[pattern_key][-10:]
    
    async def _collaborate(self):
        """Collaborate with other agents"""
        # Find potential collaboration partners
        if random.random() < 0.1:  # 10% chance to initiate collaboration
            await self._initiate_collaboration()
    
    async def _initiate_collaboration(self):
        """Initiate collaboration with other agents"""
        # Simple collaboration: share knowledge
        knowledge_to_share = {
            'agent_type': self.agent_type.value,
            'fitness': self.state.fitness,
            'specialization': self.state.specialization_score,
            'recent_insights': list(self.local_memory.keys())
        }
        
        self.swarm_communication.broadcast_message(
            self.agent_id,
            "knowledge_sharing",
            knowledge_to_share
        )
    
    async def _handle_collaboration_request(self, message: SwarmMessage):
        """Handle collaboration request from another agent"""
        # Respond with willingness to collaborate based on compatibility
        sender_fitness = message.content.get('fitness', 0.5)
        compatibility = 1 - abs(self.state.fitness - sender_fitness)
        
        if compatibility > 0.5:  # Compatible agents
            response = SwarmMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type="collaboration_response",
                content={
                    'willing': True,
                    'compatibility': compatibility,
                    'specialization': self.state.specialization_score
                }
            )
            self.swarm_communication.send_message(response)
    
    def _incorporate_shared_knowledge(self, knowledge: Dict[str, Any]):
        """Incorporate knowledge shared by other agents"""
        # Update collaboration score
        self.state.collaboration_score = min(1.0, self.state.collaboration_score + 0.01)
        
        # Store shared insights
        if 'recent_insights' in knowledge:
            for insight in knowledge['recent_insights']:
                if insight not in self.local_memory:
                    self.local_memory[f"shared_{insight}"] = knowledge
    
    async def _handle_broadcast_message(self, message: SwarmMessage):
        """Handle broadcast message"""
        channel = message.content.get('channel', '')
        data = message.content.get('data', {})
        
        if channel == "market_update":
            await self._handle_market_update(data)
        elif channel == "emergency_signal":
            await self._handle_emergency_signal(data)
    
    async def _handle_market_update(self, market_data: Dict[str, Any]):
        """Handle market update broadcast"""
        # Evaluate market data according to specialization
        evaluation = self.evaluate_market_data(market_data)
        
        # Share evaluation if significant
        if evaluation.get('significance', 0) > 0.7:
            self.swarm_communication.broadcast_message(
                self.agent_id,
                f"{self.agent_type.value}_insights",
                evaluation
            )
    
    async def _handle_emergency_signal(self, emergency_data: Dict[str, Any]):
        """Handle emergency signal"""
        # Increase energy and prioritize emergency tasks
        self.state.energy = min(1.0, self.state.energy + 0.2)
        
        # Add emergency task to front of queue
        emergency_task = {
            'type': 'emergency_response',
            'data': emergency_data,
            'priority': 10
        }
        self.task_queue.appendleft(emergency_task)

class ScoutAgent(BaseAgent):
    """Scout agent for market exploration"""
    
    def __init__(self, agent_id: str, swarm_communication: SwarmCommunicationProtocol,
                 collective_memory: CollectiveMemoryBank):
        super().__init__(agent_id, AgentType.SCOUT, swarm_communication, collective_memory)
        self.exploration_radius = 0.1
        self.discovered_opportunities = []
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scouting task"""
        task_type = task.get('type', 'general_exploration')
        
        if task_type == 'market_exploration':
            return await self._explore_market_opportunities(task.get('market_data', {}))
        elif task_type == 'pattern_discovery':
            return await self._discover_patterns(task.get('data', {}))
        elif task_type == 'opportunity_validation':
            return await self._validate_opportunity(task.get('opportunity', {}))
        else:
            return await self._general_exploration(task)
    
    def evaluate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate market data for opportunities"""
        opportunities = []
        significance = 0.0
        
        # Look for volatility opportunities
        if 'volatility' in market_data:
            volatility = market_data['volatility']
            if volatility > 0.02:  # High volatility threshold
                opportunities.append({
                    'type': 'high_volatility',
                    'value': volatility,
                    'confidence': min(1.0, volatility / 0.05)
                })
                significance += 0.3
        
        # Look for volume anomalies
        if 'volume_ratio' in market_data:
            volume_ratio = market_data['volume_ratio']
            if volume_ratio > 1.5:  # High volume
                opportunities.append({
                    'type': 'volume_spike',
                    'value': volume_ratio,
                    'confidence': min(1.0, (volume_ratio - 1) / 2)
                })
                significance += 0.4
        
        # Look for price gaps
        if 'price_gap' in market_data:
            price_gap = abs(market_data['price_gap'])
            if price_gap > 0.001:  # Significant gap
                opportunities.append({
                    'type': 'price_gap',
                    'value': price_gap,
                    'confidence': min(1.0, price_gap / 0.005)
                })
                significance += 0.3
        
        return {
            'opportunities': opportunities,
            'significance': significance,
            'scout_confidence': self.state.specialization_score
        }
    
    async def _explore_market_opportunities(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explore market for trading opportunities"""
        evaluation = self.evaluate_market_data(market_data)
        opportunities = evaluation['opportunities']
        
        # Validate each opportunity
        validated_opportunities = []
        for opportunity in opportunities:
            validation_score = await self._validate_opportunity_internal(opportunity)
            if validation_score > 0.6:
                opportunity['validation_score'] = validation_score
                validated_opportunities.append(opportunity)
        
        # Store discovered opportunities
        self.discovered_opportunities.extend(validated_opportunities)
        
        # Keep only recent opportunities
        if len(self.discovered_opportunities) > 50:
            self.discovered_opportunities = self.discovered_opportunities[-50:]
        
        performance = len(validated_opportunities) / max(1, len(opportunities))
        
        return {
            'opportunities_found': len(validated_opportunities),
            'total_opportunities': len(opportunities),
            'validated_opportunities': validated_opportunities,
            'performance': performance,
            'exploration_quality': evaluation['significance']
        }
    
    async def _discover_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Discover patterns in market data"""
        patterns_found = []
        
        # Simple pattern discovery (can be enhanced with ML)
        if 'price_series' in data:
            price_series = data['price_series']
            
            # Look for trend patterns
            if len(price_series) >= 10:
                recent_trend = np.polyfit(range(10), price_series[-10:], 1)[0]
                if abs(recent_trend) > np.std(price_series) * 0.1:
                    patterns_found.append({
                        'type': 'trend_pattern',
                        'direction': 'up' if recent_trend > 0 else 'down',
                        'strength': abs(recent_trend),
                        'confidence': min(1.0, abs(recent_trend) / np.std(price_series))
                    })
            
            # Look for reversal patterns
            if len(price_series) >= 5:
                recent_changes = np.diff(price_series[-5:])
                if len(set(np.sign(recent_changes))) > 1:  # Direction changes
                    patterns_found.append({
                        'type': 'reversal_pattern',
                        'volatility': np.std(recent_changes),
                        'confidence': 0.6
                    })
        
        # Store patterns in collective memory
        for pattern in patterns_found:
            self.collective_memory.store_pattern(pattern['type'], pattern)
        
        performance = len(patterns_found) / 10  # Normalize to [0, 1]
        
        return {
            'patterns_discovered': len(patterns_found),
            'patterns': patterns_found,
            'performance': min(1.0, performance)
        }
    
    async def _validate_opportunity(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trading opportunity"""
        validation_score = await self._validate_opportunity_internal(opportunity)
        
        return {
            'opportunity': opportunity,
            'validation_score': validation_score,
            'is_valid': validation_score > 0.6,
            'performance': validation_score
        }
    
    async def _validate_opportunity_internal(self, opportunity: Dict[str, Any]) -> float:
        """Internal opportunity validation"""
        base_confidence = opportunity.get('confidence', 0.5)
        opportunity_type = opportunity.get('type', 'unknown')
        
        # Validation based on historical success
        similar_experiences = self.collective_memory.retrieve_similar_experiences({
            'type': 'opportunity_validation',
            'opportunity_type': opportunity_type
        }, limit=10)
        
        if similar_experiences:
            historical_success = np.mean([exp.get('performance', 0.5) for exp in similar_experiences])
            validation_score = 0.7 * base_confidence + 0.3 * historical_success
        else:
            validation_score = base_confidence
        
        # Add some randomness for exploration
        validation_score += random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, validation_score))
    
    async def _general_exploration(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General exploration task"""
        # Move in solution space
        self.state.position = (
            max(0, min(1, self.state.position[0] + random.uniform(-self.exploration_radius, self.exploration_radius))),
            max(0, min(1, self.state.position[1] + random.uniform(-self.exploration_radius, self.exploration_radius)))
        )
        
        # Simulate exploration result
        exploration_quality = random.uniform(0.3, 0.9)
        
        return {
            'exploration_completed': True,
            'new_position': self.state.position,
            'exploration_quality': exploration_quality,
            'performance': exploration_quality
        }

class AnalystAgent(BaseAgent):
    """Analyst agent for deep market analysis"""
    
    def __init__(self, agent_id: str, swarm_communication: SwarmCommunicationProtocol,
                 collective_memory: CollectiveMemoryBank):
        super().__init__(agent_id, AgentType.ANALYST, swarm_communication, collective_memory)
        self.analysis_models = {
            'trend_analyzer': self._create_trend_model(),
            'volatility_analyzer': self._create_volatility_model(),
            'sentiment_analyzer': self._create_sentiment_model()
        }
        self.analysis_history = deque(maxlen=100)
    
    def _create_trend_model(self):
        """Create trend analysis model"""
        return MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=1000, random_state=42)
    
    def _create_volatility_model(self):
        """Create volatility analysis model"""
        return RandomForestClassifier(n_estimators=50, random_state=42)
    
    def _create_sentiment_model(self):
        """Create sentiment analysis model"""
        return MLPClassifier(hidden_layer_sizes=(30, 20), max_iter=1000, random_state=42)
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis task"""
        task_type = task.get('type', 'general_analysis')
        
        if task_type == 'deep_market_analysis':
            return await self._deep_market_analysis(task.get('market_data', {}))
        elif task_type == 'trend_analysis':
            return await self._analyze_trend(task.get('price_data', []))
        elif task_type == 'volatility_analysis':
            return await self._analyze_volatility(task.get('price_data', []))
        elif task_type == 'sentiment_analysis':
            return await self._analyze_sentiment(task.get('sentiment_data', {}))
        elif task_type == 'correlation_analysis':
            return await self._analyze_correlations(task.get('data', {}))
        else:
            return await self._general_analysis(task)
    
    def evaluate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate market data with deep analysis"""
        analysis_results = {}
        significance = 0.0
        
        # Trend analysis
        if 'price_series' in market_data:
            trend_result = self._quick_trend_analysis(market_data['price_series'])
            analysis_results['trend'] = trend_result
            significance += trend_result.get('strength', 0) * 0.4
        
        # Volatility analysis
        if 'volatility' in market_data:
            vol_result = self._quick_volatility_analysis(market_data['volatility'])
            analysis_results['volatility'] = vol_result
            significance += vol_result.get('significance', 0) * 0.3
        
        # Volume analysis
        if 'volume_data' in market_data:
            volume_result = self._quick_volume_analysis(market_data['volume_data'])
            analysis_results['volume'] = volume_result
            significance += volume_result.get('significance', 0) * 0.3
        
        return {
            'analysis_results': analysis_results,
            'significance': significance,
            'analyst_confidence': self.state.specialization_score,
            'analysis_depth': 'deep'
        }
    
    def _quick_trend_analysis(self, price_series: List[float]) -> Dict[str, Any]:
        """Quick trend analysis"""
        if len(price_series) < 5:
            return {'trend': 'insufficient_data', 'strength': 0}
        
        # Calculate trend using linear regression
        x = np.arange(len(price_series))
        slope, intercept = np.polyfit(x, price_series, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((price_series - y_pred) ** 2)
        ss_tot = np.sum((price_series - np.mean(price_series)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine trend direction and strength
        if abs(slope) < np.std(price_series) * 0.01:
            trend = 'sideways'
        elif slope > 0:
            trend = 'uptrend'
        else:
            trend = 'downtrend'
        
        strength = min(1.0, abs(slope) / np.std(price_series))
        
        return {
            'trend': trend,
            'strength': strength,
            'slope': slope,
            'r_squared': r_squared,
            'confidence': r_squared
        }
    
    def _quick_volatility_analysis(self, volatility: float) -> Dict[str, Any]:
        """Quick volatility analysis"""
        # Classify volatility levels
        if volatility < 0.01:
            level = 'low'
            significance = 0.3
        elif volatility < 0.02:
            level = 'normal'
            significance = 0.5
        elif volatility < 0.04:
            level = 'high'
            significance = 0.8
        else:
            level = 'extreme'
            significance = 1.0
        
        return {
            'level': level,
            'value': volatility,
            'significance': significance,
            'trading_impact': 'high' if significance > 0.7 else 'medium' if significance > 0.4 else 'low'
        }
    
    def _quick_volume_analysis(self, volume_data: List[float]) -> Dict[str, Any]:
        """Quick volume analysis"""
        if len(volume_data) < 5:
            return {'significance': 0, 'pattern': 'insufficient_data'}
        
        recent_volume = np.mean(volume_data[-5:])
        historical_volume = np.mean(volume_data[:-5]) if len(volume_data) > 5 else recent_volume
        
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
        
        if volume_ratio > 1.5:
            pattern = 'volume_spike'
            significance = min(1.0, (volume_ratio - 1) / 2)
        elif volume_ratio < 0.7:
            pattern = 'volume_decline'
            significance = min(1.0, (1 - volume_ratio) / 0.3)
        else:
            pattern = 'normal_volume'
            significance = 0.3
        
        return {
            'pattern': pattern,
            'volume_ratio': volume_ratio,
            'significance': significance,
            'recent_volume': recent_volume
        }
    
    async def _deep_market_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep market analysis"""
        analysis_components = {}
        
        # Comprehensive trend analysis
        if 'price_series' in market_data:
            trend_analysis = await self._analyze_trend(market_data['price_series'])
            analysis_components['trend'] = trend_analysis
        
        # Volatility regime analysis
        if 'volatility_series' in market_data:
            vol_analysis = await self._analyze_volatility(market_data['volatility_series'])
            analysis_components['volatility'] = vol_analysis
        
        # Market microstructure analysis
        if 'order_book' in market_data:
            microstructure = self._analyze_microstructure(market_data['order_book'])
            analysis_components['microstructure'] = microstructure
        
        # Correlation analysis
        if 'correlation_data' in market_data:
            correlation = await self._analyze_correlations(market_data['correlation_data'])
            analysis_components['correlation'] = correlation
        
        # Synthesize analysis
        overall_assessment = self._synthesize_analysis(analysis_components)
        
        # Store analysis in history
        analysis_record = {
            'timestamp': datetime.now(),
            'components': analysis_components,
            'assessment': overall_assessment,
            'market_data_quality': self._assess_data_quality(market_data)
        }
        self.analysis_history.append(analysis_record)
        
        performance = overall_assessment.get('confidence', 0.5)
        
        return {
            'analysis_components': analysis_components,
            'overall_assessment': overall_assessment,
            'performance': performance,
            'analysis_quality': 'comprehensive'
        }
    
    async def _analyze_trend(self, price_data: List[float]) -> Dict[str, Any]:
        """Analyze trend in price data"""
        if len(price_data) < 10:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Multiple timeframe analysis
        short_term = self._quick_trend_analysis(price_data[-10:])
        medium_term = self._quick_trend_analysis(price_data[-20:]) if len(price_data) >= 20 else short_term
        long_term = self._quick_trend_analysis(price_data) if len(price_data) >= 30 else medium_term
        
        # Trend consistency
        trends = [short_term['trend'], medium_term['trend'], long_term['trend']]
        trend_consistency = len(set(trends)) == 1
        
        # Trend strength
        avg_strength = np.mean([short_term['strength'], medium_term['strength'], long_term['strength']])
        
        # Trend momentum
        momentum = self._calculate_momentum(price_data)
        
        return {
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term,
            'trend_consistency': trend_consistency,
            'average_strength': avg_strength,
            'momentum': momentum,
            'confidence': avg_strength * (1.1 if trend_consistency else 0.9)
        }
    
    def _calculate_momentum(self, price_data: List[float]) -> Dict[str, Any]:
        """Calculate price momentum"""
        if len(price_data) < 5:
            return {'value': 0, 'direction': 'neutral'}
        
        # Rate of change
        roc_5 = (price_data[-1] - price_data[-5]) / price_data[-5] if len(price_data) >= 5 else 0
        roc_10 = (price_data[-1] - price_data[-10]) / price_data[-10] if len(price_data) >= 10 else roc_5
        
        avg_momentum = (roc_5 + roc_10) / 2
        
        if avg_momentum > 0.01:
            direction = 'bullish'
        elif avg_momentum < -0.01:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        return {
            'value': abs(avg_momentum),
            'direction': direction,
            'roc_5': roc_5,
            'roc_10': roc_10
        }
    
    async def _analyze_volatility(self, volatility_data: List[float]) -> Dict[str, Any]:
        """Analyze volatility patterns"""
        if len(volatility_data) < 10:
            return {'error': 'Insufficient data for volatility analysis'}
        
        # Volatility statistics
        current_vol = volatility_data[-1]
        avg_vol = np.mean(volatility_data)
        vol_std = np.std(volatility_data)
        
        # Volatility regime
        if current_vol > avg_vol + vol_std:
            regime = 'high_volatility'
        elif current_vol < avg_vol - vol_std:
            regime = 'low_volatility'
        else:
            regime = 'normal_volatility'
        
        # Volatility trend
        vol_trend = self._quick_trend_analysis(volatility_data[-20:]) if len(volatility_data) >= 20 else {'trend': 'unknown'}
        
        # Volatility clustering
        clustering_score = self._detect_volatility_clustering(volatility_data)
        
        return {
            'current_volatility': current_vol,
            'average_volatility': avg_vol,
            'volatility_std': vol_std,
            'regime': regime,
            'volatility_trend': vol_trend,
            'clustering_score': clustering_score,
            'percentile': self._calculate_percentile(current_vol, volatility_data)
        }
    
    def _detect_volatility_clustering(self, volatility_data: List[float]) -> float:
        """Detect volatility clustering (GARCH effects)"""
        if len(volatility_data) < 10:
            return 0.0
        
        # Simple clustering detection using autocorrelation
        vol_changes = np.diff(volatility_data)
        if len(vol_changes) < 2:
            return 0.0
        
        # Calculate autocorrelation at lag 1
        autocorr = np.corrcoef(vol_changes[:-1], vol_changes[1:])[0, 1]
        
        return abs(autocorr) if not np.isnan(autocorr) else 0.0
    
    def _calculate_percentile(self, value: float, data: List[float]) -> float:
        """Calculate percentile of value in data"""
        if not data:
            return 0.5
        
        sorted_data = sorted(data)
        position = sum(1 for x in sorted_data if x <= value)
        
        return position / len(sorted_data)
    
    async def _analyze_sentiment(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market sentiment"""
        sentiment_score = 0.5  # Neutral default
        confidence = 0.5
        
        # News sentiment
        if 'news_sentiment' in sentiment_data:
            news_scores = sentiment_data['news_sentiment']
            if news_scores:
                sentiment_score = 0.4 * sentiment_score + 0.6 * np.mean(news_scores)
                confidence += 0.2
        
        # Social media sentiment
        if 'social_sentiment' in sentiment_data:
            social_scores = sentiment_data['social_sentiment']
            if social_scores:
                sentiment_score = 0.7 * sentiment_score + 0.3 * np.mean(social_scores)
                confidence += 0.1
        
        # Market sentiment indicators
        if 'market_indicators' in sentiment_data:
            indicators = sentiment_data['market_indicators']
            if indicators:
                indicator_score = np.mean(list(indicators.values()))
                sentiment_score = 0.8 * sentiment_score + 0.2 * indicator_score
                confidence += 0.2
        
        # Classify sentiment
        if sentiment_score > 0.6:
            sentiment_label = 'bullish'
        elif sentiment_score < 0.4:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': min(1.0, confidence),
            'components': sentiment_data
        }
    
    async def _analyze_correlations(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze correlations between assets"""
        correlations = {}
        
        if 'correlation_matrix' in correlation_data:
            corr_matrix = correlation_data['correlation_matrix']
            
            # Find highest correlations
            high_correlations = []
            for i, row in enumerate(corr_matrix):
                for j, corr in enumerate(row):
                    if i != j and abs(corr) > 0.7:  # High correlation threshold
                        high_correlations.append({
                            'asset1': i,
                            'asset2': j,
                            'correlation': corr,
                            'strength': 'high' if abs(corr) > 0.8 else 'medium'
                        })
            
            correlations['high_correlations'] = high_correlations
            correlations['average_correlation'] = np.mean(np.abs(corr_matrix))
        
        # Time-varying correlations
        if 'time_series_data' in correlation_data:
            time_series = correlation_data['time_series_data']
            rolling_correlations = self._calculate_rolling_correlations(time_series)
            correlations['rolling_correlations'] = rolling_correlations
        
        return correlations
    
    def _calculate_rolling_correlations(self, time_series: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate rolling correlations"""
        if len(time_series) < 2:
            return {}
        
        assets = list(time_series.keys())
        rolling_corrs = {}
        
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                series1, series2 = time_series[asset1], time_series[asset2]
                
                if len(series1) >= 20 and len(series2) >= 20:
                    # Calculate rolling correlation
                    window = 20
                    correlations = []
                    
                    for k in range(window, min(len(series1), len(series2))):
                        corr = np.corrcoef(series1[k-window:k], series2[k-window:k])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                    
                    if correlations:
                        rolling_corrs[f"{asset1}_{asset2}"] = {
                            'current_correlation': correlations[-1],
                            'average_correlation': np.mean(correlations),
                            'correlation_volatility': np.std(correlations),
                            'correlation_trend': 'increasing' if correlations[-1] > correlations[0] else 'decreasing'
                        }
        
        return rolling_corrs
    
    def _analyze_microstructure(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market microstructure"""
        microstructure = {}
        
        # Bid-ask spread analysis
        if 'bids' in order_book and 'asks' in order_book:
            bids = order_book['bids']
            asks = order_book['asks']
            
            if bids and asks:
                best_bid = max(bids, key=lambda x: x[0])[0]  # Highest bid price
                best_ask = min(asks, key=lambda x: x[0])[0]  # Lowest ask price
                
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
                spread_bps = (spread / mid_price) * 10000  # Basis points
                
                microstructure['spread_analysis'] = {
                    'spread': spread,
                    'spread_bps': spread_bps,
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'liquidity_quality': 'good' if spread_bps < 5 else 'poor'
                }
        
        # Order book depth
        if 'bids' in order_book and 'asks' in order_book:
            bid_depth = sum(order[1] for order in order_book['bids'])  # Total bid volume
            ask_depth = sum(order[1] for order in order_book['asks'])  # Total ask volume
            
            imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
            
            microstructure['depth_analysis'] = {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': bid_depth + ask_depth,
                'imbalance': imbalance,
                'imbalance_direction': 'bullish' if imbalance > 0.1 else 'bearish' if imbalance < -0.1 else 'neutral'
            }
        
        return microstructure
    
    def _synthesize_analysis(self, analysis_components: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize all analysis components into overall assessment"""
        signals = []
        confidence_scores = []
        
        # Extract signals from each component
        for component, analysis in analysis_components.items():
            if component == 'trend':
                trend_signal = self._extract_trend_signal(analysis)
                signals.append(trend_signal)
                confidence_scores.append(analysis.get('confidence', 0.5))
            
            elif component == 'volatility':
                vol_signal = self._extract_volatility_signal(analysis)
                signals.append(vol_signal)
                confidence_scores.append(0.7)  # Default confidence for volatility
            
            elif component == 'sentiment':
                sentiment_signal = self._extract_sentiment_signal(analysis)
                signals.append(sentiment_signal)
                confidence_scores.append(analysis.get('confidence', 0.5))
        
        # Aggregate signals
        if signals:
            bullish_signals = sum(1 for s in signals if s == 'bullish')
            bearish_signals = sum(1 for s in signals if s == 'bearish')
            neutral_signals = sum(1 for s in signals if s == 'neutral')
            
            total_signals = len(signals)
            
            if bullish_signals > bearish_signals and bullish_signals / total_signals > 0.5:
                overall_signal = 'bullish'
                signal_strength = bullish_signals / total_signals
            elif bearish_signals > bullish_signals and bearish_signals / total_signals > 0.5:
                overall_signal = 'bearish'
                signal_strength = bearish_signals / total_signals
            else:
                overall_signal = 'neutral'
                signal_strength = neutral_signals / total_signals
        else:
            overall_signal = 'neutral'
            signal_strength = 0.5
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return {
            'overall_signal': overall_signal,
            'signal_strength': signal_strength,
            'confidence': overall_confidence,
            'signal_breakdown': {
                'bullish': bullish_signals,
                'bearish': bearish_signals,
                'neutral': neutral_signals
            },
            'recommendation': self._generate_recommendation(overall_signal, signal_strength, overall_confidence)
        }
    
    def _extract_trend_signal(self, trend_analysis: Dict[str, Any]) -> str:
        """Extract signal from trend analysis"""
        if 'short_term' in trend_analysis:
            short_trend = trend_analysis['short_term']['trend']
            if short_trend in ['uptrend']:
                return 'bullish'
            elif short_trend in ['downtrend']:
                return 'bearish'
        
        return 'neutral'
    
    def _extract_volatility_signal(self, volatility_analysis: Dict[str, Any]) -> str:
        """Extract signal from volatility analysis"""
        regime = volatility_analysis.get('regime', 'normal_volatility')
        
        if regime == 'high_volatility':
            return 'neutral'  # High volatility is generally neutral for direction
        elif regime == 'low_volatility':
            return 'neutral'  # Low volatility suggests consolidation
        
        return 'neutral'
    
    def _extract_sentiment_signal(self, sentiment_analysis: Dict[str, Any]) -> str:
        """Extract signal from sentiment analysis"""
        return sentiment_analysis.get('sentiment_label', 'neutral')
    
    def _generate_recommendation(self, signal: str, strength: float, confidence: float) -> Dict[str, Any]:
        """Generate trading recommendation"""
        if signal == 'bullish' and strength > 0.6 and confidence > 0.7:
            action = 'BUY'
            conviction = 'high'
        elif signal == 'bearish' and strength > 0.6 and confidence > 0.7:
            action = 'SELL'
            conviction = 'high'
        elif signal in ['bullish', 'bearish'] and strength > 0.5 and confidence > 0.5:
            action = 'BUY' if signal == 'bullish' else 'SELL'
            conviction = 'medium'
        else:
            action = 'HOLD'
            conviction = 'low'
        
        return {
            'action': action,
            'conviction': conviction,
            'signal_strength': strength,
            'confidence': confidence
        }
    
    def _assess_data_quality(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality of market data"""
        quality_score = 0.0
        quality_factors = []
        
        # Check data completeness
        expected_fields = ['price_series', 'volume_data', 'volatility']
        present_fields = sum(1 for field in expected_fields if field in market_data)
        completeness = present_fields / len(expected_fields)
        quality_score += completeness * 0.4
        quality_factors.append(f"Completeness: {completeness:.1%}")
        
        # Check data recency
        if 'timestamp' in market_data:
            data_age = (datetime.now() - market_data['timestamp']).total_seconds() / 3600  # Hours
            recency_score = max(0, 1 - data_age / 24)  # Decay over 24 hours
            quality_score += recency_score * 0.3
            quality_factors.append(f"Recency: {recency_score:.1%}")
        
        # Check data consistency
        consistency_score = 0.8  # Default assumption
        if 'price_series' in market_data:
            price_series = market_data['price_series']
            if len(price_series) > 1:
                price_changes = np.diff(price_series)
                outliers = np.abs(price_changes) > 3 * np.std(price_changes)
                consistency_score = 1 - (np.sum(outliers) / len(price_changes))
        
        quality_score += consistency_score * 0.3
        quality_factors.append(f"Consistency: {consistency_score:.1%}")
        
        return {
            'overall_quality': quality_score,
            'quality_factors': quality_factors,
            'quality_grade': 'A' if quality_score > 0.8 else 'B' if quality_score > 0.6 else 'C'
        }
    
    async def _general_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General analysis task"""
        # Simulate analysis work
        analysis_complexity = task.get('complexity', 0.5)
        analysis_time = analysis_complexity * 2  # Seconds
        
        await asyncio.sleep(min(analysis_time, 5))  # Cap at 5 seconds
        
        # Generate analysis result
        analysis_quality = random.uniform(0.6, 0.95)  # Analysts are generally good
        
        return {
            'analysis_completed': True,
            'analysis_quality': analysis_quality,
            'complexity_handled': analysis_complexity,
            'performance': analysis_quality
        }

class RiskAgent(BaseAgent):
    """Risk management agent"""
    
    def __init__(self, agent_id: str, swarm_communication: SwarmCommunicationProtocol,
                 collective_memory: CollectiveMemoryBank):
        super().__init__(agent_id, AgentType.RISK_MANAGER, swarm_communication, collective_memory)
        self.risk_models = {
            'var_model': self._create_var_model(),
            'stress_test_model': self._create_stress_test_model(),
            'correlation_model': self._create_correlation_model()
        }
        self.risk_limits = {
            'max_position_size': 0.1,  # 10% of portfolio
            'max_daily_var': 0.02,     # 2% daily VaR
            'max_drawdown': 0.15,      # 15% max drawdown
            'max_correlation': 0.8     # 80% max correlation
        }
        self.current_risks = {}
        self.risk_alerts = deque(maxlen=100)
    
    def _create_var_model(self):
        """Create VaR calculation model"""
        return {
            'confidence_level': 0.95,
            'time_horizon': 1,  # 1 day
            'method': 'historical_simulation'
        }
    
    def _create_stress_test_model(self):
        """Create stress testing model"""
        return {
            'scenarios': [
                {'name': 'market_crash', 'shock': -0.2},
                {'name': 'volatility_spike', 'vol_multiplier': 3},
                {'name': 'liquidity_crisis', 'spread_multiplier': 5},
                {'name': 'correlation_breakdown', 'correlation_shock': 0.9}
            ]
        }
    
    def _create_correlation_model(self):
        """Create correlation monitoring model"""
        return {
            'lookback_period': 30,
            'correlation_threshold': 0.8,
            'monitoring_frequency': 'daily'
        }
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk management task"""
        task_type = task.get('type', 'general_risk_assessment')
        
        if task_type == 'portfolio_risk_assessment':
            return await self._assess_portfolio_risk(task.get('portfolio_data', {}))
        elif task_type == 'position_risk_check':
            return await self._check_position_risk(task.get('position_data', {}))
        elif task_type == 'var_calculation':
            return await self._calculate_var(task.get('price_data', []))
        elif task_type == 'stress_testing':
            return await self._perform_stress_test(task.get('portfolio_data', {}))
        elif task_type == 'correlation_monitoring':
            return await self._monitor_correlations(task.get('correlation_data', {}))
        elif task_type == 'risk_limit_monitoring':
            return await self._monitor_risk_limits(task.get('current_positions', {}))
        else:
            return await self._general_risk_assessment(task)
    
    def evaluate_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate market data for risk factors"""
        risk_factors = []
        overall_risk = 0.0
        
        # Volatility risk
        if 'volatility' in market_data:
            volatility = market_data['volatility']
            if volatility > 0.03:  # 3% daily volatility threshold
                risk_factors.append({
                    'type': 'high_volatility',
                    'value': volatility,
                    'severity': min(1.0, volatility / 0.05)
                })
                overall_risk += 0.4
        
        # Liquidity risk
        if 'bid_ask_spread' in market_data:
            spread = market_data['bid_ask_spread']
            if spread > 0.0005:  # 5 pips for major pairs
                risk_factors.append({
                    'type': 'liquidity_risk',
                    'value': spread,
                    'severity': min(1.0, spread / 0.002)
                })
                overall_risk += 0.3
        
        # Market stress indicators
        if 'market_stress_indicators' in market_data:
            stress_indicators = market_data['market_stress_indicators']
            avg_stress = np.mean(list(stress_indicators.values()))
            if avg_stress > 0.7:
                risk_factors.append({
                    'type': 'market_stress',
                    'value': avg_stress,
                    'severity': avg_stress
                })
                overall_risk += 0.3
        
        return {
            'risk_factors': risk_factors,
            'overall_risk_level': min(1.0, overall_risk),
            'risk_assessment': 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low',
            'risk_manager_confidence': self.state.specialization_score
        }
    
    async def _assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall portfolio risk"""
        risk_metrics = {}
        
        # Calculate portfolio VaR
        if 'positions' in portfolio_data and 'price_history' in portfolio_data:
            var_result = await self._calculate_portfolio_var(
                portfolio_data['positions'], 
                portfolio_data['price_history']
            )
            risk_metrics['var'] = var_result
        
        # Calculate portfolio concentration
        if 'positions' in portfolio_data:
            concentration = self._calculate_concentration_risk(portfolio_data['positions'])
            risk_metrics['concentration'] = concentration
        
        # Calculate correlation risk
        if 'correlation_matrix' in portfolio_data:
            correlation_risk = self._calculate_correlation_risk(
                portfolio_data['correlation_matrix'],
                portfolio_data.get('positions', {})
            )
            risk_metrics['correlation'] = correlation_risk
        
        # Calculate maximum drawdown
        if 'portfolio_returns' in portfolio_data:
            drawdown = self._calculate_max_drawdown(portfolio_data['portfolio_returns'])
            risk_metrics['drawdown'] = drawdown
        
        # Overall risk assessment
        overall_assessment = self._synthesize_risk_assessment(risk_metrics)
        
        # Check against risk limits
        limit_violations = self._check_risk_limits(risk_metrics)
        
        performance = 1.0 - overall_assessment.get('risk_score', 0.5)  # Lower risk = better performance
        
        return {
            'risk_metrics': risk_metrics,
            'overall_assessment': overall_assessment,
            'limit_violations': limit_violations,
            'performance': performance,
            'recommendations': self._generate_risk_recommendations(risk_metrics, limit_violations)
        }
    
    async def _check_position_risk(self, position_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check risk for individual position"""
        position_risks = {}
        
        # Position size risk
        position_size = position_data.get('size', 0)
        portfolio_value = position_data.get('portfolio_value', 1)
        size_ratio = abs(position_size) / portfolio_value if portfolio_value > 0 else 0
        
        position_risks['size_risk'] = {
            'size_ratio': size_ratio,
            'risk_level': 'high' if size_ratio > self.risk_limits['max_position_size'] else 'normal',
            'limit_utilization': size_ratio / self.risk_limits['max_position_size']
        }
        
        # Stop loss risk
        if 'entry_price' in position_data and 'stop_loss' in position_data:
            entry_price = position_data['entry_price']
            stop_loss = position_data['stop_loss']
            
            if entry_price > 0:
                stop_loss_distance = abs(stop_loss - entry_price) / entry_price
                position_risks['stop_loss_risk'] = {
                    'distance': stop_loss_distance,
                    'risk_level': 'high' if stop_loss_distance > 0.05 else 'normal'  # 5% threshold
                }
        
        # Volatility risk
        if 'volatility' in position_data:
            volatility = position_data['volatility']
            position_risks['volatility_risk'] = {
                'volatility': volatility,
                'risk_level': 'high' if volatility > 0.03 else 'normal'
            }
        
        # Overall position risk
        risk_scores = []
        for risk_type, risk_data in position_risks.items():
            if risk_data.get('risk_level') == 'high':
                risk_scores.append(0.8)
            else:
                risk_scores.append(0.3)
        
        overall_risk = np.mean(risk_scores) if risk_scores else 0.3
        
        return {
            'position_risks': position_risks,
            'overall_risk': overall_risk,
            'risk_level': 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low',
            'performance': 1.0 - overall_risk
        }
    
    async def _calculate_var(self, price_data: List[float]) -> Dict[str, Any]:
        """Calculate Value at Risk"""
        if len(price_data) < 30:
            return {'error': 'Insufficient data for VaR calculation'}
        
        # Calculate returns
        returns = np.diff(price_data) / price_data[:-1]
        
        # Historical simulation VaR
        confidence_level = self.risk_models['var_model']['confidence_level']
        var_percentile = (1 - confidence_level) * 100
        
        historical_var = np.percentile(returns, var_percentile)
        
        # Parametric VaR (assuming normal distribution)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)
        parametric_var = mean_return + z_score * std_return
        
        # Expected Shortfall (Conditional VaR)
        tail_returns = returns[returns <= historical_var]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else historical_var
        
        # VaR in monetary terms (assuming $1 position)
        var_dollar = abs(historical_var)
        es_dollar = abs(expected_shortfall)
        
        return {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'expected_shortfall': expected_shortfall,
            'var_dollar': var_dollar,
            'es_dollar': es_dollar,
            'confidence_level': confidence_level,
            'sample_size': len(returns)
        }
    
    async def _calculate_portfolio_var(self, positions: Dict[str, float], 
                                     price_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate portfolio-level VaR"""
        if not positions or not price_history:
            return {'error': 'Insufficient data for portfolio VaR'}
        
        # Calculate individual asset returns
        asset_returns = {}
        for asset, prices in price_history.items():
            if len(prices) >= 30:
                returns = np.diff(prices) / prices[:-1]
                asset_returns[asset] = returns
        
        if not asset_returns:
            return {'error': 'No sufficient price history for any asset'}
        
        # Align return series (use shortest series length)
        min_length = min(len(returns) for returns in asset_returns.values())
        aligned_returns = {asset: returns[-min_length:] for asset, returns in asset_returns.items()}
        
        # Create portfolio returns
        portfolio_returns = np.zeros(min_length)
        total_position_value = sum(abs(pos) for pos in positions.values())
        
        for asset, position in positions.items():
            if asset in aligned_returns and total_position_value > 0:
                weight = position / total_position_value
                portfolio_returns += weight * aligned_returns[asset]
        
        # Calculate portfolio VaR
        confidence_level = 0.95
        var_percentile = (1 - confidence_level) * 100
        portfolio_var = np.percentile(portfolio_returns, var_percentile)
        
        # Expected Shortfall
        tail_returns = portfolio_returns[portfolio_returns <= portfolio_var]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else portfolio_var
        
        return {
            'portfolio_var': abs(portfolio_var),
            'expected_shortfall': abs(expected_shortfall),
            'portfolio_volatility': np.std(portfolio_returns),
            'confidence_level': confidence_level,
            'observation_period': min_length
        }
    
    def _calculate_concentration_risk(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate portfolio concentration risk"""
        if not positions:
            return {'concentration_score': 0, 'risk_level': 'low'}
        
        # Calculate position weights
        total_value = sum(abs(pos) for pos in positions.values())
        weights = {asset: abs(pos) / total_value for asset, pos in positions.items()} if total_value > 0 else {}
        
        if not weights:
            return {'concentration_score': 0, 'risk_level': 'low'}
        
        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(weight ** 2 for weight in weights.values())
        
        # Concentration score (0 = perfectly diversified, 1 = fully concentrated)
        max_weight = max(weights.values())
        
        # Risk assessment
        if hhi > 0.5 or max_weight > 0.4:
            risk_level = 'high'
        elif hhi > 0.25 or max_weight > 0.2:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'hhi': hhi,
            'max_weight': max_weight,
            'concentration_score': hhi,
            'risk_level': risk_level,
            'number_of_positions': len(positions),
            'effective_positions': 1 / hhi if hhi > 0 else len(positions)
        }
    
    def _calculate_correlation_risk(self, correlation_matrix: List[List[float]], 
                                  positions: Dict[str, float]) -> Dict[str, Any]:
        """Calculate correlation risk"""
        if not correlation_matrix or not positions:
            return {'correlation_risk': 0, 'risk_level': 'low'}
        
        # Convert to numpy array
        corr_matrix = np.array(correlation_matrix)
        
        # Calculate average correlation
        n = corr_matrix.shape[0]
        if n <= 1:
            return {'correlation_risk': 0, 'risk_level': 'low'}
        
        # Average correlation (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        avg_correlation = np.mean(np.abs(corr_matrix[mask]))
        
        # Maximum correlation
        max_correlation = np.max(np.abs(corr_matrix[mask]))
        
        # Risk assessment
        if avg_correlation > 0.7 or max_correlation > 0.9:
            risk_level = 'high'
        elif avg_correlation > 0.5 or max_correlation > 0.8:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'average_correlation': avg_correlation,
            'maximum_correlation': max_correlation,
            'correlation_risk': avg_correlation,
            'risk_level': risk_level,
            'diversification_ratio': 1 - avg_correlation
        }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> Dict[str, Any]:
        """Calculate maximum drawdown"""
        if len(returns) < 2:
            return {'max_drawdown': 0, 'risk_level': 'low'}
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + np.array(returns))
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = np.min(drawdowns)
        
        # Current drawdown
        current_drawdown = drawdowns[-1]
        
        # Risk assessment
        if abs(max_drawdown) > 0.2:  # 20% drawdown
            risk_level = 'high'
        elif abs(max_drawdown) > 0.1:  # 10% drawdown
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'max_drawdown': abs(max_drawdown),
            'current_drawdown': abs(current_drawdown),
            'risk_level': risk_level,
            'drawdown_duration': self._calculate_drawdown_duration(drawdowns)
        }
    
    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate current drawdown duration"""
        # Find the last peak (drawdown = 0)
        last_peak_index = -1
        for i in range(len(drawdowns) - 1, -1, -1):
            if drawdowns[i] >= -0.001:  # Close to zero (allowing for small numerical errors)
                last_peak_index = i
                break
        
        if last_peak_index == -1:
            return len(drawdowns)  # Entire period is in drawdown
        
        return len(drawdowns) - 1 - last_peak_index
    
    def _synthesize_risk_assessment(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize overall risk assessment"""
        risk_scores = []
        risk_components = []
        
        # VaR component
        if 'var' in risk_metrics:
            var_data = risk_metrics['var']
            if 'portfolio_var' in var_data:
                var_score = min(1.0, var_data['portfolio_var'] / self.risk_limits['max_daily_var'])
                risk_scores.append(var_score)
                risk_components.append(f"VaR: {var_score:.1%}")
        
        # Concentration component
        if 'concentration' in risk_metrics:
            conc_data = risk_metrics['concentration']
            conc_score = conc_data.get('concentration_score', 0)
            risk_scores.append(conc_score)
            risk_components.append(f"Concentration: {conc_score:.1%}")
        
        # Correlation component
        if 'correlation' in risk_metrics:
            corr_data = risk_metrics['correlation']
            corr_score = corr_data.get('correlation_risk', 0)
            risk_scores.append(corr_score)
            risk_components.append(f"Correlation: {corr_score:.1%}")
        
        # Drawdown component
        if 'drawdown' in risk_metrics:
            dd_data = risk_metrics['drawdown']
            dd_score = dd_data.get('max_drawdown', 0) / self.risk_limits['max_drawdown']
            risk_scores.append(dd_score)
            risk_components.append(f"Drawdown: {dd_score:.1%}")
        
        # Overall risk score
        overall_risk = np.mean(risk_scores) if risk_scores else 0.5
        
        # Risk level classification
        if overall_risk > 0.8:
            risk_level = 'critical'
        elif overall_risk > 0.6:
            risk_level = 'high'
        elif overall_risk > 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': overall_risk,
            'risk_level': risk_level,
            'risk_components': risk_components,
            'risk_grade': 'A' if overall_risk < 0.3 else 'B' if overall_risk < 0.6 else 'C' if overall_risk < 0.8 else 'D'
        }
    
    def _check_risk_limits(self, risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for risk limit violations"""
        violations = []
        
        # Check VaR limit
        if 'var' in risk_metrics:
            var_data = risk_metrics['var']
            if 'portfolio_var' in var_data:
                portfolio_var = var_data['portfolio_var']
                if portfolio_var > self.risk_limits['max_daily_var']:
                    violations.append({
                        'type': 'var_limit',
                        'current_value': portfolio_var,
                        'limit': self.risk_limits['max_daily_var'],
                        'severity': 'high',
                        'excess': portfolio_var - self.risk_limits['max_daily_var']
                    })
        
        # Check concentration limit
        if 'concentration' in risk_metrics:
            conc_data = risk_metrics['concentration']
            max_weight = conc_data.get('max_weight', 0)
            if max_weight > self.risk_limits['max_position_size']:
                violations.append({
                    'type': 'concentration_limit',
                    'current_value': max_weight,
                    'limit': self.risk_limits['max_position_size'],
                    'severity': 'medium',
                    'excess': max_weight - self.risk_limits['max_position_size']
                })
        
        # Check drawdown limit
        if 'drawdown' in risk_metrics:
            dd_data = risk_metrics['drawdown']
            max_drawdown = dd_data.get('max_drawdown', 0)
            if max_drawdown > self.risk_limits['max_drawdown']:
                violations.append({
                    'type': 'drawdown_limit',
                    'current_value': max_drawdown,
                    'limit': self.risk_limits['max_drawdown'],
                    'severity': 'critical',
                    'excess': max_drawdown - self.risk_limits['max_drawdown']
                })
        
        # Check correlation limit
        if 'correlation' in risk_metrics:
            corr_data = risk_metrics['correlation']
            max_correlation = corr_data.get('maximum_correlation', 0)
            if max_correlation > self.risk_limits['max_correlation']:
                violations.append({
                    'type': 'correlation_limit',
                    'current_value': max_correlation,
                    'limit': self.risk_limits['max_correlation'],
                    'severity': 'medium',
                    'excess': max_correlation - self.risk_limits['max_correlation']
                })
        
        return violations
    
    def _generate_risk_recommendations(self, risk_metrics: Dict[str, Any], 
                                     violations: List[Dict[str, Any]]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # Recommendations based on violations
        for violation in violations:
            if violation['type'] == 'var_limit':
                recommendations.append("Reduce position sizes to lower portfolio VaR")
            elif violation['type'] == 'concentration_limit':
                recommendations.append("Diversify portfolio to reduce concentration risk")
            elif violation['type'] == 'drawdown_limit':
                recommendations.append("Implement stop-loss measures to limit further drawdown")
            elif violation['type'] == 'correlation_limit':
                recommendations.append("Add uncorrelated assets to reduce correlation risk")
        
        # General recommendations based on risk levels
        if 'concentration' in risk_metrics:
            conc_data = risk_metrics['concentration']
            if conc_data.get('risk_level') == 'high':
                recommendations.append("Consider adding more positions to improve diversification")
        
        if 'correlation' in risk_metrics:
            corr_data = risk_metrics['correlation']
            if corr_data.get('risk_level') == 'high':
                recommendations.append("Monitor correlation changes and adjust positions accordingly")
        
        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append("Current risk levels are within acceptable limits")
        
        return recommendations
    
    async def _perform_stress_test(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform stress testing on portfolio"""
        stress_results = {}
        
        scenarios = self.risk_models['stress_test_model']['scenarios']
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            scenario_result = await self._run_stress_scenario(portfolio_data, scenario)
            stress_results[scenario_name] = scenario_result
        
        # Find worst-case scenario
        worst_scenario = min(stress_results.items(), 
                           key=lambda x: x[1].get('portfolio_value_change', 0))
        
        # Calculate stress test summary
        avg_loss = np.mean([result.get('portfolio_value_change', 0) 
                           for result in stress_results.values()])
        
        return {
            'scenario_results': stress_results,
            'worst_case_scenario': {
                'name': worst_scenario[0],
                'result': worst_scenario[1]
            },
            'average_loss': avg_loss,
            'stress_test_grade': 'A' if avg_loss > -0.1 else 'B' if avg_loss > -0.2 else 'C',
            'performance': max(0, 1 + avg_loss)  # Convert loss to performance score
        }
    
    async def _run_stress_scenario(self, portfolio_data: Dict[str, Any], 
                                 scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run individual stress test scenario"""
        scenario_name = scenario['name']
        
        if scenario_name == 'market_crash':
            # Apply market shock
            shock = scenario['shock']
            portfolio_change = shock  # Simplified: assume all positions move together
            
        elif scenario_name == 'volatility_spike':
            # Increase volatility
            vol_multiplier = scenario['vol_multiplier']
            # Simplified: higher volatility increases potential loss
            portfolio_change = -0.05 * vol_multiplier
            
        elif scenario_name == 'liquidity_crisis':
            # Increase spreads
            spread_multiplier = scenario['spread_multiplier']
            # Simplified: wider spreads increase trading costs
            portfolio_change = -0.01 * spread_multiplier
            
        elif scenario_name == 'correlation_breakdown':
            # Correlations go to extreme
            correlation_shock = scenario['correlation_shock']
            # Simplified: high correlation reduces diversification benefit
            portfolio_change = -0.03 * correlation_shock
            
        else:
            portfolio_change = 0
        
        return {
            'scenario_name': scenario_name,
            'portfolio_value_change': portfolio_change,
            'scenario_parameters': scenario,
            'impact_severity': 'high' if portfolio_change < -0.1 else 'medium' if portfolio_change < -0.05 else 'low'
        }
    
    async def _monitor_correlations(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor correlation changes"""
        if 'current_correlations' not in correlation_data:
            return {'error': 'No correlation data provided'}
        
        current_corr = correlation_data['current_correlations']
        historical_corr = correlation_data.get('historical_correlations', current_corr)
        
        # Calculate correlation changes
        correlation_changes = {}
        alerts = []
        
        for asset_pair, current_value in current_corr.items():
            historical_value = historical_corr.get(asset_pair, current_value)
            change = current_value - historical_value
            
            correlation_changes[asset_pair] = {
                'current': current_value,
                'historical': historical_value,
                'change': change,
                'change_magnitude': abs(change)
            }
            
            # Check for significant changes
            if abs(change) > 0.2:  # 20% correlation change
                alerts.append({
                    'asset_pair': asset_pair,
                    'type': 'correlation_change',
                    'severity': 'high' if abs(change) > 0.4 else 'medium',
                    'change': change,
                    'current_correlation': current_value
                })
        
        # Overall correlation monitoring result
        avg_correlation_change = np.mean([data['change_magnitude'] 
                                        for data in correlation_changes.values()])
        
        return {
            'correlation_changes': correlation_changes,
            'alerts': alerts,
            'average_change': avg_correlation_change,
            'monitoring_status': 'alert' if alerts else 'normal',
            'performance': max(0, 1 - avg_correlation_change)
        }
    
    async def _monitor_risk_limits(self, current_positions: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor risk limits in real-time"""
        limit_status = {}
        violations = []
        warnings = []
        
        # Check position size limits
        for asset, position_data in current_positions.items():
            position_size = position_data.get('size', 0)
            portfolio_value = position_data.get('portfolio_value', 1)
            
            size_ratio = abs(position_size) / portfolio_value if portfolio_value > 0 else 0
            limit_utilization = size_ratio / self.risk_limits['max_position_size']
            
            limit_status[f"{asset}_position_size"] = {
                'current_value': size_ratio,
                'limit': self.risk_limits['max_position_size'],
                'utilization': limit_utilization,
                'status': 'violation' if limit_utilization > 1 else 'warning' if limit_utilization > 0.8 else 'normal'
            }
            
            if limit_utilization > 1:
                violations.append({
                    'asset': asset,
                    'type': 'position_size',
                    'current': size_ratio,
                    'limit': self.risk_limits['max_position_size']
                })
            elif limit_utilization > 0.8:
                warnings.append({
                    'asset': asset,
                    'type': 'position_size_warning',
                    'current': size_ratio,
                    'limit': self.risk_limits['max_position_size']
                })
        
        # Overall monitoring status
        if violations:
            monitoring_status = 'critical'
        elif warnings:
            monitoring_status = 'warning'
        else:
            monitoring_status = 'normal'
        
        return {
            'limit_status': limit_status,
            'violations': violations,
            'warnings': warnings,
            'monitoring_status': monitoring_status,
            'performance': 1.0 if monitoring_status == 'normal' else 0.5 if monitoring_status == 'warning' else 0.0
        }
    
    async def _general_risk_assessment(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General risk assessment task"""
        # Simulate risk assessment work
        risk_complexity = task.get('complexity', 0.5)
        assessment_time = risk_complexity * 1.5  # Seconds
        
        await asyncio.sleep(min(assessment_time, 3))  # Cap at 3 seconds
        
        # Generate risk assessment result
        risk_level = random.uniform(0.2, 0.8)  # Risk managers are conservative
        
        return {
            'risk_assessment_completed': True,
            'risk_level': risk_level,
            'risk_grade': 'A' if risk_level < 0.3 else 'B' if risk_level < 0.6 else 'C',
            'performance': 1.0 - risk_level
        }

class SwarmCoordinator:
    """Coordinator for managing swarm operations"""
    
    def __init__(self, swarm_communication: SwarmCommunicationProtocol,
                 collective_memory: CollectiveMemoryBank):
        self.swarm_communication = swarm_communication
        self.collective_memory = collective_memory
        self.agents = {}
        self.task_queue = deque()
        self.active_decisions = {}
        self.coordination_history = deque(maxlen=1000)
        
    def register_agent(self, agent: BaseAgent):
        """Register agent with coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Agent {agent.agent_id} ({agent.agent_type.value}) registered")
    
    async def coordinate_swarm_decision(self, decision_request: Dict[str, Any]) -> SwarmDecision:
        """Coordinate swarm decision making"""
        decision_id = f"decision_{int(time.time() * 1000)}"
        
        # Phase 1: Broadcast decision request
        self.swarm_communication.broadcast_message(
            "coordinator",
            "decision_request",
            {
                'decision_id': decision_id,
                'request': decision_request,
                'deadline': (datetime.now() + timedelta(seconds=30)).isoformat()
            }
        )
        
        # Phase 2: Collect agent responses
        responses = await self._collect_agent_responses(decision_id, timeout=30)
        
        # Phase 3: Analyze responses and reach consensus
        consensus_result = self._analyze_consensus(responses, decision_request)
        
        # Phase 4: Create swarm decision
        swarm_decision = SwarmDecision(
            decision_type=decision_request.get('type', 'general'),
            action=consensus_result['action'],
            confidence=consensus_result['confidence'],
            consensus_level=consensus_result['consensus_level'],
            participating_agents=list(responses.keys()),
            reasoning=consensus_result['reasoning'],
            expected_outcome=consensus_result['expected_outcome'],
            risk_assessment=consensus_result['risk_assessment'],
            timestamp=datetime.now(),
            execution_priority=consensus_result.get('priority', 1)
        )
        
        # Store decision in history
        self.coordination_history.append({
            'decision': swarm_decision,
            'responses': responses,
            'consensus_analysis': consensus_result
        })
        
        return swarm_decision
    
    async def _collect_agent_responses(self, decision_id: str, timeout: int = 30) -> Dict[str, Any]:
        """Collect responses from agents"""
        responses = {}
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for responses from each agent
            for agent_id, agent in self.agents.items():
                if agent_id not in responses:
                    # Check if agent has responded
                    messages = self.swarm_communication.receive_messages("coordinator")
                    
                    for message in messages:
                        if (message.message_type == "decision_response" and 
                            message.content.get('decision_id') == decision_id):
                            responses[message.sender_id] = message.content['response']
            
            # Check if we have enough responses
            if len(responses) >= len(self.agents) * 0.7:  # 70% response rate
                break
            
            await asyncio.sleep(0.5)
        
        return responses
    
    def _analyze_consensus(self, responses: Dict[str, Any], 
                          decision_request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent responses to reach consensus"""
        
        if not responses:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'consensus_level': 0.0,
                'reasoning': {'error': 'No agent responses received'},
                'expected_outcome': {},
                'risk_assessment': {'risk_level': 'unknown'}
            }
        
        # Collect votes
        votes = {}
        confidence_scores = []
        risk_assessments = []
        
        for agent_id, response in responses.items():
            action = response.get('action', 'HOLD')
            confidence = response.get('confidence', 0.5)
            risk = response.get('risk_assessment', {})
            
            if action not in votes:
                votes[action] = []
            votes[action].append({
                'agent_id': agent_id,
                'confidence': confidence,
                'reasoning': response.get('reasoning', '')
            })
            
            confidence_scores.append(confidence)
            risk_assessments.append(risk)
        
        # Determine winning action
        if not votes:
            winning_action = 'HOLD'
            consensus_level = 0.0
        else:
            # Weight votes by confidence
            weighted_votes = {}
            for action, agent_votes in votes.items():
                total_weight = sum(vote['confidence'] for vote in agent_votes)
                weighted_votes[action] = total_weight
            
            winning_action = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
            
            # Calculate consensus level
            winning_votes = len(votes[winning_action])
            total_votes = sum(len(agent_votes) for agent_votes in votes.values())
            consensus_level = winning_votes / total_votes if total_votes > 0 else 0
        
        # Calculate overall confidence
        overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.5
        
        # Aggregate risk assessment
        aggregated_risk = self._aggregate_risk_assessments(risk_assessments)
        
        # Generate reasoning
        reasoning = {
            'vote_breakdown': votes,
            'weighted_votes': weighted_votes if 'weighted_votes' in locals() else {},
            'consensus_mechanism': 'weighted_voting',
            'participating_agents': len(responses)
        }
        
        # Expected outcome
        expected_outcome = {
            'success_probability': overall_confidence * consensus_level,
            'expected_return': self._estimate_expected_return(winning_action, responses),
            'time_horizon': decision_request.get('time_horizon', 'short_term')
        }
        
        return {
            'action': winning_action,
            'confidence': overall_confidence,
            'consensus_level': consensus_level,
            'reasoning': reasoning,
            'expected_outcome': expected_outcome,
            'risk_assessment': aggregated_risk,
            'priority': self._calculate_priority(winning_action, consensus_level, overall_confidence)
        }
    
    def _aggregate_risk_assessments(self, risk_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate risk assessments from multiple agents"""
        if not risk_assessments:
            return {'risk_level': 'unknown', 'confidence': 0.0}
        
        # Extract risk levels
        risk_levels = []
        for assessment in risk_assessments:
            risk_level = assessment.get('risk_level', 'medium')
            if risk_level == 'low':
                risk_levels.append(0.2)
            elif risk_level == 'medium':
                risk_levels.append(0.5)
            elif risk_level == 'high':
                risk_levels.append(0.8)
            else:
                risk_levels.append(0.5)  # Default
        
        # Calculate average risk
        avg_risk = np.mean(risk_levels)
        
        # Convert back to categorical
        if avg_risk < 0.35:
            aggregated_level = 'low'
        elif avg_risk < 0.65:
            aggregated_level = 'medium'
        else:
            aggregated_level = 'high'
        
        # Calculate confidence in risk assessment
        risk_std = np.std(risk_levels)
        risk_confidence = max(0, 1 - risk_std)  # Lower std = higher confidence
        
        return {
            'risk_level': aggregated_level,
            'risk_score': avg_risk,
            'confidence': risk_confidence,
            'assessment_count': len(risk_assessments)
        }
    
    def _estimate_expected_return(self, action: str, responses: Dict[str, Any]) -> float:
        """Estimate expected return for the action"""
        if action == 'HOLD':
            return 0.0
        
        # Extract return estimates from agent responses
        return_estimates = []
        for response in responses.values():
            if response.get('action') == action:
                expected_return = response.get('expected_return', 0.0)
                return_estimates.append(expected_return)
        
        return np.mean(return_estimates) if return_estimates else 0.0
    
    def _calculate_priority(self, action: str, consensus_level: float, confidence: float) -> int:
        """Calculate execution priority for decision"""
        base_priority = 1
        
        # Higher priority for strong consensus
        if consensus_level > 0.8:
            base_priority += 2
        elif consensus_level > 0.6:
            base_priority += 1
        
        # Higher priority for high confidence
        if confidence > 0.8:
            base_priority += 2
        elif confidence > 0.6:
            base_priority += 1
        
        # Action-specific priorities
        if action in ['BUY', 'SELL']:
            base_priority += 1  # Trading actions get higher priority
        
        return min(10, base_priority)  # Cap at 10
    
    async def distribute_tasks(self, tasks: List[Dict[str, Any]]):
        """Distribute tasks to appropriate agents"""
        for task in tasks:
            await self._assign_task_to_agent(task)
    
    async def _assign_task_to_agent(self, task: Dict[str, Any]):
        """Assign task to most suitable agent"""
        task_type = task.get('type', 'general')
        
        # Find suitable agents based on task type
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            if self._is_agent_suitable_for_task(agent, task):
                suitability_score = self._calculate_agent_suitability(agent, task)
                suitable_agents.append((agent_id, suitability_score))
        
        if not suitable_agents:
            logger.warning(f"No suitable agents found for task: {task_type}")
            return
        
        # Sort by suitability score
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Assign to best agent
        best_agent_id = suitable_agents[0][0]
        
        # Send task assignment message
        task_message = SwarmMessage(
            sender_id="coordinator",
            receiver_id=best_agent_id,
            message_type="task_assignment",
            content={'task': task},
            priority=task.get('priority', 1)
        )
        
        self.swarm_communication.send_message(task_message)
        
        logger.info(f"Task {task_type} assigned to agent {best_agent_id}")
    
    def _is_agent_suitable_for_task(self, agent: BaseAgent, task: Dict[str, Any]) -> bool:
        """Check if agent is suitable for task"""
        task_type = task.get('type', 'general')
        agent_type = agent.agent_type
        
        # Task-agent type mapping
        task_agent_mapping = {
            'market_exploration': [AgentType.SCOUT],
            'pattern_discovery': [AgentType.SCOUT, AgentType.ANALYST],
            'deep_analysis': [AgentType.ANALYST],
            'risk_assessment': [AgentType.RISK_MANAGER],
            'portfolio_optimization': [AgentType.ANALYST, AgentType.RISK_MANAGER],
            'trade_execution': [AgentType.EXECUTOR],
            'general': list(AgentType)  # All agents can handle general tasks
        }
        
        suitable_types = task_agent_mapping.get(task_type, [agent_type])
        return agent_type in suitable_types
    
    def _calculate_agent_suitability(self, agent: BaseAgent, task: Dict[str, Any]) -> float:
        """Calculate agent suitability score for task"""
        base_score = 0.5
        
        # Factor in agent fitness
        base_score += agent.state.fitness * 0.3
        
        # Factor in specialization
        base_score += agent.state.specialization_score * 0.3
        
        # Factor in agent energy
        base_score += agent.state.energy * 0.2
        
        # Factor in agent experience
        experience_factor = min(1.0, agent.state.experience / 1000)
        base_score += experience_factor * 0.2
        
        return min(1.0, base_score)
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get overall swarm status"""
        if not self.agents:
            return {'status': 'no_agents', 'agent_count': 0}
        
        # Agent statistics
        agent_stats = {
            'total_agents': len(self.agents),
            'agents_by_type': {},
            'average_fitness': 0.0,
            'average_energy': 0.0,
            'average_experience': 0.0
        }
        
        fitness_scores = []
        energy_levels = []
        experience_levels = []
        
        for agent in self.agents.values():
            agent_type = agent.agent_type.value
            if agent_type not in agent_stats['agents_by_type']:
                agent_stats['agents_by_type'][agent_type] = 0
            agent_stats['agents_by_type'][agent_type] += 1
            
            fitness_scores.append(agent.state.fitness)
            energy_levels.append(agent.state.energy)
            experience_levels.append(agent.state.experience)
        
        agent_stats['average_fitness'] = np.mean(fitness_scores)
        agent_stats['average_energy'] = np.mean(energy_levels)
        agent_stats['average_experience'] = np.mean(experience_levels)
        
        # Communication statistics
        comm_stats = self.swarm_communication.get_communication_stats()
        
        # Memory statistics
        memory_stats = self.collective_memory.get_collective_insights()
        
        # Recent decisions
        recent_decisions = len([d for d in self.coordination_history 
                              if (datetime.now() - d['decision']['timestamp']).total_seconds() < 3600])
        
        # Overall swarm health
        swarm_health = (
            agent_stats['average_fitness'] * 0.4 +
            agent_stats['average_energy'] * 0.3 +
            min(1.0, agent_stats['average_experience'] / 1000) * 0.3
        )
        
        return {
            'status': 'active',
            'swarm_health': swarm_health,
            'agent_statistics': agent_stats,
            'communication_statistics': comm_stats,
            'memory_statistics': memory_stats,
            'recent_decisions': recent_decisions,
            'coordination_efficiency': min(1.0, recent_decisions / 10),  # Normalize to decisions per hour
            'collective_intelligence_level': swarm_health * memory_stats.get('memory_efficiency', 0.5)
        }

class SwarmTradingNetwork:
    """Main Swarm Intelligence Trading Network"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.swarm_communication = SwarmCommunicationProtocol()
        self.collective_memory = CollectiveMemoryBank()
        self.coordinator = SwarmCoordinator(self.swarm_communication, self.collective_memory)
        
        self.agents = {}
        self.is_running = False
        self.decision_history = deque(maxlen=1000)
        
        # Initialize agents
        self._initialize_swarm()
        
        logger.info("🐝 Swarm Intelligence Trading Network initialized")
    
    def _initialize_swarm(self):
        """Initialize swarm with different types of agents"""
        agent_config = self.config.get('agents', {})
        
        # Scout agents
        num_scouts = agent_config.get('scouts', 100)
        for i in range(num_scouts):
            agent_id = f"scout_{i:03d}"
            agent = ScoutAgent(agent_id, self.swarm_communication, self.collective_memory)
            self.agents[agent_id] = agent
            self.coordinator.register_agent(agent)
        
        # Analyst agents
        num_analysts = agent_config.get('analysts', 200)
        for i in range(num_analysts):
            agent_id = f"analyst_{i:03d}"
            agent = AnalystAgent(agent_id, self.swarm_communication, self.collective_memory)
            self.agents[agent_id] = agent
            self.coordinator.register_agent(agent)
        
        # Risk manager agents
        num_risk_managers = agent_config.get('risk_managers', 50)
        for i in range(num_risk_managers):
            agent_id = f"risk_{i:03d}"
            agent = RiskAgent(agent_id, self.swarm_communication, self.collective_memory)
            self.agents[agent_id] = agent
            self.coordinator.register_agent(agent)
        
        logger.info(f"Swarm initialized with {len(self.agents)} agents")
    
    async def start_swarm(self):
        """Start the swarm network"""
        self.is_running = True
        
        # Start all agents
        agent_tasks = []
        for agent in self.agents.values():
            task = asyncio.create_task(agent.run_agent_loop())
            agent_tasks.append(task)
        
        logger.info("🚀 Swarm network started")
        
        # Keep swarm running
        try:
            await asyncio.gather(*agent_tasks)
        except Exception as e:
            logger.error(f"Error in swarm execution: {e}")
        finally:
            self.is_running = False
    
    async def stop_swarm(self):
        """Stop the swarm network"""
        self.is_running = False
        
        # Stop all agents
        for agent in self.agents.values():
            agent.is_active = False
        
        logger.info("🛑 Swarm network stopped")
    
    async def swarm_decision_making(self, market_data: Dict[str, Any]) -> SwarmDecision:
        """Main swarm decision making process"""
        logger.info("🧠 Starting swarm decision making process...")
        
        # Phase 1: Distribute market data to all agents
        await self._distribute_market_data(market_data)
        
        # Phase 2: Collect initial evaluations
        evaluations = await self._collect_agent_evaluations(market_data)
        
        # Phase 3: Coordinate swarm decision
        decision_request = {
            'type': 'trading_decision',
            'market_data': market_data,
            'agent_evaluations': evaluations,
            'time_horizon': 'short_term',
            'urgency': 'normal'
        }
        
        swarm_decision = await self.coordinator.coordinate_swarm_decision(decision_request)
        
        # Phase 4: Store decision in history
        self.decision_history.append({
            'decision': swarm_decision,
            'market_data': market_data,
            'evaluations': evaluations,
            'timestamp': datetime.now()
        })
        
        logger.info(f"✅ Swarm decision: {swarm_decision.action} (confidence: {swarm_decision.confidence:.1%})")
        
        return swarm_decision
    
    async def _distribute_market_data(self, market_data: Dict[str, Any]):
        """Distribute market data to all agents"""
        self.swarm_communication.broadcast_message(
            "swarm_network",
            "market_update",
            market_data
        )
    
    async def _collect_agent_evaluations(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect evaluations from agents"""
        evaluations = {}
        
        # Give agents time to process
        await asyncio.sleep(2)
        
        # Collect evaluations by agent type
        for agent_id, agent in self.agents.items():
            try:
                evaluation = agent.evaluate_market_data(market_data)
                evaluations[agent_id] = {
                    'agent_type': agent.agent_type.value,
                    'evaluation': evaluation,
                    'agent_fitness': agent.state.fitness,
                    'agent_specialization': agent.state.specialization_score
                }
            except Exception as e:
                logger.error(f"Error getting evaluation from agent {agent_id}: {e}")
        
        return evaluations
    
    async def swarm_learning_evolution(self):
        """Evolve and improve the swarm through learning"""
        logger.info("🧬 Starting swarm learning evolution...")
        
        # Analyze recent performance
        performance_analysis = self._analyze_swarm_performance()
        
        # Evolve agent capabilities
        await self._evolve_agent_capabilities(performance_analysis)
        
        # Update collective knowledge
        self._update_collective_knowledge(performance_analysis)
        
        # Optimize swarm structure
        await self._optimize_swarm_structure(performance_analysis)
        
        logger.info("✅ Swarm evolution completed")
    
    def _analyze_swarm_performance(self) -> Dict[str, Any]:
        """Analyze recent swarm performance"""
        if not self.decision_history:
            return {'performance_score': 0.5, 'decisions_analyzed': 0}
        
        recent_decisions = list(self.decision_history)[-50:]  # Last 50 decisions
        
        # Analyze decision quality
        confidence_scores = [d['decision'].confidence for d in recent_decisions]
        consensus_scores = [d['decision'].consensus_level for d in recent_decisions]
        
        avg_confidence = np.mean(confidence_scores)
        avg_consensus = np.mean(consensus_scores)
        
        # Analyze agent participation
        participating_agents = []
        for decision_record in recent_decisions:
            participating_agents.extend(decision_record['decision'].participating_agents)
        
        unique_participants = len(set(participating_agents))
        participation_rate = unique_participants / len(self.agents) if self.agents else 0
        
        # Calculate overall performance score
        performance_score = (avg_confidence * 0.4 + avg_consensus * 0.4 + participation_rate * 0.2)
        
        return {
            'performance_score': performance_score,
            'average_confidence': avg_confidence,
            'average_consensus': avg_consensus,
            'participation_rate': participation_rate,
            'decisions_analyzed': len(recent_decisions),
            'unique_participants': unique_participants
        }
    
    async def _evolve_agent_capabilities(self, performance_analysis: Dict[str, Any]):
        """Evolve individual agent capabilities"""
        performance_threshold = 0.7
        
        for agent_id, agent in self.agents.items():
            # Agents with high fitness get capability boost
            if agent.state.fitness > performance_threshold:
                agent.state.specialization_score = min(1.0, agent.state.specialization_score + 0.01)
                agent.state.energy = min(1.0, agent.state.energy + 0.05)
            
            # Agents with low fitness get learning boost
            elif agent.state.fitness < 0.3:
                # Force learning from successful agents
                await self._transfer_knowledge_to_agent(agent)
        
        logger.info("🔄 Agent capabilities evolved")
    
    async def _transfer_knowledge_to_agent(self, target_agent: BaseAgent):
        """Transfer knowledge from successful agents to target agent"""
        # Find top performing agents of the same type
        same_type_agents = [agent for agent in self.agents.values() 
                           if agent.agent_type == target_agent.agent_type and agent.state.fitness > 0.7]
        
        if same_type_agents:
            # Transfer knowledge from best performing agent
            best_agent = max(same_type_agents, key=lambda a: a.state.fitness)
            
            # Simple knowledge transfer (copy successful patterns)
            for key, value in best_agent.local_memory.items():
                if key.startswith('successful_'):
                    target_agent.local_memory[f"transferred_{key}"] = value
            
            # Boost target agent's learning
            target_agent.state.specialization_score = min(1.0, target_agent.state.specialization_score + 0.05)
    
    def _update_collective_knowledge(self, performance_analysis: Dict[str, Any]):
        """Update collective knowledge based on performance"""
        # Store performance insights
        performance_insight = {
            'type': 'swarm_performance',
            'performance_score': performance_analysis['performance_score'],
            'timestamp': datetime.now(),
            'insights': {
                'high_performance_factors': self._identify_success_factors(),
                'improvement_areas': self._identify_improvement_areas(performance_analysis)
            }
        }
        
        self.collective_memory.store_experience("swarm_network", performance_insight)
    
    def _identify_success_factors(self) -> List[str]:
        """Identify factors contributing to swarm success"""
        success_factors = []
        
        # Analyze recent successful decisions
        successful_decisions = [d for d in self.decision_history 
                              if d['decision'].confidence > 0.8 and d['decision'].consensus_level > 0.7]
        
        if len(successful_decisions) > len(self.decision_history) * 0.3:  # 30% success rate
            success_factors.append("high_consensus_decisions")
        
        # Check agent diversity
        agent_types = set(agent.agent_type for agent in self.agents.values())
        if len(agent_types) >= 3:
            success_factors.append("agent_diversity")
        
        # Check communication efficiency
        comm_stats = self.swarm_communication.get_communication_stats()
        if comm_stats.get('network_density', 0) > 0.5:
            success_factors.append("effective_communication")
        
        return success_factors
    
    def _identify_improvement_areas(self, performance_analysis: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        improvement_areas = []
        
        if performance_analysis['average_confidence'] < 0.6:
            improvement_areas.append("decision_confidence")
        
        if performance_analysis['average_consensus'] < 0.6:
            improvement_areas.append("consensus_building")
        
        if performance_analysis['participation_rate'] < 0.7:
            improvement_areas.append("agent_participation")
        
        return improvement_areas
    
    async def _optimize_swarm_structure(self, performance_analysis: Dict[str, Any]):
        """Optimize swarm structure based on performance"""
        # If performance is low, consider adding more agents
        if performance_analysis['performance_score'] < 0.5:
            await self._expand_swarm()
        
        # If participation is low, improve communication
        if performance_analysis['participation_rate'] < 0.6:
            self._improve_communication_structure()
    
    async def _expand_swarm(self):
        """Add more agents to improve performance"""
        # Add a few more analyst agents (they tend to improve decision quality)
        for i in range(5):
            agent_id = f"analyst_boost_{i:03d}"
            agent = AnalystAgent(agent_id, self.swarm_communication, self.collective_memory)
            self.agents[agent_id] = agent
            self.coordinator.register_agent(agent)
            
            # Start the new agent
            asyncio.create_task(agent.run_agent_loop())
        
        logger.info("📈 Swarm expanded with additional agents")
    
    def _improve_communication_structure(self):
        """Improve communication structure"""
        # Create additional communication channels
        specialized_channels = [
            "high_priority_signals",
            "risk_alerts",
            "market_opportunities",
            "coordination_urgent"
        ]
        
        for channel in specialized_channels:
            for agent_id in self.agents.keys():
                self.swarm_communication.subscribe_to_channel(agent_id, channel)
        
        logger.info("📡 Communication structure improved")
    
    def get_swarm_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive swarm performance summary"""
        # Basic swarm status
        swarm_status = self.coordinator.get_swarm_status()
        
        # Performance analysis
        performance_analysis = self._analyze_swarm_performance()
        
        # Memory insights
        memory_insights = self.collective_memory.get_collective_insights()
        
        # Communication efficiency
        comm_stats = self.swarm_communication.get_communication_stats()
        
        # Recent decision quality
        recent_decisions = list(self.decision_history)[-10:]
        if recent_decisions:
            avg_recent_confidence = np.mean([d['decision'].confidence for d in recent_decisions])
            avg_recent_consensus = np.mean([d['decision'].consensus_level for d in recent_decisions])
        else:
            avg_recent_confidence = 0.5
            avg_recent_consensus = 0.5
        
        # Calculate swarm intelligence quotient
        swarm_iq = (
            swarm_status['swarm_health'] * 0.3 +
            performance_analysis['performance_score'] * 0.3 +
            memory_insights.get('memory_efficiency', 0.5) * 0.2 +
            comm_stats.get('network_density', 0.5) * 0.2
        )
        
        return {
            'swarm_status': swarm_status,
            'performance_analysis': performance_analysis,
            'memory_insights': memory_insights,
            'communication_stats': comm_stats,
            'recent_decision_quality': {
                'average_confidence': avg_recent_confidence,
                'average_consensus': avg_recent_consensus,
                'decisions_count': len(recent_decisions)
            },
            'swarm_intelligence_quotient': swarm_iq,
            'swarm_grade': 'A' if swarm_iq > 0.8 else 'B' if swarm_iq > 0.6 else 'C' if swarm_iq > 0.4 else 'D',
            'collective_intelligence_level': 'revolutionary' if swarm_iq > 0.9 else 'advanced' if swarm_iq > 0.7 else 'developing'
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Test swarm intelligence network
    config = {
        'agents': {
            'scouts': 10,      # Reduced for testing
            'analysts': 20,    # Reduced for testing
            'risk_managers': 5 # Reduced for testing
        }
    }
    
    swarm_network = SwarmTradingNetwork(config)
    
    # Sample market data
    sample_market_data = {
        'EURUSD': {
            'price': 1.1000,
            'volatility': 0.015,
            'volume_ratio': 1.2,
            'bid_ask_spread': 0.0001
        },
        'GBPUSD': {
            'price': 1.3000,
            'volatility': 0.020,
            'volume_ratio': 1.5,
            'bid_ask_spread': 0.0002
        },
        'market_stress_indicators': {
            'vix': 0.6,
            'credit_spreads': 0.4,
            'liquidity': 0.8
        }
    }
    
    async def test_swarm_intelligence():
        # Start swarm (in background)
        swarm_task = asyncio.create_task(swarm_network.start_swarm())
        
        # Wait for swarm to initialize
        await asyncio.sleep(3)
        
        # Test swarm decision making
        decision = await swarm_network.swarm_decision_making(sample_market_data)
        
        print("🐝 Swarm Intelligence Decision:")
        print(f"Action: {decision.action}")
        print(f"Confidence: {decision.confidence:.1%}")
        print(f"Consensus Level: {decision.consensus_level:.1%}")
        print(f"Participating Agents: {len(decision.participating_agents)}")
        
        # Test swarm evolution
        await swarm_network.swarm_learning_evolution()
        
        # Get performance summary
        summary = swarm_network.get_swarm_performance_summary()
        print(f"\n📊 Swarm Performance Summary:")
        print(f"Swarm IQ: {summary['swarm_intelligence_quotient']:.3f}")
        print(f"Swarm Grade: {summary['swarm_grade']}")
        print(f"Collective Intelligence: {summary['collective_intelligence_level']}")
        
        # Stop swarm
        await swarm_network.stop_swarm()
        swarm_task.cancel()
    
    # asyncio.run(test_swarm_intelligence())