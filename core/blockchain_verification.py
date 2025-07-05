"""
Blockchain-Based Signal Verification & Trading Reputation System
================================================================

Revolutionary Immutable Signal Tracking menggunakan blockchain untuk mencatat
setiap signal, performance, dan decision-making process. Sistem ini menciptakan
"trading DNA" yang tidak dapat dimanipulasi dan membangun reputation system
yang transparan.
"""

import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import logging
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
import base64
import sqlite3
import threading
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from enum import Enum
import uuid
import pickle
from concurrent.futures import ThreadPoolExecutor
import requests
import os

logger = logging.getLogger(__name__)

class TransactionType(Enum):
    """Types of blockchain transactions"""
    SIGNAL_CREATION = "signal_creation"
    SIGNAL_VERIFICATION = "signal_verification"
    PERFORMANCE_UPDATE = "performance_update"
    REPUTATION_UPDATE = "reputation_update"
    STRATEGY_REGISTRATION = "strategy_registration"
    NFT_CREATION = "nft_creation"
    CONSENSUS_VOTE = "consensus_vote"
    REWARD_DISTRIBUTION = "reward_distribution"

class ConsensusAlgorithm(Enum):
    """Consensus algorithms for blockchain"""
    PROOF_OF_PERFORMANCE = "proof_of_performance"
    PROOF_OF_STAKE = "proof_of_stake"
    DELEGATED_PROOF_OF_STAKE = "delegated_proof_of_stake"
    PRACTICAL_BYZANTINE_FAULT_TOLERANCE = "pbft"

@dataclass
class BlockchainTransaction:
    """Individual blockchain transaction"""
    transaction_id: str
    transaction_type: TransactionType
    sender: str
    receiver: str
    data: Dict[str, Any]
    timestamp: datetime
    signature: str
    gas_fee: float
    nonce: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary"""
        return {
            'transaction_id': self.transaction_id,
            'transaction_type': self.transaction_type.value,
            'sender': self.sender,
            'receiver': self.receiver,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'signature': self.signature,
            'gas_fee': self.gas_fee,
            'nonce': self.nonce
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BlockchainTransaction':
        """Create transaction from dictionary"""
        return cls(
            transaction_id=data['transaction_id'],
            transaction_type=TransactionType(data['transaction_type']),
            sender=data['sender'],
            receiver=data['receiver'],
            data=data['data'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            signature=data['signature'],
            gas_fee=data['gas_fee'],
            nonce=data['nonce']
        )

@dataclass
class Block:
    """Blockchain block"""
    block_number: int
    previous_hash: str
    merkle_root: str
    timestamp: datetime
    transactions: List[BlockchainTransaction]
    nonce: int
    difficulty: int
    miner: str
    block_reward: float
    gas_used: float
    block_hash: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            'block_number': self.block_number,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp.isoformat(),
            'transactions': [tx.to_dict() for tx in self.transactions],
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'miner': self.miner,
            'gas_used': self.gas_used
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            return hashlib.sha256(b'').hexdigest()
        
        # Get transaction hashes
        tx_hashes = [hashlib.sha256(json.dumps(tx.to_dict(), sort_keys=True).encode()).hexdigest() 
                    for tx in self.transactions]
        
        # Build Merkle tree
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])  # Duplicate last hash if odd number
            
            new_level = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i + 1]
                new_level.append(hashlib.sha256(combined.encode()).hexdigest())
            
            tx_hashes = new_level
        
        return tx_hashes[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary"""
        return {
            'block_number': self.block_number,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp.isoformat(),
            'transactions': [tx.to_dict() for tx in self.transactions],
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'miner': self.miner,
            'block_reward': self.block_reward,
            'gas_used': self.gas_used,
            'block_hash': self.block_hash
        }

@dataclass
class TradingSignal:
    """Trading signal for blockchain storage"""
    signal_id: str
    pair: str
    action: str  # BUY, SELL, HOLD
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timeframe: str
    strategy_name: str
    ai_model_version: str
    market_conditions: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    fundamental_factors: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    created_at: datetime
    creator: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return asdict(self)
    
    def calculate_signal_hash(self) -> str:
        """Calculate unique hash for signal"""
        signal_string = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(signal_string.encode()).hexdigest()

@dataclass
class PerformanceRecord:
    """Performance record for signal verification"""
    signal_id: str
    actual_outcome: str  # WIN, LOSS, BREAKEVEN
    entry_executed: bool
    exit_executed: bool
    actual_entry_price: float
    actual_exit_price: float
    profit_loss: float
    profit_loss_percentage: float
    duration_hours: float
    slippage: float
    execution_quality: float
    market_impact: float
    verified_at: datetime
    verifier: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance record to dictionary"""
        return asdict(self)

@dataclass
class TradingNFT:
    """NFT for exceptional trading strategies"""
    nft_id: str
    strategy_name: str
    strategy_dna: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    rarity_score: float
    creation_date: datetime
    creator: str
    owner: str
    metadata: Dict[str, Any]
    image_url: str
    attributes: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NFT to dictionary"""
        return asdict(self)

class CryptographicManager:
    """Manages cryptographic operations for blockchain"""
    
    def __init__(self):
        self.private_key = None
        self.public_key = None
        self.key_pair_generated = False
        
    def generate_key_pair(self) -> Tuple[str, str]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        self.private_key = private_key
        self.public_key = public_key
        self.key_pair_generated = True
        
        return private_pem.decode(), public_pem.decode()
    
    def sign_data(self, data: str, private_key_pem: str = None) -> str:
        """Sign data with private key"""
        if private_key_pem:
            private_key = load_pem_private_key(private_key_pem.encode(), password=None)
        elif self.private_key:
            private_key = self.private_key
        else:
            raise ValueError("No private key available for signing")
        
        signature = private_key.sign(
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
    
    def verify_signature(self, data: str, signature: str, public_key_pem: str) -> bool:
        """Verify signature with public key"""
        try:
            public_key = load_pem_public_key(public_key_pem.encode())
            signature_bytes = base64.b64decode(signature.encode())
            
            public_key.verify(
                signature_bytes,
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def hash_data(self, data: str) -> str:
        """Hash data using SHA-256"""
        return hashlib.sha256(data.encode()).hexdigest()

class ProofOfPerformance:
    """Proof of Performance consensus mechanism"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.reputation_scores = defaultdict(float)
        self.stake_amounts = defaultdict(float)
        
    def calculate_performance_score(self, trader_id: str, 
                                  recent_signals: List[PerformanceRecord]) -> float:
        """Calculate performance score for trader"""
        if not recent_signals:
            return 0.5  # Neutral score for new traders
        
        # Calculate win rate
        wins = sum(1 for signal in recent_signals if signal.actual_outcome == 'WIN')
        total_signals = len(recent_signals)
        win_rate = wins / total_signals if total_signals > 0 else 0
        
        # Calculate average profit
        profits = [signal.profit_loss_percentage for signal in recent_signals]
        avg_profit = np.mean(profits) if profits else 0
        
        # Calculate consistency (lower std deviation = higher consistency)
        consistency = 1 - (np.std(profits) / 100) if len(profits) > 1 else 0.5
        consistency = max(0, min(1, consistency))
        
        # Calculate execution quality
        execution_scores = [signal.execution_quality for signal in recent_signals]
        avg_execution = np.mean(execution_scores) if execution_scores else 0.5
        
        # Composite performance score
        performance_score = (
            win_rate * 0.3 +
            (avg_profit + 1) / 2 * 0.3 +  # Normalize profit to [0, 1]
            consistency * 0.2 +
            avg_execution * 0.2
        )
        
        return max(0, min(1, performance_score))
    
    def update_reputation(self, trader_id: str, performance_record: PerformanceRecord):
        """Update trader reputation based on performance"""
        self.performance_history[trader_id].append(performance_record)
        
        # Keep only recent performance (last 100 signals)
        if len(self.performance_history[trader_id]) > 100:
            self.performance_history[trader_id] = self.performance_history[trader_id][-100:]
        
        # Recalculate reputation score
        recent_performance = self.performance_history[trader_id][-20:]  # Last 20 signals
        new_score = self.calculate_performance_score(trader_id, recent_performance)
        
        # Update with exponential moving average
        alpha = 0.1  # Learning rate
        current_score = self.reputation_scores.get(trader_id, 0.5)
        self.reputation_scores[trader_id] = alpha * new_score + (1 - alpha) * current_score
    
    def get_mining_probability(self, trader_id: str) -> float:
        """Get probability of being selected as miner"""
        reputation = self.reputation_scores.get(trader_id, 0.5)
        stake = self.stake_amounts.get(trader_id, 0)
        
        # Combine reputation and stake
        base_probability = reputation * 0.7 + min(stake / 10000, 1) * 0.3
        
        return base_probability
    
    def select_miner(self, candidates: List[str]) -> str:
        """Select miner based on proof of performance"""
        if not candidates:
            return "system"
        
        # Calculate probabilities for all candidates
        probabilities = []
        for candidate in candidates:
            prob = self.get_mining_probability(candidate)
            probabilities.append(prob)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob == 0:
            return np.random.choice(candidates)
        
        normalized_probs = [p / total_prob for p in probabilities]
        
        # Select miner based on weighted probability
        selected_miner = np.random.choice(candidates, p=normalized_probs)
        
        return selected_miner
    
    def validate_block(self, block: Block, miner: str) -> bool:
        """Validate block based on miner's performance"""
        miner_reputation = self.reputation_scores.get(miner, 0.5)
        
        # Higher reputation miners have more lenient validation
        min_reputation_threshold = 0.3
        
        if miner_reputation < min_reputation_threshold:
            logger.warning(f"Miner {miner} has low reputation: {miner_reputation}")
            return False
        
        # Additional validation checks
        if not block.transactions:
            return False
        
        if block.gas_used < 0:
            return False
        
        return True

class SmartContract:
    """Smart contract for automated trading operations"""
    
    def __init__(self, contract_id: str, contract_code: str):
        self.contract_id = contract_id
        self.contract_code = contract_code
        self.state = {}
        self.execution_history = []
        
    def execute(self, function_name: str, parameters: Dict[str, Any], 
               caller: str) -> Dict[str, Any]:
        """Execute smart contract function"""
        execution_result = {
            'success': False,
            'result': None,
            'gas_used': 0,
            'error': None
        }
        
        try:
            if function_name == "signal_verification":
                result = self._verify_signal_performance(parameters, caller)
            elif function_name == "reputation_update":
                result = self._update_reputation(parameters, caller)
            elif function_name == "reward_distribution":
                result = self._distribute_rewards(parameters, caller)
            elif function_name == "nft_creation":
                result = self._create_trading_nft(parameters, caller)
            else:
                raise ValueError(f"Unknown function: {function_name}")
            
            execution_result['success'] = True
            execution_result['result'] = result
            execution_result['gas_used'] = self._calculate_gas_usage(function_name, parameters)
            
        except Exception as e:
            execution_result['error'] = str(e)
            logger.error(f"Smart contract execution failed: {e}")
        
        # Record execution
        self.execution_history.append({
            'function': function_name,
            'parameters': parameters,
            'caller': caller,
            'result': execution_result,
            'timestamp': datetime.now()
        })
        
        return execution_result
    
    def _verify_signal_performance(self, parameters: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """Verify signal performance and update records"""
        signal_id = parameters['signal_id']
        actual_outcome = parameters['actual_outcome']
        performance_data = parameters['performance_data']
        
        # Create performance record
        performance_record = PerformanceRecord(
            signal_id=signal_id,
            actual_outcome=actual_outcome,
            entry_executed=performance_data.get('entry_executed', False),
            exit_executed=performance_data.get('exit_executed', False),
            actual_entry_price=performance_data.get('actual_entry_price', 0),
            actual_exit_price=performance_data.get('actual_exit_price', 0),
            profit_loss=performance_data.get('profit_loss', 0),
            profit_loss_percentage=performance_data.get('profit_loss_percentage', 0),
            duration_hours=performance_data.get('duration_hours', 0),
            slippage=performance_data.get('slippage', 0),
            execution_quality=performance_data.get('execution_quality', 0.5),
            market_impact=performance_data.get('market_impact', 0),
            verified_at=datetime.now(),
            verifier=caller
        )
        
        # Store in contract state
        if 'performance_records' not in self.state:
            self.state['performance_records'] = {}
        
        self.state['performance_records'][signal_id] = performance_record.to_dict()
        
        return {
            'signal_id': signal_id,
            'verification_status': 'verified',
            'performance_record': performance_record.to_dict()
        }
    
    def _update_reputation(self, parameters: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """Update trader reputation"""
        trader_id = parameters['trader_id']
        performance_score = parameters['performance_score']
        
        # Update reputation in contract state
        if 'reputations' not in self.state:
            self.state['reputations'] = {}
        
        current_reputation = self.state['reputations'].get(trader_id, 0.5)
        
        # Exponential moving average update
        alpha = 0.1
        new_reputation = alpha * performance_score + (1 - alpha) * current_reputation
        
        self.state['reputations'][trader_id] = new_reputation
        
        return {
            'trader_id': trader_id,
            'old_reputation': current_reputation,
            'new_reputation': new_reputation,
            'updated_by': caller
        }
    
    def _distribute_rewards(self, parameters: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """Distribute rewards based on performance"""
        reward_pool = parameters['reward_pool']
        performance_data = parameters['performance_data']
        
        # Calculate reward distribution
        total_performance = sum(data['score'] for data in performance_data.values())
        
        if total_performance == 0:
            return {'error': 'No performance data for reward distribution'}
        
        rewards = {}
        for trader_id, data in performance_data.items():
            reward_percentage = data['score'] / total_performance
            reward_amount = reward_pool * reward_percentage
            rewards[trader_id] = reward_amount
        
        # Update balances in contract state
        if 'balances' not in self.state:
            self.state['balances'] = defaultdict(float)
        
        for trader_id, reward in rewards.items():
            self.state['balances'][trader_id] += reward
        
        return {
            'total_reward_pool': reward_pool,
            'rewards_distributed': rewards,
            'distribution_timestamp': datetime.now().isoformat()
        }
    
    def _create_trading_nft(self, parameters: Dict[str, Any], caller: str) -> Dict[str, Any]:
        """Create NFT for exceptional trading strategy"""
        strategy_data = parameters['strategy_data']
        performance_metrics = parameters['performance_metrics']
        
        # Calculate rarity score
        rarity_score = self._calculate_nft_rarity(performance_metrics)
        
        # Create NFT
        nft = TradingNFT(
            nft_id=str(uuid.uuid4()),
            strategy_name=strategy_data['name'],
            strategy_dna=strategy_data['dna'],
            performance_metrics=performance_metrics,
            rarity_score=rarity_score,
            creation_date=datetime.now(),
            creator=caller,
            owner=caller,
            metadata=strategy_data.get('metadata', {}),
            image_url=self._generate_nft_image_url(strategy_data, performance_metrics),
            attributes=self._generate_nft_attributes(strategy_data, performance_metrics)
        )
        
        # Store NFT in contract state
        if 'nfts' not in self.state:
            self.state['nfts'] = {}
        
        self.state['nfts'][nft.nft_id] = nft.to_dict()
        
        return {
            'nft_id': nft.nft_id,
            'rarity_score': rarity_score,
            'creation_status': 'success',
            'nft_data': nft.to_dict()
        }
    
    def _calculate_nft_rarity(self, performance_metrics: Dict[str, Any]) -> float:
        """Calculate NFT rarity score based on performance"""
        win_rate = performance_metrics.get('win_rate', 0)
        profit_factor = performance_metrics.get('profit_factor', 1)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 1)
        
        # Rarity factors
        win_rate_rarity = min(1, win_rate / 0.8)  # 80% win rate = max rarity
        profit_rarity = min(1, profit_factor / 3)  # 3.0 profit factor = max rarity
        sharpe_rarity = min(1, sharpe_ratio / 2)   # 2.0 Sharpe = max rarity
        drawdown_rarity = max(0, 1 - max_drawdown / 0.1)  # <10% drawdown = max rarity
        
        # Composite rarity score
        rarity_score = (
            win_rate_rarity * 0.3 +
            profit_rarity * 0.3 +
            sharpe_rarity * 0.2 +
            drawdown_rarity * 0.2
        )
        
        return rarity_score
    
    def _generate_nft_image_url(self, strategy_data: Dict[str, Any], 
                               performance_metrics: Dict[str, Any]) -> str:
        """Generate NFT image URL"""
        # In a real implementation, this would generate or reference an actual image
        strategy_hash = hashlib.md5(str(strategy_data).encode()).hexdigest()[:8]
        return f"https://nft.agi-forex.com/strategy/{strategy_hash}.png"
    
    def _generate_nft_attributes(self, strategy_data: Dict[str, Any], 
                                performance_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate NFT attributes"""
        attributes = [
            {
                "trait_type": "Strategy Type",
                "value": strategy_data.get('type', 'Unknown')
            },
            {
                "trait_type": "Win Rate",
                "value": f"{performance_metrics.get('win_rate', 0) * 100:.1f}%"
            },
            {
                "trait_type": "Profit Factor",
                "value": f"{performance_metrics.get('profit_factor', 1):.2f}"
            },
            {
                "trait_type": "Sharpe Ratio",
                "value": f"{performance_metrics.get('sharpe_ratio', 0):.2f}"
            },
            {
                "trait_type": "Max Drawdown",
                "value": f"{performance_metrics.get('max_drawdown', 1) * 100:.1f}%"
            },
            {
                "trait_type": "Total Trades",
                "value": performance_metrics.get('total_trades', 0)
            }
        ]
        
        return attributes
    
    def _calculate_gas_usage(self, function_name: str, parameters: Dict[str, Any]) -> float:
        """Calculate gas usage for function execution"""
        base_gas = {
            'signal_verification': 1000,
            'reputation_update': 500,
            'reward_distribution': 2000,
            'nft_creation': 5000
        }
        
        # Base gas cost
        gas_used = base_gas.get(function_name, 1000)
        
        # Additional gas based on data size
        data_size = len(json.dumps(parameters))
        gas_used += data_size // 100  # 1 gas per 100 bytes
        
        return gas_used

class TradingBlockchain:
    """Main blockchain implementation for trading signals"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chain: List[Block] = []
        self.pending_transactions: List[BlockchainTransaction] = []
        self.difficulty = config.get('difficulty', 4)
        self.block_reward = config.get('block_reward', 10.0)
        self.max_transactions_per_block = config.get('max_transactions_per_block', 100)
        
        # Consensus mechanism
        self.consensus = ProofOfPerformance()
        
        # Cryptographic manager
        self.crypto_manager = CryptographicManager()
        
        # Smart contracts
        self.smart_contracts = {}
        
        # Database for persistence
        self.db_path = config.get('db_path', 'trading_blockchain.db')
        self._init_database()
        
        # Network nodes (for distributed blockchain)
        self.network_nodes = set()
        self.node_id = config.get('node_id', str(uuid.uuid4()))
        
        # Transaction pool
        self.transaction_pool = deque(maxlen=10000)
        
        # Mining status
        self.is_mining = False
        self.mining_thread = None
        
        # Create genesis block
        if not self.chain:
            self._create_genesis_block()
        
        logger.info("🔗 Trading Blockchain initialized")
    
    def _init_database(self):
        """Initialize SQLite database for blockchain persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                block_number INTEGER PRIMARY KEY,
                block_hash TEXT UNIQUE,
                previous_hash TEXT,
                merkle_root TEXT,
                timestamp TEXT,
                nonce INTEGER,
                difficulty INTEGER,
                miner TEXT,
                block_reward REAL,
                gas_used REAL,
                block_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                block_number INTEGER,
                transaction_type TEXT,
                sender TEXT,
                receiver TEXT,
                timestamp TEXT,
                signature TEXT,
                gas_fee REAL,
                nonce INTEGER,
                transaction_data TEXT,
                FOREIGN KEY (block_number) REFERENCES blocks (block_number)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                signal_hash TEXT UNIQUE,
                pair TEXT,
                action TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                timeframe TEXT,
                strategy_name TEXT,
                creator TEXT,
                created_at TEXT,
                signal_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_records (
                signal_id TEXT PRIMARY KEY,
                actual_outcome TEXT,
                profit_loss REAL,
                profit_loss_percentage REAL,
                execution_quality REAL,
                verified_at TEXT,
                verifier TEXT,
                performance_data TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reputations (
                trader_id TEXT PRIMARY KEY,
                reputation_score REAL,
                total_signals INTEGER,
                successful_signals INTEGER,
                total_profit_loss REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_transaction = BlockchainTransaction(
            transaction_id="genesis",
            transaction_type=TransactionType.SIGNAL_CREATION,
            sender="system",
            receiver="system",
            data={"message": "AGI Forex Trading Blockchain Genesis Block"},
            timestamp=datetime.now(),
            signature="genesis_signature",
            gas_fee=0.0,
            nonce=0
        )
        
        genesis_block = Block(
            block_number=0,
            previous_hash="0",
            merkle_root="",
            timestamp=datetime.now(),
            transactions=[genesis_transaction],
            nonce=0,
            difficulty=self.difficulty,
            miner="system",
            block_reward=0.0,
            gas_used=0.0
        )
        
        genesis_block.merkle_root = genesis_block.calculate_merkle_root()
        genesis_block.block_hash = genesis_block.calculate_hash()
        
        self.chain.append(genesis_block)
        self._save_block_to_db(genesis_block)
        
        logger.info("🎯 Genesis block created")
    
    def create_immutable_signal(self, signal_data: TradingSignal, creator_private_key: str) -> str:
        """Create immutable trading signal on blockchain"""
        
        # Create signal hash
        signal_hash = signal_data.calculate_signal_hash()
        
        # Create transaction data
        transaction_data = {
            'signal': signal_data.to_dict(),
            'signal_hash': signal_hash,
            'ai_model_version': signal_data.ai_model_version,
            'market_snapshot': signal_data.market_conditions,
            'confidence_score': signal_data.confidence,
            'predicted_outcome': {
                'direction': signal_data.action,
                'target_price': signal_data.take_profit,
                'risk_price': signal_data.stop_loss
            }
        }
        
        # Sign transaction
        transaction_string = json.dumps(transaction_data, sort_keys=True)
        signature = self.crypto_manager.sign_data(transaction_string, creator_private_key)
        
        # Create blockchain transaction
        transaction = BlockchainTransaction(
            transaction_id=str(uuid.uuid4()),
            transaction_type=TransactionType.SIGNAL_CREATION,
            sender=signal_data.creator,
            receiver="blockchain",
            data=transaction_data,
            timestamp=datetime.now(),
            signature=signature,
            gas_fee=self._calculate_gas_fee(transaction_data),
            nonce=self._get_next_nonce(signal_data.creator)
        )
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        self.transaction_pool.append(transaction)
        
        # Store signal in database
        self._save_signal_to_db(signal_data, signal_hash)
        
        logger.info(f"📝 Signal {signal_data.signal_id} added to blockchain")
        
        return signal_hash
    
    def verify_signal_performance(self, signal_id: str, actual_outcome: Dict[str, Any], 
                                 verifier: str, verifier_private_key: str) -> Dict[str, Any]:
        """Verify signal performance and update blockchain"""
        
        # Get original signal
        signal = self._get_signal_from_db(signal_id)
        if not signal:
            raise ValueError(f"Signal {signal_id} not found")
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(signal, actual_outcome)
        
        # Create performance record
        performance_record = PerformanceRecord(
            signal_id=signal_id,
            actual_outcome=actual_outcome['outcome'],
            entry_executed=actual_outcome.get('entry_executed', False),
            exit_executed=actual_outcome.get('exit_executed', False),
            actual_entry_price=actual_outcome.get('actual_entry_price', 0),
            actual_exit_price=actual_outcome.get('actual_exit_price', 0),
            profit_loss=performance_metrics['profit_loss'],
            profit_loss_percentage=performance_metrics['profit_loss_percentage'],
            duration_hours=performance_metrics['duration_hours'],
            slippage=performance_metrics['slippage'],
            execution_quality=performance_metrics['execution_quality'],
            market_impact=performance_metrics.get('market_impact', 0),
            verified_at=datetime.now(),
            verifier=verifier
        )
        
        # Create verification transaction
        verification_data = {
            'signal_id': signal_id,
            'performance_record': performance_record.to_dict(),
            'verification_timestamp': datetime.now().isoformat(),
            'original_signal_hash': signal['signal_hash']
        }
        
        # Sign verification
        verification_string = json.dumps(verification_data, sort_keys=True)
        signature = self.crypto_manager.sign_data(verification_string, verifier_private_key)
        
        # Create blockchain transaction
        transaction = BlockchainTransaction(
            transaction_id=str(uuid.uuid4()),
            transaction_type=TransactionType.SIGNAL_VERIFICATION,
            sender=verifier,
            receiver="blockchain",
            data=verification_data,
            timestamp=datetime.now(),
            signature=signature,
            gas_fee=self._calculate_gas_fee(verification_data),
            nonce=self._get_next_nonce(verifier)
        )
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        self.transaction_pool.append(transaction)
        
        # Update reputation
        self.consensus.update_reputation(signal['creator'], performance_record)
        
        # Save performance record
        self._save_performance_to_db(performance_record)
        
        # Execute smart contract for reputation update
        if 'reputation_contract' in self.smart_contracts:
            contract = self.smart_contracts['reputation_contract']
            contract.execute('reputation_update', {
                'trader_id': signal['creator'],
                'performance_score': performance_metrics['performance_score']
            }, verifier)
        
        logger.info(f"✅ Signal {signal_id} performance verified")
        
        return {
            'verification_status': 'success',
            'performance_record': performance_record.to_dict(),
            'performance_metrics': performance_metrics,
            'transaction_id': transaction.transaction_id
        }
    
    def create_trading_nft(self, strategy_data: Dict[str, Any], 
                          performance_metrics: Dict[str, Any], 
                          creator: str, creator_private_key: str) -> Dict[str, Any]:
        """Create NFT for exceptional trading strategy"""
        
        # Check if strategy qualifies for NFT
        if not self._qualifies_for_nft(performance_metrics):
            raise ValueError("Strategy does not meet NFT creation criteria")
        
        # Execute NFT creation smart contract
        if 'nft_contract' not in self.smart_contracts:
            raise ValueError("NFT smart contract not deployed")
        
        contract = self.smart_contracts['nft_contract']
        nft_result = contract.execute('nft_creation', {
            'strategy_data': strategy_data,
            'performance_metrics': performance_metrics
        }, creator)
        
        if not nft_result['success']:
            raise ValueError(f"NFT creation failed: {nft_result['error']}")
        
        # Create NFT transaction
        nft_data = {
            'nft_id': nft_result['result']['nft_id'],
            'strategy_data': strategy_data,
            'performance_metrics': performance_metrics,
            'rarity_score': nft_result['result']['rarity_score'],
            'creation_timestamp': datetime.now().isoformat()
        }
        
        # Sign NFT creation
        nft_string = json.dumps(nft_data, sort_keys=True)
        signature = self.crypto_manager.sign_data(nft_string, creator_private_key)
        
        # Create blockchain transaction
        transaction = BlockchainTransaction(
            transaction_id=str(uuid.uuid4()),
            transaction_type=TransactionType.NFT_CREATION,
            sender=creator,
            receiver="blockchain",
            data=nft_data,
            timestamp=datetime.now(),
            signature=signature,
            gas_fee=self._calculate_gas_fee(nft_data),
            nonce=self._get_next_nonce(creator)
        )
        
        # Add to pending transactions
        self.pending_transactions.append(transaction)
        self.transaction_pool.append(transaction)
        
        logger.info(f"🎨 NFT {nft_result['result']['nft_id']} created for strategy")
        
        return nft_result['result']
    
    def mine_block(self, miner: str) -> Optional[Block]:
        """Mine a new block"""
        if not self.pending_transactions:
            return None
        
        # Select transactions for block
        transactions_to_include = self.pending_transactions[:self.max_transactions_per_block]
        
        # Create new block
        previous_block = self.chain[-1] if self.chain else None
        previous_hash = previous_block.block_hash if previous_block else "0"
        
        new_block = Block(
            block_number=len(self.chain),
            previous_hash=previous_hash,
            merkle_root="",
            timestamp=datetime.now(),
            transactions=transactions_to_include,
            nonce=0,
            difficulty=self.difficulty,
            miner=miner,
            block_reward=self.block_reward,
            gas_used=sum(tx.gas_fee for tx in transactions_to_include)
        )
        
        # Calculate Merkle root
        new_block.merkle_root = new_block.calculate_merkle_root()
        
        # Mine block (find nonce that satisfies difficulty)
        target = "0" * self.difficulty
        
        while True:
            new_block.nonce += 1
            block_hash = new_block.calculate_hash()
            
            if block_hash.startswith(target):
                new_block.block_hash = block_hash
                break
            
            # Prevent infinite mining in case of high difficulty
            if new_block.nonce > 1000000:
                logger.warning("Mining timeout reached")
                return None
        
        # Validate block with consensus mechanism
        if not self.consensus.validate_block(new_block, miner):
            logger.warning(f"Block validation failed for miner {miner}")
            return None
        
        # Add block to chain
        self.chain.append(new_block)
        
        # Remove mined transactions from pending
        mined_tx_ids = {tx.transaction_id for tx in transactions_to_include}
        self.pending_transactions = [tx for tx in self.pending_transactions 
                                   if tx.transaction_id not in mined_tx_ids]
        
        # Save block to database
        self._save_block_to_db(new_block)
        
        logger.info(f"⛏️  Block {new_block.block_number} mined by {miner}")
        
        return new_block
    
    def start_mining(self, miner_id: str):
        """Start mining process"""
        if self.is_mining:
            return
        
        self.is_mining = True
        
        def mining_loop():
            while self.is_mining:
                try:
                    # Check if this node should mine
                    candidates = [miner_id]  # In a real network, this would be all nodes
                    selected_miner = self.consensus.select_miner(candidates)
                    
                    if selected_miner == miner_id:
                        block = self.mine_block(miner_id)
                        if block:
                            # Broadcast block to network (simplified)
                            self._broadcast_block(block)
                    
                    time.sleep(10)  # Mining interval
                    
                except Exception as e:
                    logger.error(f"Mining error: {e}")
                    time.sleep(5)
        
        self.mining_thread = threading.Thread(target=mining_loop)
        self.mining_thread.start()
        
        logger.info(f"⛏️  Mining started for {miner_id}")
    
    def stop_mining(self):
        """Stop mining process"""
        self.is_mining = False
        if self.mining_thread:
            self.mining_thread.join()
        
        logger.info("🛑 Mining stopped")
    
    def deploy_smart_contract(self, contract_name: str, contract_code: str) -> str:
        """Deploy smart contract to blockchain"""
        contract_id = str(uuid.uuid4())
        
        contract = SmartContract(contract_id, contract_code)
        self.smart_contracts[contract_name] = contract
        
        logger.info(f"📜 Smart contract '{contract_name}' deployed")
        
        return contract_id
    
    def get_signal_history(self, signal_id: str) -> Dict[str, Any]:
        """Get complete history of a signal from blockchain"""
        signal_transactions = []
        
        for block in self.chain:
            for transaction in block.transactions:
                if (transaction.transaction_type in [TransactionType.SIGNAL_CREATION, 
                                                   TransactionType.SIGNAL_VERIFICATION] and
                    (transaction.data.get('signal_id') == signal_id or 
                     transaction.data.get('signal', {}).get('signal_id') == signal_id)):
                    signal_transactions.append({
                        'transaction': transaction.to_dict(),
                        'block_number': block.block_number,
                        'block_hash': block.block_hash,
                        'timestamp': transaction.timestamp.isoformat()
                    })
        
        return {
            'signal_id': signal_id,
            'transaction_history': signal_transactions,
            'total_transactions': len(signal_transactions)
        }
    
    def get_trader_reputation(self, trader_id: str) -> Dict[str, Any]:
        """Get trader reputation from blockchain"""
        reputation_score = self.consensus.reputation_scores.get(trader_id, 0.5)
        performance_history = self.consensus.performance_history.get(trader_id, [])
        
        # Calculate additional metrics
        if performance_history:
            total_signals = len(performance_history)
            successful_signals = sum(1 for p in performance_history if p.actual_outcome == 'WIN')
            win_rate = successful_signals / total_signals
            
            total_profit = sum(p.profit_loss_percentage for p in performance_history)
            avg_profit = total_profit / total_signals
            
            # Calculate consistency
            profits = [p.profit_loss_percentage for p in performance_history]
            consistency = 1 - (np.std(profits) / 100) if len(profits) > 1 else 0.5
        else:
            total_signals = 0
            successful_signals = 0
            win_rate = 0
            avg_profit = 0
            consistency = 0.5
        
        # Get reputation grade
        if reputation_score >= 0.9:
            grade = 'A+'
        elif reputation_score >= 0.8:
            grade = 'A'
        elif reputation_score >= 0.7:
            grade = 'B+'
        elif reputation_score >= 0.6:
            grade = 'B'
        elif reputation_score >= 0.5:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'trader_id': trader_id,
            'reputation_score': reputation_score,
            'reputation_grade': grade,
            'total_signals': total_signals,
            'successful_signals': successful_signals,
            'win_rate': win_rate,
            'average_profit': avg_profit,
            'consistency_score': consistency,
            'performance_history': [p.to_dict() for p in performance_history[-10:]],  # Last 10
            'blockchain_verified': True
        }
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        total_blocks = len(self.chain)
        total_transactions = sum(len(block.transactions) for block in self.chain)
        
        # Calculate average block time
        if len(self.chain) > 1:
            time_diffs = []
            for i in range(1, len(self.chain)):
                diff = (self.chain[i].timestamp - self.chain[i-1].timestamp).total_seconds()
                time_diffs.append(diff)
            avg_block_time = np.mean(time_diffs)
        else:
            avg_block_time = 0
        
        # Get unique miners
        miners = set(block.miner for block in self.chain)
        
        # Calculate total gas used
        total_gas = sum(block.gas_used for block in self.chain)
        
        # Get transaction types distribution
        tx_types = defaultdict(int)
        for block in self.chain:
            for tx in block.transactions:
                tx_types[tx.transaction_type.value] += 1
        
        return {
            'total_blocks': total_blocks,
            'total_transactions': total_transactions,
            'average_block_time_seconds': avg_block_time,
            'unique_miners': len(miners),
            'total_gas_used': total_gas,
            'current_difficulty': self.difficulty,
            'pending_transactions': len(self.pending_transactions),
            'transaction_types': dict(tx_types),
            'blockchain_size_mb': self._calculate_blockchain_size(),
            'network_nodes': len(self.network_nodes),
            'consensus_algorithm': 'proof_of_performance'
        }
    
    def validate_blockchain(self) -> Dict[str, Any]:
        """Validate entire blockchain integrity"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'blocks_validated': 0,
            'transactions_validated': 0
        }
        
        for i, block in enumerate(self.chain):
            # Validate block hash
            calculated_hash = block.calculate_hash()
            if calculated_hash != block.block_hash:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Block {i} hash mismatch")
            
            # Validate previous hash (except genesis)
            if i > 0:
                if block.previous_hash != self.chain[i-1].block_hash:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Block {i} previous hash mismatch")
            
            # Validate Merkle root
            calculated_merkle = block.calculate_merkle_root()
            if calculated_merkle != block.merkle_root:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Block {i} Merkle root mismatch")
            
            # Validate transactions
            for j, tx in enumerate(block.transactions):
                # Validate transaction signature (simplified)
                if not tx.signature:
                    validation_results['warnings'].append(f"Block {i}, Transaction {j} missing signature")
                
                validation_results['transactions_validated'] += 1
            
            validation_results['blocks_validated'] += 1
        
        return validation_results
    
    def _calculate_performance_metrics(self, signal: Dict[str, Any], 
                                     actual_outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for signal verification"""
        
        entry_price = signal.get('entry_price', 0)
        actual_entry = actual_outcome.get('actual_entry_price', entry_price)
        actual_exit = actual_outcome.get('actual_exit_price', entry_price)
        
        # Calculate profit/loss
        if signal.get('action') == 'BUY':
            profit_loss = actual_exit - actual_entry
        elif signal.get('action') == 'SELL':
            profit_loss = actual_entry - actual_exit
        else:
            profit_loss = 0
        
        profit_loss_percentage = (profit_loss / actual_entry) * 100 if actual_entry > 0 else 0
        
        # Calculate duration
        created_at = datetime.fromisoformat(signal.get('created_at', datetime.now().isoformat()))
        exit_time = actual_outcome.get('exit_time', datetime.now())
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)
        
        duration_hours = (exit_time - created_at).total_seconds() / 3600
        
        # Calculate slippage
        expected_entry = signal.get('entry_price', 0)
        slippage = abs(actual_entry - expected_entry) / expected_entry if expected_entry > 0 else 0
        
        # Calculate execution quality
        execution_quality = max(0, 1 - slippage * 10)  # Penalize high slippage
        
        # Calculate performance score
        outcome = actual_outcome.get('outcome', 'LOSS')
        if outcome == 'WIN':
            performance_score = 0.7 + min(0.3, profit_loss_percentage / 10)  # Bonus for high profit
        elif outcome == 'BREAKEVEN':
            performance_score = 0.5
        else:
            performance_score = max(0, 0.3 - abs(profit_loss_percentage) / 10)  # Penalty for loss
        
        return {
            'profit_loss': profit_loss,
            'profit_loss_percentage': profit_loss_percentage,
            'duration_hours': duration_hours,
            'slippage': slippage,
            'execution_quality': execution_quality,
            'performance_score': performance_score
        }
    
    def _qualifies_for_nft(self, performance_metrics: Dict[str, Any]) -> bool:
        """Check if strategy qualifies for NFT creation"""
        win_rate = performance_metrics.get('win_rate', 0)
        profit_factor = performance_metrics.get('profit_factor', 1)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        total_trades = performance_metrics.get('total_trades', 0)
        
        # NFT qualification criteria
        criteria = [
            win_rate >= 0.7,           # 70% win rate
            profit_factor >= 2.0,      # 2.0 profit factor
            sharpe_ratio >= 1.5,       # 1.5 Sharpe ratio
            total_trades >= 50         # Minimum 50 trades
        ]
        
        # Must meet at least 3 out of 4 criteria
        return sum(criteria) >= 3
    
    def _calculate_gas_fee(self, data: Dict[str, Any]) -> float:
        """Calculate gas fee for transaction"""
        base_fee = 0.001  # Base fee
        data_size = len(json.dumps(data))
        size_fee = data_size * 0.00001  # Fee per byte
        
        return base_fee + size_fee
    
    def _get_next_nonce(self, sender: str) -> int:
        """Get next nonce for sender"""
        # Count transactions from this sender
        nonce = 0
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == sender:
                    nonce = max(nonce, tx.nonce + 1)
        
        # Check pending transactions
        for tx in self.pending_transactions:
            if tx.sender == sender:
                nonce = max(nonce, tx.nonce + 1)
        
        return nonce
    
    def _save_block_to_db(self, block: Block):
        """Save block to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save block
        cursor.execute('''
            INSERT OR REPLACE INTO blocks 
            (block_number, block_hash, previous_hash, merkle_root, timestamp, 
             nonce, difficulty, miner, block_reward, gas_used, block_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            block.block_number, block.block_hash, block.previous_hash,
            block.merkle_root, block.timestamp.isoformat(), block.nonce,
            block.difficulty, block.miner, block.block_reward, block.gas_used,
            json.dumps(block.to_dict())
        ))
        
        # Save transactions
        for tx in block.transactions:
            cursor.execute('''
                INSERT OR REPLACE INTO transactions
                (transaction_id, block_number, transaction_type, sender, receiver,
                 timestamp, signature, gas_fee, nonce, transaction_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tx.transaction_id, block.block_number, tx.transaction_type.value,
                tx.sender, tx.receiver, tx.timestamp.isoformat(),
                tx.signature, tx.gas_fee, tx.nonce, json.dumps(tx.to_dict())
            ))
        
        conn.commit()
        conn.close()
    
    def _save_signal_to_db(self, signal: TradingSignal, signal_hash: str):
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO signals
            (signal_id, signal_hash, pair, action, entry_price, stop_loss,
             take_profit, confidence, timeframe, strategy_name, creator,
             created_at, signal_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.signal_id, signal_hash, signal.pair, signal.action,
            signal.entry_price, signal.stop_loss, signal.take_profit,
            signal.confidence, signal.timeframe, signal.strategy_name,
            signal.creator, signal.created_at.isoformat(),
            json.dumps(signal.to_dict())
        ))
        
        conn.commit()
        conn.close()
    
    def _save_performance_to_db(self, performance: PerformanceRecord):
        """Save performance record to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO performance_records
            (signal_id, actual_outcome, profit_loss, profit_loss_percentage,
             execution_quality, verified_at, verifier, performance_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance.signal_id, performance.actual_outcome,
            performance.profit_loss, performance.profit_loss_percentage,
            performance.execution_quality, performance.verified_at.isoformat(),
            performance.verifier, json.dumps(performance.to_dict())
        ))
        
        conn.commit()
        conn.close()
    
    def _get_signal_from_db(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Get signal from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM signals WHERE signal_id = ?', (signal_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return {
                'signal_id': row[0],
                'signal_hash': row[1],
                'pair': row[2],
                'action': row[3],
                'entry_price': row[4],
                'stop_loss': row[5],
                'take_profit': row[6],
                'confidence': row[7],
                'timeframe': row[8],
                'strategy_name': row[9],
                'creator': row[10],
                'created_at': row[11],
                'signal_data': json.loads(row[12])
            }
        
        return None
    
    def _broadcast_block(self, block: Block):
        """Broadcast block to network nodes"""
        # In a real implementation, this would send the block to all network nodes
        logger.info(f"📡 Broadcasting block {block.block_number} to network")
    
    def _calculate_blockchain_size(self) -> float:
        """Calculate blockchain size in MB"""
        total_size = 0
        for block in self.chain:
            block_data = json.dumps(block.to_dict())
            total_size += len(block_data.encode())
        
        return total_size / (1024 * 1024)  # Convert to MB

class BlockchainSignalVerification:
    """Main blockchain signal verification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.blockchain = TradingBlockchain(config)
        self.crypto_manager = CryptographicManager()
        
        # Generate system key pair
        self.private_key, self.public_key = self.crypto_manager.generate_key_pair()
        
        # Deploy smart contracts
        self._deploy_smart_contracts()
        
        # Start mining if enabled
        if config.get('enable_mining', True):
            self.blockchain.start_mining(config.get('miner_id', 'system'))
        
        logger.info("🔗 Blockchain Signal Verification System initialized")
    
    def _deploy_smart_contracts(self):
        """Deploy necessary smart contracts"""
        
        # Signal verification contract
        signal_contract_code = """
        contract SignalVerification {
            function verifySignal(signal_id, performance_data) {
                // Verify signal performance and update reputation
                return updatePerformanceRecord(signal_id, performance_data);
            }
        }
        """
        
        self.blockchain.deploy_smart_contract('signal_contract', signal_contract_code)
        
        # Reputation management contract
        reputation_contract_code = """
        contract ReputationManagement {
            function updateReputation(trader_id, performance_score) {
                // Update trader reputation based on performance
                return calculateNewReputation(trader_id, performance_score);
            }
        }
        """
        
        self.blockchain.deploy_smart_contract('reputation_contract', reputation_contract_code)
        
        # NFT creation contract
        nft_contract_code = """
        contract TradingNFT {
            function createNFT(strategy_data, performance_metrics) {
                // Create NFT for exceptional trading strategies
                return mintTradingNFT(strategy_data, performance_metrics);
            }
        }
        """
        
        self.blockchain.deploy_smart_contract('nft_contract', nft_contract_code)
        
        logger.info("📜 Smart contracts deployed")
    
    async def create_verified_signal(self, signal_data: Dict[str, Any], 
                                   creator: str) -> Dict[str, Any]:
        """Create verified trading signal on blockchain"""
        
        # Create TradingSignal object
        trading_signal = TradingSignal(
            signal_id=str(uuid.uuid4()),
            pair=signal_data['pair'],
            action=signal_data['action'],
            entry_price=signal_data['entry_price'],
            stop_loss=signal_data['stop_loss'],
            take_profit=signal_data['take_profit'],
            confidence=signal_data['confidence'],
            timeframe=signal_data['timeframe'],
            strategy_name=signal_data['strategy_name'],
            ai_model_version=signal_data.get('ai_model_version', '1.0'),
            market_conditions=signal_data.get('market_conditions', {}),
            technical_indicators=signal_data.get('technical_indicators', {}),
            fundamental_factors=signal_data.get('fundamental_factors', {}),
            risk_assessment=signal_data.get('risk_assessment', {}),
            created_at=datetime.now(),
            creator=creator
        )
        
        # Create immutable signal on blockchain
        signal_hash = self.blockchain.create_immutable_signal(trading_signal, self.private_key)
        
        return {
            'signal_id': trading_signal.signal_id,
            'signal_hash': signal_hash,
            'blockchain_status': 'pending',
            'created_at': trading_signal.created_at.isoformat(),
            'verification_url': f"blockchain://signal/{signal_hash}"
        }
    
    async def verify_signal_outcome(self, signal_id: str, actual_outcome: Dict[str, Any], 
                                  verifier: str) -> Dict[str, Any]:
        """Verify signal outcome and update blockchain"""
        
        verification_result = self.blockchain.verify_signal_performance(
            signal_id, actual_outcome, verifier, self.private_key
        )
        
        return verification_result
    
    async def create_strategy_nft(self, strategy_data: Dict[str, Any], 
                                performance_metrics: Dict[str, Any], 
                                creator: str) -> Dict[str, Any]:
        """Create NFT for exceptional trading strategy"""
        
        nft_result = self.blockchain.create_trading_nft(
            strategy_data, performance_metrics, creator, self.private_key
        )
        
        return nft_result
    
    def get_signal_verification_status(self, signal_id: str) -> Dict[str, Any]:
        """Get verification status of signal"""
        
        signal_history = self.blockchain.get_signal_history(signal_id)
        
        # Determine verification status
        has_creation = any(tx['transaction']['transaction_type'] == 'signal_creation' 
                          for tx in signal_history['transaction_history'])
        has_verification = any(tx['transaction']['transaction_type'] == 'signal_verification' 
                             for tx in signal_history['transaction_history'])
        
        if has_verification:
            status = 'verified'
        elif has_creation:
            status = 'pending_verification'
        else:
            status = 'not_found'
        
        return {
            'signal_id': signal_id,
            'verification_status': status,
            'transaction_count': signal_history['total_transactions'],
            'blockchain_history': signal_history['transaction_history'],
            'immutable': True,
            'tamper_proof': True
        }
    
    def get_trader_blockchain_reputation(self, trader_id: str) -> Dict[str, Any]:
        """Get trader reputation from blockchain"""
        
        reputation_data = self.blockchain.get_trader_reputation(trader_id)
        
        # Add blockchain verification info
        reputation_data.update({
            'verification_method': 'blockchain',
            'tamper_proof': True,
            'decentralized': True,
            'consensus_verified': True
        })
        
        return reputation_data
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get blockchain system status"""
        
        blockchain_stats = self.blockchain.get_blockchain_stats()
        validation_result = self.blockchain.validate_blockchain()
        
        return {
            'blockchain_stats': blockchain_stats,
            'validation_result': validation_result,
            'mining_status': self.blockchain.is_mining,
            'smart_contracts': list(self.blockchain.smart_contracts.keys()),
            'system_health': 'healthy' if validation_result['is_valid'] else 'degraded',
            'consensus_algorithm': 'proof_of_performance',
            'decentralization_level': 'high'
        }

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    # Test blockchain verification system
    config = {
        'difficulty': 2,  # Lower difficulty for testing
        'block_reward': 10.0,
        'max_transactions_per_block': 10,
        'db_path': 'test_blockchain.db',
        'node_id': 'test_node_001',
        'enable_mining': True,
        'miner_id': 'test_miner'
    }
    
    verification_system = BlockchainSignalVerification(config)
    
    # Sample signal data
    sample_signal = {
        'pair': 'EURUSD',
        'action': 'BUY',
        'entry_price': 1.1000,
        'stop_loss': 1.0950,
        'take_profit': 1.1100,
        'confidence': 0.85,
        'timeframe': 'H1',
        'strategy_name': 'AI_Momentum_Strategy',
        'ai_model_version': '2.0',
        'market_conditions': {
            'volatility': 0.015,
            'trend': 'bullish',
            'volume': 'high'
        },
        'technical_indicators': {
            'rsi': 65,
            'macd': 0.0015,
            'bollinger_position': 0.7
        }
    }
    
    async def test_blockchain_verification():
        # Create verified signal
        signal_result = await verification_system.create_verified_signal(
            sample_signal, 'trader_001'
        )
        
        print("🔗 Blockchain Signal Created:")
        print(f"Signal ID: {signal_result['signal_id']}")
        print(f"Signal Hash: {signal_result['signal_hash']}")
        
        # Wait for mining
        await asyncio.sleep(5)
        
        # Verify signal outcome
        actual_outcome = {
            'outcome': 'WIN',
            'actual_entry_price': 1.1005,
            'actual_exit_price': 1.1095,
            'exit_time': datetime.now().isoformat()
        }
        
        verification_result = await verification_system.verify_signal_outcome(
            signal_result['signal_id'], actual_outcome, 'verifier_001'
        )
        
        print(f"\n✅ Signal Verification:")
        print(f"Status: {verification_result['verification_status']}")
        print(f"Profit: {verification_result['performance_metrics']['profit_loss_percentage']:.2f}%")
        
        # Get verification status
        status = verification_system.get_signal_verification_status(signal_result['signal_id'])
        print(f"\n📊 Verification Status: {status['verification_status']}")
        
        # Get trader reputation
        reputation = verification_system.get_trader_blockchain_reputation('trader_001')
        print(f"\n🏆 Trader Reputation:")
        print(f"Score: {reputation['reputation_score']:.3f}")
        print(f"Grade: {reputation['reputation_grade']}")
        
        # Get system status
        system_status = verification_system.get_system_status()
        print(f"\n🔗 Blockchain Status:")
        print(f"Blocks: {system_status['blockchain_stats']['total_blocks']}")
        print(f"Transactions: {system_status['blockchain_stats']['total_transactions']}")
        print(f"Health: {system_status['system_health']}")
    
    # asyncio.run(test_blockchain_verification())