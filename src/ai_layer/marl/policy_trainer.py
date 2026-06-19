"""
MARL Policy Training Module

Per Research Plan Section 2.4:
- Centralized Training, Decentralized Execution (CTDE) paradigm
- Trains policies on historical campaign data
- Saves policies for OPE validation before deployment
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    policy_id: str
    learning_rate: float = 0.001
    hidden_dim: int = 128
    batch_size: int = 64
    epochs: int = 100
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    target_update_freq: int = 10
    min_samples: int = 500
    

class MARLPolicyNetwork(nn.Module):
    """
    Deep Q-Network for MARL policy
    
    Input: State features (campaign context, audience, timing, etc.)
    Output: Q-values for each action (content hooks, budget allocation)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(MARLPolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        self.to(DEVICE)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_proba(self, state_features: np.ndarray) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state_features).to(DEVICE)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            q_values = self.forward(x)
            probs = torch.softmax(q_values, dim=-1).cpu().numpy()[0]
        
        return {f'action_{i}': float(p) for i, p in enumerate(probs)}


class ReplayBuffer:
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(DEVICE)
        actions = torch.LongTensor([b[1] for b in batch]).to(DEVICE)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(DEVICE)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(DEVICE)
        dones = torch.FloatTensor([b[4] for b in batch]).to(DEVICE)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class MARLPolicyTrainer:
    """
    Trainer for MARL policies using DQN with experience replay
    
    Per research plan: Centralized Training, Decentralized Execution (CTDE)
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state_dim = 13
        self.action_dim = 5
        
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
        self.replay_buffer = ReplayBuffer()
        self.epsilon = config.epsilon_start
        self.training_history = []
        
        self.model_dir = Path("models/marl_policies") / config.policy_id
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MARL Policy Trainer for {config.policy_id}")
        
    def _init_networks(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = MARLPolicyNetwork(state_dim, action_dim, self.config.hidden_dim)
        self.target_net = MARLPolicyNetwork(state_dim, action_dim, self.config.hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config.learning_rate
        )
        
    def _state_to_features(self, state: Dict[str, Any]) -> np.ndarray:
        features = []
        
        for key in ['budget_remaining', 'campaign_day', 'impressions', 'clicks', 'ctr']:
            val = state.get(key, 0)
            if isinstance(val, (int, float)):
                features.append(float(val))
            else:
                features.append(0.0)
                
        # Cyclical encoding for time features
        hour = state.get('hour', 12)
        day_of_week = state.get('day_of_week', 0)
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        
        platform = state.get('platform', 'linkedin')
        features.append(1.0 if platform == 'linkedin' else 0.0)
        features.append(1.0 if platform == 'twitter' else 0.0)
        features.append(1.0 if platform == 'email' else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _action_to_index(self, action: Dict[str, Any]) -> int:
        hook_map = {
            'productivity_boost': 0,
            'time_savings': 1,
            'roi_focus': 2,
            'innovation': 3,
            'competitive_edge': 4
        }
        
        hook = action.get('selected_hook', action.get('hook', 'productivity_boost'))
        return hook_map.get(hook, 0)
    
    def prepare_training_data(self, historical_decisions: List[Dict[str, Any]]) -> int:
        if not historical_decisions:
            logger.warning("No historical decisions provided for training")
            return 0
            
        sample_state = historical_decisions[0].get('state', {})
        state_features = self._state_to_features(sample_state)
        state_dim = len(state_features)
        
        action_indices = set()
        for d in historical_decisions:
            action_idx = self._action_to_index(d.get('action', {}))
            action_indices.add(action_idx)
        action_dim = max(5, max(action_indices) + 1)
        
        self._init_networks(state_dim, action_dim)
        
        for i, decision in enumerate(historical_decisions):
            state = self._state_to_features(decision.get('state', {}))
            action = self._action_to_index(decision.get('action', {}))
            reward = float(decision.get('reward', 0.0))
            
            # Get next state (or same state if last in sequence)
            if i < len(historical_decisions) - 1:
                next_state = self._state_to_features(historical_decisions[i + 1].get('state', {}))
                done = False
            else:
                next_state = state.copy()
                done = True
                
            self.replay_buffer.push(state, action, reward, next_state, done)
            
        logger.info(f"Prepared {len(self.replay_buffer)} training samples")
        return len(self.replay_buffer)
    
    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        if self.policy_net is None:
            raise ValueError("Call prepare_training_data first")
            
        epochs = epochs or self.config.epochs
        batch_size = min(self.config.batch_size, len(self.replay_buffer))
        
        if len(self.replay_buffer) < self.config.min_samples:
            logger.warning(
                f"Insufficient samples: {len(self.replay_buffer)} < {self.config.min_samples}"
            )
            
        logger.info(f"Starting MARL policy training for {epochs} epochs")
        
        losses = []
        rewards = []
        
        for epoch in range(epochs):
            self.policy_net.train()
            epoch_loss = 0.0
            num_batches = max(1, len(self.replay_buffer) // batch_size)
            
            for _ in range(num_batches):
                states, actions, batch_rewards, next_states, dones = \
                    self.replay_buffer.sample(batch_size)
                
                # Current Q values
                current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Double DQN: select actions with policy net, evaluate with target net
                with torch.no_grad():
                    next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                    next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
                    target_q = batch_rewards + (1 - dones) * self.config.gamma * next_q
                
                loss = nn.functional.smooth_l1_loss(current_q.squeeze(), target_q)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            if epoch % self.config.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")
                
        metrics = {
            'policy_id': self.config.policy_id,
            'epochs_trained': epochs,
            'final_loss': losses[-1] if losses else 0.0,
            'avg_loss': np.mean(losses) if losses else 0.0,
            'samples_used': len(self.replay_buffer),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'trained_at': datetime.now().isoformat()
        }
        
        self.training_history.append(metrics)
        
        logger.info(f"Training complete: {metrics}")
        return metrics
    
    def save(self) -> Path:
        if self.policy_net is None:
            raise ValueError("No trained policy to save")
            
        policy_path = self.model_dir / "policy.pkl"
        with open(policy_path, 'wb') as f:
            pickle.dump(self.policy_net, f)
            
        metadata_path = self.model_dir / "metadata.json"
        metadata = {
            'policy_id': self.config.policy_id,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.config.hidden_dim,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Policy saved to {self.model_dir}")
        return policy_path
    
    @classmethod
    def load(cls, policy_id: str) -> 'MARLPolicyNetwork':
        model_dir = Path("models/marl_policies") / policy_id
        policy_path = model_dir / "policy.pkl"
        
        if not policy_path.exists():
            raise FileNotFoundError(f"No policy found at {policy_path}")
            
        with open(policy_path, 'rb') as f:
            policy = pickle.load(f)
            
        logger.info(f"Loaded policy from {policy_path}")
        return policy


async def train_marl_policy_from_db(
    policy_id: str = "marl_policy_v1",
    min_samples: int = 500,
    epochs: int = 100
) -> Dict[str, Any]:
    from ...data_layer.database.connection import async_session_maker
    from ...data_layer.repositories.campaign_repo import CampaignRepository
    
    async with async_session_maker() as session:
        repo = CampaignRepository(session)
        
        historical_data = await repo.get_recent_decisions_for_ope(
            limit=min_samples * 2,
            min_impressions=50,
            lookback_days=60
        )
        
        if len(historical_data) < min_samples:
            return {
                'success': False,
                'error': f'Insufficient training data: {len(historical_data)} < {min_samples}',
                'samples_found': len(historical_data)
            }
            
        config = TrainingConfig(
            policy_id=policy_id,
            epochs=epochs,
            min_samples=min_samples
        )
        
        trainer = MARLPolicyTrainer(config)
        samples_prepared = trainer.prepare_training_data(historical_data)
        
        if samples_prepared < min_samples:
            return {
                'success': False,
                'error': f'Insufficient prepared samples: {samples_prepared}',
                'samples_prepared': samples_prepared
            }
            
        metrics = trainer.train()
        
        save_path = trainer.save()
        
        return {
            'success': True,
            'policy_id': policy_id,
            'save_path': str(save_path),
            'training_metrics': metrics
        }
