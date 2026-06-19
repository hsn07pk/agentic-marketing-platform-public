"""
LinUCB contextual bandit implementation with GPU optimization
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinUCBBandit:
    """
    Linear Upper Confidence Bound contextual bandit
    GPU-optimized for fast decision making
    """
    
    def __init__(
        self,
        n_arms: int,
        n_features: int,
        alpha: float = 1.0,
        use_gpu: bool = True
    ):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.device = DEVICE if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        
        self.A = torch.zeros((n_arms, n_features, n_features), device=self.device)
        self.b = torch.zeros((n_arms, n_features, 1), device=self.device)
        
        for i in range(n_arms):
            self.A[i] = torch.eye(n_features, device=self.device)
        
        self.t = 0
        self.arm_counts = torch.zeros(n_arms, device=self.device)
        self.rewards = []
        
    def select_arm(self, context: np.ndarray) -> Tuple[int, float]:
        x = torch.tensor(context, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        ucb_values = torch.zeros(self.n_arms, device=self.device)
        
        for a in range(self.n_arms):
            A_inv = torch.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            
            mean = (theta.T @ x).squeeze()
            variance = torch.sqrt(x.T @ A_inv @ x).squeeze()
            ucb_values[a] = mean + self.alpha * variance
        
        selected_arm = torch.argmax(ucb_values).cpu().item()
        confidence = torch.softmax(ucb_values, dim=0)[selected_arm].cpu().item()
        
        self.t += 1
        self.arm_counts[selected_arm] += 1
        
        return selected_arm, confidence
    
    def update(self, arm: int, context: np.ndarray, reward: float):
        x = torch.tensor(context, dtype=torch.float32, device=self.device).reshape(-1, 1)
        r = torch.tensor(reward, dtype=torch.float32, device=self.device)
        
        self.A[arm] += x @ x.T
        self.b[arm] += r * x
        
        self.rewards.append(reward)
    
    def batch_update(self, arms: List[int], contexts: np.ndarray, rewards: np.ndarray):
        X = torch.tensor(contexts, dtype=torch.float32, device=self.device)
        R = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        for i, arm in enumerate(arms):
            x = X[i].reshape(-1, 1)
            r = R[i]
            
            self.A[arm] += x @ x.T
            self.b[arm] += r * x
        
        self.rewards.extend(rewards.tolist())
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_pulls': self.t,
            'arm_counts': self.arm_counts.cpu().numpy().tolist(),
            'average_reward': np.mean(self.rewards) if self.rewards else 0,
            'cumulative_reward': sum(self.rewards),
            'device': str(self.device),
            'gpu_memory_allocated': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }
    
    def reset(self):
        self.A = torch.zeros((self.n_arms, self.n_features, self.n_features), device=self.device)
        self.b = torch.zeros((self.n_arms, self.n_features, 1), device=self.device)
        
        for i in range(self.n_arms):
            self.A[i] = torch.eye(self.n_features, device=self.device)
        
        self.t = 0
        self.arm_counts = torch.zeros(self.n_arms, device=self.device)
        self.rewards = []