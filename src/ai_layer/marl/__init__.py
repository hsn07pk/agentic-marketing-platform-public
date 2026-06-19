"""
Multi-Agent Reinforcement Learning (MARL) module

Components:
- OffPolicyEvaluator: Doubly-robust OPE for policy validation
- MARLGatekeeper: Two-stage promotion gate (OPE + Canary)
- MARLPolicyTrainer: DQN-based policy training
"""
from .ope_gating import OffPolicyEvaluator, MARLGatekeeper
from .policy_trainer import MARLPolicyTrainer, TrainingConfig, MARLPolicyNetwork

__all__ = [
    'OffPolicyEvaluator', 
    'MARLGatekeeper',
    'MARLPolicyTrainer',
    'TrainingConfig',
    'MARLPolicyNetwork'
]
