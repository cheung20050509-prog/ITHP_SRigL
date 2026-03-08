"""
Neuroplastic Scheduler for ITHP
Implements bio-inspired dynamic topology optimization:
1. Activity-based Pruning - remove inactive connections
2. Hebbian Growth - grow connections between co-active neurons
3. Skip Connections - add cross-layer shortcuts

Key difference from SRigL: parameters can increase, goal is performance not compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math
import numpy as np

# 统一的神经元级拓扑管理
try:
    from .neuron_topology import NeuronLevelTopology
    from .global_neuron_graph import GlobalNeuronGraph
except ImportError:
    from neuron_topology import NeuronLevelTopology
    from global_neuron_graph import GlobalNeuronGraph


class ActivityTracker:
    """Track connection activity using EMA of |weight| × |input| × |gradient|"""
    
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.activity = {}  # layer_name -> (out, in) tensor
        self.input_cache = {}  # layer_name -> cached input activation
        
    def cache_input(self, layer_name: str, input_act: torch.Tensor):
        """Cache input activation during forward pass"""
        # input_act: (batch, in_features)
        # Store mean absolute activation per input neuron
        self.input_cache[layer_name] = input_act.abs().mean(dim=0).detach()
        
    def update(self, layer_name: str, weight: torch.Tensor, gradient: torch.Tensor):
        """Update activity after backward pass"""
        if layer_name not in self.input_cache:
            return
            
        input_act_mean = self.input_cache[layer_name]  # (in_features,)
        
        # activity[i,j] = |weight[i,j]| × |input[j]| × |gradient[i,j]|
        current_activity = (
            weight.abs() * 
            input_act_mean.unsqueeze(0) * 
            gradient.abs()
        ).detach()
        
        # EMA update
        if layer_name not in self.activity:
            self.activity[layer_name] = current_activity
        else:
            self.activity[layer_name] = (
                (1 - self.ema_alpha) * self.activity[layer_name] +
                self.ema_alpha * current_activity
            )
            
    def get_inactive_connections(self, layer_name: str, mask: torch.Tensor, 
                                  threshold: float) -> torch.Tensor:
        """Return mask of connections that should be pruned"""
        if layer_name not in self.activity:
            return torch.zeros_like(mask)
            
        activity = self.activity[layer_name]
        # Only consider currently active connections
        inactive = (activity < threshold) & mask
        return inactive
    
    def reset(self):
        """Reset all tracking"""
        self.activity.clear()
        self.input_cache.clear()


class HebbianTracker:
    """Track co-activation for Hebbian growth"""
    
    def __init__(self):
        self.co_activation = {}  # layer_name -> (out, in) tensor
        self.sample_count = {}
        
    def record(self, layer_name: str, input_act: torch.Tensor, output_act: torch.Tensor):
        """Record co-activation during forward pass"""
        # input_act: (batch, in_features)
        # output_act: (batch, out_features)
        
        # Compute co-activation matrix: (out, in)
        # co_act[i,j] = mean over batch of |output[:,i]| × |input[:,j]|
        co_act = torch.einsum('bi,bj->ij', output_act.abs(), input_act.abs())
        co_act = co_act / input_act.shape[0]  # Normalize by batch size
        co_act = co_act.detach()
        
        if layer_name not in self.co_activation:
            self.co_activation[layer_name] = co_act
            self.sample_count[layer_name] = 1
        else:
            self.co_activation[layer_name] += co_act
            self.sample_count[layer_name] += 1
            
    def get_growth_candidates(self, layer_name: str, mask: torch.Tensor, 
                               top_k: int) -> torch.Tensor:
        """Find connections to grow based on co-activation. Returns flat indices tensor."""
        if layer_name not in self.co_activation:
            return torch.tensor([], dtype=torch.long, device=mask.device)
            
        # Average co-activation
        co_act = self.co_activation[layer_name] / self.sample_count[layer_name]
        
        # Only consider positions without connections
        co_act_masked = co_act * (~mask).float()
        
        # Find top-k positions
        flat = co_act_masked.flatten()
        k = min(top_k, (flat > 0).sum().item())
        if k == 0:
            return torch.tensor([], dtype=torch.long, device=mask.device)
            
        _, indices = torch.topk(flat, k)
        return indices  # Return tensor, not list
    
    def reset(self):
        """Reset statistics"""
        self.co_activation.clear()
        self.sample_count.clear()


class ContinuousScalePolicy:
    """
    连续动作空间的强化学习策略，用于学习最优scale。
    
    比离散UCB更精细，使用高斯策略 + 自然策略梯度：
    - 动作: scale ~ N(μ, σ²)，clamp到有效范围
    - 状态: 当前loss, loss变化率, 稀疏度等
    - 奖励: loss下降 + 稀疏度维持
    
    使用REINFORCE with baseline进行策略更新。
    """
    
    def __init__(
        self, 
        name: str = "weight",  # "weight" or "graph"
        init_mean: float = 0.05,
        init_std: float = 0.02,
        min_scale: float = 0.005,
        max_scale: float = 0.15,
        lr_mean: float = 0.01,
        lr_std: float = 0.005,
        baseline_ema: float = 0.9,
    ):
        self.name = name
        
        # 策略参数: scale ~ N(μ, σ²)
        self.mean = init_mean
        self.std = init_std
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # 学习率
        self.lr_mean = lr_mean
        self.lr_std = lr_std
        
        # Baseline for variance reduction (EMA of rewards)
        self.baseline = 0.0
        self.baseline_ema = baseline_ema
        
        # 历史记录
        self.history = []  # [(scale, reward), ...]
        self.current_scale = None
        self.pre_update_loss = None
        
        # 统计
        self.total_updates = 0
        self.cumulative_reward = 0.0
        
    def sample_scale(self) -> float:
        """采样scale从当前策略分布"""
        # 从高斯分布采样
        scale = np.random.normal(self.mean, self.std)
        # Clamp到有效范围
        scale = np.clip(scale, self.min_scale, self.max_scale)
        self.current_scale = scale
        return scale
    
    def begin_update(self, current_loss: float):
        """记录更新前的loss"""
        self.pre_update_loss = current_loss
        
    def end_update(self, post_loss: float, sparsity_penalty: float = 0.0):
        """
        更新完成后计算reward并更新策略。
        
        Args:
            post_loss: 更新后的loss
            sparsity_penalty: 稀疏度惩罚（如果稀疏度偏离目标）
        """
        if self.pre_update_loss is None or self.current_scale is None:
            return
            
        # 计算reward: loss下降 - 稀疏度惩罚
        reward = (self.pre_update_loss - post_loss) - 0.1 * sparsity_penalty
        
        # 更新baseline (EMA)
        self.baseline = self.baseline_ema * self.baseline + (1 - self.baseline_ema) * reward
        
        # Advantage = reward - baseline
        advantage = reward - self.baseline
        
        # ============ 策略梯度更新 ============
        # REINFORCE: ∇J = E[∇log π(a|s) × A]
        # 对于高斯策略 π(a) = N(μ, σ²):
        #   ∇_μ log π = (a - μ) / σ²
        #   ∇_σ log π = ((a - μ)² - σ²) / σ³
        
        a = self.current_scale
        
        # 梯度计算
        grad_mean = (a - self.mean) / (self.std ** 2)
        grad_std = ((a - self.mean) ** 2 - self.std ** 2) / (self.std ** 3)
        
        # 策略更新
        self.mean += self.lr_mean * advantage * grad_mean
        self.std += self.lr_std * advantage * grad_std
        
        # 约束
        self.mean = np.clip(self.mean, self.min_scale, self.max_scale)
        self.std = np.clip(self.std, 0.005, 0.05)  # 不要太确定也不要太随机
        
        # 记录
        self.history.append((self.current_scale, reward))
        self.total_updates += 1
        self.cumulative_reward += reward
        
        self.pre_update_loss = None
        
    def get_stats(self) -> Dict:
        """获取策略统计信息"""
        recent_rewards = [r for _, r in self.history[-10:]] if self.history else []
        return {
            'name': self.name,
            'mean': float(self.mean),
            'std': float(self.std),
            'baseline': float(self.baseline),
            'total_updates': self.total_updates,
            'avg_reward': float(self.cumulative_reward / max(1, self.total_updates)),
            'recent_avg_reward': float(np.mean(recent_rewards)) if recent_rewards else 0.0,
            'last_scale': float(self.current_scale) if self.current_scale else None,
        }


class AdaptiveUpdatePolicy:
    """UCB-based policy for selecting topology update magnitude.
    
    Uses Upper Confidence Bound (UCB1) algorithm to balance
    exploration (trying different scales) and exploitation (using best known).
    
    This is a simple reinforcement learning approach:
    - Arms: different update scales [0.02, 0.05, 0.08, 0.1]
    - Reward: negative delta loss (loss decrease = positive reward)
    """
    
    def __init__(self, scales: List[float] = None):
        self.scales = scales or [0.02, 0.05, 0.08, 0.1]
        self.n_arms = len(self.scales)
        
        # UCB statistics
        self.counts = np.zeros(self.n_arms)  # times each arm was pulled
        self.rewards = np.zeros(self.n_arms)  # cumulative reward
        self.total_pulls = 0
        
        # Current state
        self.current_arm = 1  # Start with 0.05
        self.pre_update_loss = None
        
    def select_scale(self) -> float:
        """Select scale using UCB1 algorithm."""
        self.total_pulls += 1
        
        # Exploration: try each arm at least once
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                self.current_arm = i
                return self.scales[i]
        
        # UCB1 formula: mean_reward + sqrt(2 * ln(t) / n_i)
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            mean_reward = self.rewards[i] / self.counts[i]
            exploration_bonus = np.sqrt(2 * np.log(self.total_pulls) / self.counts[i])
            ucb_values[i] = mean_reward + exploration_bonus
            
        self.current_arm = np.argmax(ucb_values)
        return self.scales[self.current_arm]
    
    def begin_update(self, current_loss: float):
        """Record loss before update."""
        self.pre_update_loss = current_loss
        self.counts[self.current_arm] += 1
        
    def end_update(self, post_loss: float):
        """Record reward after observing effect."""
        if self.pre_update_loss is not None:
            # Reward = negative delta loss (loss decrease is good)
            reward = self.pre_update_loss - post_loss
            self.rewards[self.current_arm] += reward
            
    def get_stats(self) -> Dict:
        """Get policy statistics."""
        stats = {}
        for i, scale in enumerate(self.scales):
            if self.counts[i] > 0:
                avg_reward = self.rewards[i] / self.counts[i]
                stats[f'scale_{scale}'] = {
                    'count': int(self.counts[i]),
                    'avg_reward': float(avg_reward),
                }
        stats['current_scale'] = self.scales[self.current_arm]
        return stats


class StabilityGuard:
    """Monitor training stability and control topology updates.
    
    Key insight: 拓扑更新应该在权重稳定后进行，而不是固定间隔。
    训练过程：训练 -> 稳定 -> 更新拓扑 -> 再训练 -> ...
    
    增强版：要求连续稳定一段时间才允许拓扑更新
    """
    
    def __init__(self, patience: int = 50, variance_threshold: float = 0.20,
                 min_stable_steps: int = 15):
        # Loss tracking
        self.loss_history = []
        self.ib_loss_history = []
        self.patience = patience
        self.variance_threshold = variance_threshold  # CV < 20% is stable
        
        # Stability state
        self.is_stable = False
        self.consecutive_stable_steps = 0  # 连续稳定的步数
        self.min_stable_steps = min_stable_steps  # 至少稳定这么多步
        self.steps_since_topology_update = 0
        self.min_steps_between_updates = 80  # 更频繁更新以测试
        
        # Performance tracking for adaptive updates
        self.pre_update_loss = None
        self.post_update_losses = []
        self.update_budget_multiplier = 1.0  # 自适应调整幅度
        
        # History of topology update effects
        self.update_history = []  # [(delta_loss, prune_count, grow_count), ...]
        
    def update(self, loss: float, ib_loss: float = None):
        """Update with current loss values"""
        self.loss_history.append(loss)
        if ib_loss is not None:
            self.ib_loss_history.append(ib_loss)
        self.steps_since_topology_update += 1
        
        # Track post-update performance
        if self.pre_update_loss is not None and len(self.post_update_losses) < 100:
            self.post_update_losses.append(loss)
        
        # Check stability (loss converging)
        was_stable = self.is_stable
        if len(self.loss_history) >= self.patience:
            recent = self.loss_history[-self.patience:]
            mean_loss = np.mean(recent)
            std_loss = np.std(recent)
            
            # Stable = low variance relative to mean
            if mean_loss > 0:
                cv = std_loss / mean_loss  # coefficient of variation
                self.is_stable = cv < self.variance_threshold
            else:
                self.is_stable = False
        
        # 追踪连续稳定步数
        if self.is_stable:
            self.consecutive_stable_steps += 1
        else:
            self.consecutive_stable_steps = 0
                
    def should_update_topology(self) -> bool:
        """Check if it's time to update topology.
        
        条件（增强版）：
        1. 权重已稳定 (loss converged)
        2. 连续稳定足够长时间 (避免偶然稳定)
        3. 距离上次更新足够久
        4. IB loss 健康
        """
        if not self.is_stable:
            return False
        if self.consecutive_stable_steps < self.min_stable_steps:
            return False  # 必须连续稳定足够久
        if self.steps_since_topology_update < self.min_steps_between_updates:
            return False
        if not self._check_ib_health():
            return False
        return True
    
    def _check_ib_health(self) -> bool:
        """Check if information bottleneck is healthy"""
        if len(self.ib_loss_history) < 10:
            return True
        recent = np.mean(self.ib_loss_history[-5:])
        baseline = np.mean(self.ib_loss_history[:5])
        if baseline > 0 and recent > baseline * 2.0:
            return False
        return True
    
    def begin_topology_update(self):
        """Called before topology update starts"""
        self.pre_update_loss = np.mean(self.loss_history[-20:]) if len(self.loss_history) >= 20 else None
        self.post_update_losses = []
        
    def end_topology_update(self, prune_count: int, grow_count: int):
        """Called after topology update finishes. Evaluate effect."""
        self.steps_since_topology_update = 0
        self.is_stable = False  # Reset, need to re-stabilize
        self.consecutive_stable_steps = 0  # 重置连续稳定计数
        
    def evaluate_last_update(self) -> float:
        """Evaluate the effect of last topology update.
        
        Returns adjustment factor for next update:
        - > 1.0: last update was beneficial, can be more aggressive
        - < 1.0: last update was harmful, be more conservative
        - 1.0: neutral
        """
        if self.pre_update_loss is None or len(self.post_update_losses) < 50:
            return 1.0  # Not enough data
            
        post_mean = np.mean(self.post_update_losses[-20:])
        delta = post_mean - self.pre_update_loss
        
        if delta < -0.1:  # Loss decreased (good!)
            self.update_budget_multiplier = min(1.5, self.update_budget_multiplier * 1.1)
            return 1.1
        elif delta > 0.1:  # Loss increased (bad)
            self.update_budget_multiplier = max(0.3, self.update_budget_multiplier * 0.8)
            return 0.8
        else:  # Neutral
            return 1.0
    
    def get_update_scale(self) -> float:
        """Get the scale factor for topology update amounts."""
        return self.update_budget_multiplier
    
    def can_modify_topology(self) -> bool:
        """Legacy interface - now uses should_update_topology logic"""
        return self.should_update_topology()


class ForwardHook:
    """Hook to capture layer input/output during forward pass"""
    
    def __init__(self, layer_name: str, scheduler: 'NeuroplasticScheduler'):
        self.layer_name = layer_name
        self.scheduler = scheduler
        
    def __call__(self, module, input, output):
        if len(input) > 0 and isinstance(input[0], torch.Tensor):
            input_act = input[0]
            output_act = output
            
            # 兼容旧的tracker
            if input_act.dim() == 3:
                input_act_mean = input_act.mean(dim=1)
            else:
                input_act_mean = input_act
                
            self.scheduler.activity_tracker.cache_input(self.layer_name, input_act_mean)
            
            if output_act.dim() == 3:
                output_act_mean = output_act.mean(dim=1)
            else:
                output_act_mean = output_act
                
            self.scheduler.hebbian_tracker.record(self.layer_name, input_act_mean, output_act_mean)
            
            # 新的NeuronLevelTopology: 缓存激活
            if self.layer_name in self.scheduler.topologies:
                self.scheduler.topologies[self.layer_name].cache_activations(
                    input_act, output_act
                )


class BackwardHook:
    """Hook to capture gradients and update activity"""
    
    def __init__(self, layer_name: str, scheduler: 'NeuroplasticScheduler'):
        self.layer_name = layer_name
        self.scheduler = scheduler
        
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        weight = self.scheduler.weights[self.layer_name]
        self.scheduler.activity_tracker.update(self.layer_name, weight, grad)
        
        # 新的NeuronLevelTopology: 更新importance
        if self.layer_name in self.scheduler.topologies:
            self.scheduler.topologies[self.layer_name].update_importance(grad)
        
        # Apply mask to gradient
        mask = self.scheduler.masks[self.layer_name]
        return grad * mask.float()


class NeuroplasticScheduler:
    """
    Main scheduler for neuroplastic training.
    Manages pruning, growth, and topology updates.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        config: Dict = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.total_steps = total_steps
        
        # Default config
        default_config = {
            # Activity tracking
            'ema_alpha': 0.1,
            
            # Warmup - train this many steps before any topology changes
            'warmup_steps': 500,
            
            # Pruning
            'prune_interval': 200,
            'prune_threshold': 0.001,
            'min_fan_in': 3,
            'min_fan_out': 3,  # Protect against dead neurons (no output)
            'max_prune_ratio': 0.1,
            
            # Growth
            'growth_interval': 200,
            'max_density': 1.5,
            'growth_ratio': 0.05,
            
            # Stability - 拓扑更新前需要微调到稳定状态
            'stability_patience': 50,      # 检测50步内的稳定性
            'variance_threshold': 0.35,    # CV<35%认为稳定（测试更宽松）
            'min_stable_steps': 10,        # 连续稳定10步才允许更新
            
            # Layer patterns - now covers ITHP + DeBERTa FFN (51% of params)
            'target_patterns': [
                # ITHP modules (~450K connections)
                'ithp.encoder1',
                'ithp.encoder2', 
                'ithp.MLP1',
                'ithp.MLP2',
                'expand.0',
                # DeBERTa FFN layers (~56M connections per direction)
                # intermediate.dense: 768 -> 3072 (expansion)
                # output.dense: 3072 -> 768 (compression, not LayerNorm)
                'intermediate.dense',
                'output.dense',
            ],
        }
        self.config = {**default_config, **(config or {})}
        
        # Trackers
        self.activity_tracker = ActivityTracker(self.config['ema_alpha'])
        self.hebbian_tracker = HebbianTracker()
        self.stability_guard = StabilityGuard(
            patience=self.config['stability_patience'],
            variance_threshold=self.config['variance_threshold'],
            min_stable_steps=self.config.get('min_stable_steps', 50)
        )
        
        # ============ 独立的连续RL策略 (替代离散UCB) ============
        # Prune策略：独立学习最优prune scale
        self.prune_policy = ContinuousScalePolicy(
            name="prune",
            init_mean=0.05,      # 初始均值5%
            init_std=0.02,       # 初始探索范围
            min_scale=0.005,     # 最小0.5%
            max_scale=0.15,      # 最大15%
            lr_mean=0.01,
            lr_std=0.005,
        )
        
        # Grow策略：独立学习最优grow scale（不再与prune绑定）
        self.grow_policy = ContinuousScalePolicy(
            name="grow",
            init_mean=0.03,      # 初始比prune保守
            init_std=0.015,      # 探索范围
            min_scale=0.005,     # 最小0.5%
            max_scale=0.12,      # 最大12%（比prune保守）
            lr_mean=0.01,
            lr_std=0.005,
        )
        
        # 图边连接：独立策略，学习图边的最优更新幅度（保留供graph_fusion使用）
        self.graph_policy = ContinuousScalePolicy(
            name="graph",
            init_mean=0.10,      # 图边初始均值10%（比权重激进一些）
            init_std=0.03,       # 更大探索范围
            min_scale=0.02,      # 最小2%（避免太保守）
            max_scale=0.25,      # 最大25%（允许激进更新）
            lr_mean=0.015,       # 稍快学习
            lr_std=0.008,
        )
        
        # 保留旧的UCB策略作为备用
        self.update_policy = AdaptiveUpdatePolicy(
            scales=[0.02, 0.05, 0.08, 0.1]
        )
        self.pending_ucb_reward_step = None  # Step at which to provide UCB reward
        self.ucb_reward_delay = 100  # Wait this many steps to evaluate effect
        
        # 连续策略reward收集
        self.pending_prune_reward_step = None
        self.pending_grow_reward_step = None
        self.pending_graph_reward_step = None
        self.reward_delay = 50  # 50步后评估效果
        
        # Storage
        self.masks: Dict[str, torch.Tensor] = {}
        self.weights: Dict[str, nn.Parameter] = {}
        self.initial_counts: Dict[str, int] = {}
        self.layer_info: Dict[str, Dict] = {}
        
        # Counters
        self.step_count = 0
        self.prune_count = 0
        self.growth_count = 0
        
        # Graph fusion neuroplasticity (if model has graph_fusion)
        self.graph_fusion = None
        self._init_graph_fusion()
        
        # Initialize
        self._init_layers()
        self._register_hooks()
        self._wrap_optimizer()
        
        # ============ 全局神经元图（统一prune/grow） ============
        self.global_graph = GlobalNeuronGraph(sw_beta=2.0, sf_gamma=1.0)
        self.global_graph.build_from_model(self.model, self.config['target_patterns'])
        # 共享mask引用
        for name in self.masks:
            self.global_graph.masks[name] = self.masks[name]
            self.global_graph.weights[name] = self.weights[name]
        
    def _init_graph_fusion(self):
        """Initialize graph fusion neuroplasticity if model has graph_fusion module."""
        # Search for graph_fusion in model
        for name, module in self.model.named_modules():
            if hasattr(module, 'graph_fusion') and module.graph_fusion is not None:
                self.graph_fusion = module.graph_fusion
                break
        
        if self.graph_fusion is not None:
            stats = self.graph_fusion.get_graph_stats()
            print(f"[Neuroplastic] Graph fusion enabled: {stats['n_nodes']} nodes, {stats['n_edges']:.0f} edges")
            print(f"  Cross-modal: text↔visual={stats['text_visual_edges']}, "
                  f"text↔acoustic={stats['text_acoustic_edges']}, "
                  f"visual↔acoustic={stats['visual_acoustic_edges']}")
        else:
            print(f"[Neuroplastic] Graph fusion: not found or disabled")
    
    def _should_track(self, name: str, module: nn.Module) -> bool:
        """Check if layer should be tracked"""
        if not isinstance(module, nn.Linear):
            return False
        for pattern in self.config['target_patterns']:
            if pattern.lower() in name.lower():
                return True
        return False
    
    def _init_layers(self):
        """Initialize masks, tracking, and NeuronLevelTopology for target layers"""
        # 神经元级拓扑管理器
        self.topologies: Dict[str, NeuronLevelTopology] = {}
        
        for name, module in self.model.named_modules():
            if not self._should_track(name, module):
                continue
                
            weight = module.weight
            out_features, in_features = weight.shape
            
            # Start with full connectivity (all True)
            mask = torch.ones(out_features, in_features, 
                            dtype=torch.bool, device=weight.device)
            
            self.masks[name] = mask
            self.weights[name] = weight
            self.initial_counts[name] = mask.sum().item()
            
            self.layer_info[name] = {
                'out_features': out_features,
                'in_features': in_features,
                'initial_connections': self.initial_counts[name],
            }
            
            # 创建 NeuronLevelTopology 实例
            self.topologies[name] = NeuronLevelTopology(
                name=name,
                weight=weight,
                mask=mask,
                sw_beta=2.0,       # Small-World距离衰减
                sf_alpha=0.5,      # SW与SF混合比例
                ema_alpha=self.config['ema_alpha'],
                min_density=0.01,
                max_density=self.config['max_density'],
                protect_period=5,
            )
            
        print(f"[Neuroplastic] Tracking {len(self.masks)} layers with NeuronLevelTopology")
        for name, info in self.layer_info.items():
            print(f"  - {name}: {info['out_features']}x{info['in_features']} "
                  f"= {info['initial_connections']} connections")
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        for name, module in self.model.named_modules():
            if name not in self.masks:
                continue
                
            # Forward hook
            fwd_hook = ForwardHook(name, self)
            module.register_forward_hook(fwd_hook)
            
            # Backward hook on weight
            bwd_hook = BackwardHook(name, self)
            module.weight.register_hook(bwd_hook)
            
    def _wrap_optimizer(self):
        """Wrap optimizer step to apply masks"""
        original_step = self.optimizer.step
        scheduler = self
        
        def wrapped_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            scheduler._apply_masks()
            return result
            
        self.optimizer.step = wrapped_step
        
    def _apply_masks(self):
        """Apply masks to weights"""
        for name, mask in self.masks.items():
            self.weights[name].data *= mask.float()
    
    def reset_momentum(self):
        """Reset optimizer momentum for pruned/grown connections (RigL-style)."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # Find which weight this parameter corresponds to
                param_name = None
                for name, weight in self.weights.items():
                    if p is weight:
                        param_name = name
                        break
                if param_name is None:
                    continue
                    
                mask = self.masks[param_name]
                state = self.optimizer.state.get(p, {})
                
                # Reset momentum buffers for masked positions
                if 'momentum_buffer' in state:  # SGD
                    state['momentum_buffer'] *= mask.float()
                if 'exp_avg' in state:  # Adam first moment
                    state['exp_avg'] *= mask.float()
                if 'exp_avg_sq' in state:  # Adam second moment
                    state['exp_avg_sq'] *= mask.float()
    
    def apply_mask_to_gradients(self):
        """Ensure gradients are zero for pruned connections (RigL-style)."""
        for name, mask in self.masks.items():
            weight = self.weights[name]
            if weight.grad is not None:
                weight.grad *= mask.float()
            
    def _get_modification_strength(self) -> float:
        """Get current modification strength (decays over training)"""
        progress = self.step_count / self.total_steps
        return max(0.01, 1.0 - progress)
    
    @torch.no_grad()
    def _prune_inactive(self):
        """Prune inactive connections - VECTORIZED for speed"""
        strength = self._get_modification_strength()
        threshold = self.config['prune_threshold']
        min_fan_in = self.config['min_fan_in']
        min_fan_out = self.config['min_fan_out']
        max_ratio = self.config['max_prune_ratio'] * strength
        
        total_pruned = 0
        
        for name in self.masks:
            mask = self.masks[name]
            weight = self.weights[name]
            
            if name not in self.activity_tracker.activity:
                continue
            
            activity = self.activity_tracker.activity[name]
            
            # Vectorized: compute fan_in per row and fan_out per column
            fan_in_per_row = mask.sum(dim=1)  # (out_features,)
            fan_out_per_col = mask.sum(dim=0)  # (in_features,)
            
            # Create protection mask: can only prune if fan_in > min AND fan_out > min
            row_ok = (fan_in_per_row > min_fan_in).unsqueeze(1)  # (out, 1)
            col_ok = (fan_out_per_col > min_fan_out).unsqueeze(0)  # (1, in)
            can_prune = row_ok & col_ok  # (out, in)
            
            # Get inactive connections that can be pruned
            inactive = (activity < threshold) & mask & can_prune
            
            n_inactive = inactive.sum().item()
            if n_inactive == 0:
                continue
            
            # Limit pruning amount
            n_active = mask.sum().item()
            max_prune = int(n_active * max_ratio)
            
            if n_inactive <= max_prune:
                # Prune all inactive
                prune_mask = inactive
            else:
                # Prune lowest activity ones up to max_prune
                activity_masked = torch.where(inactive, activity, 
                                             torch.tensor(float('inf'), device=activity.device))
                flat = activity_masked.flatten()
                _, indices = torch.topk(flat, max_prune, largest=False)
                prune_mask = torch.zeros_like(mask)
                prune_mask.view(-1)[indices] = True
            
            # Apply pruning
            n_pruned = prune_mask.sum().item()
            mask[prune_mask] = False
            weight.data[prune_mask] = 0
            total_pruned += n_pruned
            
        if total_pruned > 0:
            self.prune_count += 1
            print(f"[Neuroplastic] Step {self.step_count}: Pruned {total_pruned} connections")
            
    @torch.no_grad()
    def _grow_hebbian(self):
        """Grow new connections based on co-activation - VECTORIZED for speed"""
        strength = self._get_modification_strength()
        max_density = self.config['max_density']
        growth_ratio = self.config['growth_ratio'] * strength
        
        total_grown = 0
        
        for name in self.masks:
            mask = self.masks[name]
            weight = self.weights[name]
            
            # Check density limit
            current = mask.sum().item()
            initial = self.initial_counts[name]
            if current >= initial * max_density:
                continue
                
            # How many to grow
            n_grow = min(
                int(current * growth_ratio),
                int(initial * max_density - current)
            )
            
            if n_grow <= 0:
                continue
                
            # Get growth candidates (flat indices tensor)
            indices = self.hebbian_tracker.get_growth_candidates(name, mask, n_grow)
            
            if len(indices) == 0:
                continue
            
            # Vectorized growth
            n_cols = mask.shape[1]
            rows = indices // n_cols
            cols = indices % n_cols
            
            # Create growth mask
            grow_mask = torch.zeros_like(mask)
            grow_mask.view(-1)[indices] = True
            
            # Only grow where not already connected
            grow_mask = grow_mask & (~mask)
            n_grown = grow_mask.sum().item()
            
            if n_grown > 0:
                # Apply mask updates
                mask[grow_mask] = True
                
                # Initialize new weights with small random values
                # Use average fan_in for initialization scale
                avg_fan_in = mask.sum(dim=1).float().mean().item()
                std = 0.1 / math.sqrt(max(avg_fan_in, 1))
                weight.data[grow_mask] = torch.randn(n_grown, device=weight.device) * std
                
                total_grown += n_grown
                    
        if total_grown > 0:
            self.growth_count += 1
            print(f"[Neuroplastic] Step {self.step_count}: Grew {total_grown} connections")
            
        # Reset Hebbian tracker
        self.hebbian_tracker.reset()
    
    # NOTE: _manage_cross_layer_skip函数已删除
    # 统一框架只保留神经元级操作：
    # - weight prune/grow: 通过NeuronLevelTopology处理
    # - 不再有跨层skip connection
        
    def step(self, loss: float = None, ib_loss: float = None):
        """Call after each optimizer step.
        
        新策略：基于稳定性的自适应拓扑更新
        1. 先让权重训练到稳定
        2. 稳定后进行拓扑更新（net-zero: prune K, grow K）
        3. 评估效果，自适应调整下次更新幅度
        """
        self.step_count += 1
        
        # ==================== Unified Activity Tracking ====================
        # Update graph edge activity from gradients (before they get cleared)
        # This enables gradient-based prune/grow for graph edges, same as weights
        if self.graph_fusion is not None and hasattr(self.graph_fusion, 'update_edge_activity'):
            self.graph_fusion.update_edge_activity()
        
        # Update stability guard
        if loss is not None:
            self.stability_guard.update(loss, ib_loss)
        
        # ============ 连续策略 Reward 反馈 ============
        # Prune策略reward
        if (self.pending_prune_reward_step is not None and 
            self.step_count >= self.pending_prune_reward_step and
            len(self.stability_guard.loss_history) >= 20):
            post_loss = np.mean(self.stability_guard.loss_history[-20:])
            sparsity_penalty = 0.0
            self.prune_policy.end_update(post_loss, sparsity_penalty)
            stats = self.prune_policy.get_stats()
            print(f"[RL-Prune] μ={stats['mean']:.4f}, σ={stats['std']:.4f}, "
                  f"reward={stats['recent_avg_reward']:.4f}")
            self.pending_prune_reward_step = None
        
        # Grow策略reward
        if (self.pending_grow_reward_step is not None and 
            self.step_count >= self.pending_grow_reward_step and
            len(self.stability_guard.loss_history) >= 20):
            post_loss = np.mean(self.stability_guard.loss_history[-20:])
            sparsity_penalty = 0.0
            self.grow_policy.end_update(post_loss, sparsity_penalty)
            stats = self.grow_policy.get_stats()
            print(f"[RL-Grow] μ={stats['mean']:.4f}, σ={stats['std']:.4f}, "
                  f"reward={stats['recent_avg_reward']:.4f}")
            self.pending_grow_reward_step = None
        
        # 图边策略reward
        if (self.pending_graph_reward_step is not None and 
            self.step_count >= self.pending_graph_reward_step and
            len(self.stability_guard.loss_history) >= 20):
            post_loss = np.mean(self.stability_guard.loss_history[-20:])
            self.graph_policy.end_update(post_loss, sparsity_penalty=0.0)
            stats = self.graph_policy.get_stats()
            print(f"[RL-Graph] μ={stats['mean']:.4f}, σ={stats['std']:.4f}, "
                  f"reward={stats['recent_avg_reward']:.4f}")
            self.pending_graph_reward_step = None
        
        # Warmup: skip topology changes during initial training
        warmup_steps = self.config.get('warmup_steps', 500)
        if self.step_count <= warmup_steps:
            if self.step_count == warmup_steps:
                print(f"[Neuroplastic] Warmup complete at step {warmup_steps}")
                print(f"[Neuroplastic] Mode: Continuous RL policies (weight + graph independent)")
            return
        
        # 每100步打印稳定性状态
        if self.step_count % 100 == 0:
            sg = self.stability_guard
            cv = np.std(sg.loss_history[-50:]) / (np.mean(sg.loss_history[-50:]) + 1e-8) if len(sg.loss_history) >= 50 else 999
            print(f"[StabilityCheck] step={self.step_count}, stable={sg.is_stable}, "
                  f"consec={sg.consecutive_stable_steps}/{sg.min_stable_steps}, "
                  f"since_update={sg.steps_since_topology_update}/{sg.min_steps_between_updates}, cv={cv:.3f}")
        
        # 新策略：基于稳定性触发，而非固定间隔
        if self.stability_guard.should_update_topology():
            self._do_adaptive_topology_update()
            
    def _do_adaptive_topology_update(self):
        """Perform adaptive topology update using independent RL policies.
        
        使用三个完全独立的连续策略:
        - prune_policy: 学习权重连接的最优prune scale
        - grow_policy: 学习权重连接的最优grow scale (不再与prune绑定)
        - graph_policy: 学习图边的最优prune/grow scale
        
        比离散UCB更精细：scale ~ N(μ, σ²)，REINFORCE更新
        """
        # Get current loss for RL reward calculation
        current_loss = np.mean(self.stability_guard.loss_history[-20:]) if len(self.stability_guard.loss_history) >= 20 else 0
        
        # ============ 独立采样三个策略的scale ============
        prune_scale = self.prune_policy.sample_scale()
        grow_scale = self.grow_policy.sample_scale()
        graph_scale = self.graph_policy.sample_scale()
        
        # 记录更新前的loss
        self.prune_policy.begin_update(current_loss)
        self.grow_policy.begin_update(current_loss)
        self.graph_policy.begin_update(current_loss)
        self.stability_guard.begin_topology_update()
        
        # 调度reward收集
        self.pending_prune_reward_step = self.step_count + self.reward_delay
        self.pending_grow_reward_step = self.step_count + self.reward_delay
        self.pending_graph_reward_step = self.step_count + self.reward_delay
        
        # Print policy stats
        prune_stats = self.prune_policy.get_stats()
        grow_stats = self.grow_policy.get_stats()
        graph_stats = self.graph_policy.get_stats()
        print(f"\n[Neuroplastic] Step {self.step_count}: Topology update (independent prune/grow)")
        print(f"  Prune Policy: μ={prune_stats['mean']:.4f}, σ={prune_stats['std']:.4f} -> scale={prune_scale:.4f}")
        print(f"  Grow Policy:  μ={grow_stats['mean']:.4f}, σ={grow_stats['std']:.4f} -> scale={grow_scale:.4f}")
        print(f"  Graph Policy: μ={graph_stats['mean']:.4f}, σ={graph_stats['std']:.4f} -> scale={graph_scale:.4f}")
        
        # ============ 全局统一 prune/grow ============
        # 收集每层的importance和co_activation
        importance_dict = {}
        co_activation_dict = {}
        
        for name, topology in self.topologies.items():
            if topology.importance is not None:
                importance_dict[name] = topology.importance
            else:
                # Fallback: 用权重绝对值
                importance_dict[name] = topology.weight.abs()
            
            if topology.co_activation is not None:
                co_activation_dict[name] = topology.co_activation
            else:
                # Fallback: uniform
                co_activation_dict[name] = torch.ones_like(topology.weight)
            
            # 增加connection age（保护新连接）
            topology.step_age()
        
        # 全局图统一prune/grow
        topo_result = self.global_graph.topology_step(
            importance_dict=importance_dict,
            co_activation_dict=co_activation_dict,
            prune_scale=prune_scale * self.config['max_prune_ratio'],
            grow_scale=grow_scale * self.config['growth_ratio'],
        )
        
        actual_pruned = topo_result['pruned']
        actual_grown = topo_result['grown']
        
        print(f"  Result: pruned {actual_pruned}, grew {actual_grown} (GlobalNeuronGraph unified)")
        print(f"  Active edges: {topo_result['active_edges']}, density: {topo_result['density']:.4f}")
        
        # Update counters
        if actual_pruned > 0:
            self.prune_count += 1
        if actual_grown > 0:
            self.growth_count += 1
        
        # TODO: 已删除skip connection逻辑（只保留神经元级操作）
        # Cross-layer skip updates removed - neuron-level only
        
        # Graph fusion neuroplastic update (uses independent graph_policy)
        if self.graph_fusion is not None:
            # 使用独立的graph_policy scale（不再是权重scale的系数）
            edge_stats = self.graph_fusion.neuroplastic_step(
                prune_ratio=graph_scale,   # 直接使用图策略学到的scale
                grow_ratio=graph_scale * 0.9  # 稍微保守的growth（避免边数膨胀）
            )
            activity_info = "activity-based" if edge_stats.get('has_activity', False) else "fallback"
            print(f"  Graph edges: pruned {edge_stats['n_pruned']}, grew {edge_stats['n_grown']}, "
                  f"total={edge_stats['n_edges']:.0f}, cross-modal={edge_stats['cross_modal_edges']} "
                  f"(mode: {activity_info}, scale={graph_scale:.4f})")
        
        # RigL-style cleanup: reset momentum and apply gradient masking
        self.reset_momentum()
        self.apply_mask_to_gradients()
        
        # Reset trackers for next cycle
        self.hebbian_tracker.reset()
        for topology in self.topologies.values():
            topology.reset_activations()
        self.stability_guard.end_topology_update(actual_pruned, actual_grown)
        
        # Evaluate effect of previous update
        adjustment = self.stability_guard.evaluate_last_update()
        if adjustment != 1.0:
            print(f"  Adaptive: next update scale adjusted to {self.stability_guard.update_budget_multiplier:.2f}")
            
    def get_stats(self) -> Dict:
        """Get current statistics"""
        stats = {
            'step': self.step_count,
            'prune_count': self.prune_count,
            'growth_count': self.growth_count,
            'layers': {},
            'cross_layer_skip': {},
        }
        
        total_current = 0
        total_initial = 0
        
        for name, mask in self.masks.items():
            current = mask.sum().item()
            initial = self.initial_counts[name]
            stats['layers'][name] = {
                'current': current,
                'initial': initial,
                'ratio': current / initial if initial > 0 else 1.0,
            }
            total_current += current
            total_initial += initial
            
        stats['total'] = {
            'current': total_current,
            'initial': total_initial,
            'ratio': total_current / total_initial if total_initial > 0 else 1.0,
        }
        
        # Add cross-layer skip statistics
        if hasattr(self.model, 'get_all_neuroplastic_blocks'):
            try:
                blocks = self.model.get_all_neuroplastic_blocks()
                total_skip_active = 0
                total_skip_possible = 0
                for i, block in enumerate(blocks):
                    if hasattr(block, 'skip_mask'):
                        active = block.skip_mask.sum().item()
                        possible = block.skip_mask.numel()
                        total_skip_active += active
                        total_skip_possible += possible
                        stats['cross_layer_skip'][f'block_{i}'] = {
                            'active': int(active),
                            'possible': possible,
                            'ratio': active / possible if possible > 0 else 0,
                        }
                stats['cross_layer_skip']['total'] = {
                    'active': total_skip_active,
                    'possible': total_skip_possible,
                    'ratio': total_skip_active / total_skip_possible if total_skip_possible > 0 else 0,
                }
            except Exception:
                pass
        
        # Add graph fusion statistics
        if self.graph_fusion is not None:
            try:
                graph_stats = self.graph_fusion.get_graph_stats()
                stats['graph_fusion'] = {
                    'n_nodes': graph_stats['n_nodes'],
                    'n_edges': graph_stats['n_edges'],
                    'edge_density': graph_stats['edge_density'],
                    'text_visual_edges': graph_stats['text_visual_edges'],
                    'text_acoustic_edges': graph_stats['text_acoustic_edges'],
                    'visual_acoustic_edges': graph_stats['visual_acoustic_edges'],
                    'cross_modal_total': (graph_stats['text_visual_edges'] + 
                                          graph_stats['text_acoustic_edges'] + 
                                          graph_stats['visual_acoustic_edges']),
                }
            except Exception:
                pass
        
        return stats
    
    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        print(f"\n[Neuroplastic] Statistics at step {stats['step']}:")
        print(f"  Prune operations: {stats['prune_count']}")
        print(f"  Growth operations: {stats['growth_count']}")
        print(f"  Total connections: {stats['total']['current']}/{stats['total']['initial']} "
              f"({stats['total']['ratio']:.1%})")
        for name, info in stats['layers'].items():
            print(f"    {name}: {info['current']}/{info['initial']} ({info['ratio']:.1%})")
        
        # Print cross-layer skip stats
        if 'cross_layer_skip' in stats and stats['cross_layer_skip']:
            skip_total = stats['cross_layer_skip'].get('total', {})
            if skip_total:
                print(f"  Cross-layer skip: {skip_total.get('active', 0)}/{skip_total.get('possible', 0)} "
                      f"({skip_total.get('ratio', 0):.2%})")
        
        # Print graph fusion stats
        if 'graph_fusion' in stats:
            gf = stats['graph_fusion']
            print(f"  Graph fusion: {gf['n_edges']:.0f} edges, density={gf['edge_density']:.3f}")
            print(f"    Cross-modal: text↔visual={gf['text_visual_edges']}, "
                  f"text↔acoustic={gf['text_acoustic_edges']}, "
                  f"visual↔acoustic={gf['visual_acoustic_edges']}")
