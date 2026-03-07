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
    """
    
    def __init__(self, patience: int = 50, variance_threshold: float = 0.05):
        # Loss tracking
        self.loss_history = []
        self.ib_loss_history = []
        self.patience = patience
        self.variance_threshold = variance_threshold
        
        # Stability state
        self.is_stable = False
        self.steps_since_topology_update = 0
        self.min_steps_between_updates = 200  # 不能太频繁
        
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
                
    def should_update_topology(self) -> bool:
        """Check if it's time to update topology.
        
        条件：
        1. 权重已稳定 (loss converged)
        2. 距离上次更新足够久
        3. IB loss 健康
        """
        if not self.is_stable:
            return False
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
            # Flatten if needed (handle sequence dimension)
            if input_act.dim() == 3:
                input_act = input_act.mean(dim=1)  # (batch, seq, dim) -> (batch, dim)
            
            self.scheduler.activity_tracker.cache_input(self.layer_name, input_act)
            
            output_act = output
            if output_act.dim() == 3:
                output_act = output_act.mean(dim=1)
                
            self.scheduler.hebbian_tracker.record(self.layer_name, input_act, output_act)


class BackwardHook:
    """Hook to capture gradients and update activity"""
    
    def __init__(self, layer_name: str, scheduler: 'NeuroplasticScheduler'):
        self.layer_name = layer_name
        self.scheduler = scheduler
        
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        weight = self.scheduler.weights[self.layer_name]
        self.scheduler.activity_tracker.update(self.layer_name, weight, grad)
        
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
            
            # Stability
            'stability_patience': 5,
            'variance_threshold': 0.3,
            
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
            self.config['stability_patience'],
            self.config['variance_threshold']
        )
        
        # RL-based adaptive policy for update magnitude
        self.update_policy = AdaptiveUpdatePolicy(
            scales=[0.02, 0.05, 0.08, 0.1]
        )
        self.pending_ucb_reward_step = None  # Step at which to provide UCB reward
        self.ucb_reward_delay = 100  # Wait this many steps to evaluate effect
        
        # Storage
        self.masks: Dict[str, torch.Tensor] = {}
        self.weights: Dict[str, nn.Parameter] = {}
        self.initial_counts: Dict[str, int] = {}
        self.layer_info: Dict[str, Dict] = {}
        
        # Counters
        self.step_count = 0
        self.prune_count = 0
        self.growth_count = 0
        
        # Initialize
        self._init_layers()
        self._register_hooks()
        self._wrap_optimizer()
        
    def _should_track(self, name: str, module: nn.Module) -> bool:
        """Check if layer should be tracked"""
        if not isinstance(module, nn.Linear):
            return False
        for pattern in self.config['target_patterns']:
            if pattern.lower() in name.lower():
                return True
        return False
    
    def _init_layers(self):
        """Initialize masks and tracking for target layers"""
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
            
        print(f"[Neuroplastic] Tracking {len(self.masks)} layers")
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
    
    @torch.no_grad()
    def _manage_cross_layer_skip(self, grow: bool = False, prune: bool = False):
        """Manage cross-layer skip connections in NeuroplasticBlocks.
        
        These are the within-block input->output skip connections that 
        bypass the hidden layer, providing true synaptic-level plasticity.
        """
        # Check if model has neuroplastic blocks
        if not hasattr(self.model, 'get_all_neuroplastic_blocks'):
            return
        
        strength = self._get_modification_strength()
        
        if grow:
            # Grow skip connections based on co-activation
            growth_count = max(1, int(5 * strength))  # 5 -> 1 as training progresses
            try:
                self.model.grow_skip_connections(growth_count)
            except Exception as e:
                print(f"[Neuroplastic] Skip growth error: {e}")
        
        if prune:
            # Prune weak skip connections
            prune_ratio = 0.1 * strength  # 10% -> 1% as training progresses
            try:
                self.model.prune_skip_connections(prune_ratio)
            except Exception as e:
                print(f"[Neuroplastic] Skip prune error: {e}")
        
    def step(self, loss: float = None, ib_loss: float = None):
        """Call after each optimizer step.
        
        新策略：基于稳定性的自适应拓扑更新
        1. 先让权重训练到稳定
        2. 稳定后进行拓扑更新（net-zero: prune K, grow K）
        3. 评估效果，自适应调整下次更新幅度
        """
        self.step_count += 1
        
        # Update stability guard
        if loss is not None:
            self.stability_guard.update(loss, ib_loss)
        
        # UCB reward feedback: evaluate effect of previous topology update
        if (self.pending_ucb_reward_step is not None and 
            self.step_count >= self.pending_ucb_reward_step and
            len(self.stability_guard.loss_history) >= 20):
            post_loss = np.mean(self.stability_guard.loss_history[-20:])
            self.update_policy.end_update(post_loss)
            arm = self.update_policy.current_arm
            scale = self.update_policy.scales[arm]
            avg_reward = self.update_policy.rewards[arm] / max(1, self.update_policy.counts[arm])
            print(f"[Neuroplastic] UCB reward for scale={scale:.2f}: avg_reward={avg_reward:.4f}")
            self.pending_ucb_reward_step = None
        
        # Warmup: skip topology changes during initial training
        warmup_steps = self.config.get('warmup_steps', 500)
        if self.step_count <= warmup_steps:
            if self.step_count == warmup_steps:
                print(f"[Neuroplastic] Warmup complete at step {warmup_steps}")
                print(f"[Neuroplastic] Mode: Stability-based adaptive updates (net-zero rewiring)")
            return
        
        # 新策略：基于稳定性触发，而非固定间隔
        if self.stability_guard.should_update_topology():
            self._do_adaptive_topology_update()
            # Schedule UCB reward collection
            self.pending_ucb_reward_step = self.step_count + self.ucb_reward_delay
            
    def _do_adaptive_topology_update(self):
        """Perform adaptive topology update (net-zero rewiring).
        
        Uses UCB1 (reinforcement learning) to select update magnitude.
        """
        # Get current loss for RL reward calculation
        current_loss = np.mean(self.stability_guard.loss_history[-20:]) if len(self.stability_guard.loss_history) >= 20 else 0
        
        # RL: Select scale using UCB1 policy
        scale = self.update_policy.select_scale()
        self.update_policy.begin_update(current_loss)
        self.stability_guard.begin_topology_update()
        
        # Print policy stats periodically
        policy_stats = self.update_policy.get_stats()
        print(f"\n[Neuroplastic] Step {self.step_count}: Topology update (UCB scale={scale:.2f})")
        print(f"  UCB policy: {policy_stats}")
        
        # Phase 1: Calculate how many to prune
        total_to_prune = 0
        prune_candidates = {}  # layer_name -> (indices, activities)
        
        for name in self.masks:
            mask = self.masks[name]
            if name not in self.activity_tracker.activity:
                continue
            
            activity = self.activity_tracker.activity[name]
            n_active = mask.sum().item()
            
            # Calculate pruning amount with adaptive scale
            base_prune_ratio = self.config['max_prune_ratio'] * scale
            n_prune = int(n_active * base_prune_ratio)
            
            if n_prune > 0:
                # Get lowest activity connections
                threshold = self.config['prune_threshold']
                fan_in_per_row = mask.sum(dim=1)
                fan_out_per_col = mask.sum(dim=0)
                
                row_ok = (fan_in_per_row > self.config['min_fan_in']).unsqueeze(1)
                col_ok = (fan_out_per_col > self.config['min_fan_out']).unsqueeze(0)
                can_prune = row_ok & col_ok & mask
                
                activity_masked = torch.where(can_prune, activity, 
                                             torch.tensor(float('inf'), device=activity.device))
                flat = activity_masked.flatten()
                _, indices = torch.topk(flat, min(n_prune, (flat < float('inf')).sum().item()), largest=False)
                
                prune_candidates[name] = indices
                total_to_prune += len(indices)
        
        # Phase 2: Grow exactly the same amount (net-zero!)
        total_to_grow = total_to_prune
        
        print(f"  Target: prune {total_to_prune}, grow {total_to_grow} (net-zero)")
        
        # Phase 3: Execute pruning
        actual_pruned = 0
        for name, indices in prune_candidates.items():
            mask = self.masks[name]
            weight = self.weights[name]
            
            prune_mask = torch.zeros_like(mask)
            prune_mask.view(-1)[indices] = True
            
            n_pruned = prune_mask.sum().item()
            mask[prune_mask] = False
            weight.data[prune_mask] = 0
            actual_pruned += n_pruned
        
        # Phase 4: Execute growth
        actual_grown = 0
        remaining_to_grow = total_to_grow
        
        for name in self.masks:
            if remaining_to_grow <= 0:
                break
                
            mask = self.masks[name]
            weight = self.weights[name]
            initial = self.initial_counts[name]
            current = mask.sum().item()
            
            # Allow growing up to max_density
            max_allowed = int(initial * self.config['max_density'])
            space = max_allowed - current
            
            if space <= 0:
                continue
            
            # Proportional allocation
            layer_grow = min(space, remaining_to_grow * current // max(actual_pruned, 1))
            layer_grow = max(1, layer_grow)  # At least 1
            
            indices = self.hebbian_tracker.get_growth_candidates(name, mask, layer_grow)
            if len(indices) == 0:
                continue
            
            grow_mask = torch.zeros_like(mask)
            grow_mask.view(-1)[indices] = True
            grow_mask = grow_mask & (~mask)
            
            n_grown = grow_mask.sum().item()
            if n_grown > 0:
                mask[grow_mask] = True
                avg_fan_in = mask.sum(dim=1).float().mean().item()
                std = 0.1 / math.sqrt(max(avg_fan_in, 1))
                weight.data[grow_mask] = torch.randn(n_grown, device=weight.device) * std
                actual_grown += n_grown
                remaining_to_grow -= n_grown
        
        # Update counters
        if actual_pruned > 0:
            self.prune_count += 1
        if actual_grown > 0:
            self.growth_count += 1
            
        print(f"  Result: pruned {actual_pruned}, grew {actual_grown}")
        
        # Cross-layer skip updates
        self._manage_cross_layer_skip(prune=True, grow=True)
        
        # Reset trackers for next cycle
        self.hebbian_tracker.reset()
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
