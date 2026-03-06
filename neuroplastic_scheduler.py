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
                               top_k: int) -> List[Tuple[int, int]]:
        """Find connections to grow based on co-activation"""
        if layer_name not in self.co_activation:
            return []
            
        # Average co-activation
        co_act = self.co_activation[layer_name] / self.sample_count[layer_name]
        
        # Only consider positions without connections
        co_act_masked = co_act * (~mask).float()
        
        # Find top-k positions
        flat = co_act_masked.flatten()
        k = min(top_k, (flat > 0).sum().item())
        if k == 0:
            return []
            
        _, indices = torch.topk(flat, k)
        
        # Convert to (row, col) pairs
        n_cols = mask.shape[1]
        candidates = [(idx.item() // n_cols, idx.item() % n_cols) for idx in indices]
        return candidates
    
    def reset(self):
        """Reset statistics"""
        self.co_activation.clear()
        self.sample_count.clear()


class StabilityGuard:
    """Monitor training stability and prevent degradation"""
    
    def __init__(self, patience: int = 5, variance_threshold: float = 0.3):
        self.loss_history = []
        self.ib_loss_history = []
        self.patience = patience
        self.variance_threshold = variance_threshold
        self.topology_frozen = False
        self.freeze_reason = None
        
    def update(self, loss: float, ib_loss: float = None):
        """Update with current loss values"""
        self.loss_history.append(loss)
        if ib_loss is not None:
            self.ib_loss_history.append(ib_loss)
        
        # Check stability
        if len(self.loss_history) >= self.patience:
            recent = self.loss_history[-self.patience:]
            mean_loss = np.mean(recent)
            std_loss = np.std(recent)
            
            # Freeze if loss is oscillating too much
            if mean_loss > 0 and std_loss / mean_loss > self.variance_threshold:
                self.topology_frozen = True
                self.freeze_reason = f"Loss variance too high: {std_loss/mean_loss:.2%}"
            else:
                self.topology_frozen = False
                self.freeze_reason = None
                
    def check_ib_health(self) -> bool:
        """Check if information bottleneck is healthy"""
        if len(self.ib_loss_history) < 10:
            return True
            
        recent = np.mean(self.ib_loss_history[-5:])
        baseline = np.mean(self.ib_loss_history[:5])
        
        # IB loss increased too much
        if baseline > 0 and recent > baseline * 1.5:
            return False
        return True
    
    def can_modify_topology(self) -> bool:
        """Check if it's safe to modify topology"""
        if self.topology_frozen:
            return False
        if not self.check_ib_health():
            return False
        return True


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
            
            # Pruning
            'prune_interval': 200,
            'prune_threshold': 0.001,
            'min_fan_in': 3,
            'max_prune_ratio': 0.1,
            
            # Growth
            'growth_interval': 200,
            'max_density': 1.5,
            'growth_ratio': 0.05,
            
            # Stability
            'stability_patience': 5,
            'variance_threshold': 0.3,
            
            # Layer patterns
            'target_patterns': [
                'ithp.encoder1',
                'ithp.encoder2', 
                'ithp.MLP1',
                'ithp.MLP2',
                'expand.0',
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
        """Prune inactive connections"""
        strength = self._get_modification_strength()
        threshold = self.config['prune_threshold']
        min_fan_in = self.config['min_fan_in']
        max_ratio = self.config['max_prune_ratio'] * strength
        
        total_pruned = 0
        
        for name in self.masks:
            mask = self.masks[name]
            weight = self.weights[name]
            
            # Get inactive connections
            inactive = self.activity_tracker.get_inactive_connections(
                name, mask, threshold
            )
            
            # Limit pruning amount
            n_active = mask.sum().item()
            max_prune = int(n_active * max_ratio)
            
            # Sort by activity (prune lowest first)
            if name in self.activity_tracker.activity:
                activity = self.activity_tracker.activity[name]
                activity_masked = torch.where(inactive, activity, 
                                             torch.tensor(float('inf'), device=activity.device))
                flat = activity_masked.flatten()
                _, indices = torch.topk(flat, min(max_prune, inactive.sum().item()), largest=False)
            else:
                continue
                
            # Apply pruning with min_fan_in protection
            n_pruned = 0
            for idx in indices:
                i = idx.item() // mask.shape[1]
                j = idx.item() % mask.shape[1]
                
                # Check min fan-in
                if mask[i].sum() <= min_fan_in:
                    continue
                    
                mask[i, j] = False
                weight.data[i, j] = 0
                n_pruned += 1
                
            total_pruned += n_pruned
            
        if total_pruned > 0:
            self.prune_count += 1
            print(f"[Neuroplastic] Step {self.step_count}: Pruned {total_pruned} connections")
            
    @torch.no_grad()
    def _grow_hebbian(self):
        """Grow new connections based on co-activation"""
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
                
            # Get growth candidates
            candidates = self.hebbian_tracker.get_growth_candidates(name, mask, n_grow)
            
            # Apply growth
            for i, j in candidates:
                if not mask[i, j]:
                    mask[i, j] = True
                    # Initialize with small random value
                    fan_in = mask[i].sum().item()
                    std = 1.0 / math.sqrt(fan_in)
                    weight.data[i, j] = std * torch.randn(1, device=weight.device).item() * 0.1
                    total_grown += 1
                    
        if total_grown > 0:
            self.growth_count += 1
            print(f"[Neuroplastic] Step {self.step_count}: Grew {total_grown} connections")
            
        # Reset Hebbian tracker
        self.hebbian_tracker.reset()
        
    def step(self, loss: float = None, ib_loss: float = None):
        """Call after each optimizer step"""
        self.step_count += 1
        
        # Update stability guard
        if loss is not None:
            self.stability_guard.update(loss, ib_loss)
        
        # Check if safe to modify
        if not self.stability_guard.can_modify_topology():
            if self.stability_guard.freeze_reason:
                if self.step_count % 100 == 0:
                    print(f"[Neuroplastic] Topology frozen: {self.stability_guard.freeze_reason}")
            return
            
        # Prune at intervals
        if self.step_count % self.config['prune_interval'] == 0:
            self._prune_inactive()
            
        # Grow at intervals
        if self.step_count % self.config['growth_interval'] == 0:
            self._grow_hebbian()
            
    def get_stats(self) -> Dict:
        """Get current statistics"""
        stats = {
            'step': self.step_count,
            'prune_count': self.prune_count,
            'growth_count': self.growth_count,
            'layers': {},
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
