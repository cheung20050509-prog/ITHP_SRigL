"""
SRigL Scheduler for DeBERTa/Transformer models
Based on: https://github.com/calgaryml/condensed-sparsity

Implements Dynamic Sparse Training with constant fan-in structure.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import math


class IndexMaskHook:
    """Backward hook for accumulating dense gradients and masking sparse gradients.
    
    Reference: condensed-sparsity/src/rigl_torch/rigl_scheduler.py L21-57
    """
    
    def __init__(self, layer_idx: int, scheduler: 'DeBertaRigLScheduler'):
        self.layer_idx = layer_idx
        self.scheduler = scheduler
        self.dense_grad = None
        
    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        """Called during backward pass."""
        # Accumulate dense gradient for topology decisions
        if self.dense_grad is None:
            self.dense_grad = torch.zeros_like(grad)
        self.dense_grad += grad / self.scheduler.grad_accumulation_n
        
        # Return masked gradient (only update active weights)
        mask = self.scheduler.masks[self.layer_idx]
        return grad * mask.float()
    
    def reset(self):
        """Reset accumulated gradients after topology update."""
        self.dense_grad = None


class DeBertaRigLScheduler:
    """RigL scheduler adapted for DeBERTa Transformer architecture.
    
    Supports sparse training of Linear layers in:
    - ITHP encoder and MLP layers
    - Optionally DeBERTa FFN layers
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        dense_allocation: Fraction of weights to keep (1 - sparsity)
        delta: Steps between topology updates
        alpha: Initial fraction of weights to update each step
        T_end: Step to stop topology updates (default: 75% of training)
        sparsify_patterns: List of module name patterns to sparsify
        exclude_patterns: List of module name patterns to exclude
        grad_accumulation_n: Number of gradients to accumulate
        const_fan_in: Whether to use constant fan-in constraint
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dense_allocation: float = 0.1,
        delta: int = 100,
        alpha: float = 0.3,
        T_end: Optional[int] = None,
        sparsify_patterns: List[str] = None,
        exclude_patterns: List[str] = None,
        grad_accumulation_n: int = 1,
        const_fan_in: bool = True,
        min_layer_size: int = 1000,
        device: str = 'cuda',
    ):
        self.model = model
        self.optimizer = optimizer
        self.dense_allocation = dense_allocation
        self.S = 1 - dense_allocation  # Sparsity
        self.delta = delta
        self.alpha = alpha
        self.T_end = T_end
        self.grad_accumulation_n = grad_accumulation_n
        self.const_fan_in = const_fan_in
        self.min_layer_size = min_layer_size
        self.device = device
        
        self.step_count = 0
        self.rigl_steps = 0
        
        # Default patterns to sparsify (ITHP layers)
        # Note: actual names are like 'deberta.ithp.encoder1.0' 
        if sparsify_patterns is None:
            sparsify_patterns = [
                'ithp.encoder1',    # ITHP encoder1
                'ithp.encoder2',    # ITHP encoder2
                'ithp.MLP1',        # ITHP MLP1
                'ithp.MLP2',        # ITHP MLP2
                'expand.0',         # ITHP expand layer (expand.0 is the Linear)
            ]
        self.sparsify_patterns = sparsify_patterns
        
        # Patterns to exclude (keep dense)
        # Note: exclude_patterns are checked ONLY if sparsify_patterns don't match
        if exclude_patterns is None:
            exclude_patterns = [
                'deberta.model',    # Keep DeBERTa backbone dense (NOT deberta.ithp)
                'embeddings',       # Keep embeddings dense
                'pooler',           # Keep pooler dense
                'LayerNorm',        # Keep normalization dense
                'classifier',       # Keep classifier dense
            ]
        self.exclude_patterns = exclude_patterns
        
        # Storage
        self.masks: List[torch.Tensor] = []
        self.backward_hooks: List[IndexMaskHook] = []
        self.layer_info: List[Dict] = []
        self.W: List[nn.Parameter] = []
        self.layer_names: List[str] = []
        
        # Initialize
        self._init_sparse_layers()
        self._wrap_optimizer_step()
        
    def _should_sparsify(self, name: str, module: nn.Module) -> bool:
        """Check if a layer should be sparsified.
        
        Priority: sparsify_patterns > exclude_patterns
        If a layer matches sparsify_patterns, it's sparsified regardless of exclude_patterns.
        """
        if not isinstance(module, nn.Linear):
            return False
            
        # Check minimum size first
        if module.weight.numel() < self.min_layer_size:
            return False
            
        # Check if matches sparsify patterns (takes priority)
        for pattern in self.sparsify_patterns:
            if pattern.lower() in name.lower():
                return True
        
        # If no sparsify pattern matched, check exclusions
        # (This is for when user specifies custom patterns)
        return False
    
    def _init_sparse_layers(self):
        """Initialize sparse masks for eligible layers."""
        total_params = 0
        total_sparse_params = 0
        
        layer_idx = 0
        for name, module in self.model.named_modules():
            if not self._should_sparsify(name, module):
                continue
                
            weight = module.weight
            out_features, in_features = weight.shape
            n_params = weight.numel()
            
            # Calculate fan-in for this layer
            if self.const_fan_in:
                fan_in = max(1, int(in_features * self.dense_allocation))
                n_keep = fan_in * out_features
            else:
                n_keep = max(1, int(n_params * self.dense_allocation))
                fan_in = n_keep // out_features
            
            # Create initial mask with constant fan-in
            mask = torch.zeros(out_features, in_features, dtype=torch.bool, device=weight.device)
            for i in range(out_features):
                indices = torch.randperm(in_features, device=weight.device)[:fan_in]
                mask[i, indices] = True
            
            self.masks.append(mask)
            self.W.append(weight)
            self.layer_names.append(name)
            
            # Store layer info
            self.layer_info.append({
                'name': name,
                'out_features': out_features,
                'in_features': in_features,
                'fan_in': fan_in,
                'n_params': n_params,
                'n_keep': n_keep,
                'sparsity': 1 - n_keep / n_params,
            })
            
            # Apply initial mask
            weight.data *= mask.float()
            
            # Register backward hook
            hook = IndexMaskHook(layer_idx, self)
            weight.register_hook(hook)
            self.backward_hooks.append(hook)
            
            total_params += n_params
            total_sparse_params += n_keep
            layer_idx += 1
            
        overall_sparsity = 1 - total_sparse_params / total_params if total_params > 0 else 0
        print(f"[SRigL] Initialized {len(self.masks)} sparse layers")
        print(f"[SRigL] Total params: {total_params:,}, Non-zero: {total_sparse_params:,}")
        print(f"[SRigL] Overall sparsity: {overall_sparsity*100:.2f}%")
        for info in self.layer_info:
            print(f"  - {info['name']}: {info['n_params']:,} params, "
                  f"fan_in={info['fan_in']}, sparsity={info['sparsity']*100:.1f}%")
        
    def _wrap_optimizer_step(self):
        """Wrap optimizer.step() to apply masks after update."""
        original_step = self.optimizer.step
        scheduler = self
        
        def wrapped_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            scheduler._reset_momentum()
            scheduler._apply_masks()
            return result
        
        self.optimizer.step = wrapped_step
        
    def _reset_momentum(self):
        """Zero out momentum for inactive weights."""
        for idx, weight in enumerate(self.W):
            mask = self.masks[idx]
            
            # Find parameter in optimizer state
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p is weight:
                        state = self.optimizer.state.get(p, {})
                        # Adam/AdamW momentum
                        if 'exp_avg' in state:
                            state['exp_avg'] *= mask.float()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'] *= mask.float()
                        # SGD momentum
                        if 'momentum_buffer' in state:
                            state['momentum_buffer'] *= mask.float()
                        break
                        
    def _apply_masks(self):
        """Apply sparse masks to all registered weights."""
        for idx, weight in enumerate(self.W):
            weight.data *= self.masks[idx].float()
                
    def cosine_annealing(self) -> float:
        """Cosine annealing schedule for drop fraction."""
        if self.T_end is None:
            return self.alpha
        if self.step_count >= self.T_end:
            return 0.0
        return self.alpha / 2 * (1 + math.cos(math.pi * self.step_count / self.T_end))
    
    def __call__(self):
        """Called after each optimizer step. Manages topology updates."""
        self.step_count += 1
        
        # Check if time for topology update
        if self.step_count % self.delta == 0:
            if self.T_end is None or self.step_count < self.T_end:
                self._rigl_step()
                self.rigl_steps += 1
                
    def step(self):
        """Alias for __call__."""
        self()
                
    @torch.no_grad()
    def _rigl_step(self):
        """Perform RigL prune/regrow step with constant fan-in.
        
        Reference: condensed-sparsity/src/rigl_torch/rigl_scheduler.py L725-822
        """
        drop_fraction = self.cosine_annealing()
        if drop_fraction <= 0:
            return
            
        for idx, weight in enumerate(self.W):
            mask = self.masks[idx]
            info = self.layer_info[idx]
            
            out_features = info['out_features']
            in_features = info['in_features']
            fan_in = info['fan_in']
            
            # Get dense gradient
            hook = self.backward_hooks[idx]
            if hook.dense_grad is None:
                continue
            dense_grad = hook.dense_grad
            
            # Process each output neuron independently (constant fan-in)
            new_mask = torch.zeros_like(mask)
            
            for i in range(out_features):
                row_mask = mask[i]
                row_weight = weight[i]
                row_grad = dense_grad[i]
                
                n_active = row_mask.sum().item()
                n_prune = int(n_active * drop_fraction)
                n_keep = n_active - n_prune
                
                # Ensure minimum connectivity
                n_prune = min(n_prune, n_active - 1)
                n_keep = n_active - n_prune
                
                if n_prune <= 0:
                    new_mask[i] = row_mask
                    continue
                
                # Score for dropping: magnitude of weights (lower = drop)
                score_drop = torch.abs(row_weight)
                score_drop = torch.where(
                    row_mask, 
                    score_drop, 
                    torch.tensor(float('inf'), device=score_drop.device)
                )
                
                # Keep top-k by magnitude
                _, keep_indices = torch.topk(score_drop, n_keep, largest=True)
                keep_mask = torch.zeros(in_features, dtype=torch.bool, device=weight.device)
                keep_mask[keep_indices] = True
                
                # Score for growing: gradient magnitude (higher = grow)
                score_grow = torch.abs(row_grad)
                # Set active positions to -inf so they won't be selected
                score_grow = torch.where(
                    keep_mask, 
                    torch.tensor(float('-inf'), device=score_grow.device), 
                    score_grow
                )
                
                # Grow top-k by gradient
                _, grow_indices = torch.topk(score_grow, n_prune, largest=True)
                grow_mask = torch.zeros(in_features, dtype=torch.bool, device=weight.device)
                grow_mask[grow_indices] = True
                
                # Combine keep + grow
                new_mask[i] = keep_mask | grow_mask
                
            # Update mask
            self.masks[idx] = new_mask
            
            # Zero out pruned weights
            weight.data *= new_mask.float()
            
            # Reset hook gradient
            hook.reset()
        
        # Log
        print(f"[SRigL] Step {self.step_count}: topology update "
              f"(drop_frac={drop_fraction:.4f}, rigl_step={self.rigl_steps})")
        
    def get_sparsity_stats(self) -> Dict:
        """Get current sparsity statistics."""
        stats = {'layers': {}}
        total_params = 0
        total_nonzero = 0
        
        for idx, mask in enumerate(self.masks):
            name = self.layer_names[idx]
            n_params = mask.numel()
            n_nonzero = mask.sum().item()
            sparsity = 1 - n_nonzero / n_params
            
            stats['layers'][name] = {
                'params': n_params,
                'nonzero': n_nonzero,
                'sparsity': sparsity,
                'fan_in': self.layer_info[idx]['fan_in'],
            }
            total_params += n_params
            total_nonzero += n_nonzero
            
        stats['total'] = {
            'params': total_params,
            'nonzero': total_nonzero,
            'sparsity': 1 - total_nonzero / total_params if total_params > 0 else 0,
        }
        stats['step'] = self.step_count
        stats['rigl_steps'] = self.rigl_steps
        
        return stats
    
    def __repr__(self) -> str:
        """String representation with sparsity stats."""
        stats = self.get_sparsity_stats()
        lines = [
            f"DeBertaRigLScheduler(",
            f"  step={stats['step']}, rigl_steps={stats['rigl_steps']},",
            f"  total_sparsity={stats['total']['sparsity']*100:.2f}%,",
            f"  layers={len(self.masks)},",
            f"  delta={self.delta}, alpha={self.alpha}, T_end={self.T_end}",
            f")",
        ]
        return "\n".join(lines)
    
    def state_dict(self) -> Dict:
        """Save scheduler state."""
        return {
            'masks': [m.cpu() for m in self.masks],
            'step_count': self.step_count,
            'rigl_steps': self.rigl_steps,
            'layer_info': self.layer_info,
            'layer_names': self.layer_names,
            'dense_allocation': self.dense_allocation,
            'delta': self.delta,
            'alpha': self.alpha,
            'T_end': self.T_end,
        }
        
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state."""
        self.masks = [m.to(self.device) for m in state_dict['masks']]
        self.step_count = state_dict['step_count']
        self.rigl_steps = state_dict['rigl_steps']
        self.layer_info = state_dict['layer_info']
        self.layer_names = state_dict['layer_names']
        self._apply_masks()
