"""
Utility functions for ITHP + SRigL sparse training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import os


def get_sparsity_stats(model: nn.Module, rigl_scheduler) -> Dict:
    """Get comprehensive sparsity statistics.
    
    Args:
        model: The neural network model
        rigl_scheduler: DeBertaRigLScheduler instance
        
    Returns:
        Dictionary with per-layer and total statistics
    """
    if rigl_scheduler is None:
        return {'total': {'sparsity': 0.0, 'message': 'Dense model (no SRigL)'}}
    
    return rigl_scheduler.get_sparsity_stats()


def count_parameters(model: nn.Module) -> Dict:
    """Count model parameters by category.
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by layer type
    linear_params = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            linear_params += m.weight.numel()
            if m.bias is not None:
                linear_params += m.bias.numel()
    
    return {
        'total': total,
        'trainable': trainable,
        'linear': linear_params,
        'other': trainable - linear_params,
    }


def print_model_sparsity(model: nn.Module, rigl_scheduler=None):
    """Print detailed sparsity information."""
    print("\n" + "="*60)
    print("Model Sparsity Report")
    print("="*60)
    
    if rigl_scheduler is None:
        print("Dense model (SRigL not enabled)")
        return
    
    stats = rigl_scheduler.get_sparsity_stats()
    
    print(f"\nTotal Statistics:")
    print(f"  Total sparse params: {stats['total']['params']:,}")
    print(f"  Non-zero params: {stats['total']['nonzero']:,}")
    print(f"  Overall sparsity: {stats['total']['sparsity']*100:.2f}%")
    print(f"  RigL steps completed: {stats['rigl_steps']}")
    
    print(f"\nPer-Layer Statistics:")
    for name, layer_stats in stats['layers'].items():
        print(f"  {name}:")
        print(f"    Params: {layer_stats['params']:,}, "
              f"Non-zero: {layer_stats['nonzero']:,}, "
              f"Sparsity: {layer_stats['sparsity']*100:.1f}%, "
              f"Fan-in: {layer_stats['fan_in']}")
    
    print("="*60 + "\n")


def validate_constant_fan_in(rigl_scheduler) -> bool:
    """Validate that constant fan-in constraint is satisfied.
    
    Returns:
        True if all layers have constant fan-in, False otherwise
    """
    if rigl_scheduler is None:
        return True
    
    all_valid = True
    for idx, mask in enumerate(rigl_scheduler.masks):
        fan_ins = mask.sum(dim=1)  # Sum across input dimension
        expected = rigl_scheduler.layer_info[idx]['fan_in']
        
        if not torch.all(fan_ins == expected):
            print(f"[Warning] Layer {rigl_scheduler.layer_names[idx]}: "
                  f"inconsistent fan-in (expected {expected}, got {fan_ins.unique().tolist()})")
            all_valid = False
    
    return all_valid


def sparse_to_dense_model(model: nn.Module, rigl_scheduler) -> nn.Module:
    """Convert sparse model back to dense (for inference).
    
    Note: This keeps the sparse weight values, just removes masking.
    For true densification, you'd need to keep only non-zero weights.
    """
    if rigl_scheduler is None:
        return model
    
    # Masks are already applied to weights, so just return model
    return model


def export_sparse_weights(model: nn.Module, rigl_scheduler, path: str):
    """Export model with sparse weight statistics.
    
    Saves:
    - Model state dict
    - RigL scheduler state (masks, etc.)
    - Sparsity statistics
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    export_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if rigl_scheduler is not None:
        export_dict['rigl_state_dict'] = rigl_scheduler.state_dict()
        export_dict['sparsity_stats'] = rigl_scheduler.get_sparsity_stats()
    
    torch.save(export_dict, path)
    print(f"[Export] Saved sparse model to {path}")


def compute_flops_reduction(rigl_scheduler) -> float:
    """Estimate FLOPs reduction from sparsity.
    
    For sparse matrix-vector multiply, FLOPs scales with non-zeros.
    
    Returns:
        Estimated FLOPs reduction factor (1.0 = no reduction)
    """
    if rigl_scheduler is None:
        return 1.0
    
    stats = rigl_scheduler.get_sparsity_stats()
    # FLOPs roughly proportional to non-zero weights
    density = 1 - stats['total']['sparsity']
    return density


def format_sparsity_summary(rigl_scheduler) -> str:
    """Format a one-line sparsity summary."""
    if rigl_scheduler is None:
        return "Dense (no SRigL)"
    
    stats = rigl_scheduler.get_sparsity_stats()
    return (f"Sparsity: {stats['total']['sparsity']*100:.1f}%, "
            f"Non-zero: {stats['total']['nonzero']:,}/{stats['total']['params']:,}, "
            f"RigL steps: {stats['rigl_steps']}")
