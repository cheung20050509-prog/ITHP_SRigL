"""
Neuron-Level Topology Management

将每层的权重矩阵视为二分图：
- 输入神经元 → 输出神经元
- 边 = 权重连接
- 用图的视角管理prune/grow

统一框架：
- Importance = |weight| × |pre_act| × |post_act| × |gradient|
- Growth Score = Hebbian co-activation × topology prior
- Topology Prior = α × Small-World + (1-α) × Scale-Free
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import math


class NeuronLevelTopology:
    """
    神经元级的拓扑管理器。
    
    将Linear层视为二分图：
    - 节点：输入神经元 + 输出神经元
    - 边：权重连接 weight[i,j]
    
    支持：
    - Activity-based importance tracking
    - Hebbian co-activation for growth
    - Small-World + Scale-Free topology prior
    - Independent prune/grow (no net-zero constraint)
    """
    
    def __init__(
        self,
        name: str,
        weight: nn.Parameter,
        mask: torch.Tensor,
        sw_beta: float = 2.0,       # Small-World距离衰减
        sf_alpha: float = 0.5,      # SW与SF的混合比例
        ema_alpha: float = 0.1,     # EMA衰减率
        min_density: float = 0.01,  # 最小密度（防止过度稀疏）
        max_density: float = 1.0,   # 最大密度
        protect_period: int = 5,    # 新连接的保护周期
    ):
        self.name = name
        self.weight = weight  # 共享引用
        self.mask = mask      # 共享引用
        
        self.out_dim, self.in_dim = weight.shape
        self.n_total = self.out_dim * self.in_dim
        
        # Topology prior参数
        self.sw_beta = sw_beta
        self.sf_alpha = sf_alpha
        
        # Activity tracking
        self.ema_alpha = ema_alpha
        self.importance = None      # [out, in] importance EMA
        self.co_activation = None   # [out, in] growth候选分数
        self.pre_activation = None  # [in] 输入激活
        self.post_activation = None # [out] 输出激活
        
        # Connection age tracking (保护新连接)
        self.connection_age = torch.zeros_like(mask, dtype=torch.float)
        self.protect_period = protect_period
        
        # Density constraints
        self.min_density = min_density
        self.max_density = max_density
        
        # 预计算topology prior (静态部分)
        self._topology_prior_cache = None
        
    # ==================== Activity Tracking ====================
    
    def cache_activations(self, pre_act: torch.Tensor, post_act: torch.Tensor):
        """
        缓存前向传播的激活值。
        
        Args:
            pre_act: 输入激活 [batch, in_dim] 或 [batch, seq, in_dim]
            post_act: 输出激活 [batch, out_dim] 或 [batch, seq, out_dim]
        """
        # 处理sequence维度
        if pre_act.dim() == 3:
            pre_act = pre_act.mean(dim=1)
        if post_act.dim() == 3:
            post_act = post_act.mean(dim=1)
        
        # 平均绝对激活 per neuron
        self.pre_activation = pre_act.abs().mean(dim=0).detach()   # [in_dim]
        self.post_activation = post_act.abs().mean(dim=0).detach() # [out_dim]
        
        # 更新Hebbian co-activation (用于growth)
        # co_act[i,j] = |post[i]| × |pre[j]|
        current_co = torch.outer(self.post_activation, self.pre_activation)
        
        if self.co_activation is None:
            self.co_activation = current_co
        else:
            self.co_activation = (
                (1 - self.ema_alpha) * self.co_activation + 
                self.ema_alpha * current_co
            )
    
    def update_importance(self, gradient: torch.Tensor):
        """
        用梯度更新importance。调用时机：backward后。
        
        Importance = |weight| × |pre_act| × |post_act| × |gradient|
        """
        if self.pre_activation is None or self.post_activation is None:
            return
        
        # importance[i,j] = |w[i,j]| × |pre[j]| × |post[i]| × |grad[i,j]|
        current_importance = (
            self.weight.abs() *
            self.pre_activation.unsqueeze(0) *    # [1, in]
            self.post_activation.unsqueeze(1) *   # [out, 1]
            gradient.abs()
        ).detach()
        
        if self.importance is None:
            self.importance = current_importance
        else:
            self.importance = (
                (1 - self.ema_alpha) * self.importance +
                self.ema_alpha * current_importance
            )
    
    # ==================== Topology Prior ====================
    
    def get_topology_prior(self) -> torch.Tensor:
        """
        计算SW/SF混合拓扑先验。
        
        Returns:
            prior: [out_dim, in_dim] 每个位置的先验分数
        """
        if self._topology_prior_cache is not None:
            return self._topology_prior_cache
        
        device = self.weight.device
        prior = torch.zeros(self.out_dim, self.in_dim, device=device)
        
        # ===== Small-World: 偏好位置相近的连接 =====
        # 用相对位置作为"距离"
        out_pos = torch.arange(self.out_dim, device=device).float() / self.out_dim
        in_pos = torch.arange(self.in_dim, device=device).float() / self.in_dim
        
        # 距离矩阵 [out, in]
        distance = (out_pos.unsqueeze(1) - in_pos.unsqueeze(0)).abs()
        sw_prior = torch.exp(-self.sw_beta * distance)
        
        # ===== Scale-Free: 动态计算（依赖当前度） =====
        # 这里用静态近似：uniform，实际prune/grow时动态计算
        sf_prior = torch.ones(self.out_dim, self.in_dim, device=device)
        
        # 混合
        prior = self.sf_alpha * sw_prior + (1 - self.sf_alpha) * sf_prior
        
        # Normalize to [0, 1]
        prior = prior / prior.max()
        
        self._topology_prior_cache = prior
        return prior
    
    def get_dynamic_sf_prior(self) -> torch.Tensor:
        """
        动态计算Scale-Free先验（基于当前度分布）。
        
        Scale-Free: 偏好连接到高度节点（富者更富）
        """
        device = self.weight.device
        
        # 当前度
        in_degree = self.mask.sum(dim=1).float()   # [out] 每个输出神经元的入度
        out_degree = self.mask.sum(dim=0).float()  # [in] 每个输入神经元的出度
        
        # Normalize
        in_degree_norm = in_degree / (in_degree.sum() + 1e-8)
        out_degree_norm = out_degree / (out_degree.sum() + 1e-8)
        
        # sf_prior[i,j] = hub_score(i) + hub_score(j)
        sf_prior = in_degree_norm.unsqueeze(1) + out_degree_norm.unsqueeze(0)
        
        return sf_prior
    
    # ==================== Prune ====================
    
    def prune(self, scale: float) -> int:
        """
        剪掉最低importance的连接。
        
        Args:
            scale: 要剪掉的比例 (0.0 ~ 1.0)
        
        Returns:
            实际剪掉的数量
        """
        if scale <= 0:
            return 0
        
        n_active = self.mask.sum().item()
        n_prune = int(n_active * scale)
        
        if n_prune == 0:
            return 0
        
        # 检查最小密度约束
        min_connections = int(self.n_total * self.min_density)
        if n_active - n_prune < min_connections:
            n_prune = max(0, n_active - min_connections)
        
        if n_prune == 0:
            return 0
        
        # 计算prune分数 = importance × prior × age_factor
        if self.importance is not None:
            importance = self.importance
        else:
            # Fallback: 用权重绝对值
            importance = self.weight.abs()
        
        # 混合静态SW + 动态SF prior
        sw_prior = self.get_topology_prior()
        sf_prior = self.get_dynamic_sf_prior()
        prior = self.sf_alpha * sw_prior + (1 - self.sf_alpha) * sf_prior
        
        # Age factor: 新连接受保护（分数人为抬高）
        age_factor = torch.clamp(self.connection_age / self.protect_period, 0, 1)
        # 新连接(age=0): factor=0, 乘以后importance被忽略（不会被选中prune）
        # 老连接(age>=protect): factor=1, 正常参与
        
        prune_score = importance * prior * (0.1 + 0.9 * age_factor)
        
        # 只考虑活跃连接
        prune_score = prune_score * self.mask.float()
        prune_score[~self.mask] = float('inf')  # 不活跃的不参与排序
        
        # 找最低分的n_prune个
        flat_scores = prune_score.view(-1)
        _, prune_indices = torch.topk(flat_scores, n_prune, largest=False)
        
        # 执行prune
        with torch.no_grad():
            prune_mask = torch.zeros_like(self.mask)
            prune_mask.view(-1)[prune_indices] = True
            
            self.mask[prune_mask] = False
            self.weight.data[prune_mask] = 0.0
            self.connection_age[prune_mask] = 0  # 重置age
        
        return n_prune
    
    # ==================== Grow ====================
    
    def grow(self, scale: float) -> int:
        """
        长出最高co-activation的新连接。
        
        Args:
            scale: 要长出的比例 (相对于当前active数量)
        
        Returns:
            实际长出的数量
        """
        if scale <= 0:
            return 0
        
        n_active = self.mask.sum().item()
        n_grow = int(n_active * scale)
        
        if n_grow == 0:
            return 0
        
        # 检查最大密度约束
        max_connections = int(self.n_total * self.max_density)
        if n_active + n_grow > max_connections:
            n_grow = max(0, max_connections - n_active)
        
        if n_grow == 0:
            return 0
        
        # 计算grow分数 = co_activation × prior
        if self.co_activation is not None:
            growth_score = self.co_activation
        else:
            # Fallback: 用prior
            growth_score = torch.ones_like(self.weight)
        
        # 混合prior
        sw_prior = self.get_topology_prior()
        sf_prior = self.get_dynamic_sf_prior()
        prior = self.sf_alpha * sw_prior + (1 - self.sf_alpha) * sf_prior
        
        grow_score = growth_score * prior
        
        # 只考虑不活跃位置
        grow_score = grow_score * (~self.mask).float()
        grow_score[self.mask] = float('-inf')  # 已有的不参与
        
        # 找最高分的n_grow个
        flat_scores = grow_score.view(-1)
        n_candidates = (~self.mask).sum().item()
        k = min(n_grow, n_candidates)
        
        if k == 0:
            return 0
        
        _, grow_indices = torch.topk(flat_scores, k, largest=True)
        
        # 执行grow
        with torch.no_grad():
            grow_mask = torch.zeros_like(self.mask)
            grow_mask.view(-1)[grow_indices] = True
            
            self.mask[grow_mask] = True
            
            # 初始化新权重
            n_grown = grow_mask.sum().item()
            fan_in = self.mask.sum(dim=1).float().mean().item()
            std = 0.1 / math.sqrt(max(fan_in, 1))
            self.weight.data[grow_mask] = torch.randn(n_grown, device=self.weight.device) * std
            
            # 新连接age=0（受保护）
            self.connection_age[grow_mask] = 0
        
        return n_grown
    
    # ==================== Utilities ====================
    
    def step_age(self):
        """每次topology update后调用，增加所有活跃连接的age"""
        with torch.no_grad():
            self.connection_age[self.mask] += 1
    
    def reset_activations(self):
        """重置激活缓存"""
        self.pre_activation = None
        self.post_activation = None
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        n_active = self.mask.sum().item()
        density = n_active / self.n_total
        
        in_degree = self.mask.sum(dim=1).float()
        out_degree = self.mask.sum(dim=0).float()
        
        return {
            'name': self.name,
            'shape': (self.out_dim, self.in_dim),
            'n_active': n_active,
            'density': density,
            'in_degree_mean': in_degree.mean().item(),
            'in_degree_std': in_degree.std().item(),
            'out_degree_mean': out_degree.mean().item(),
            'out_degree_std': out_degree.std().item(),
            'importance_mean': self.importance.mean().item() if self.importance is not None else 0,
            'avg_age': self.connection_age[self.mask].mean().item() if n_active > 0 else 0,
        }
