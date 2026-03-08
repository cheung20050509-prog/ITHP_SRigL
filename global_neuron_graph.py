"""
Global Neuron Graph - 统一的神经元级拓扑管理

将整个网络视为一个50K+神经元、64M边的全局图：
- 每个神经元有全局唯一ID
- 边 = 权重连接
- SW/SF prior基于全局拓扑
- 统一的prune/grow决策
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class LayerInfo:
    """单层信息"""
    name: str
    layer_idx: int      # 全局层索引
    in_start: int       # 输入神经元起始ID
    in_end: int         # 输入神经元结束ID
    out_start: int      # 输出神经元起始ID
    out_end: int        # 输出神经元结束ID
    weight: nn.Parameter
    mask: torch.Tensor


class GlobalNeuronGraph:
    """
    全局神经元图。
    
    将所有层的神经元统一编号，所有权重连接作为边。
    
    例如 DeBERTa + ITHP:
    - 神经元: ~50K (768 embed + 12×(768+3072) + ITHP)
    - 边: ~64M 权重连接
    """
    
    def __init__(self, sw_beta: float = 1.0, sf_gamma: float = 1.0):
        """
        Args:
            sw_beta: Small-World距离衰减系数
            sf_gamma: Scale-Free度偏好系数
        """
        self.sw_beta = sw_beta
        self.sf_gamma = sf_gamma
        
        # 神经元映射
        self.n_neurons = 0
        self.layers: List[LayerInfo] = []
        self.neuron_to_layer: Dict[int, int] = {}  # neuron_id -> layer_idx
        
        # 权重和mask引用
        self.weights: Dict[str, nn.Parameter] = {}
        self.masks: Dict[str, torch.Tensor] = {}
        
        # 全局统计
        self.total_edges = 0
        self.active_edges = 0
        
        # 缓存
        self._layer_distances = None  # 层间距离矩阵
        self._global_degrees = None   # 全局度分布
        
    def build_from_model(self, model: nn.Module, target_patterns: List[str]):
        """
        从模型构建全局神经元图。
        
        Args:
            model: PyTorch模型
            target_patterns: 要跟踪的层名称模式
        """
        neuron_id = 0
        layer_idx = 0
        
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(p.lower() in name.lower() for p in target_patterns):
                continue
            
            weight = module.weight
            out_dim, in_dim = weight.shape
            
            # 分配输入神经元ID（如果是新的）
            # 简化：每层独立分配，后续可优化为共享
            in_start = neuron_id
            in_end = neuron_id + in_dim
            neuron_id = in_end
            
            # 分配输出神经元ID
            out_start = neuron_id
            out_end = neuron_id + out_dim
            neuron_id = out_end
            
            # 创建mask（初始全连接）
            mask = torch.ones(out_dim, in_dim, dtype=torch.bool, device=weight.device)
            
            # 记录层信息
            layer_info = LayerInfo(
                name=name,
                layer_idx=layer_idx,
                in_start=in_start,
                in_end=in_end,
                out_start=out_start,
                out_end=out_end,
                weight=weight,
                mask=mask,
            )
            self.layers.append(layer_info)
            self.weights[name] = weight
            self.masks[name] = mask
            
            # 记录神经元到层的映射
            for nid in range(in_start, out_end):
                self.neuron_to_layer[nid] = layer_idx
            
            self.total_edges += out_dim * in_dim
            self.active_edges += out_dim * in_dim
            
            layer_idx += 1
        
        self.n_neurons = neuron_id
        self._build_layer_distances()
        
        print(f"[GlobalNeuronGraph] Built: {self.n_neurons} neurons, "
              f"{self.total_edges} edges, {len(self.layers)} layers")
        
    def _build_layer_distances(self):
        """构建层间距离矩阵"""
        n_layers = len(self.layers)
        self._layer_distances = torch.zeros(n_layers, n_layers)
        
        for i in range(n_layers):
            for j in range(n_layers):
                # 层距离 = |i - j|，归一化到[0, 1]
                self._layer_distances[i, j] = abs(i - j) / max(n_layers - 1, 1)
    
    def get_neuron_distance(self, n1: int, n2: int) -> float:
        """
        计算两个神经元之间的"距离"。
        
        距离定义：
        - 同层内：基于位置的距离
        - 跨层：层距离 + 层内位置距离
        """
        layer1 = self.neuron_to_layer.get(n1, 0)
        layer2 = self.neuron_to_layer.get(n2, 0)
        
        # 层距离
        layer_dist = self._layer_distances[layer1, layer2].item()
        
        # 层内位置距离（简化：使用相对位置）
        info1 = self.layers[layer1]
        info2 = self.layers[layer2]
        
        # 神经元在其层内的相对位置 [0, 1]
        if n1 >= info1.out_start:
            pos1 = (n1 - info1.out_start) / max(info1.out_end - info1.out_start, 1)
        else:
            pos1 = (n1 - info1.in_start) / max(info1.in_end - info1.in_start, 1)
            
        if n2 >= info2.out_start:
            pos2 = (n2 - info2.out_start) / max(info2.out_end - info2.out_start, 1)
        else:
            pos2 = (n2 - info2.in_start) / max(info2.in_end - info2.in_start, 1)
        
        pos_dist = abs(pos1 - pos2)
        
        # 总距离 = 层距离权重 + 位置距离权重
        return 0.7 * layer_dist + 0.3 * pos_dist
    
    def compute_sw_prior_for_layer(self, layer_name: str) -> torch.Tensor:
        """
        计算某层的SW prior矩阵。
        
        SW prior: 偏好距离近的连接
        prior[i,j] = exp(-beta * distance(out_neuron_i, in_neuron_j))
        """
        layer_info = None
        for info in self.layers:
            if info.name == layer_name:
                layer_info = info
                break
        
        if layer_info is None:
            return None
        
        out_dim = layer_info.out_end - layer_info.out_start
        in_dim = layer_info.in_end - layer_info.in_start
        device = layer_info.weight.device
        
        # 输出神经元位置
        out_positions = torch.arange(out_dim, device=device).float() / out_dim
        # 输入神经元位置
        in_positions = torch.arange(in_dim, device=device).float() / in_dim
        
        # 距离矩阵
        dist_matrix = torch.abs(out_positions.unsqueeze(1) - in_positions.unsqueeze(0))
        
        # SW prior
        sw_prior = torch.exp(-self.sw_beta * dist_matrix)
        
        return sw_prior
    
    def compute_sf_prior_for_layer(self, layer_name: str) -> torch.Tensor:
        """
        计算某层的SF prior矩阵。
        
        SF prior: 偏好高度节点（hub）
        prior[i,j] ∝ degree(i)^gamma + degree(j)^gamma
        """
        layer_info = None
        for info in self.layers:
            if info.name == layer_name:
                layer_info = info
                break
        
        if layer_info is None:
            return None
        
        mask = layer_info.mask
        
        # 当前度
        out_degree = mask.sum(dim=1).float()  # 每个输出神经元的出度
        in_degree = mask.sum(dim=0).float()   # 每个输入神经元的入度
        
        # 归一化
        out_degree = out_degree / (out_degree.max() + 1e-8)
        in_degree = in_degree / (in_degree.max() + 1e-8)
        
        # SF prior: hub优先
        sf_prior = (out_degree.unsqueeze(1) ** self.sf_gamma + 
                    in_degree.unsqueeze(0) ** self.sf_gamma)
        
        # 归一化到 [0, 1]
        sf_prior = sf_prior / (sf_prior.max() + 1e-8)
        
        return sf_prior
    
    def get_global_degrees(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取全局度分布。
        
        Returns:
            in_degrees: 每个神经元的入度
            out_degrees: 每个神经元的出度
        """
        in_degrees = torch.zeros(self.n_neurons)
        out_degrees = torch.zeros(self.n_neurons)
        
        for info in self.layers:
            mask = info.mask
            
            # 输出神经元的出度
            out_deg = mask.sum(dim=1).cpu()
            for i, d in enumerate(out_deg):
                out_degrees[info.out_start + i] = d
            
            # 输入神经元的入度
            in_deg = mask.sum(dim=0).cpu()
            for j, d in enumerate(in_deg):
                in_degrees[info.in_start + j] += d  # 可能多层共享输入
        
        return in_degrees, out_degrees
    
    def get_stats(self) -> Dict:
        """获取全局图统计"""
        active = sum(m.sum().item() for m in self.masks.values())
        
        in_deg, out_deg = self.get_global_degrees()
        
        return {
            'n_neurons': self.n_neurons,
            'n_layers': len(self.layers),
            'total_edges': self.total_edges,
            'active_edges': int(active),
            'density': active / self.total_edges if self.total_edges > 0 else 0,
            'avg_in_degree': in_deg.mean().item(),
            'avg_out_degree': out_deg.mean().item(),
            'max_in_degree': in_deg.max().item(),
            'max_out_degree': out_deg.max().item(),
        }
    
    # ==================== 统一的 Prune/Grow (向量化) ====================
    
    def global_prune(
        self, 
        importance_dict: Dict[str, torch.Tensor],
        prune_ratio: float = 0.05,
        min_density: float = 0.01,
    ) -> Dict[str, int]:
        """
        全局统一剪枝 (向量化版本)。
        
        所有层的importance放到一个池子，全局排序后剪掉最低的。
        """
        # 1. 收集所有层的分数到一个大tensor中
        all_scores_list = []
        layer_info_list = []  # [(layer_name, start_idx, end_idx, flat_mask)]
        global_offset = 0
        
        for info in self.layers:
            name = info.name
            mask = self.masks[name]
            
            if name not in importance_dict:
                continue
            
            importance = importance_dict[name]
            
            # 计算prior
            sw_prior = self.compute_sw_prior_for_layer(name)
            sf_prior = self.compute_sf_prior_for_layer(name)
            prior = 0.5 * sw_prior + 0.5 * sf_prior
            
            # 综合分数 = importance × prior
            score = (importance * prior).view(-1)
            
            # 把非活跃位置的分数设为极大值（不会被选中剪枝）
            flat_mask = mask.view(-1)
            score = torch.where(flat_mask, score, torch.full_like(score, float('inf')))
            
            all_scores_list.append(score)
            layer_info_list.append((name, global_offset, global_offset + score.numel(), flat_mask))
            global_offset += score.numel()
        
        if not all_scores_list:
            return {}
        
        # 拼接成一个大tensor
        all_scores = torch.cat(all_scores_list)
        
        # 2. 计算活跃边数和剪枝数量
        n_active = sum(self.masks[name].sum().item() for name in self.masks)
        n_prune = int(n_active * prune_ratio)
        
        # 最小密度保护
        min_edges = int(self.total_edges * min_density)
        if n_active - n_prune < min_edges:
            n_prune = max(0, int(n_active) - min_edges)
        
        if n_prune == 0:
            return {}
        
        # 3. 用topk找最低分的n_prune个 (smallest)
        _, prune_indices = torch.topk(all_scores, n_prune, largest=False)
        prune_indices = prune_indices.cpu().numpy()
        
        # 4. 根据全局索引反推回各层并执行剪枝
        prune_counts = {}
        for global_idx in prune_indices:
            # 找到这个索引属于哪一层
            for layer_name, start_idx, end_idx, flat_mask in layer_info_list:
                if start_idx <= global_idx < end_idx:
                    local_idx = global_idx - start_idx
                    
                    mask = self.masks[layer_name]
                    weight = self.weights[layer_name]
                    
                    mask.view(-1)[local_idx] = False
                    weight.data.view(-1)[local_idx] = 0.0
                    
                    prune_counts[layer_name] = prune_counts.get(layer_name, 0) + 1
                    break
        
        return prune_counts
    
    def global_grow(
        self,
        co_activation_dict: Dict[str, torch.Tensor],
        grow_ratio: float = 0.03,
        max_density: float = 1.5,
    ) -> Dict[str, int]:
        """
        全局统一生长 (向量化版本)。
        
        所有层的growth候选放到一个池子，全局排序后选最高分的。
        """
        # 1. 收集所有层的growth分数
        all_scores_list = []
        layer_info_list = []  # [(layer_name, start_idx, end_idx, flat_mask)]
        global_offset = 0
        
        for info in self.layers:
            name = info.name
            mask = self.masks[name]
            weight = self.weights[name]
            
            if name not in co_activation_dict:
                continue
            
            co_act = co_activation_dict[name]
            
            # 计算prior
            sw_prior = self.compute_sw_prior_for_layer(name)
            sf_prior = self.compute_sf_prior_for_layer(name)
            prior = 0.5 * sw_prior + 0.5 * sf_prior
            
            # Growth分数 = co_activation × prior
            score = (co_act * prior).view(-1)
            
            # 把已活跃位置的分数设为极小值（不会被选中生长）
            flat_mask = mask.view(-1)
            score = torch.where(~flat_mask, score, torch.full_like(score, float('-inf')))
            
            all_scores_list.append(score)
            layer_info_list.append((name, global_offset, global_offset + score.numel(), flat_mask, weight))
            global_offset += score.numel()
        
        if not all_scores_list:
            return {}
        
        # 拼接成一个大tensor
        all_scores = torch.cat(all_scores_list)
        
        # 2. 计算活跃边数和生长数量
        n_active = sum(self.masks[name].sum().item() for name in self.masks)
        n_grow = int(n_active * grow_ratio)
        
        # 最大密度限制
        max_edges = int(self.total_edges * max_density)
        if n_active + n_grow > max_edges:
            n_grow = max(0, max_edges - int(n_active))
        
        # 可用位置数
        n_inactive = int(self.total_edges - n_active)
        n_grow = min(n_grow, n_inactive)
        
        if n_grow == 0:
            return {}
        
        # 3. 用topk找最高分的n_grow个 (largest)
        _, grow_indices = torch.topk(all_scores, n_grow, largest=True)
        grow_indices = grow_indices.cpu().numpy()
        
        # 4. 根据全局索引反推回各层并执行生长
        grow_counts = {}
        for global_idx in grow_indices:
            # 找到这个索引属于哪一层
            for layer_name, start_idx, end_idx, flat_mask, weight in layer_info_list:
                if start_idx <= global_idx < end_idx:
                    local_idx = global_idx - start_idx
                    
                    mask = self.masks[layer_name]
                    
                    mask.view(-1)[local_idx] = True
                    
                    # 初始化新权重（小随机值）
                    fan_in = mask.sum(dim=1).float().mean().item()
                    std = 0.02 / math.sqrt(max(fan_in, 1))
                    weight.data.view(-1)[local_idx] = torch.randn(1, device=weight.device).item() * std
                    
                    grow_counts[layer_name] = grow_counts.get(layer_name, 0) + 1
                    break
        
        return grow_counts
    
    def topology_step(
        self,
        importance_dict: Dict[str, torch.Tensor],
        co_activation_dict: Dict[str, torch.Tensor],
        prune_scale: float = 0.05,
        grow_scale: float = 0.03,
    ) -> Dict:
        """
        统一的拓扑更新步骤。
        
        Args:
            importance_dict: 每层的importance分数
            co_activation_dict: 每层的co_activation分数
            prune_scale: 剪枝比例
            grow_scale: 生长比例
            
        Returns:
            统计信息
        """
        # 全局剪枝
        prune_counts = self.global_prune(importance_dict, prune_scale)
        total_pruned = sum(prune_counts.values())
        
        # 全局生长
        grow_counts = self.global_grow(co_activation_dict, grow_scale)
        total_grown = sum(grow_counts.values())
        
        # 更新活跃边计数
        self.active_edges = sum(m.sum().item() for m in self.masks.values())
        
        return {
            'pruned': total_pruned,
            'grown': total_grown,
            'active_edges': self.active_edges,
            'density': self.active_edges / self.total_edges,
            'prune_per_layer': prune_counts,
            'grow_per_layer': grow_counts,
        }
