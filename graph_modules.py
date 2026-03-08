"""
Small-World + Scale-Free Graph Fusion Module for Multimodal Learning.

Implements bio-inspired graph topology for multimodal token fusion:
- Small-World: High clustering + short path lengths (Watts-Strogatz)
- Scale-Free: Power-law degree distribution (Barabási-Albert)
- Neuroplastic edges: Gradient-based prune/grow unified with weight plasticity

Reference:
- Watts & Strogatz (1998): Collective dynamics of 'small-world' networks
- Barabási & Albert (1999): Emergence of scaling in random networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class EdgeActivityTracker:
    """
    Track edge activity using the same principle as weight activity tracking.
    
    Edge Activity = |edge_logit| × |node_i_activation| × |node_j_activation| × |gradient|
    
    This ensures graph edges are treated identically to weight connections:
    - Low activity edges are candidates for pruning
    - High co-activation node pairs are candidates for growth
    """
    
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.node_activations = None  # Cached from forward pass
        self.edge_activity = None     # EMA of edge activity
        self.co_activation = None     # Node pair co-activation for growth
        self.sample_count = 0
        
    def cache_activations(self, node_embeddings: torch.Tensor):
        """Cache node activations during forward pass.
        
        Args:
            node_embeddings: [batch, n_nodes, hidden_dim]
        """
        # Average over batch, store [n_nodes, hidden_dim]
        self.node_activations = node_embeddings.detach().abs().mean(dim=0)
        
    def update_activity(self, edge_logits: torch.Tensor, n_nodes: int):
        """Update edge activity after backward pass.
        
        Uses gradient information from edge_logits to compute importance.
        Activity[i,j] = |edge_logit[i,j]| × |node_i| × |node_j| × |grad[i,j]|
        
        Args:
            edge_logits: nn.Parameter with .grad attribute
            n_nodes: number of nodes in graph
        """
        if edge_logits.grad is None or self.node_activations is None:
            return
        
        with torch.no_grad():
            # Node activation magnitudes per node (reduce hidden_dim)
            node_mag = self.node_activations.norm(dim=-1)  # [n_nodes]
            
            # Compute pairwise node activation product: |node_i| × |node_j|
            node_pair_act = torch.outer(node_mag, node_mag)  # [n_nodes, n_nodes]
            
            # Edge activity = |edge_logit| × |node_i| × |node_j| × |grad|
            current_activity = (
                edge_logits.abs() * 
                node_pair_act * 
                edge_logits.grad.abs()
            )
            
            # EMA update
            if self.edge_activity is None:
                self.edge_activity = current_activity.clone()
            else:
                self.edge_activity = (
                    (1 - self.ema_alpha) * self.edge_activity +
                    self.ema_alpha * current_activity
                )
            
            # Update co-activation for growth (Hebbian)
            # Co-activation[i,j] = |node_i| × |node_j| (gradient-weighted)
            hebbian_signal = node_pair_act * edge_logits.grad.abs()
            
            if self.co_activation is None:
                self.co_activation = hebbian_signal.clone()
            else:
                self.co_activation = (
                    (1 - self.ema_alpha) * self.co_activation +
                    self.ema_alpha * hebbian_signal
                )
            
            self.sample_count += 1
    
    def get_prune_scores(self, active_mask: torch.Tensor) -> torch.Tensor:
        """Get activity scores for active edges (used for pruning).
        
        Low activity = prune candidate.
        """
        if self.edge_activity is None:
            return torch.zeros_like(active_mask, dtype=torch.float)
        
        return self.edge_activity * active_mask.float()
    
    def get_growth_scores(self, inactive_mask: torch.Tensor) -> torch.Tensor:
        """Get co-activation scores for inactive edges (used for growth).
        
        High co-activation = growth candidate (Hebbian principle).
        """
        if self.co_activation is None:
            return torch.zeros_like(inactive_mask, dtype=torch.float)
        
        return self.co_activation * inactive_mask.float()
    
    def reset(self):
        """Reset cached activations (call at start of each step)."""
        self.node_activations = None


def build_small_world_adjacency(n_nodes: int, k: int = 4, p: float = 0.1) -> torch.Tensor:
    """
    Build Watts-Strogatz small-world graph adjacency matrix.
    
    Args:
        n_nodes: Number of nodes
        k: Each node connects to k nearest neighbors (must be even)
        p: Rewiring probability (0 = ring, 1 = random)
    
    Returns:
        adj: [n_nodes, n_nodes] adjacency matrix
    """
    # Start with ring lattice
    adj = torch.zeros(n_nodes, n_nodes)
    
    # Connect each node to k/2 neighbors on each side
    for i in range(n_nodes):
        for j in range(1, k // 2 + 1):
            # Connect to neighbors
            adj[i, (i + j) % n_nodes] = 1
            adj[i, (i - j) % n_nodes] = 1
    
    # Rewire edges with probability p
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj[i, j] == 1 and torch.rand(1).item() < p:
                # Rewire: remove edge (i,j), add edge to random node
                adj[i, j] = 0
                adj[j, i] = 0
                
                # Find new target (not i, not existing neighbor)
                candidates = torch.where(adj[i] == 0)[0]
                candidates = candidates[candidates != i]
                if len(candidates) > 0:
                    new_j = candidates[torch.randint(len(candidates), (1,)).item()]
                    adj[i, new_j] = 1
                    adj[new_j, i] = 1
    
    return adj


def build_scale_free_adjacency(n_nodes: int, m: int = 2) -> torch.Tensor:
    """
    Build Barabási-Albert scale-free graph adjacency matrix.
    
    Args:
        n_nodes: Number of nodes
        m: Number of edges to attach from new node to existing nodes
    
    Returns:
        adj: [n_nodes, n_nodes] adjacency matrix
    """
    adj = torch.zeros(n_nodes, n_nodes)
    
    # Start with m+1 fully connected nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = 1
            adj[j, i] = 1
    
    # Add remaining nodes with preferential attachment
    degrees = adj.sum(dim=1)
    
    for new_node in range(m + 1, n_nodes):
        # Probability proportional to degree (preferential attachment)
        prob = degrees[:new_node] / degrees[:new_node].sum()
        
        # Select m nodes to connect to (without replacement)
        targets = torch.multinomial(prob + 1e-8, min(m, new_node), replacement=False)
        
        for t in targets:
            adj[new_node, t] = 1
            adj[t, new_node] = 1
        
        # Update degrees
        degrees = adj.sum(dim=1)
    
    return adj


def build_hybrid_adjacency(n_nodes: int, sw_k: int = 4, sw_p: float = 0.1, 
                            sf_m: int = 2, alpha: float = 0.5) -> torch.Tensor:
    """
    Build hybrid Small-World + Scale-Free adjacency matrix.
    
    Args:
        n_nodes: Number of nodes
        sw_k: Small-world k parameter
        sw_p: Small-world rewiring probability  
        sf_m: Scale-free m parameter
        alpha: Weight for small-world (1-alpha for scale-free)
    
    Returns:
        adj: [n_nodes, n_nodes] adjacency matrix
    """
    sw_adj = build_small_world_adjacency(n_nodes, sw_k, sw_p)
    sf_adj = build_scale_free_adjacency(n_nodes, sf_m)
    
    # Combine: union of edges, weighted
    hybrid = alpha * sw_adj + (1 - alpha) * sf_adj
    # Binarize
    hybrid = (hybrid > 0.3).float()
    
    return hybrid


class GraphAttentionLayer(nn.Module):
    """
    Single-head Graph Attention Layer (GAT).
    Simplified implementation without torch_geometric dependency for batch processing.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_nodes, in_features]
            adj: [n_nodes, n_nodes] or [batch, n_nodes, n_nodes]
        
        Returns:
            out: [batch, n_nodes, out_features]
        """
        batch_size, n_nodes, _ = x.shape
        
        # Linear transformation
        h = self.W(x)  # [batch, n_nodes, out_features]
        
        # Compute attention coefficients
        # For each pair (i,j), compute attention a([h_i || h_j])
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # [batch, n, n, out]
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # [batch, n, n, out]
        
        # Concatenate
        h_cat = torch.cat([h_i, h_j], dim=-1)  # [batch, n, n, 2*out]
        
        # Attention scores
        e = self.leaky_relu(self.a(h_cat).squeeze(-1))  # [batch, n, n]
        
        # Mask with adjacency (only attend to neighbors)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Set non-neighbors to -inf
        attention = e.masked_fill(adj == 0, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)  # Handle all -inf rows
        attention = self.dropout(attention)
        
        # Aggregate
        out = torch.bmm(attention, h)  # [batch, n_nodes, out_features]
        
        return out


class MultiHeadGraphAttention(nn.Module):
    """Multi-head Graph Attention."""
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4, 
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        
        if concat:
            assert out_features % n_heads == 0
            head_dim = out_features // n_heads
        else:
            head_dim = out_features
        
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, head_dim, dropout)
            for _ in range(n_heads)
        ])
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x, adj) for head in self.heads]
        
        if self.concat:
            return torch.cat(head_outputs, dim=-1)
        else:
            return torch.mean(torch.stack(head_outputs), dim=0)


class SmallWorldScaleFreeGraphFusion(nn.Module):
    """
    Graph-based multimodal fusion with Small-World + Scale-Free topology.
    
    Properly treats multimodal sequences as graph nodes with temporal segmentation:
    - Text tokens: n_text nodes (each token is a node)
    - Visual segments: n_visual_segments nodes (temporal segments preserve structure)
    - Acoustic segments: n_acoustic_segments nodes (temporal segments preserve structure)
    
    Graph topology combines:
    - Small-World: efficient local clustering + shortcuts (Watts-Strogatz)
    - Scale-Free: hub nodes for important features (Barabási-Albert)
    
    Cross-modal edges:
    - Text←→Visual: text CLS/key tokens connect to visual segments
    - Text←→Acoustic: text CLS/key tokens connect to acoustic segments
    - Visual←→Acoustic: temporal alignment (same time segments connected)
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        visual_dim: int = 47,
        acoustic_dim: int = 74,
        hidden_dim: int = 256,
        n_text_nodes: int = 50,
        n_visual_segments: int = 8,
        n_acoustic_segments: int = 8,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        sw_k: int = 6,
        sw_p: float = 0.15,
        sf_m: int = 3,
        topology_alpha: float = 0.5,
        learnable_topology: bool = True,
        cross_modal_connectivity: float = 0.3,
    ):
        super().__init__()
        
        self.n_text_nodes = n_text_nodes
        self.n_visual_segments = n_visual_segments
        self.n_acoustic_segments = n_acoustic_segments
        self.n_total_nodes = n_text_nodes + n_visual_segments + n_acoustic_segments
        self.hidden_dim = hidden_dim
        
        # Node type indices for graph structure
        self.text_start = 0
        self.text_end = n_text_nodes
        self.visual_start = n_text_nodes
        self.visual_end = n_text_nodes + n_visual_segments
        self.acoustic_start = n_text_nodes + n_visual_segments
        self.acoustic_end = self.n_total_nodes
        
        # Project all modalities to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.acoustic_proj = nn.Linear(acoustic_dim, hidden_dim)
        
        # Learnable segment position embeddings for temporal ordering
        self.visual_pos_emb = nn.Parameter(torch.randn(n_visual_segments, hidden_dim) * 0.02)
        self.acoustic_pos_emb = nn.Parameter(torch.randn(n_acoustic_segments, hidden_dim) * 0.02)
        
        # Modality type embeddings
        self.modality_emb = nn.Embedding(3, hidden_dim)  # 0=text, 1=visual, 2=acoustic
        
        # Build initial adjacency (Small-World + Scale-Free hybrid with cross-modal edges)
        init_adj = self._build_multimodal_graph(
            sw_k, sw_p, sf_m, topology_alpha, cross_modal_connectivity
        )
        
        # Make adjacency learnable (soft edges)
        if learnable_topology:
            # Initialize edge logits from adjacency
            edge_logits = torch.zeros(self.n_total_nodes, self.n_total_nodes)
            edge_logits[init_adj > 0] = 2.0  # High logit for initial edges
            edge_logits[init_adj == 0] = -2.0  # Low logit for non-edges
            self.edge_logits = nn.Parameter(edge_logits)
        else:
            self.register_buffer('adj', init_adj)
            self.edge_logits = None
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            MultiHeadGraphAttention(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # FFN for richer node representations
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        
        # Output projections (back to original dimensions)
        self.text_out = nn.Linear(hidden_dim, text_dim)
        self.visual_out = nn.Linear(hidden_dim, visual_dim)
        self.acoustic_out = nn.Linear(hidden_dim, acoustic_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # ==================== Neuroplastic Edge Tracking ====================
        # Track edge activity for gradient-based prune/grow (unified with weights)
        self.edge_tracker = EdgeActivityTracker(ema_alpha=0.1)
        
        # Track edge age for fair comparison (new edges need time to develop)
        # All edges start at age 0, incremented each neuroplastic step
        self.register_buffer('edge_age', torch.zeros(self.n_total_nodes, self.n_total_nodes))
        
    def _build_multimodal_graph(
        self, sw_k: int, sw_p: float, sf_m: int, 
        topology_alpha: float, cross_modal: float
    ) -> torch.Tensor:
        """
        Build a multimodal graph with intra-modal and cross-modal connections.
        
        Structure:
        1. Intra-modal: Small-World + Scale-Free within each modality
        2. Cross-modal: Strategic connections between modalities
        """
        n = self.n_total_nodes
        adj = torch.zeros(n, n)
        
        # 1. Intra-modality connections (Small-World + Scale-Free within each modality)
        # Text subgraph
        text_adj = build_hybrid_adjacency(
            self.n_text_nodes, sw_k, sw_p, sf_m, topology_alpha
        )
        adj[self.text_start:self.text_end, self.text_start:self.text_end] = text_adj
        
        # Visual subgraph (smaller, so use smaller k)
        if self.n_visual_segments >= 4:
            visual_adj = build_hybrid_adjacency(
                self.n_visual_segments, min(sw_k, 4), sw_p, min(sf_m, 2), topology_alpha
            )
            adj[self.visual_start:self.visual_end, self.visual_start:self.visual_end] = visual_adj
        else:
            # Fully connect small visual subgraph
            adj[self.visual_start:self.visual_end, self.visual_start:self.visual_end] = 1.0
            
        # Acoustic subgraph
        if self.n_acoustic_segments >= 4:
            acoustic_adj = build_hybrid_adjacency(
                self.n_acoustic_segments, min(sw_k, 4), sw_p, min(sf_m, 2), topology_alpha
            )
            adj[self.acoustic_start:self.acoustic_end, self.acoustic_start:self.acoustic_end] = acoustic_adj
        else:
            adj[self.acoustic_start:self.acoustic_end, self.acoustic_start:self.acoustic_end] = 1.0
        
        # 2. Cross-modal connections
        
        # Text ↔ Visual: CLS token (first) + periodic text tokens connect to visual
        # Connect text[0] (CLS) to all visual segments
        adj[0, self.visual_start:self.visual_end] = 1.0
        adj[self.visual_start:self.visual_end, 0] = 1.0
        
        # Connect every ~5 text tokens to corresponding visual segments (temporal alignment)
        text_stride = max(1, self.n_text_nodes // self.n_visual_segments)
        for i in range(self.n_visual_segments):
            text_idx = min(i * text_stride, self.n_text_nodes - 1)
            adj[text_idx, self.visual_start + i] = 1.0
            adj[self.visual_start + i, text_idx] = 1.0
        
        # Text ↔ Acoustic: similar to visual
        adj[0, self.acoustic_start:self.acoustic_end] = 1.0
        adj[self.acoustic_start:self.acoustic_end, 0] = 1.0
        
        text_stride = max(1, self.n_text_nodes // self.n_acoustic_segments)
        for i in range(self.n_acoustic_segments):
            text_idx = min(i * text_stride, self.n_text_nodes - 1)
            adj[text_idx, self.acoustic_start + i] = 1.0
            adj[self.acoustic_start + i, text_idx] = 1.0
        
        # Visual ↔ Acoustic: temporal alignment (same time segments connected)
        # If n_visual == n_acoustic, align 1:1; else, proportional
        for i in range(self.n_visual_segments):
            # Find corresponding acoustic segment
            acoustic_idx = int(i * self.n_acoustic_segments / self.n_visual_segments)
            acoustic_idx = min(acoustic_idx, self.n_acoustic_segments - 1)
            adj[self.visual_start + i, self.acoustic_start + acoustic_idx] = 1.0
            adj[self.acoustic_start + acoustic_idx, self.visual_start + i] = 1.0
        
        # Add random cross-modal edges based on cross_modal probability
        for i in range(self.n_text_nodes):
            for j in range(self.n_visual_segments):
                if torch.rand(1).item() < cross_modal * 0.1:
                    adj[i, self.visual_start + j] = 1.0
                    adj[self.visual_start + j, i] = 1.0
            for j in range(self.n_acoustic_segments):
                if torch.rand(1).item() < cross_modal * 0.1:
                    adj[i, self.acoustic_start + j] = 1.0
                    adj[self.acoustic_start + j, i] = 1.0
        
        # Remove self-loops
        adj.fill_diagonal_(0)
        
        return adj
        
    def get_adjacency(self, temperature: float = 1.0) -> torch.Tensor:
        """Get adjacency matrix (soft or hard)."""
        if self.edge_logits is not None:
            # Soft adjacency via sigmoid
            adj = torch.sigmoid(self.edge_logits / temperature)
            # Make symmetric
            adj = (adj + adj.T) / 2
            return adj
        else:
            return self.adj
    
    def compute_topology_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to encourage graph properties:
        - Sparsity: not too many edges
        - Scale-free: high degree variance
        - Cross-modal connectivity: ensure modalities are connected
        """
        if self.edge_logits is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        adj = self.get_adjacency()
        
        # 1. Sparsity: penalize too many edges
        edge_density = adj.mean()
        target_density = 0.12  # Aim for ~12% edge density
        sparsity_loss = (edge_density - target_density) ** 2
        
        # 2. Degree variance: encourage scale-free (high variance in degrees)
        degrees = adj.sum(dim=1)
        degree_var = degrees.var()
        scale_free_loss = 1.0 / (degree_var + 1.0)
        
        # 3. Cross-modal connectivity: ensure there are edges between modalities
        text_visual_conn = adj[self.text_start:self.text_end, self.visual_start:self.visual_end].mean()
        text_acoustic_conn = adj[self.text_start:self.text_end, self.acoustic_start:self.acoustic_end].mean()
        visual_acoustic_conn = adj[self.visual_start:self.visual_end, self.acoustic_start:self.acoustic_end].mean()
        
        # Penalize if cross-modal connectivity is too low
        min_cross_modal = 0.05
        cross_modal_loss = (
            F.relu(min_cross_modal - text_visual_conn) +
            F.relu(min_cross_modal - text_acoustic_conn) +
            F.relu(min_cross_modal - visual_acoustic_conn)
        )
        
        return sparsity_loss + 0.1 * scale_free_loss + 0.5 * cross_modal_loss
    
    def _segment_sequence(self, seq: torch.Tensor, n_segments: int) -> torch.Tensor:
        """
        Segment a sequence into n_segments nodes by pooling.
        
        Args:
            seq: [batch, seq_len, dim]
            n_segments: number of output segments
        
        Returns:
            segmented: [batch, n_segments, dim]
        """
        batch, seq_len, dim = seq.shape
        
        if seq_len <= n_segments:
            # If sequence is shorter than segments, pad or repeat
            if seq_len == n_segments:
                return seq
            # Interpolate
            seq = seq.transpose(1, 2)  # [batch, dim, seq_len]
            seq = F.interpolate(seq, size=n_segments, mode='linear', align_corners=True)
            return seq.transpose(1, 2)  # [batch, n_segments, dim]
        
        # Split sequence into segments and pool each
        segment_size = seq_len // n_segments
        segments = []
        
        for i in range(n_segments):
            start = i * segment_size
            if i == n_segments - 1:
                # Last segment takes the rest
                end = seq_len
            else:
                end = start + segment_size
            
            # Mean pool within segment
            segment = seq[:, start:end, :].mean(dim=1)  # [batch, dim]
            segments.append(segment)
        
        return torch.stack(segments, dim=1)  # [batch, n_segments, dim]
    
    def _expand_segments(self, segments: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Expand segmented representation back to original sequence length.
        
        Args:
            segments: [batch, n_segments, dim]
            target_len: target sequence length
        
        Returns:
            expanded: [batch, target_len, dim]
        """
        batch, n_segments, dim = segments.shape
        
        # Use linear interpolation
        segments = segments.transpose(1, 2)  # [batch, dim, n_segments]
        expanded = F.interpolate(segments, size=target_len, mode='linear', align_corners=True)
        return expanded.transpose(1, 2)  # [batch, target_len, dim]
    
    def forward(
        self, 
        text_emb: torch.Tensor,  # [batch, seq, text_dim]
        visual: torch.Tensor,    # [batch, seq, visual_dim] or [batch, visual_dim]
        acoustic: torch.Tensor,  # [batch, seq, acoustic_dim] or [batch, acoustic_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply graph-based multimodal fusion with proper temporal segmentation.
        
        Graph Structure (e.g., 50 text + 8 visual + 8 acoustic = 66 nodes):
        - Text tokens: preserve fine-grained token-level information
        - Visual/Acoustic segments: preserve temporal structure with coarser granularity
        - Cross-modal edges: enable information flow between modalities
        
        Returns:
            text_out: [batch, seq, text_dim] - enhanced text representation
            visual_out: same shape as input visual
            acoustic_out: same shape as input acoustic
            topology_loss: scalar loss for regularization
        """
        batch_size = text_emb.size(0)
        seq_len = text_emb.size(1)
        device = text_emb.device
        
        # Handle different input shapes
        visual_is_seq = visual.dim() == 3
        acoustic_is_seq = acoustic.dim() == 3
        
        if visual_is_seq:
            visual_seq_len = visual.size(1)
        else:
            visual = visual.unsqueeze(1)  # [batch, 1, dim]
            visual_seq_len = 1
            
        if acoustic_is_seq:
            acoustic_seq_len = acoustic.size(1)
        else:
            acoustic = acoustic.unsqueeze(1)  # [batch, 1, dim]
            acoustic_seq_len = 1
        
        # 1. Project all modalities to hidden dim
        text_h = self.text_proj(text_emb)  # [batch, seq, hidden]
        visual_h = self.visual_proj(visual)  # [batch, visual_seq_len, hidden]
        acoustic_h = self.acoustic_proj(acoustic)  # [batch, acoustic_seq_len, hidden]
        
        # 2. Segment visual and acoustic into fixed number of nodes
        visual_segments = self._segment_sequence(visual_h, self.n_visual_segments)  # [batch, n_visual, hidden]
        acoustic_segments = self._segment_sequence(acoustic_h, self.n_acoustic_segments)  # [batch, n_acoustic, hidden]
        
        # 3. Add position embeddings for segments
        visual_segments = visual_segments + self.visual_pos_emb.unsqueeze(0)
        acoustic_segments = acoustic_segments + self.acoustic_pos_emb.unsqueeze(0)
        
        # 4. Add modality type embeddings
        text_type = self.modality_emb(torch.zeros(batch_size, self.n_text_nodes, dtype=torch.long, device=device))
        visual_type = self.modality_emb(torch.ones(batch_size, self.n_visual_segments, dtype=torch.long, device=device))
        acoustic_type = self.modality_emb(torch.full((batch_size, self.n_acoustic_segments), 2, dtype=torch.long, device=device))
        
        text_h = text_h + text_type
        visual_segments = visual_segments + visual_type
        acoustic_segments = acoustic_segments + acoustic_type
        
        # 5. Concatenate all nodes: [batch, n_total_nodes, hidden]
        # Order: text tokens, visual segments, acoustic segments
        nodes = torch.cat([text_h, visual_segments, acoustic_segments], dim=1)
        
        # 6. Get adjacency
        adj = self.get_adjacency()
        
        # 7. Graph attention layers with residual
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            nodes_new = gat(nodes, adj)
            nodes = ln(nodes + self.dropout(nodes_new))
        
        # 8. FFN
        nodes = self.ffn_ln(nodes + self.ffn(nodes))
        
        # 8.5 Cache node activations for neuroplastic edge tracking
        # This enables gradient-based prune/grow after backward pass
        self.edge_tracker.cache_activations(nodes)
        
        # 9. Split back by modality
        text_nodes = nodes[:, self.text_start:self.text_end, :]  # [batch, n_text, hidden]
        visual_nodes = nodes[:, self.visual_start:self.visual_end, :]  # [batch, n_visual, hidden]
        acoustic_nodes = nodes[:, self.acoustic_start:self.acoustic_end, :]  # [batch, n_acoustic, hidden]
        
        # 10. Project back to original dimensions
        text_out = self.text_out(text_nodes)  # [batch, seq, text_dim]
        visual_out_seg = self.visual_out(visual_nodes)  # [batch, n_visual, visual_dim]
        acoustic_out_seg = self.acoustic_out(acoustic_nodes)  # [batch, n_acoustic, acoustic_dim]
        
        # 11. Expand segments back to original sequence length
        if visual_is_seq:
            visual_out = self._expand_segments(visual_out_seg, visual_seq_len)
        else:
            visual_out = visual_out_seg.mean(dim=1)  # [batch, visual_dim]
            
        if acoustic_is_seq:
            acoustic_out = self._expand_segments(acoustic_out_seg, acoustic_seq_len)
        else:
            acoustic_out = acoustic_out_seg.mean(dim=1)  # [batch, acoustic_dim]
        
        # 12. Topology regularization loss
        topology_loss = self.compute_topology_loss()
        
        return text_out, visual_out, acoustic_out, topology_loss
    
    def get_graph_stats(self) -> dict:
        """Get current graph topology statistics."""
        adj = self.get_adjacency().detach()
        degrees = adj.sum(dim=1)
        
        # Per-modality stats
        text_degrees = degrees[self.text_start:self.text_end]
        visual_degrees = degrees[self.visual_start:self.visual_end]
        acoustic_degrees = degrees[self.acoustic_start:self.acoustic_end]
        
        # Cross-modal edge counts
        text_visual_edges = (adj[self.text_start:self.text_end, self.visual_start:self.visual_end] > 0.5).sum().item()
        text_acoustic_edges = (adj[self.text_start:self.text_end, self.acoustic_start:self.acoustic_end] > 0.5).sum().item()
        visual_acoustic_edges = (adj[self.visual_start:self.visual_end, self.acoustic_start:self.acoustic_end] > 0.5).sum().item()
        
        return {
            'n_nodes': self.n_total_nodes,
            'n_text_nodes': self.n_text_nodes,
            'n_visual_segments': self.n_visual_segments,
            'n_acoustic_segments': self.n_acoustic_segments,
            'n_edges': (adj > 0.5).sum().item() / 2,  # undirected
            'edge_density': adj.mean().item(),
            'avg_degree': degrees.mean().item(),
            'max_degree': degrees.max().item(),
            'min_degree': degrees.min().item(),
            'degree_std': degrees.std().item(),
            'text_avg_degree': text_degrees.mean().item(),
            'visual_avg_degree': visual_degrees.mean().item(),
            'acoustic_avg_degree': acoustic_degrees.mean().item(),
            'text_visual_edges': text_visual_edges,
            'text_acoustic_edges': text_acoustic_edges,
            'visual_acoustic_edges': visual_acoustic_edges,
        }
    
    # ==================== Neuroplastic Edge Methods ====================
    
    def prune_edges(self, prune_ratio: float = 0.1, min_edges_per_node: int = 2) -> int:
        """
        Prune weak edges based on edge_logits magnitude.
        
        Uses magnitude-based pruning similar to weight pruning in NeuroplasticBlock:
        - Compute edge importance as |edge_logits|
        - Remove bottom prune_ratio of edges
        - Preserve minimum connectivity per node
        
        Args:
            prune_ratio: Fraction of active edges to prune (0.0 to 1.0)
            min_edges_per_node: Minimum edges each node must keep
        
        Returns:
            n_pruned: Number of edges pruned
        """
        if self.edge_logits is None:
            return 0
        
        with torch.no_grad():
            # Get current adjacency (soft)
            adj = self.get_adjacency()
            
            # Find active edges (above threshold)
            active_mask = adj > 0.5
            n_active = active_mask.sum().item()
            
            if n_active == 0:
                return 0
            
            # Number of edges to prune
            n_prune = int(n_active * prune_ratio)
            if n_prune == 0:
                return 0
            
            # ==================== Activity-based Pruning ====================
            # Use the same principle as weight pruning:
            # Activity = |edge_logit| × |node_i| × |node_j| × |gradient|
            # Low activity edges are candidates for pruning
            
            activity_scores = self.edge_tracker.get_prune_scores(active_mask)
            has_activity = activity_scores.sum() > 0
            
            if has_activity:
                # Use activity-based importance (same as weight neuroplastic)
                importance = activity_scores
            else:
                # Fallback to magnitude-based if no gradient info available yet
                importance = self.edge_logits.abs() * active_mask.float()
            
            # Protect young edges: edges need time to develop before being pruned
            # Scale down importance of mature edges slightly to give new edges a chance
            edge_maturity = torch.clamp(self.edge_age / 10.0, 0, 1)  # Mature after 10 steps
            # Young edges get artificially boosted importance (protected from pruning)
            importance = importance * (0.5 + 0.5 * edge_maturity)
            
            # Find edges to prune (lowest importance among active)
            flat_importance = importance.view(-1)
            flat_active = active_mask.view(-1)
            
            # Get indices of active edges sorted by importance
            active_indices = torch.where(flat_active)[0]
            active_importance = flat_importance[active_indices]
            
            # Sort by importance (ascending) - lowest activity = prune first
            _, sort_order = torch.sort(active_importance)
            prune_candidates = active_indices[sort_order[:n_prune]]
            
            # Check minimum connectivity constraint
            n_pruned = 0
            for idx in prune_candidates:
                i = idx // self.n_total_nodes
                j = idx % self.n_total_nodes
                
                # Check if node i or j would fall below minimum degree
                current_degree_i = (self.edge_logits[i] > 0).sum().item()
                current_degree_j = (self.edge_logits[j] > 0).sum().item()
                
                if current_degree_i > min_edges_per_node and current_degree_j > min_edges_per_node:
                    # Prune: set logit to very negative value
                    self.edge_logits.data[i, j] = -5.0
                    self.edge_logits.data[j, i] = -5.0  # symmetric
                    # Reset edge age for pruned edges
                    self.edge_age[i, j] = 0
                    self.edge_age[j, i] = 0
                    n_pruned += 1
            
            return n_pruned
    
    def grow_edges(self, grow_ratio: float = 0.1, cross_modal_bias: float = 2.0) -> int:
        """
        Grow new edges with preference for cross-modal connections.
        
        Uses Hebbian growth: grow edges between highly co-active node pairs.
        Same principle as weight growth in neuroplastic networks.
        
        Args:
            grow_ratio: Fraction of current edges to add as new edges
            cross_modal_bias: Multiplier for cross-modal edge probability
        
        Returns:
            n_grown: Number of edges grown
        """
        if self.edge_logits is None:
            return 0
        
        with torch.no_grad():
            # Get current adjacency
            adj = self.get_adjacency()
            
            # Find inactive edges (below threshold)
            inactive_mask = adj < 0.3
            # Also must not be self-loop
            inactive_mask.fill_diagonal_(False)
            
            n_active = (adj > 0.5).sum().item()
            n_grow = int(n_active * grow_ratio)
            
            if n_grow == 0:
                return 0
            
            # ==================== Hebbian Growth ====================
            # Use co-activation scores from gradient-based tracking
            # High co-activation = strong candidate for new edge (Hebbian principle)
            
            hebbian_scores = self.edge_tracker.get_growth_scores(inactive_mask)
            has_hebbian = hebbian_scores.sum() > 0
            
            # Create cross-modal bias mask
            modal_bias = torch.ones_like(self.edge_logits)
            
            # Cross-modal edges get higher priority
            modal_bias[self.text_start:self.text_end, self.visual_start:self.visual_end] = cross_modal_bias
            modal_bias[self.visual_start:self.visual_end, self.text_start:self.text_end] = cross_modal_bias
            modal_bias[self.text_start:self.text_end, self.acoustic_start:self.acoustic_end] = cross_modal_bias
            modal_bias[self.acoustic_start:self.acoustic_end, self.text_start:self.text_end] = cross_modal_bias
            modal_bias[self.visual_start:self.visual_end, self.acoustic_start:self.acoustic_end] = cross_modal_bias * 1.5
            modal_bias[self.acoustic_start:self.acoustic_end, self.visual_start:self.visual_end] = cross_modal_bias * 1.5
            
            if has_hebbian:
                # Combine Hebbian scores with cross-modal bias
                # Hebbian determines WHERE to grow, bias encourages cross-modal
                grow_scores = hebbian_scores * modal_bias
            else:
                # Fallback: random with cross-modal bias
                grow_scores = modal_bias * inactive_mask.float()
            
            # Apply inactive mask
            grow_scores = grow_scores * inactive_mask.float()
            
            if grow_scores.sum() == 0:
                return 0
            
            # Normalize to probability
            grow_prob = grow_scores / grow_scores.sum()
            
            # Sample edges to grow based on Hebbian + cross-modal scores
            flat_prob = grow_prob.view(-1)
            n_candidates = (flat_prob > 0).sum().item()
            grow_indices = torch.multinomial(flat_prob + 1e-8, min(n_grow, n_candidates), replacement=False)
            
            # Activate new edges
            n_grown = 0
            for idx in grow_indices:
                i = idx // self.n_total_nodes
                j = idx % self.n_total_nodes
                
                if i != j:  # No self-loops
                    # Initialize with small positive logit
                    # Higher initial logit for edges with high hebbian score
                    if has_hebbian:
                        init_logit = 1.5 + 0.5 * (hebbian_scores[i, j] / (hebbian_scores.max() + 1e-8))
                    else:
                        init_logit = 1.0 + torch.randn(1).item() * 0.1
                    
                    self.edge_logits.data[i, j] = init_logit
                    self.edge_logits.data[j, i] = init_logit  # symmetric
                    
                    # New edges start at age 0 (protected from immediate pruning)
                    self.edge_age[i, j] = 0
                    self.edge_age[j, i] = 0
                    
                    n_grown += 1
            
            return n_grown
    
    def neuroplastic_step(self, prune_ratio: float = 0.05, grow_ratio: float = 0.05) -> dict:
        """
        Perform one neuroplastic update step: prune weak edges, grow new ones.
        
        Unified with weight neuroplasticity:
        1. Update edge activity from accumulated gradients
        2. Prune low-activity edges (like weight pruning)
        3. Grow edges based on Hebbian co-activation (like weight growth)
        4. Increment edge age for fair comparison
        
        Args:
            prune_ratio: Fraction of edges to prune
            grow_ratio: Fraction of edges to grow
        
        Returns:
            dict with n_pruned, n_grown, edge_stats, activity_stats
        """
        # 1. Update edge activity from gradients (must be called after backward)
        if self.edge_logits is not None and self.edge_logits.grad is not None:
            self.edge_tracker.update_activity(self.edge_logits, self.n_total_nodes)
        
        # 2. Prune low-activity edges
        n_pruned = self.prune_edges(prune_ratio)
        
        # 3. Grow high co-activation edges (Hebbian)
        n_grown = self.grow_edges(grow_ratio)
        
        # 4. Increment edge age for all active edges
        # This ensures new edges get a chance to develop before being pruned
        with torch.no_grad():
            adj = self.get_adjacency()
            active_mask = adj > 0.5
            self.edge_age[active_mask] += 1
        
        # 5. Reset tracker for next accumulation period
        self.edge_tracker.reset()
        
        stats = self.get_graph_stats()
        
        return {
            'n_pruned': n_pruned,
            'n_grown': n_grown,
            'n_edges': stats['n_edges'],
            'edge_density': stats['edge_density'],
            'cross_modal_edges': (
                stats['text_visual_edges'] + 
                stats['text_acoustic_edges'] + 
                stats['visual_acoustic_edges']
            ),
            'tracker_samples': self.edge_tracker.sample_count,
            'has_activity': self.edge_tracker.edge_activity is not None,
        }
    
    def get_edge_gradient_stats(self) -> dict:
        """Get gradient statistics for edge_logits (for monitoring)."""
        if self.edge_logits is None or self.edge_logits.grad is None:
            return {'has_grad': False}
        
        grad = self.edge_logits.grad
        adj = self.get_adjacency().detach()
        active_mask = adj > 0.5
        
        return {
            'has_grad': True,
            'grad_mean': grad.mean().item(),
            'grad_std': grad.std().item(),
            'grad_max': grad.abs().max().item(),
            'active_grad_mean': (grad * active_mask.float()).sum().item() / max(1, active_mask.sum().item()),
        }
    
    def update_edge_activity(self):
        """
        Update edge activity tracking from current gradients.
        
        Call this after backward() and before neuroplastic_step().
        This accumulates gradient information for activity-based pruning.
        """
        if self.edge_logits is not None and self.edge_logits.grad is not None:
            self.edge_tracker.update_activity(self.edge_logits, self.n_total_nodes)
    
    def get_unified_importance(self) -> Dict[str, torch.Tensor]:
        """
        Get importance scores in format compatible with weight importance.
        
        Returns dict with:
        - 'edge_importance': [n_nodes, n_nodes] activity-based importance
        - 'active_mask': [n_nodes, n_nodes] which edges are active
        
        This allows external systems to compare edge vs weight importance.
        """
        if self.edge_logits is None:
            return {}
        
        adj = self.get_adjacency().detach()
        active_mask = adj > 0.5
        
        # Use activity if available, else magnitude
        if self.edge_tracker.edge_activity is not None:
            importance = self.edge_tracker.edge_activity.clone()
        else:
            importance = self.edge_logits.abs().detach()
        
        # Normalize to [0, 1] range
        if importance.max() > 0:
            importance = importance / importance.max()
        
        return {
            'edge_importance': importance,
            'active_mask': active_mask,
            'n_edges': active_mask.sum().item() // 2,
        }
