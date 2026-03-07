"""
ITHP + DeBERTa model with Neuroplastic training support.
Implements synaptic-level connection reorganization mimicking biological plasticity.

Key features:
1. Within-layer connection pruning/growth based on activity
2. Cross-layer skip connections that can grow dynamically (like axonal sprouting)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
import global_configs
from global_configs import DEVICE

from typing import List, Tuple, Dict, Optional


class NeuroplasticBlock(nn.Module):
    """A block with dynamic cross-layer skip connections.
    
    Supports:
    1. Standard forward: input -> fc1 -> hidden_activation -> dropout -> fc2 -> [output_activation] -> [output_dropout]
    2. Dynamic skip: input -> skip_weight -> output (bypassing hidden layer)
    
    The skip connections start empty and can grow based on co-activation patterns.
    This mimics axonal sprouting in biological neural plasticity.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 hidden_activation: str = 'relu', dropout: float = 0.0,
                 output_activation: str = None, output_dropout: float = 0.0,
                 max_skip_ratio: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.max_skip_connections = int(in_dim * out_dim * max_skip_ratio)
        
        # Standard layers
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
        # Hidden layer activation (after fc1)
        if hidden_activation == 'relu':
            self.hidden_activation = nn.ReLU()
        elif hidden_activation == 'sigmoid':
            self.hidden_activation = nn.Sigmoid()
        elif hidden_activation == 'tanh':
            self.hidden_activation = nn.Tanh()
        else:
            self.hidden_activation = nn.Identity()
        
        # Hidden layer dropout (after hidden_activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output activation (after fc2) - for MLP modules
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'relu':
            self.output_activation = nn.ReLU()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()
        
        # Output dropout (after output_activation) - for MLP modules
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout > 0 else nn.Identity()
        
        # Cross-layer skip: input -> output directly
        # Starts empty (all False), connections grow based on Hebbian learning
        self.register_buffer('skip_mask', 
                           torch.zeros(out_dim, in_dim, dtype=torch.bool))
        self.skip_weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        
        # Cache for co-activation tracking
        self.register_buffer('input_cache', torch.zeros(1))
        self.register_buffer('output_cache', torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional cross-layer skip."""
        # Cache input for Hebbian tracking (ensure 1D)
        if self.training:
            if x.dim() == 3:  # (batch, seq, dim)
                self.input_cache = x.detach().mean(dim=(0, 1))  # -> (dim,)
            elif x.dim() == 2:  # (batch, dim)
                self.input_cache = x.detach().mean(dim=0)  # -> (dim,)
            else:
                self.input_cache = x.detach()
        
        # Standard path: fc1 -> hidden_activation -> dropout -> fc2
        hidden = self.fc1(x)
        hidden = self.hidden_activation(hidden)
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        
        # Add cross-layer skip contribution if any connections exist
        if self.skip_mask.any():
            effective_skip = self.skip_weight * self.skip_mask.float()
            skip_out = F.linear(x, effective_skip)
            out = out + skip_out
        
        # Output activation and dropout (for MLP modules)
        out = self.output_activation(out)
        out = self.output_dropout(out)
            
        # Cache output for Hebbian tracking (ensure 1D)
        if self.training:
            if out.dim() == 3:  # (batch, seq, dim)
                self.output_cache = out.detach().mean(dim=(0, 1))  # -> (dim,)
            elif out.dim() == 2:  # (batch, dim)
                self.output_cache = out.detach().mean(dim=0)  # -> (dim,)
            else:
                self.output_cache = out.detach()
            
        return out
    
    @property
    def n_skip_connections(self) -> int:
        return self.skip_mask.sum().item()
    
    @property
    def skip_density(self) -> float:
        return self.n_skip_connections / (self.in_dim * self.out_dim)
    
    def add_skip_connection(self, in_idx: int, out_idx: int, init_std: float = 0.01) -> bool:
        """Add a skip connection from input[in_idx] to output[out_idx]."""
        if self.n_skip_connections >= self.max_skip_connections:
            return False
        if self.skip_mask[out_idx, in_idx]:
            return False  # Already exists
            
        self.skip_mask[out_idx, in_idx] = True
        self.skip_weight.data[out_idx, in_idx] = init_std * torch.randn(1).item()
        return True
    
    def remove_skip_connection(self, in_idx: int, out_idx: int) -> bool:
        """Remove a skip connection."""
        if not self.skip_mask[out_idx, in_idx]:
            return False
            
        self.skip_mask[out_idx, in_idx] = False
        self.skip_weight.data[out_idx, in_idx] = 0
        return True
    
    def get_co_activation_scores(self) -> torch.Tensor:
        """Compute co-activation scores for potential skip connections.
        
        Returns: (out_dim, in_dim) tensor of co-activation scores.
        High scores indicate strong candidates for new skip connections.
        """
        if self.input_cache.numel() == 1 or self.output_cache.numel() == 1:
            return torch.zeros(self.out_dim, self.in_dim, device=self.skip_weight.device)
            
        # Co-activation: |output_i| * |input_j|
        scores = torch.outer(self.output_cache.abs(), self.input_cache.abs())
        return scores


class ITHP_Neuroplastic(nn.Module):
    """ITHP module with true synaptic plasticity support.
    
    Uses NeuroplasticBlock for each encoder, enabling:
    1. Within-layer connection pruning/growth (via NeuroplasticScheduler)
    2. Cross-layer skip connections that grow based on co-activation
    
    Structure:
        X0 -> encoder1 -> B0 -> encoder2 -> B1 -> output
              (VAE)            (VAE)
    """
    
    def __init__(self, ITHP_args):
        super().__init__()
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM, 
            global_configs.ACOUSTIC_DIM,
            global_configs.VISUAL_DIM
        )

        self.X0_dim = ITHP_args['X0_dim']
        self.X1_dim = ITHP_args['X1_dim']
        self.X2_dim = ITHP_args['X2_dim']
        self.inter_dim = ITHP_args['inter_dim']
        self.drop_prob = ITHP_args['drop_prob']
        self.max_sen_len = ITHP_args['max_sen_len']
        self.B0_dim = ITHP_args['B0_dim']
        self.B1_dim = ITHP_args['B1_dim']
        self.p_beta = ITHP_args['p_beta']
        self.p_gamma = ITHP_args['p_gamma']
        self.p_lambda = ITHP_args['p_lambda']

        # Encoder1: X0 -> B0 (with cross-layer skip capability)
        # Original: Linear -> ReLU -> Dropout -> Linear
        self.encoder1 = NeuroplasticBlock(
            in_dim=self.X0_dim,
            hidden_dim=self.inter_dim,
            out_dim=self.B0_dim * 2,  # mu and logvar
            hidden_activation='relu',
            dropout=self.drop_prob,
            output_activation=None,
            output_dropout=0.0,
            max_skip_ratio=0.05
        )

        # MLP1: B0 -> predict acoustic
        # Original: Linear -> ReLU -> Dropout -> Linear -> Sigmoid -> Dropout
        self.MLP1 = NeuroplasticBlock(
            in_dim=self.B0_dim,
            hidden_dim=self.inter_dim,
            out_dim=self.X1_dim,
            hidden_activation='relu',  # ReLU on hidden layer
            dropout=self.drop_prob,
            output_activation='sigmoid',  # Sigmoid after fc2
            output_dropout=self.drop_prob,  # Dropout after Sigmoid
            max_skip_ratio=0.05
        )

        # Encoder2: B0 -> B1 (with cross-layer skip capability)
        # Original: Linear -> ReLU -> Dropout -> Linear
        self.encoder2 = NeuroplasticBlock(
            in_dim=self.B0_dim,
            hidden_dim=self.inter_dim,
            out_dim=self.B1_dim * 2,  # mu and logvar
            hidden_activation='relu',
            dropout=self.drop_prob,
            output_activation=None,
            output_dropout=0.0,
            max_skip_ratio=0.05
        )

        # MLP2: B1 -> predict visual
        # Original: Linear -> ReLU -> Dropout -> Linear -> Sigmoid -> Dropout
        self.MLP2 = NeuroplasticBlock(
            in_dim=self.B1_dim,
            hidden_dim=self.inter_dim,
            out_dim=self.X2_dim,
            hidden_activation='relu',  # ReLU on hidden layer
            dropout=self.drop_prob,
            output_activation='sigmoid',  # Sigmoid after fc2
            output_dropout=self.drop_prob,  # Dropout after Sigmoid
            max_skip_ratio=0.05
        )

        self.criterion = nn.MSELoss()

    def kl_loss(self, mu, logvar):
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x, visual, acoustic):
        # Encoder 1: X0 -> B0
        h1 = self.encoder1(x)
        mu1, logvar1 = h1.chunk(2, dim=-1)
        kl_loss_0 = self.kl_loss(mu1, logvar1)
        b0 = self.reparameterise(mu1, logvar1)
        
        # MLP1: B0 -> predict acoustic
        output1 = self.MLP1(b0)
        mse_0 = self.criterion(output1, acoustic)
        IB0 = kl_loss_0 + self.p_beta * mse_0

        # Encoder 2: B0 -> B1
        h2 = self.encoder2(b0)
        mu2, logvar2 = h2.chunk(2, dim=-1)
        kl_loss_1 = self.kl_loss(mu2, logvar2)
        b1 = self.reparameterise(mu2, logvar2)
        
        # MLP2: B1 -> predict visual
        output2 = self.MLP2(b1)
        mse_1 = self.criterion(output2, visual)
        IB1 = kl_loss_1 + self.p_gamma * mse_1
        
        IB_total = IB0 + self.p_lambda * IB1

        return b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1
    
    def get_neuroplastic_blocks(self) -> List[NeuroplasticBlock]:
        """Return all NeuroplasticBlock modules for scheduler management."""
        return [self.encoder1, self.MLP1, self.encoder2, self.MLP2]


class ITHP_DebertaModel_Neuroplastic(nn.Module):
    """DeBERTa + ITHP with true synaptic plasticity.
    
    Structure (matches original ITHP exactly):
        input_ids -> DeBERTa -> x (batch, seq, 768)
                                ↓
                            ITHP -> b1 (batch, seq, B1_dim)
                                ↓
                            expand -> h_m (batch, seq, 768)
                                ↓
                            LayerNorm(h_m + x) -> pooler -> output (batch, 768)
    """
    
    def __init__(self, config, args):
        super().__init__()
        from transformers.models.bert.modeling_bert import BertPooler
        
        TEXT_DIM = global_configs.TEXT_DIM
        ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
        VISUAL_DIM = global_configs.VISUAL_DIM
        
        self.text_encoder = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.text_encoder.config = config
        
        ITHP_args = {
            'X0_dim': TEXT_DIM,  # 768
            'X1_dim': ACOUSTIC_DIM,  # 74
            'X2_dim': VISUAL_DIM,  # 47
            'inter_dim': args.inter_dim,  # 256
            'drop_prob': args.drop_prob,  # 0.3
            'max_sen_len': args.max_seq_length,
            'B0_dim': args.B0_dim,  # 128
            'B1_dim': args.B1_dim,  # 64
            'p_beta': args.p_beta,
            'p_gamma': args.p_gamma,
            'p_lambda': args.p_lambda,
        }
        
        self.ithp = ITHP_Neuroplastic(ITHP_args)
        
        # Expand: B1_dim -> TEXT_DIM (matches original)
        self.expand = nn.Linear(args.B1_dim, TEXT_DIM)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.pooler = BertPooler(config)
        self.beta_shift = args.beta_shift
        self.config = config

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
    ):
        # Text encoding: (batch, seq, 768)
        embedding_output = self.text_encoder(input_ids)
        x = embedding_output[0]
        
        # ITHP: process full sequence, outputs b1 (batch, seq, B1_dim)
        b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.ithp(x, visual, acoustic)
        
        # Expand b1 back to text dimension
        h_m = self.expand(b1)  # (batch, seq, 768)
        
        # Multimodal fusion
        acoustic_vis_embedding = self.beta_shift * h_m
        sequence_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + x)
        )
        
        # Pool to get sentence representation
        pooled_output = self.pooler(sequence_output)  # (batch, 768)
        
        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1
    
    def get_all_neuroplastic_blocks(self) -> List[NeuroplasticBlock]:
        """Return all NeuroplasticBlocks for scheduler management."""
        return self.ithp.get_neuroplastic_blocks()


class ITHP_DeBertaForSequenceClassification_Neuroplastic(nn.Module):
    """Full model for sequence classification with neuroplastic capability.
    
    Matches original ITHP_DeBertaForSequenceClassification structure.
    """
    
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args
        self.num_labels = config.num_labels
        
        self.deberta_ithp = ITHP_DebertaModel_Neuroplastic(config, args)
        
        # Classifier: same as original
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
    ):
        pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.deberta_ithp(
            input_ids,
            visual,
            acoustic,
        )
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, IB_total
    
    def get_all_neuroplastic_blocks(self) -> List[NeuroplasticBlock]:
        """Return all NeuroplasticBlocks for scheduler management."""
        return self.deberta_ithp.get_all_neuroplastic_blocks()
    
    def grow_skip_connections(self, growth_count: int = 5):
        """Grow top-k skip connections based on co-activation scores."""
        for block in self.get_all_neuroplastic_blocks():
            scores = block.get_co_activation_scores()
            # Mask already active connections
            scores = scores * (~block.skip_mask.bool()).float()
            
            # Get top-k candidates
            flat_scores = scores.flatten()
            _, top_indices = torch.topk(flat_scores, min(growth_count, flat_scores.numel()))
            
            for idx in top_indices:
                out_idx = idx // block.in_dim
                in_idx = idx % block.in_dim
                if scores[out_idx, in_idx] > 0:  # Only grow if non-zero co-activation
                    block.add_skip_connection(in_idx.item(), out_idx.item())
    
    def prune_skip_connections(self, prune_ratio: float = 0.1):
        """Prune weak skip connections based on weight magnitude."""
        for block in self.get_all_neuroplastic_blocks():
            if block.skip_mask.sum() == 0:
                continue
                
            # Get active skip weights
            active_weights = block.skip_weight.data[block.skip_mask]
            if active_weights.numel() == 0:
                continue
            
            # Prune bottom prune_ratio by magnitude
            threshold_idx = max(1, int(active_weights.numel() * prune_ratio))
            sorted_weights, _ = torch.sort(active_weights.abs())
            threshold = sorted_weights[threshold_idx - 1]
            
            # Remove connections below threshold
            weak_mask = (block.skip_weight.data.abs() <= threshold) & block.skip_mask
            weak_indices = weak_mask.nonzero()
            
            for idx in weak_indices:
                out_idx, in_idx = idx[0].item(), idx[1].item()
                block.remove_skip_connection(in_idx, out_idx)


