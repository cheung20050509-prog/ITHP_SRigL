"""
ITHP + DeBERTa model with Neuroplastic training support.
Includes dynamic skip connections for cross-layer shortcuts.
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


class DynamicSkipConnection(nn.Module):
    """Dynamic skip connection that can grow/shrink during training.
    
    Starts empty and connections can be added dynamically.
    """
    
    def __init__(self, in_dim: int, out_dim: int, max_connections: int = 100):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_connections = max_connections
        
        # Sparse representation - initially empty
        self.register_buffer('active_mask', 
                           torch.zeros(out_dim, in_dim, dtype=torch.bool))
        self.skip_weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        
        # Dropout for capacity limiting
        self.dropout = nn.Dropout(0.5)
        
    def add_connection(self, from_idx: int, to_idx: int, init_std: float = 0.01):
        """Add a skip connection"""
        if self.active_mask.sum() >= self.max_connections:
            return False
            
        if not self.active_mask[to_idx, from_idx]:
            self.active_mask[to_idx, from_idx] = True
            self.skip_weight.data[to_idx, from_idx] = init_std * torch.randn(1).item()
            return True
        return False
        
    def remove_connection(self, from_idx: int, to_idx: int):
        """Remove a skip connection"""
        if self.active_mask[to_idx, from_idx]:
            self.active_mask[to_idx, from_idx] = False
            self.skip_weight.data[to_idx, from_idx] = 0
            return True
        return False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - only computes active connections"""
        if not self.active_mask.any():
            # Return zeros if no connections
            batch_size = x.shape[0]
            if x.dim() == 3:  # (batch, seq, dim)
                seq_len = x.shape[1]
                return torch.zeros(batch_size, seq_len, self.out_dim, device=x.device)
            return torch.zeros(batch_size, self.out_dim, device=x.device)
        
        # Apply skip connection with mask
        effective_weight = self.skip_weight * self.active_mask.float()
        out = F.linear(x, effective_weight)
        return self.dropout(out)
    
    @property
    def n_connections(self) -> int:
        return self.active_mask.sum().item()


class ITHP_Neuroplastic(nn.Module):
    """ITHP module with dynamic skip connections.
    
    Structure:
        X0 -> encoder1 -> B0 -> encoder2 -> B1 -> output
                          |                  ^
                          +---(skip_b0)------+
    """
    
    def __init__(self, ITHP_args, enable_skip: bool = True):
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
        
        self.enable_skip = enable_skip

        # Standard ITHP layers
        self.encoder1 = nn.Sequential(
            nn.Linear(self.X0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.B0_dim * 2),
        )

        self.MLP1 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X1_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.B1_dim * 2),
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(self.B1_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X2_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        # Dynamic skip connection: B0 -> B1 
        # (safe skip that doesn't bypass information bottleneck)
        if enable_skip:
            self.skip_b0_to_b1 = DynamicSkipConnection(
                in_dim=self.B0_dim,
                out_dim=self.B1_dim,
                max_connections=50
            )
        else:
            self.skip_b0_to_b1 = None

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
        b0_sample = b0
        h2 = self.encoder2(b0_sample)
        mu2, logvar2 = h2.chunk(2, dim=-1)
        kl_loss_1 = self.kl_loss(mu2, logvar2)
        b1 = self.reparameterise(mu2, logvar2)
        
        # Add skip connection contribution: B0 -> B1
        if self.skip_b0_to_b1 is not None and self.skip_b0_to_b1.n_connections > 0:
            b1 = b1 + self.skip_b0_to_b1(b0)
        
        # MLP2: B1 -> predict visual
        output2 = self.MLP2(b1)
        mse_1 = self.criterion(output2, visual)
        IB1 = kl_loss_1 + self.p_gamma * mse_1
        
        IB_total = IB0 + self.p_lambda * IB1

        return b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1


class ITHP_DebertaModel_Neuroplastic(DebertaV2PreTrainedModel):
    """ITHP + DeBERTa model with neuroplastic training support."""
    
    def __init__(self, config, multimodal_config, enable_skip: bool = True):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM, 
            global_configs.ACOUSTIC_DIM,
            global_configs.VISUAL_DIM
        )
        self.config = config
        self.pooler = BertPooler(config)
        model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.model = model.to(DEVICE)
        
        ITHP_args = {
            'X0_dim': TEXT_DIM,
            'X1_dim': ACOUSTIC_DIM,
            'X2_dim': VISUAL_DIM,
            'B0_dim': multimodal_config.B0_dim,
            'B1_dim': multimodal_config.B1_dim,
            'inter_dim': multimodal_config.inter_dim,
            'max_sen_len': multimodal_config.max_seq_length,
            'drop_prob': multimodal_config.drop_prob,
            'p_beta': multimodal_config.p_beta,
            'p_gamma': multimodal_config.p_gamma,
            'p_lambda': multimodal_config.p_lambda,
        }

        self.ithp = ITHP_Neuroplastic(ITHP_args, enable_skip=enable_skip)

        self.expand = nn.Sequential(
            nn.Linear(multimodal_config.B1_dim, TEXT_DIM),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.init_weights()
        self.beta_shift = multimodal_config.beta_shift

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)

        # ITHP processing
        b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.ithp(
            pooled_output, visual, acoustic
        )

        # Expand B1 back to text dimension
        expanded = self.expand(b1)
        
        # Gated fusion
        em_norm = self.LayerNorm(expanded + pooled_output)

        return em_norm, IB_total


class ITHP_DeBertaForSequenceClassification_Neuroplastic(DebertaV2PreTrainedModel):
    """Full classification model with neuroplastic support."""
    
    def __init__(self, config, multimodal_config, enable_skip: bool = True):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = ITHP_DebertaModel_Neuroplastic(config, multimodal_config, enable_skip)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        outputs, IB_total = self.deberta(
            input_ids,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        logits = self.classifier(outputs)

        return logits, IB_total
    
    def get_skip_connection(self) -> Optional[DynamicSkipConnection]:
        """Get the skip connection module for external management"""
        return self.deberta.ithp.skip_b0_to_b1
