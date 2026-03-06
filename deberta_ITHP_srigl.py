"""
ITHP + DeBERTa model with SRigL sparse training support.
Extends deberta_ITHP.py with sparsification utilities.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
from ITHP import ITHP
import global_configs
from global_configs import DEVICE

from typing import List, Tuple


class ITHP_DebertaModel_SRigL(DebertaV2PreTrainedModel):
    """ITHP + DeBERTa model with SRigL sparsification support.
    
    Same architecture as ITHP_DebertaModel but with:
    - get_sparsifiable_modules() for identifying layers to sparsify
    - Sparse Kaiming initialization option
    """
    
    def __init__(self, config, multimodal_config, sparse_init: bool = False):
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

        self.ithp = ITHP(ITHP_args)  # Note: lowercase for pattern matching

        self.expand = nn.Sequential(
            nn.Linear(multimodal_config.B1_dim, TEXT_DIM),
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.init_weights()
        self.beta_shift = multimodal_config.beta_shift
        
        # Optional sparse initialization
        if sparse_init:
            self._sparse_kaiming_init()

    def _sparse_kaiming_init(self, dense_allocation: float = 0.1):
        """Apply sparse Kaiming initialization to ITHP layers.
        
        Adjusts initialization std for reduced fan-in.
        """
        import math
        for name, module in self.ithp.named_modules():
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                sparse_fan_in = max(1, int(fan_in * dense_allocation))
                std = 1.0 / math.sqrt(sparse_fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Also init expand layer
        for module in self.expand.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                sparse_fan_in = max(1, int(fan_in * dense_allocation))
                std = 1.0 / math.sqrt(sparse_fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_sparsifiable_modules(self) -> List[Tuple[str, nn.Module]]:
        """Return list of (name, module) tuples for sparsifiable layers.
        
        Returns layers in:
        - ithp.encoder1, ithp.encoder2, ithp.MLP1, ithp.MLP2
        - expand layer
        """
        sparsifiable = []
        
        # ITHP layers
        for name, module in self.ithp.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"ithp.{name}"
                sparsifiable.append((full_name, module))
        
        # Expand layer
        for name, module in self.expand.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"expand.{name}"
                sparsifiable.append((full_name, module))
                
        return sparsifiable
    
    def count_parameters(self) -> dict:
        """Count total and sparse-eligible parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        sparse_eligible = 0
        for name, module in self.get_sparsifiable_modules():
            sparse_eligible += module.weight.numel()
            
        return {
            'total': total,
            'trainable': trainable,
            'sparse_eligible': sparse_eligible,
            'dense': trainable - sparse_eligible,
        }

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
    ):
        embedding_output = self.model(input_ids)
        x = embedding_output[0]
        b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.ithp(x, visual, acoustic)
        h_m = self.expand(b1)
        acoustic_vis_embedding = self.beta_shift * h_m
        sequence_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + x)
        )
        pooled_output = self.pooler(sequence_output)

        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1


class ITHP_DeBertaForSequenceClassification_SRigL(DebertaV2PreTrainedModel):
    """ITHP + DeBERTa classifier with SRigL support."""
    
    def __init__(self, config, multimodal_config, sparse_init: bool = False):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = ITHP_DebertaModel_SRigL(config, multimodal_config, sparse_init)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
        
    def get_sparsifiable_modules(self) -> List[Tuple[str, nn.Module]]:
        """Delegate to inner model."""
        return self.deberta.get_sparsifiable_modules()
    
    def count_parameters(self) -> dict:
        """Delegate to inner model."""
        return self.deberta.count_parameters()

    def forward(
            self,
            input_ids,
            visual,
            acoustic,
    ):
        pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.deberta(
            input_ids,
            visual,
            acoustic,
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1
