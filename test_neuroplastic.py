#!/usr/bin/env python
"""
Test script for Neuroplastic ITHP model - matches original ITHP methodology
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, DebertaV2Config

import global_configs
from deberta_ITHP_neuroplastic import ITHP_DeBertaForSequenceClassification_Neuroplastic

# Fix random seeds for reproducibility (same as original ITHP)
def set_seed(seed=128):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(128)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACOUSTIC_DIM = 74
VISUAL_DIM = 47


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="../neuroplastic_checkpoints/best_model.pt")
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    
    # Model args (must match training)
    parser.add_argument('--inter_dim', default=256, type=int)
    parser.add_argument("--drop_prob", default=0.3, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument('--p_lambda', default=0.3, type=float)
    parser.add_argument('--p_beta', default=8, type=float)
    parser.add_argument('--p_gamma', default=32, type=float)
    parser.add_argument('--beta_shift', default=1.0, type=float)
    parser.add_argument('--B0_dim', default=128, type=int)
    parser.add_argument('--B1_dim', default=64, type=int)
    
    return parser.parse_args()


class InputFeatures:
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def prepare_deberta_input(tokens, visual, acoustic, tokenizer, max_seq_length):
    """Prepare input exactly like original ITHP"""
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    # Pad zero vectors for acoustic / visual vectors to account for [CLS] / [SEP] tokens
    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    # Pad inputs
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def convert_to_features(examples, max_seq_length, tokenizer):
    """Convert data to features - exactly like original ITHP"""
    features = []

    for example in tqdm(examples, desc="Converting features"):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        # Check inversion
        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        # Truncate input if necessary
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_deberta_input(
            tokens, visual, acoustic, tokenizer, max_seq_length
        )

        # Check input length
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert acoustic.shape[0] == max_seq_length
        assert visual.shape[0] == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_test_data(args):
    """Prepare test data exactly like original ITHP"""
    global ACOUSTIC_DIM, VISUAL_DIM
    
    # Set dimensions
    if args.dataset == "mosi":
        ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM = 74
        VISUAL_DIM = global_configs.VISUAL_DIM = 47
    else:
        ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM = 74
        VISUAL_DIM = global_configs.VISUAL_DIM = 35
    global_configs.TEXT_DIM = 768
    
    # Load data
    with open(f"datasets/{args.dataset}.pkl", "rb") as f:
        data = pickle.load(f)
    
    test_data = data["test"]
    
    # Get tokenizer and convert to features
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model)
    features = convert_to_features(test_data, args.max_seq_length, tokenizer)
    
    # Create dataset
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_labels = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)
    
    dataset = TensorDataset(all_input_ids, all_visual, all_acoustic, all_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    return loader


def test_epoch(model, test_dataloader):
    """Test epoch - exactly like original ITHP"""
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, label_ids = batch
            # Use squeeze like train_neuroplastic.py (keep sequence dimension)
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            # Normalize like original ITHP
            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            # Forward pass
            logits, _ = model(input_ids, visual_norm, acoustic_norm)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
            
            # Handle single item case
            if not isinstance(logits, list):
                logits = [logits]
            if not isinstance(label_ids, list):
                label_ids = [label_ids]

            preds.extend(logits)
            labels.extend(label_ids)

    preds = np.array(preds)
    labels = np.array(labels)

    return preds, labels


def test_score_model(model, test_dataloader, use_zero=False):
    """Compute test scores - exactly like original ITHP"""
    preds, y_test = test_epoch(model, test_dataloader)
    
    # Non-zero only (exclude neutral sentiment)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds_nz = preds[non_zeros]
    y_test_nz = y_test[non_zeros]

    # MAE and Correlation (on raw predictions)
    mae = np.mean(np.absolute(preds_nz - y_test_nz))
    corr = np.corrcoef(preds_nz, y_test_nz)[0][1]

    # Binary classification (positive/negative) - ACC2
    preds_binary = preds_nz >= 0
    y_test_binary = y_test_nz >= 0

    f_score = f1_score(y_test_binary, preds_binary, average="weighted")
    acc2 = accuracy_score(y_test_binary, preds_binary)
    
    # 7-class accuracy - ACC7 (round to nearest integer in [-3, 3])
    preds_7 = np.clip(np.round(preds), -3, 3).astype(int)
    labels_7 = np.clip(np.round(y_test), -3, 3).astype(int)
    acc7 = accuracy_score(labels_7, preds_7)

    return acc2, acc7, mae, corr, f_score


def count_connections(model):
    """Count active connections in neuroplastic layers only (not DeBERTa backbone)"""
    total_params = 0
    sparse_params = 0
    layer_stats = []
    
    # Only count layers that were subject to neuroplastic pruning/growth
    neuroplastic_prefixes = ['ithp', 'expand', 'classifier', 'skip']
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            # Check if this is a neuroplastic layer
            is_neuroplastic = any(prefix in name.lower() for prefix in neuroplastic_prefixes)
            
            if is_neuroplastic:
                total = param.numel()
                active = (param != 0).sum().item()
                total_params += total
                sparse_params += active
                layer_stats.append((name, active, total, 100*active/total))
    
    return sparse_params, total_params, layer_stats


def main():
    args = get_args()
    
    print("=" * 60)
    print("Neuroplastic ITHP - Test Evaluation")
    print("(Following Original ITHP Methodology)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load data
    print("\nPreparing test data...")
    test_loader = prepare_test_data(args)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    config = DebertaV2Config.from_pretrained(args.model)
    config.num_labels = 1  # Regression task
    model = ITHP_DeBertaForSequenceClassification_Neuroplastic(
        config, args
    ).to(DEVICE)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    # Filter out cache buffers that may have different shapes (they're only for training)
    state_dict = ckpt['model_state_dict']
    filtered_state_dict = {k: v for k, v in state_dict.items() 
                          if 'input_cache' not in k and 'output_cache' not in k}
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"  Loaded from epoch {ckpt.get('epoch', 'N/A')}")
    if 'best_acc' in ckpt:
        print(f"  Best validation acc: {ckpt['best_acc']:.4f}")
    
    # Count connections
    sparse_params, total_params, layer_stats = count_connections(model)
    print(f"\nModel sparsity:")
    print(f"  Total: {sparse_params:,}/{total_params:,} ({100*sparse_params/total_params:.1f}%)")
    for name, active, total, pct in layer_stats:
        short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
        print(f"    {short_name}: {active:,}/{total:,} ({pct:.1f}%)")
    
    # Test
    print("\nRunning test evaluation...")
    acc2, acc7, mae, corr, f_score = test_score_model(model, test_loader)
    
    print("\n" + "=" * 60)
    print("TEST RESULTS (Original ITHP Methodology)")
    print("=" * 60)
    print(f"  Accuracy (Acc-2): {acc2*100:.2f}%")
    print(f"  Accuracy (Acc-7): {acc7*100:.2f}%")
    print(f"  MAE:              {mae:.4f}")
    print(f"  Correlation:      {corr:.4f}")
    print(f"  F1 Score:         {f_score:.4f}")
    print("=" * 60)
    
    # Compare with baseline
    print("\nComparison with Original ITHP (paper config):")
    print("-" * 50)
    print(f"| Metric      | Neuroplastic | Original ITHP |")
    print(f"|-------------|--------------|---------------|")
    print(f"| Acc-2       | {acc2*100:.2f}%       | 84.27%        |")
    print(f"| Acc-7       | {acc7*100:.2f}%       | 46.49%        |")
    print(f"| MAE         | {mae:.4f}       | 0.8049        |")
    print(f"| Correlation | {corr:.4f}       | 0.8117        |")
    print(f"| F1 Score    | {f_score:.4f}       | 0.8432        |")
    print(f"| Connections | {100*sparse_params/total_params:.1f}%        | 100%          |")
    print("-" * 50)
    
    # Save results
    results = {
        'acc2': acc2,
        'acc7': acc7,
        'mae': mae,
        'correlation': corr,
        'f1_score': f_score,
        'sparsity': sparse_params / total_params,
        'checkpoint': args.checkpoint,
        'dataset': args.dataset
    }
    
    import json
    result_file = args.checkpoint.replace('.pt', '_test_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()
