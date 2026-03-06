"""
Test script for ITHP + DeBERTa + SRigL models.
Loads a checkpoint and evaluates on test set.

Usage:
    python test_srigl.py --checkpoint ./srigl_checkpoints/best_model.pt --dataset mosi
    python test_srigl.py --checkpoint ./srigl_checkpoints/best_model.pt --dataset mosei
"""

import argparse
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import DebertaV2Tokenizer
from deberta_ITHP_srigl import ITHP_DeBertaForSequenceClassification_SRigL
from rigl_scheduler import DeBertaRigLScheduler
import global_configs
from global_configs import DEVICE


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--use_zero", action="store_true", default=False,
                        help="Include zero-sentiment samples in evaluation")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Print detailed results per class")
    return parser.parse_args()


# Will be set after parsing args
ACOUSTIC_DIM = None
VISUAL_DIM = None
TEXT_DIM = None


class InputFeatures(object):
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def prepare_deberta_input(tokens, visual, acoustic, tokenizer, max_seq_length):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

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
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []
        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_deberta_input(
            tokens, visual, acoustic, tokenizer, max_seq_length
        )

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


def get_test_dataloader(args):
    """Load test dataset."""
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    test_data = data["test"]
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(test_data, args.max_seq_length, tokenizer)
    
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    test_dataset = TensorDataset(all_input_ids, all_visual, all_acoustic, all_label_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    return test_dataloader


def test_epoch(model, test_dataloader):
    """Get predictions on test set."""
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
                input_ids, visual_norm, acoustic_norm
            )

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            preds.extend(np.squeeze(logits).tolist())
            labels.extend(np.squeeze(label_ids).tolist())

    return np.array(preds), np.array(labels)


def compute_metrics(preds, labels, use_zero=False, verbose=False):
    """Compute comprehensive evaluation metrics."""
    # Filter non-zero if needed
    if not use_zero:
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0])
        preds = preds[non_zeros]
        labels = labels[non_zeros]
        print(f"Samples after filtering zero sentiment: {len(labels)}")
    
    # Regression metrics
    mae = np.mean(np.absolute(preds - labels))
    mse = np.mean((preds - labels) ** 2)
    corr = np.corrcoef(preds, labels)[0][1]
    
    # Binary classification (positive/negative)
    preds_binary = (preds >= 0).astype(int)
    labels_binary = (labels >= 0).astype(int)
    
    acc_2 = accuracy_score(labels_binary, preds_binary)
    f1_2 = f1_score(labels_binary, preds_binary, average="weighted")
    
    # 7-class classification (for MOSI/MOSEI: -3 to +3)
    preds_7 = np.clip(np.round(preds), -3, 3).astype(int)
    labels_7 = np.clip(np.round(labels), -3, 3).astype(int)
    acc_7 = accuracy_score(labels_7, preds_7)
    f1_7 = f1_score(labels_7, preds_7, average="weighted")
    
    results = {
        'mae': mae,
        'mse': mse,
        'corr': corr,
        'acc_2': acc_2,
        'f1_2': f1_2,
        'acc_7': acc_7,
        'f1_7': f1_7,
        'n_samples': len(labels),
    }
    
    if verbose:
        print("\nConfusion Matrix (Binary):")
        print(confusion_matrix(labels_binary, preds_binary))
        print("\nClassification Report (Binary):")
        print(classification_report(labels_binary, preds_binary, target_names=['Negative', 'Positive']))
    
    return results


def load_model_from_checkpoint(checkpoint_path, args):
    """Load model and optionally RigL scheduler from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Get saved args
    saved_args = checkpoint.get('args', {})
    
    # Create config namespace
    class Config:
        pass
    config = Config()
    for k, v in saved_args.items():
        setattr(config, k, v)
    
    # Fill in defaults if missing
    for attr in ['B0_dim', 'B1_dim', 'inter_dim', 'max_seq_length', 'drop_prob', 
                 'p_beta', 'p_gamma', 'p_lambda', 'beta_shift', 'dropout_prob']:
        if not hasattr(config, attr):
            if attr == 'B0_dim': setattr(config, attr, 128)
            elif attr == 'B1_dim': setattr(config, attr, 64)
            elif attr == 'inter_dim': setattr(config, attr, 256)
            elif attr == 'max_seq_length': setattr(config, attr, 50)
            elif attr == 'drop_prob': setattr(config, attr, 0.3)
            elif attr == 'dropout_prob': setattr(config, attr, 0.5)
            else: setattr(config, attr, 1.0)
    
    # Create model
    model = ITHP_DeBertaForSequenceClassification_SRigL.from_pretrained(
        args.model,
        multimodal_config=config,
        num_labels=1,
        sparse_init=False,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    # Load sparsity stats if available
    if 'rigl_state_dict' in checkpoint:
        rigl_state = checkpoint['rigl_state_dict']
        total_params = sum(m.numel() for m in rigl_state['masks'])
        total_nonzero = sum(m.sum().item() for m in rigl_state['masks'])
        sparsity = 1 - total_nonzero / total_params if total_params > 0 else 0
        print(f"Model sparsity: {sparsity*100:.2f}%")
        print(f"RigL steps completed: {rigl_state['rigl_steps']}")
    
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model


def main():
    global ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM
    
    args = get_args()
    
    # Set dataset config
    global_configs.set_dataset_config(args.dataset)
    ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
    VISUAL_DIM = global_configs.VISUAL_DIM
    TEXT_DIM = global_configs.TEXT_DIM
    
    print("\n" + "="*60)
    print("ITHP + DeBERTa + SRigL Testing")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Use zero sentiment: {args.use_zero}")
    print("="*60 + "\n")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args)
    
    # Load test data
    test_dataloader = get_test_dataloader(args)
    
    # Run test
    preds, labels = test_epoch(model, test_dataloader)
    
    # Compute metrics
    results = compute_metrics(preds, labels, use_zero=args.use_zero, verbose=args.verbose)
    
    # Print results
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"  Samples: {results['n_samples']}")
    print(f"\nRegression Metrics:")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  Correlation: {results['corr']:.4f}")
    print(f"\nBinary Classification (Pos/Neg):")
    print(f"  Accuracy: {results['acc_2']*100:.2f}%")
    print(f"  F1 Score: {results['f1_2']:.4f}")
    print(f"\n7-class Classification (-3 to +3):")
    print(f"  Accuracy: {results['acc_7']*100:.2f}%")
    print(f"  F1 Score: {results['f1_7']:.4f}")
    print("="*60 + "\n")
    
    return results


if __name__ == "__main__":
    main()
