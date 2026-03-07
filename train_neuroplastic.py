"""
Training script for ITHP + DeBERTa with Neuroplastic training.
Implements activity-based pruning and Hebbian growth.
"""

import argparse
import os
import sys
import random
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW

from deberta_ITHP_neuroplastic import ITHP_DeBertaForSequenceClassification_Neuroplastic
from neuroplastic_scheduler import NeuroplasticScheduler
import global_configs
from global_configs import DEVICE


def get_args():
    parser = argparse.ArgumentParser()
    # Model args
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--max_seq_length", type=int, default=50)
    
    # Training args
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--dev_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--dropout_prob", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_step", type=int, default=1)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=128)
    
    # ITHP args
    parser.add_argument('--inter_dim', default=256, type=int)
    parser.add_argument("--drop_prob", default=0.3, type=float)
    parser.add_argument('--p_lambda', default=0.3, type=float)
    parser.add_argument('--p_beta', default=8, type=float)
    parser.add_argument('--p_gamma', default=32, type=float)
    parser.add_argument('--beta_shift', default=1.0, type=float)
    parser.add_argument('--IB_coef', default=10, type=float)
    parser.add_argument('--B0_dim', default=128, type=int)
    parser.add_argument('--B1_dim', default=64, type=int)
    
    # Neuroplastic args
    parser.add_argument("--warmup_steps", type=int, default=500, help="Steps before topology changes start")
    parser.add_argument("--prune_interval", type=int, default=200)
    parser.add_argument("--growth_interval", type=int, default=200)
    parser.add_argument("--prune_threshold", type=float, default=0.001)
    parser.add_argument("--max_density", type=float, default=1.5)
    parser.add_argument("--max_prune_ratio", type=float, default=0.05, help="Max fraction to prune per update")
    parser.add_argument("--growth_ratio", type=float, default=0.05, help="Max fraction to grow per update")
    parser.add_argument("--no_neuroplastic", action="store_true", default=False)
    
    # Checkpoint args
    parser.add_argument("--output_dir", type=str, default="./neuroplastic_checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override output_dir")
    parser.add_argument("--save_every", type=int, default=5)
    
    return parser.parse_args()


ACOUSTIC_DIM = None
VISUAL_DIM = None
TEXT_DIM = None


class InputFeatures:
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
    for example in examples:
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        aligned_visual = [visual[inv_idx, :] for inv_idx in inversions]
        aligned_audio = [acoustic[inv_idx, :] for inv_idx in inversions]

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_deberta_input(
            tokens, visual, acoustic, tokenizer, max_seq_length
        )

        features.append(InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            visual=visual,
            acoustic=acoustic,
            label_id=label_id,
        ))
    return features


def get_appropriate_dataset(data, args):
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)
    return TensorDataset(all_input_ids, all_visual, all_acoustic, all_label_ids)


def set_up_data_loader(args):
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_dataset = get_appropriate_dataset(data["train"], args)
    dev_dataset = get_appropriate_dataset(data["dev"], args)
    test_dataset = get_appropriate_dataset(data["test"], args)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


def prep_for_training(args, num_train_steps):
    from transformers import DebertaV2Config
    
    config = DebertaV2Config.from_pretrained(args.model)
    config.num_labels = 1
    
    model = ITHP_DeBertaForSequenceClassification_Neuroplastic(config, args)
    model = model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * args.warmup_proportion),
        num_training_steps=num_train_steps,
    )
    
    # Neuroplastic scheduler
    np_scheduler = None
    if not args.no_neuroplastic:
        np_config = {
            'warmup_steps': args.warmup_steps,
            'prune_interval': args.prune_interval,
            'growth_interval': args.growth_interval,
            'prune_threshold': args.prune_threshold,
            'max_density': args.max_density,
            'max_prune_ratio': args.max_prune_ratio,
            'growth_ratio': args.growth_ratio,
        }
        np_scheduler = NeuroplasticScheduler(
            model, optimizer, num_train_steps, np_config
        )
    
    return model, optimizer, scheduler, np_scheduler


def train_epoch(model, train_loader, optimizer, scheduler, np_scheduler, args):
    model.train()
    tr_loss = 0
    ib_loss_total = 0
    n_batches = 0
    total_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        batch = tuple(t.to(DEVICE) for t in batch)
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        
        # Match original preprocessing
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        
        # Normalize (match original)
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)
        
        logits, IB_total = model(input_ids, visual_norm, acoustic_norm)
        logits = logits.view(-1)  # Flatten to 1D
        label_ids = label_ids.view(-1)
        
        loss_fct = MSELoss()
        mse_loss = loss_fct(logits, label_ids)
        loss = mse_loss + args.IB_coef * IB_total
        
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Neuroplastic step
        if np_scheduler is not None:
            np_scheduler.step(loss.item(), IB_total.item())
        
        optimizer.zero_grad()
        
        tr_loss += loss.item()
        ib_loss_total += IB_total.item()
        n_batches += 1
        
        # Print progress every 20 batches (flush immediately)
        if n_batches % 20 == 0 or n_batches == total_batches:
            print(f"  [{n_batches}/{total_batches}] loss={tr_loss/n_batches:.4f}, IB={ib_loss_total/n_batches:.4f}", flush=True)
    
    return tr_loss / n_batches


@torch.no_grad()
def evaluate(model, data_loader):
    model.eval()
    preds, labels = [], []
    
    for batch in data_loader:
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        
        # Match original preprocessing
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        
        # Normalize (match original)
        visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)
        
        logits, _ = model(input_ids, visual_norm, acoustic_norm)
        logits = logits.squeeze(-1)
        
        preds.extend(logits.cpu().numpy().flatten())
        labels.extend(label_ids.cpu().numpy().flatten())
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    # Filter out NaN
    valid_mask = ~(np.isnan(preds) | np.isnan(labels))
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    
    if len(preds) == 0:
        return {'acc2': 0, 'acc7': 0, 'f1': 0, 'mae': float('inf'), 'corr': 0}
    
    # Acc-2: Binary accuracy (exclude zero sentiment)
    non_zero_mask = labels != 0
    if non_zero_mask.sum() > 0:
        preds_nz = preds[non_zero_mask]
        labels_nz = labels[non_zero_mask]
        preds_binary = (preds_nz > 0).astype(int)
        labels_binary = (labels_nz > 0).astype(int)
        acc2 = accuracy_score(labels_binary, preds_binary)
        f1 = f1_score(labels_binary, preds_binary, average='weighted')
    else:
        acc2, f1 = 0.5, 0.5
    
    # Acc-7: 7-class accuracy (round to nearest integer in [-3, 3])
    preds_7 = np.clip(np.round(preds), -3, 3).astype(int)
    labels_7 = np.clip(np.round(labels), -3, 3).astype(int)
    acc7 = accuracy_score(labels_7, preds_7)
    
    # MAE and correlation
    mae = np.mean(np.abs(preds - labels))
    corr = np.corrcoef(preds, labels)[0, 1] if len(preds) > 1 else 0
    
    return {'acc2': acc2, 'acc7': acc7, 'f1': f1, 'mae': mae, 'corr': corr}


def main():
    args = get_args()
    
    # Set global dimensions
    global ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM
    if args.dataset == "mosi":
        ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM = 74
        VISUAL_DIM = global_configs.VISUAL_DIM = 47
    else:
        ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM = 74
        VISUAL_DIM = global_configs.VISUAL_DIM = 35
    TEXT_DIM = global_configs.TEXT_DIM = 768
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output dir (allow override)
    if args.checkpoint_dir:
        args.output_dir = args.checkpoint_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_loader, dev_loader, test_loader = set_up_data_loader(args)
    
    num_train_steps = len(train_loader) * args.n_epochs
    
    # Setup model
    model, optimizer, scheduler, np_scheduler = prep_for_training(args, num_train_steps)
    
    print("=" * 60)
    print("ITHP + DeBERTa + Neuroplastic Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Neuroplastic: {'Enabled' if np_scheduler else 'Disabled'}")
    print(f"Synaptic targets: ITHP + DeBERTa FFN")
    print("=" * 60)
    
    best_acc = 0
    
    for epoch in range(1, args.n_epochs + 1):
        print(f"\nEpoch {epoch}/{args.n_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, np_scheduler, args)
        
        dev_metrics = evaluate(model, dev_loader)
        print(f"Dev: Acc2={dev_metrics['acc2']:.4f}, Acc7={dev_metrics['acc7']:.4f}, F1={dev_metrics['f1']:.4f}, "
              f"MAE={dev_metrics['mae']:.4f}, Corr={dev_metrics['corr']:.4f}")
        
        # Print neuroplastic stats
        if np_scheduler is not None:
            np_scheduler.print_stats()
        
        # Save best model (based on Acc2)
        if dev_metrics['acc2'] > best_acc:
            best_acc = dev_metrics['acc2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> Saved best model (acc2={best_acc:.4f})")
        
        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    # Load best model
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader)
    print(f"Test: Acc2={test_metrics['acc2']:.4f}, Acc7={test_metrics['acc7']:.4f}, F1={test_metrics['f1']:.4f}, "
          f"MAE={test_metrics['mae']:.4f}, Corr={test_metrics['corr']:.4f}")
    
    if np_scheduler is not None:
        np_scheduler.print_stats()


if __name__ == "__main__":
    main()
