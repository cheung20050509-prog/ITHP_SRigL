"""
Training script for ITHP + DeBERTa with SRigL sparse training.
Based on train.py with SRigL scheduler integration.
"""

import argparse
import os
import sys
import random
import pickle
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW

# Local imports
from deberta_ITHP_srigl import ITHP_DeBertaForSequenceClassification_SRigL
from rigl_scheduler import DeBertaRigLScheduler
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
    
    # SRigL args
    parser.add_argument("--dense_allocation", type=float, default=0.1,
                        help="Fraction of weights to keep (1 - sparsity). 0.1 = 90% sparse")
    parser.add_argument("--delta", type=int, default=100,
                        help="Steps between topology updates")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Initial fraction of weights to update each step")
    parser.add_argument("--t_end_fraction", type=float, default=0.75,
                        help="Fraction of training to stop topology updates")
    parser.add_argument("--const_fan_in", action="store_true", default=True,
                        help="Use constant fan-in constraint")
    parser.add_argument("--sparse_init", action="store_true", default=False,
                        help="Use sparse Kaiming initialization")
    parser.add_argument("--no_srigl", action="store_true", default=False,
                        help="Disable SRigL (dense baseline)")
    
    # Checkpoint args
    parser.add_argument("--output_dir", type=str, default="./srigl_checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    
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


def get_appropriate_dataset(data, args):
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(data, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual, all_acoustic, all_label_ids)
    return dataset


def set_up_data_loader(args):
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_dataset = get_appropriate_dataset(data["train"], args)
    dev_dataset = get_appropriate_dataset(data["dev"], args)
    test_dataset = get_appropriate_dataset(data["test"], args)

    num_train_optimization_steps = (
        int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step)
        * args.n_epochs
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps


def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")


def prep_for_training(args, num_train_optimization_steps: int):
    """Initialize model, optimizer, scheduler, and optionally RigL scheduler."""
    model = ITHP_DeBertaForSequenceClassification_SRigL.from_pretrained(
        args.model, 
        multimodal_config=args, 
        num_labels=1,
        sparse_init=args.sparse_init,
    )
    model.to(DEVICE)
    
    # Print parameter counts
    param_counts = model.count_parameters()
    print(f"[Model] Total params: {param_counts['total']:,}")
    print(f"[Model] Trainable params: {param_counts['trainable']:,}")
    print(f"[Model] Sparse-eligible params: {param_counts['sparse_eligible']:,}")

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
        num_training_steps=num_train_optimization_steps,
    )
    
    # Initialize RigL scheduler
    rigl_scheduler = None
    if not args.no_srigl:
        T_end = int(args.t_end_fraction * num_train_optimization_steps)
        rigl_scheduler = DeBertaRigLScheduler(
            model=model,
            optimizer=optimizer,
            dense_allocation=args.dense_allocation,
            delta=args.delta,
            alpha=args.alpha,
            T_end=T_end,
            grad_accumulation_n=args.gradient_accumulation_step,
            const_fan_in=args.const_fan_in,
            device=str(DEVICE),
        )
        print(f"[SRigL] T_end: {T_end} steps")
    else:
        print("[SRigL] Disabled (dense baseline)")
    
    return model, optimizer, scheduler, rigl_scheduler


def train_epoch(model, train_dataloader, optimizer, scheduler, rigl_scheduler, args):
    """Train for one epoch with SRigL updates."""
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())
        
        logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
            input_ids, visual_norm, acoustic_norm
        )
        
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1)) + 2 / (args.p_beta + args.p_gamma) * IB_loss

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()  # RigL wraps this to apply masks
            scheduler.step()
            
            # RigL topology update check
            if rigl_scheduler is not None:
                rigl_scheduler.step()
            
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model, dev_dataloader, args):
    """Evaluate on validation set."""
    model.eval()
    dev_loss = 0
    nb_dev_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Validation"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min())
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min())

            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1 = model(
                input_ids, visual_norm, acoustic_norm
            )
            
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


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


def test_score_model(model, test_dataloader, use_zero=False):
    """Compute test metrics."""
    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds_binary = preds >= 0
    y_test_binary = y_test >= 0

    f_score = f1_score(y_test_binary, preds_binary, average="weighted")
    acc = accuracy_score(y_test_binary, preds_binary)

    return acc, mae, corr, f_score


def save_checkpoint(model, optimizer, scheduler, rigl_scheduler, epoch, args, path):
    """Save training checkpoint with SRigL state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
    }
    if rigl_scheduler is not None:
        checkpoint['rigl_state_dict'] = rigl_scheduler.state_dict()
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(model, optimizer, scheduler, rigl_scheduler, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if rigl_scheduler is not None and 'rigl_state_dict' in checkpoint:
        rigl_scheduler.load_state_dict(checkpoint['rigl_state_dict'])
    print(f"[Checkpoint] Loaded from {path}, epoch {checkpoint['epoch']}")
    return checkpoint['epoch']


def train(model, train_dataloader, dev_dataloader, test_dataloader, 
          optimizer, scheduler, rigl_scheduler, args):
    """Main training loop."""
    start_epoch = 0
    
    # Resume from checkpoint
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, rigl_scheduler, args.resume)
    
    best_valid_loss = float('inf')
    results = []
    
    for epoch in range(start_epoch, args.n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.n_epochs}")
        print(f"{'='*60}")
        
        # Log sparsity stats
        if rigl_scheduler is not None:
            print(rigl_scheduler)
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, rigl_scheduler, args)
        
        # Validate
        valid_loss = eval_epoch(model, dev_dataloader, args)
        
        print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, rigl_scheduler, epoch + 1, args, ckpt_path)
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_path = os.path.join(args.output_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, rigl_scheduler, epoch + 1, args, best_path)
        
        # Test on final epoch
        if epoch == args.n_epochs - 1:
            test_acc, test_mae, test_corr, test_f1 = test_score_model(model, test_dataloader)
            print(f"\nFinal Test Results:")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  MAE: {test_mae:.4f}")
            print(f"  Correlation: {test_corr:.4f}")
            print(f"  F1 Score: {test_f1:.4f}")
            
            # Log final sparsity
            if rigl_scheduler is not None:
                stats = rigl_scheduler.get_sparsity_stats()
                print(f"\nFinal Sparsity: {stats['total']['sparsity']*100:.2f}%")
            
            results = {
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_acc': test_acc,
                'test_mae': test_mae,
                'test_corr': test_corr,
                'test_f1': test_f1,
            }
    
    return results


def main():
    global ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM
    
    args = get_args()
    
    # Set dataset config
    global_configs.set_dataset_config(args.dataset)
    ACOUSTIC_DIM = global_configs.ACOUSTIC_DIM
    VISUAL_DIM = global_configs.VISUAL_DIM
    TEXT_DIM = global_configs.TEXT_DIM
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    set_random_seed(args.seed)
    
    # Print config
    print("\n" + "="*60)
    print("ITHP + DeBERTa + SRigL Training")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Dense allocation: {args.dense_allocation} ({(1-args.dense_allocation)*100:.0f}% sparse)")
    print(f"Delta (topology update interval): {args.delta}")
    print(f"Alpha (initial drop fraction): {args.alpha}")
    print(f"T_end fraction: {args.t_end_fraction}")
    print(f"Output dir: {args.output_dir}")
    print("="*60 + "\n")
    
    # Setup data
    train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps = set_up_data_loader(args)
    print(f"Total optimization steps: {num_train_optimization_steps}")
    
    # Setup model and training
    model, optimizer, scheduler, rigl_scheduler = prep_for_training(args, num_train_optimization_steps)
    
    # Train
    results = train(model, train_dataloader, dev_dataloader, test_dataloader,
                   optimizer, scheduler, rigl_scheduler, args)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model.pt")
    save_checkpoint(model, optimizer, scheduler, rigl_scheduler, args.n_epochs, args, final_path)
    
    print("\nTraining complete!")
    return results


if __name__ == "__main__":
    main()
