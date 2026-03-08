"""
Optuna hyperparameter optimization for ITHP + DeBERTa + Neuroplastic.

Usage:
    python optuna_optimize.py --n_trials 100 --study_name ithp_neuroplastic
    
Results stored in: optuna_results.db (SQLite)
"""

import argparse
import os
import sys
import gc
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_neuroplastic import run_training


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function. Minimizes MAE.
    V3: Added weight_decay, label_smoothing, early_stopping for regularization
    """
    # Sample hyperparameters
    config = {
        'dataset': 'mosi',
        'n_epochs': 100,  # More epochs, rely on early stopping
        'train_batch_size': trial.suggest_categorical('train_batch_size', [16, 24, 32]),
        'seed': 128,  # Fixed seed
        
        # Learning rate and warmup (narrower range based on Trial 0: 4.55e-05)
        'learning_rate': trial.suggest_float('learning_rate', 2e-5, 8e-5, log=True),
        'warmup_proportion': trial.suggest_float('warmup_proportion', 0.08, 0.15),
        
        # NEW: Weight decay for L2 regularization (reduce overfitting)
        'weight_decay': trial.suggest_float('weight_decay', 0.001, 0.1, log=True),
        
        # NEW: Label smoothing (noise on labels for regularization)
        'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.15),
        
        # NEW: Early stopping (stop when no improvement)
        'early_stopping_patience': trial.suggest_int('early_stopping_patience', 5, 15),
        
        # ITHP IB coefficients (narrower range based on Trial 0: 0.016)
        'IB_coef': trial.suggest_float('IB_coef', 0.01, 0.05, log=True),
        'p_beta': trial.suggest_float('p_beta', 6, 12),
        'p_gamma': trial.suggest_float('p_gamma', 24, 48),
        
        # Dropout (stronger regularization to reduce overfitting)
        'drop_prob': trial.suggest_float('drop_prob', 0.3, 0.6),
        'p_lambda': trial.suggest_float('p_lambda', 0.3, 0.6),
        
        # Neuroplastic parameters (narrower based on Trial 0)
        'warmup_steps': trial.suggest_int('warmup_steps', 300, 600, step=50),
        'prune_interval': trial.suggest_int('prune_interval', 150, 300, step=50),
        'growth_interval': trial.suggest_int('growth_interval', 150, 300, step=50),
        'max_prune_ratio': trial.suggest_float('max_prune_ratio', 0.03, 0.08),
        'growth_ratio': trial.suggest_float('growth_ratio', 0.03, 0.08),
        
        # Output directory for this trial
        'output_dir': f'./optuna_checkpoints_v3/trial_{trial.number}',
    }
    
    try:
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} - Starting")
        print(f"{'='*60}")
        print(f"Key params: lr={config['learning_rate']:.2e}, IB_coef={config['IB_coef']:.4f}, "
              f"batch={config['train_batch_size']}")
        print(f"Regularization: weight_decay={config['weight_decay']:.4f}, "
              f"label_smooth={config['label_smoothing']:.3f}, early_stop={config['early_stopping_patience']}")
        print(f"Neuroplastic: warmup={config['warmup_steps']}, prune={config['max_prune_ratio']:.2f}, "
              f"grow={config['growth_ratio']:.2f}")
        
        # Run training
        results = run_training(config)
        
        # Convert numpy float32 to Python float (fix JSON serialization error)
        test_mae = float(results['test']['mae'])
        test_acc2 = float(results['test']['acc2'])
        test_acc7 = float(results['test']['acc7'])
        test_f1 = float(results['test']['f1'])
        test_corr = float(results['test']['corr'])
        dev_mae = float(results['dev']['mae'])
        
        # Log intermediate values
        trial.set_user_attr('test_acc2', test_acc2)
        trial.set_user_attr('test_acc7', test_acc7)
        trial.set_user_attr('test_f1', test_f1)
        trial.set_user_attr('test_mae', test_mae)
        trial.set_user_attr('test_corr', test_corr)
        trial.set_user_attr('dev_mae', dev_mae)
        
        # Print trial results
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} - COMPLETED")
        print(f"{'='*60}")
        print(f"  Dev  MAE: {dev_mae:.4f}")
        print(f"  Test MAE: {test_mae:.4f} (optimization target)")
        print(f"  Test Acc2: {test_acc2:.4f}")
        print(f"  Test Acc7: {test_acc7:.4f}")
        print(f"  Test F1:   {test_f1:.4f}")
        print(f"  Test Corr: {test_corr:.4f}")
        print(f"{'='*60}\n", flush=True)
        
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # Minimize MAE (return Python float)
        return test_mae
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        gc.collect()
        torch.cuda.empty_cache()
        raise optuna.TrialPruned()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--study_name', type=str, default='ithp_neuroplastic_v3', help='Optuna study name')
    parser.add_argument('--db_path', type=str, default='optuna_results_v3.db', help='SQLite database path')
    parser.add_argument('--n_jobs', type=int, default=1, help='Parallel jobs (use 1 for GPU)')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs('./optuna_checkpoints_v3', exist_ok=True)
    
    # SQLite storage
    storage = f'sqlite:///{args.db_path}'
    
    # Create or load study
    if args.resume:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=storage,
        )
        print(f"Resuming study '{args.study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction='minimize',  # Minimize MAE
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=True,
        )
        print(f"Created study '{args.study_name}'")
    
    print(f"Storage: {storage}")
    print(f"Target: {args.n_trials} trials")
    print("=" * 60)
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best MAE: {study.best_value:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\nBest trial metrics:")
    for key, value in study.best_trial.user_attrs.items():
        print(f"  {key}: {value:.4f}")
    
    # Save best config
    best_config_path = './optuna_checkpoints_v3/best_config.txt'
    with open(best_config_path, 'w') as f:
        f.write(f"Best MAE: {study.best_value:.4f}\n")
        f.write(f"Trial: {study.best_trial.number}\n\n")
        f.write("Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nMetrics:\n")
        for key, value in study.best_trial.user_attrs.items():
            f.write(f"  {key}: {value:.4f}\n")
    print(f"\nBest config saved to: {best_config_path}")
    
    # Top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value').head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number'])}: MAE={row['value']:.4f}")


if __name__ == '__main__':
    main()
