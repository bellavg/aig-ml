import argparse
import optuna
import os
import torch
import json
import numpy as np
import random
from datetime import datetime
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.optim as optim
from optuna.trial import TrialState

# Import your custom modules
from aig_dataset import AIGDataset
from model import AIGTransformer
from train import train_epoch
from test import validate


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed} for reproducibility")


def parse_args():
    """Parse command line arguments for HPO."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for AIG Transformer Model")

    # Dataset parameters
    parser.add_argument('--file_path', type=str, default=None,
                        help="Path to the original dataset file (only needed if processed data doesn't exist)")
    parser.add_argument('--root', type=str, default="",
                        help="Root directory containing the processed data.pt file")
    parser.add_argument('--processed_file', type=str, default="data.pt",
                        help="Name of the processed PyG dataset file")
    parser.add_argument('--num_graphs', type=int, default=24,
                        help="Number of graphs to load from the dataset")
    parser.add_argument('--max_nodes', type=int, default=120,
                        help="Maximum number of nodes in each graph")

    # Masking parameters
    parser.add_argument('--mask_prob', type=float, default=0.20,
                        help="Masking probability for nodes/gates")
    parser.add_argument('--mask_mode', type=str, default="node_feature",
                        choices=["node_feature", "edge_feature", "node_existence", "edge_existence", "removal"],
                        help="Masking mode to use for training")

    # For backward compatibility
    parser.add_argument('--gate_masking', action='store_true',
                        help="Enable gate masking (deprecated, use --mask_mode=node_existence)")

    # HPO parameters
    parser.add_argument('--n_trials', type=int, default=50,
                        help="Number of HPO trials to run")
    parser.add_argument('--hpo_epochs', type=int, default=10,
                        help="Number of epochs to train each trial")
    parser.add_argument('--study_name', type=str, default=None,
                        help="Name for the HPO study (default: auto-generated)")
    parser.add_argument('--optimize_batch_size', action='store_true',
                        help="Also optimize batch size during HPO")

    # Training parameters
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="Ratio of dataset used for validation")
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help="Ratio of dataset used for testing")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")

    # Pretrained model loading
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help="Path to pretrained model to continue training from")

    # Output directories
    parser.add_argument('--results_dir', type=str, default="./hpo_results/",
                        help="Directory to save HPO results")

    return parser.parse_args()


class HPOObjective:
    """Objective function for HPO."""

    def __init__(self, args, train_loader, val_loader, test_loader, input_features):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.node_features = input_features['node_features']
        self.edge_features = input_features['edge_features']
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.output_dir = os.path.join(args.results_dir, args.study_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def __call__(self, trial):
        """Run a single HPO trial."""
        # Hyperparameter suggestions


        num_layers = trial.suggest_int('num_layers', 1, 8)
        num_heads = trial.suggest_categorical("num_heads",[2, 4, 8, 16])
        hidden_dim_choices = [d for d in [32, 64, 128, 256, 512, 1024] if d % num_heads == 0]
        hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_choices)

        dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

        # Optionally optimize batch size
        if self.args.optimize_batch_size:
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
            if batch_size != self.args.batch_size:
                # Recreate data loaders with new batch size
                self.train_loader = DataLoader(self.train_loader.dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
                self.val_loader = DataLoader(self.val_loader.dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
                self.test_loader = DataLoader(self.test_loader.dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

        # Create model with trial hyperparameters
        model = AIGTransformer(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_nodes=self.args.max_nodes
        )

        # Load pretrained model if specified
        if self.args.pretrained_model and os.path.exists(self.args.pretrained_model):
            print(f"Loading pretrained model from {self.args.pretrained_model}")
            try:
                state_dict = torch.load(self.args.pretrained_model, map_location=self.device)
                model_loaded = model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded pretrained model")
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Starting with randomly initialized model instead")

        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        early_stop_count = 0
        early_stop_patience = 3  # For HPO, use lower patience

        # Training loop for this trial
        for epoch in range(self.args.hpo_epochs):
            # Create args object for the epoch with trial hyperparameters
            epoch_args = argparse.Namespace(
                **vars(self.args),
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                lr=lr
            )

            # Train for one epoch
            train_losses = train_epoch(epoch_args, model, self.train_loader, optimizer, self.device)

            # Validate
            val_losses, val_metrics = validate(epoch_args, model, self.val_loader, self.device)

            # Get validation loss - use total_loss if available, else node_loss
            current_val_loss = val_losses.get('total_loss', val_losses.get('node_loss', float('inf')))

            # Report to Optuna for pruning
            trial.report(current_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Track best validation loss
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                early_stop_count = 0
            else:
                early_stop_count += 1

            # Early stopping for efficiency
            if early_stop_count >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}/{self.args.hpo_epochs}")
                break

            epoch_log = (f"Trial {trial.number}, Epoch {epoch + 1}/{self.args.hpo_epochs}, "
                         f"Train Loss: {train_losses['total_loss']:.4f}, "
                         f"Val Loss: {current_val_loss:.4f}")
            print(epoch_log)

        # Evaluate on test set
        test_losses, test_metrics = validate(epoch_args, model, self.test_loader, self.device)
        test_loss = test_losses.get('total_loss', test_losses.get('node_loss', float('inf')))

        # Save trial results
        trial_info = {
            'trial_number': trial.number,
            'params': trial.params,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_metrics': test_metrics
        }

        with open(os.path.join(self.output_dir, f"trial_{trial.number}.json"), 'w') as f:
            json.dump(trial_info, f, indent=4)

        return best_val_loss


def run_hpo(args):
    """Run the full HPO process."""
    # Set seed for reproducibility
    set_seed(args.seed)

    # Handle backward compatibility for gate_masking
    if args.gate_masking:
        args.mask_mode = "node_existence"
        print("Warning: --gate_masking is deprecated, use --mask_mode=node_existence instead")

    # Set up study name if not provided
    if args.study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.study_name = f"hpo_{args.mask_mode}_mp{int(args.mask_prob * 100):02d}_{timestamp}"

    # Directory for this HPO run
    output_dir = os.path.join(args.results_dir, args.study_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save HPO configuration
    config_path = os.path.join(output_dir, "hpo_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Load dataset
    print(f"Loading dataset from root directory: {args.root}")
    full_dataset = AIGDataset(
        file_path=args.file_path,
        num_graphs=args.num_graphs,
        root=args.root,
        processed_file=args.processed_file
    )
    print(f"Loaded dataset with {len(full_dataset)} graphs")

    # Split dataset
    total = len(full_dataset)
    test_size = int(total * args.test_ratio)
    val_size = int(total * args.val_ratio)
    train_size = total - test_size - val_size

    print(f"Splitting dataset: Train={train_size}, Val={val_size}, Test={test_size}")
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Default batch size for initial data loaders
    args.batch_size = 32

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Get feature dimensions
    sample = full_dataset[0]
    input_features = {
        'node_features': sample.x.size(1),
        'edge_features': sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') else 0
    }

    print(f"Model input features: nodes={input_features['node_features']}, edges={input_features['edge_features']}")
    print(f"Using {args.mask_mode} masking at {args.mask_prob * 100:.1f}% probability")

    if args.pretrained_model:
        pretrained_source = os.path.basename(args.pretrained_model).split('_')[0]
        print(f"Using pretrained model from {pretrained_source}: {args.pretrained_model}")

    # Create objective function
    objective = HPOObjective(args, train_loader, val_loader, test_loader, input_features)

    # Create Optuna study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=pruner
    )

    # Run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # Get best trial
    best_trial = study.best_trial
    print(f"Best trial (#{best_trial.number}):")
    print(f"  Value: {best_trial.value:.4f}")
    print(f"  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save study results
    study_results = {
        'best_trial_number': best_trial.number,
        'best_params': best_trial.params,
        'best_value': best_trial.value,
        'study_name': args.study_name,
        'n_trials': args.n_trials,
        'completed_trials': len(study.get_trials(states=[TrialState.COMPLETE])),
        'pruned_trials': len(study.get_trials(states=[TrialState.PRUNED])),
        'failed_trials': len(study.get_trials(states=[TrialState.FAIL])),
        'mask_mode': args.mask_mode,
        'mask_prob': args.mask_prob,
        'pretrained_model': args.pretrained_model
    }

    with open(os.path.join(output_dir, "hpo_results.json"), 'w') as f:
        json.dump(study_results, f, indent=4)

    # Print completion message
    print(f"\nHPO study {args.study_name} completed!")
    print(f"Results saved to: {output_dir}")

    return study


if __name__ == "__main__":
    args = parse_args()
    study = run_hpo(args)