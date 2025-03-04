import argparse
import os
import optuna
from optuna.trial import TrialState
import torch
import torch.optim as optim
from torch.utils.data import random_split, Subset
import random
import numpy as np
from torch_geometric.loader import DataLoader
import json
from datetime import datetime
import pickle

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


def objective(trial, args, data_loaders, feature_info):
    """
    Optuna objective function for hyperparameter optimization.
    Returns validation loss to be minimized.
    """
    # Set the seed for this trial
    # seed = args.seed + trial.number
    # set_seed(seed)

    # Define the hyperparameter search space with expanded hidden_dim range
    num_heads = trial.suggest_int("num_heads", 2, 8)  # Choose between 2 and 8 heads

    # Choose hidden_dim ensuring it's divisible by num_heads
    hidden_dim_choices = [d for d in [64, 128, 256, 512, 1024] if d % num_heads == 0]
    hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_choices)
    num_layers = trial.suggest_int("num_layers", 1, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Use fixed batch size or make it a hyperparameter
    if args.optimize_batch_size:
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
        train_loader = DataLoader(data_loaders['train_dataset'], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(data_loaders['val_dataset'], batch_size=batch_size, shuffle=False)
    else:
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']

    # Unpack feature info
    node_features = feature_info['node_features']
    edge_features = feature_info['edge_features']
    max_nodes = args.max_nodes

    # Create model with trial hyperparameters
    model = AIGTransformer(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_nodes=max_nodes
    )

    # Set device and move model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set up optimizer with trial learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dictionary to store best validation scores
    best_val_loss = float('inf')

    # Set up args for masking
    args.mask_prob = args.mask_prob

    # Train for a few epochs to evaluate hyperparameters
    for epoch in range(args.hpo_epochs):
        # Train for one epoch
        train_losses = train_epoch(args, model, train_loader, optimizer, device)

        # Validate
        val_losses = validate(args, model, val_loader, device)

        # Use total_loss if available, otherwise node_loss
        current_val_loss = val_losses.get('total_loss', val_losses.get('node_loss', float('inf')))

        # Report intermediate results
        trial.report(current_val_loss, epoch)

        # Update best validation loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

        # Check for pruning (early stopping of this trial)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def parse_args():
    """Parse command line arguments for HPO."""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization for AIG Transformer Model")

    # Dataset parameters
    parser.add_argument('--file_path', type=str, default=None,
                        help="Path to the original dataset file (only needed if processed data doesn't exist)")
    parser.add_argument('--root', type=str, default="",
                        help="Directory containing the processed data.pt file")
    parser.add_argument('--processed_file', type=str, default="data.pt",
                        help="Name of the processed PyG dataset file")
    parser.add_argument('--max_nodes', type=int, default=120,
                        help="Maximum number of nodes in each graph")
    parser.add_argument('--val_ratio', type=float, default=0.25,
                        help="Ratio of dataset used for validation")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument('--num_graphs', type=int, default=500,
                        help="Number of graphs to load from the dataset")

    # Masking parameters
    parser.add_argument('--gate_masking', action='store_true',
                        help="Enable gate masking (node + connected edges)")
    parser.add_argument('--mask_prob', type=float, default=0.4,
                        help="Masking probability to use during HPO")

    # HPO parameters
    parser.add_argument('--n_trials', type=int, default=150,
                        help="Number of trials for hyperparameter optimization")
    parser.add_argument('--hpo_epochs', type=int, default=10,
                        help="Number of epochs per trial")
    parser.add_argument('--study_name', type=str, default=None,
                        help="Name of the Optuna study")
    parser.add_argument('--optimize_batch_size', action='store_true',
                        help="Whether to include batch size in optimization")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size (if not optimized)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Base random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set the base seed
    set_seed(args.seed)

    # Set up study name if not provided
    if args.study_name is None:
        masking_type = "gate" if args.gate_masking else "node"
        args.study_name = f"hpo_{masking_type}_mp{int(args.mask_prob * 100)}"

    # Create results directory
    results_dir = f"./results/hpo_results/{args.study_name}"
    os.makedirs(results_dir, exist_ok=True)

    # Create storage path for optuna study
    storage_path = os.path.join(results_dir, "study.pkl")

    # Save configuration
    config_path = os.path.join(results_dir, "hpo_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    print(f"Starting HPO study: {args.study_name}")
    print(f"Configuration saved to: {config_path}")
    print(f"Using {'gate' if args.gate_masking else 'node'} masking at {args.mask_prob * 100}% probability")
    print(f"Running {args.n_trials} trials with {args.hpo_epochs} epochs each")

    # Load dataset directly from processed data
    print(f"Loading dataset from processed directory")
    full_dataset = AIGDataset(
        num_graphs=args.num_graphs,
        processed_file=args.processed_file
    )
    print(f"Loaded dataset with {len(full_dataset)} graphs")

    # Limit the number of graphs if specified (handled within AIGDataset now)
    # The limit is applied during dataset initialization

    # Split into train and validation
    total = len(full_dataset)
    val_size = int(total * args.val_ratio)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Create dataloaders with fixed batch size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Store data loaders
    data_loaders = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'train_loader': train_loader,
        'val_loader': val_loader
    }

    # Get feature dimensions from sample
    if isinstance(full_dataset, Subset):
        sample = full_dataset.dataset[full_dataset.indices[0]]
    else:
        sample = full_dataset[0]

    node_features = sample.x.size(1)
    edge_features = sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') else 0

    feature_info = {
        'node_features': node_features,
        'edge_features': edge_features
    }

    print(f"Feature dimensions: node={node_features}, edge={edge_features}")

    # Create a local storage based study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner()
    )

    # Run the optimization
    study.optimize(lambda trial: objective(trial, args, data_loaders, feature_info), n_trials=args.n_trials)

    # Print optimization results
    pruned_trials = study.get_trials(states=[TrialState.PRUNED])
    complete_trials = study.get_trials(states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best hyperparameters to file
    best_params_path = os.path.join(results_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump({
            "best_value": trial.value,
            "best_params": trial.params,
            "study_name": args.study_name,
            "masking_type": "gate" if args.gate_masking else "node",
            "mask_prob": args.mask_prob,
            "seed": args.seed,
            "feature_info": feature_info
        }, f, indent=4)

    print(f"Best hyperparameters saved to: {best_params_path}")

    # Save study object
    with open(storage_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Study object saved to: {storage_path}")

    # Generate command for running with best parameters
    cmd = f"python main.py --hidden_dim {trial.params.get('hidden_dim')} --num_layers {trial.params.get('num_layers')} "
    cmd += f"--num_heads {trial.params.get('num_heads')} --dropout {trial.params.get('dropout')} --lr {trial.params.get('learning_rate')} "

    if args.optimize_batch_size:
        cmd += f"--batch_size {trial.params.get('batch_size')} "
    else:
        cmd += f"--batch_size {args.batch_size} "

    cmd += f"--mask_prob {args.mask_prob} "

    if args.gate_masking:
        cmd += "--gate_masking "

    # Update the command to use processed data paths
    cmd += f"--seed {args.seed} --processed_dir {args.processed_dir} --processed_file {args.processed_file} --num_graphs {args.num_graphs}"

    # Save command to file
    cmd_path = os.path.join(results_dir, "best_params_command.txt")
    with open(cmd_path, "w") as f:
        f.write(cmd)

    print(f"Command for running with best parameters saved to: {cmd_path}")
    print(f"Command: {cmd}")

    # Optional: Generate plots if visualization libraries are available
    try:
        from optuna.visualization import plot_optimization_history
        from optuna.visualization import plot_param_importances
        from optuna.visualization import plot_parallel_coordinate

        # Create visualizations directory
        viz_dir = os.path.join(results_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Plot optimization history
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(viz_dir, "optimization_history.png"))

        # Plot parameter importances
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(viz_dir, "param_importances.png"))

        # Plot parallel coordinate
        fig3 = plot_parallel_coordinate(study)
        fig3.write_image(os.path.join(viz_dir, "parallel_coordinate.png"))

        print(f"Visualization plots saved to: {viz_dir}")
    except ImportError:
        print("Visualization libraries not available. Install plotly for visualization support.")
    except Exception as e:
        print(f"Could not generate plots: {e}")


if __name__ == "__main__":
    main()