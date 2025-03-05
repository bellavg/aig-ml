import argparse
from collections import defaultdict
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import os
import numpy as np
import json
import random
from datetime import datetime

# Import your custom modules
from aig_dataset import AIGDataset
from model import AIGTransformer
from prediction import reconstruct_predictions
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate the AIG Transformer Model with various masking strategies")

    # Dataset parameters
    parser.add_argument('--file_path', type=str, default=None,
                        help="Path to the original dataset file (only needed if processed data doesn't exist)")
    parser.add_argument('--root', type=str, default="",
                        help="Root directory containing the processed data.pt file")
    parser.add_argument('--processed_file', type=str, default="data.pt",
                        help="Name of the processed PyG dataset file")

    parser.add_argument('--max_nodes', type=int, default=120,
                        help="Maximum number of nodes in each graph")

    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help="Ratio of dataset used for validation")
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help="Ratio of dataset used for testing")
    parser.add_argument('--val_freq', type=int, default=5,
                        help="Frequency (in epochs) to run validation")

    parser.add_argument('--device', type=str, default='cuda',
                        help="Device for training (e.g., 'cuda' or 'cpu')")

    parser.add_argument('--num_graphs', type=int, default=24,
                        help="Number of graphs to load from the dataset")

    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help="Hidden dimension size")
    parser.add_argument('--num_layers', type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument('--num_heads', type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate")

    # Masking parameters
    parser.add_argument('--mask_prob', type=float, default=0.20,
                        help="Masking probability for nodes/gates")
    parser.add_argument('--mask_mode', type=str, default="node_feature",
                        choices=["node_feature", "edge_feature", "connectivity"],
                        help="Masking mode to use for training")
    # For backward compatibility
    parser.add_argument('--gate_masking', action='store_true',
                        help="Enable gate masking (equivalent to --mask_mode=node_feature)")

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--early_stopping', type=int, default=10,
                        help="Number of epochs to wait for improvement before early stopping")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")

    # Experiment tracking
    parser.add_argument('--exp_name', type=str, default=None,
                        help="Experiment name for logging (default: auto-generated)")

    # Model saving parameters
    parser.add_argument('--save_model_path', type=str, default="./models/",
                        help="Path to save the best model")
    parser.add_argument('--results_dir', type=str, default="./results/",
                        help="Directory to save results")

    # Pretrained model loading
    parser.add_argument('--pretrained_model', type=str, default=None,
                        help="Path to pretrained model to continue training from")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Handle backward compatibility: if gate_masking is set, override mask_mode
    if args.gate_masking:
        args.mask_mode = "node_feature"
        print("Warning: --gate_masking is deprecated, use --mask_mode=node_feature instead")

    # Set up experiment name if not provided
    if args.exp_name is None:
        mask_prefix = f"{args.mask_mode}_mask{int(args.mask_prob * 100):02d}"
        if args.pretrained_model:
            # Include info about pretraining in experiment name
            pretrained_base = os.path.basename(args.pretrained_model).split('_')[0]
            args.exp_name = f"{mask_prefix}_from_{pretrained_base}"
        else:
            args.exp_name = f"{mask_prefix}"

    # Create directories
    os.makedirs(args.save_model_path, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Set up paths for this experiment
    exp_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Set up logging
    log_path = os.path.join(exp_dir, "training_log.txt")
    results_path = os.path.join(exp_dir, "metrics.json")
    best_model_path = os.path.join(args.save_model_path, f"{args.exp_name}_best.pt")
    final_model_path = os.path.join(args.save_model_path, f"{args.exp_name}_final.pt")

    # Save configuration
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Log start of experiment
    with open(log_path, 'w') as f:
        f.write(f"Starting experiment: {args.exp_name}\n")
        f.write(f"Masking mode: {args.mask_mode} at {args.mask_prob * 100:.1f}% probability\n")
        f.write(f"Random seed: {args.seed}\n")

        if args.pretrained_model:
            f.write(f"Using pretrained model: {args.pretrained_model}\n")

        f.write(f"Configuration saved to: {config_path}\n")
        f.write("\n")

    # Load the dataset from processed data
    print(f"Loading dataset from root directory: {args.root}")
    full_dataset = AIGDataset(
        file_path=args.file_path,  # Only used if processed data doesn't exist
        num_graphs=args.num_graphs,
        root=args.root,
        processed_file=args.processed_file
    )
    print(f"Loaded dataset with {len(full_dataset)} graphs")

    # Split the dataset
    total = len(full_dataset)
    test_size = int(total * args.test_ratio)
    val_size = int(total * args.val_ratio)
    train_size = total - test_size - val_size

    print(f"Splitting dataset: Train={train_size}, Val={val_size}, Test={test_size}")
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)  # Use the same seed for splits
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Get feature dimensions from sample
    sample = full_dataset[0]
    node_features = sample.x.size(1)
    edge_features = sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') else 0

    print(f"Model input features: nodes={node_features}, edges={edge_features}")

    # Print masking strategy
    print(f"Using {args.mask_mode} masking at {args.mask_prob * 100:.1f}% probability")

    # Create model
    model = AIGTransformer(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_nodes=args.max_nodes
    )

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained model if specified
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"Loading pretrained model from {args.pretrained_model}")

        try:
            # Load state dictionary
            state_dict = torch.load(args.pretrained_model, map_location=device)

            # Check if model architecture matches
            model_loaded = model.load_state_dict(state_dict, strict=False)

            if model_loaded.missing_keys:
                print(f"Warning: Missing keys when loading pretrained model: {model_loaded.missing_keys}")
            if model_loaded.unexpected_keys:
                print(f"Warning: Unexpected keys in pretrained model: {model_loaded.unexpected_keys}")

            print(f"Successfully loaded pretrained model")
            with open(log_path, 'a') as f:
                f.write(f"Loaded pretrained model from {args.pretrained_model}\n")
                if model_loaded.missing_keys:
                    f.write(f"Missing keys: {model_loaded.missing_keys}\n")
                if model_loaded.unexpected_keys:
                    f.write(f"Unexpected keys: {model_loaded.unexpected_keys}\n")

        except Exception as e:
            print(f"Error loading pretrained model: {e}")
            print("Starting with randomly initialized model instead")
            with open(log_path, 'a') as f:
                f.write(f"Failed to load pretrained model: {e}\n")

    # Move model to device
    model = model.to(device)

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track metrics
    best_val_loss = float('inf')
    no_improve_count = 0
    training_metrics = {
        "train_losses": [],
        "val_losses": [],
        "best_epoch": 0,
        "best_val_loss": float('inf'),
        "pretrained_model": args.pretrained_model,
        "masking_mode": args.mask_mode,
        "mask_prob": args.mask_prob,
        "seed": args.seed
    }

    # Initialize best_model_loaded to help track if the best model was successfully saved
    best_model_loaded = False

    # Training loop
    for epoch in range(args.num_epochs):

        # Train for one epoch
        train_losses = train_epoch(args, model, train_loader, optimizer, device)

        # Log training progress
        epoch_log = f"Epoch {epoch + 1}/{args.num_epochs}, "
        epoch_log += f"Train: {dict(train_losses)}"

        # Run validation
        if epoch % args.val_freq == 0 or epoch == args.num_epochs - 1:
            val_losses, val_metrics = validate(args, model, val_loader, device)
            epoch_log += f", Val: {dict(val_losses)}"
            epoch_log += f", Metrics: {val_metrics}"

            # Track metrics
            training_metrics["val_losses"].append(dict(val_losses))
            training_metrics.setdefault("val_metrics", []).append(val_metrics)

            # Check for improvement - use total_loss if available, else fall back to node_loss
            current_val_loss = val_losses.get('total_loss', val_losses.get('node_loss', val_losses.get('edge_feature_loss', float('inf'))))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                training_metrics["best_epoch"] = epoch + 1
                training_metrics["best_val_loss"] = best_val_loss

                # Save best model
                try:
                    # Ensure model path exists
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    best_model_loaded = True
                    epoch_log += f" (New best model saved to {best_model_path})"
                except Exception as e:
                    print(f"Error saving best model: {e}")
                    epoch_log += f" (Error saving best model: {e})"
                    best_model_loaded = False

                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping check
            if no_improve_count >= args.early_stopping:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Track training metrics
        training_metrics["train_losses"].append(dict(train_losses))

        # Log epoch results
        print(epoch_log)
        with open(log_path, 'a') as f:
            f.write(epoch_log + "\n")

    # Save final model
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to: {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    # Load best model for final evaluation only if it was successfully saved
    if best_model_loaded and os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model from {best_model_path} for evaluation")
        except Exception as e:
            print(f"Error loading best model for evaluation: {e}")
            print("Using final model state for evaluation instead")
    else:
        print("Best model wasn't saved successfully. Using current model state for evaluation.")

    # Evaluate on test set
    test_losses, test_metrics = validate(args, model, test_loader, device)
    test_log = f"Test results: {dict(test_losses)}"
    test_log += f", Test metrics: {test_metrics}"
    print(test_log)

    # Save final metrics
    training_metrics["test_losses"] = dict(test_losses)
    training_metrics["test_metrics"] = test_metrics
    try:
        with open(results_path, 'w') as f:
            json.dump(training_metrics, f, indent=4)
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Error saving results: {e}")

    # Log completion
    with open(log_path, 'a') as f:
        f.write("\n" + test_log + "\n")
        f.write(f"\nExperiment completed. Results saved to {results_path}\n")
        f.write(
            f"Best model: epoch {training_metrics['best_epoch']}, val_loss={training_metrics['best_val_loss']:.4f}\n")

    print(f"Experiment {args.exp_name} completed!")
    if best_model_loaded:
        print(f"Best model saved to: {best_model_path}")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()