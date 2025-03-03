import argparse
from collections import defaultdict
import torch
# Import your custom modules
from masking import create_masked_batch  # or create_masked_aig_with_edges if neede-=

def validate(args, model, val_loader, device):
    """
    Run validation with support for both node and gate masking.

    Args:
        args: Command line arguments including mask_prob and gate_masking flag
        model: AIGTransformer model
        val_loader: DataLoader for validation data
        device: Device to validate on (cuda/cpu)

    Returns:
        val_losses: Dictionary of average validation losses
    """
    model.eval()
    val_losses = defaultdict(float)

    # Track metrics
    metrics = {
        'total_masked_nodes': 0,
        'masked_nodes_per_batch': [],
        'total_masked_edges': 0,
        'masked_edges_per_batch': [],
        'batch_count': 0
    }

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            metrics['batch_count'] += 1

            # Create masked version of the batch
            masked_batch = create_masked_batch(
                batch,
                mp=args.mask_prob,
                gate_masking=args.gate_masking
            )

            # Track masking statistics
            num_masked_nodes = masked_batch.node_mask.sum().item()
            metrics['total_masked_nodes'] += num_masked_nodes
            metrics['masked_nodes_per_batch'].append(num_masked_nodes)

            if args.gate_masking and hasattr(masked_batch, 'edge_mask'):
                num_masked_edges = masked_batch.edge_mask.sum().item()
                metrics['total_masked_edges'] += num_masked_edges
                metrics['masked_edges_per_batch'].append(num_masked_edges)

            # Forward pass
            predictions = model(masked_batch)

            # Compute validation loss
            loss, loss_dict = model.compute_loss(
                predictions,
                {
                    'node_features': masked_batch.x_target,
                    'edge_index': masked_batch.edge_index_target,
                    'edge_attr': masked_batch.edge_attr_target if hasattr(masked_batch, 'edge_attr_target') else None,
                    'node_mask': masked_batch.node_mask,
                    'edge_mask': masked_batch.edge_mask if hasattr(masked_batch, 'edge_mask') else None,
                    'gate_masking': args.gate_masking
                }
            )

            # Accumulate losses
            for key, value in loss_dict.items():
                val_losses[key] += value.item() if isinstance(value, torch.Tensor) else value

    # Average losses over validation batches
    for key in val_losses:
        val_losses[key] /= metrics['batch_count']

    # Add masking statistics to the losses
    val_losses['avg_masked_nodes'] = metrics['total_masked_nodes'] / metrics['batch_count']
    if args.gate_masking:
        val_losses['avg_masked_edges'] = metrics['total_masked_edges'] / max(1, metrics['batch_count'])

    # Add a combined loss if both node and edge losses exist
    if 'node_loss' in val_losses and 'edge_existence_loss' in val_losses:
        val_losses['combined_loss'] = val_losses['node_loss'] + val_losses['edge_existence_loss']
        if 'edge_feature_loss' in val_losses:
            val_losses['combined_loss'] += val_losses['edge_feature_loss']

    return val_losses