import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from collections import Counter
from aig_dataset import AIGDataset
import os


def inspect_truth_table_values(dataset, num_samples=None, truth_table_idx=3, plot=True):
    """
    Inspect and analyze truth table values in an AIG dataset.

    Args:
        dataset: An AIGDataset instance
        num_samples: Number of samples to analyze (None for all)
        truth_table_idx: Index of the truth table value in node features (default: 3)
        plot: Whether to generate plots

    Returns:
        stats_dict: Dictionary with statistics about truth table values
    """
    # Select samples to analyze
    if num_samples is None or num_samples >= len(dataset):
        indices = range(len(dataset))
    else:
        indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Extract truth table values and node types
    all_tt_values = []
    and_gate_tt_values = []
    input_tt_values = []
    output_tt_values = []

    # Process each graph
    for idx in indices:
        data = dataset[idx]

        # Identify node types
        is_and_gate = (data.x[:, 0] == 0) & (data.x[:, 1] == 1) & (data.x[:, 2] == 0)
        is_input = (data.x[:, 0] == 1) & (data.x[:, 1] == 0) & (data.x[:, 2] == 0)
        is_output = (data.x[:, 0] == 0) & (data.x[:, 1] == 0) & (data.x[:, 2] == 1)

        # Extract truth table values for all nodes and specific types
        tt_values = data.x[:, truth_table_idx].cpu().numpy()
        all_tt_values.extend(tt_values)

        and_gate_tt_values.extend(tt_values[is_and_gate.cpu().numpy()])
        input_tt_values.extend(tt_values[is_input.cpu().numpy()])
        output_tt_values.extend(tt_values[is_output.cpu().numpy()])

    # Convert to numpy arrays for analysis
    all_tt_values = np.array(all_tt_values)
    and_gate_tt_values = np.array(and_gate_tt_values)
    input_tt_values = np.array(input_tt_values)
    output_tt_values = np.array(output_tt_values)

    # Calculate statistics
    stats_dict = {
        'all': {
            'count': len(all_tt_values),
            'min': float(np.min(all_tt_values)),
            'max': float(np.max(all_tt_values)),
            'mean': float(np.mean(all_tt_values)),
            'median': float(np.median(all_tt_values)),
            'std': float(np.std(all_tt_values)),
            'unique_values': sorted(list(set(all_tt_values)))
        },
        'and_gates': {
            'count': len(and_gate_tt_values),
            'min': float(np.min(and_gate_tt_values)) if len(and_gate_tt_values) > 0 else None,
            'max': float(np.max(and_gate_tt_values)) if len(and_gate_tt_values) > 0 else None,
            'mean': float(np.mean(and_gate_tt_values)) if len(and_gate_tt_values) > 0 else None,
            'median': float(np.median(and_gate_tt_values)) if len(and_gate_tt_values) > 0 else None,
            'std': float(np.std(and_gate_tt_values)) if len(and_gate_tt_values) > 0 else None,
            'unique_values': sorted(list(set(and_gate_tt_values)))
        },
        'inputs': {
            'count': len(input_tt_values),
            'unique_values': sorted(list(set(input_tt_values)))
        },
        'outputs': {
            'count': len(output_tt_values),
            'unique_values': sorted(list(set(output_tt_values)))
        }
    }

    # Count value frequencies for AND gates
    if len(and_gate_tt_values) > 0:
        value_counts = Counter(and_gate_tt_values)
        stats_dict['and_gates']['value_distribution'] = dict(sorted(value_counts.items()))

    # Check for binary values
    is_binary = all(v in [0, 1] for v in stats_dict['all']['unique_values'])
    stats_dict['is_binary'] = is_binary

    # Determine if values are normalized
    if len(stats_dict['all']['unique_values']) > 0:
        min_val = stats_dict['all']['min']
        max_val = stats_dict['all']['max']
        stats_dict['appears_normalized'] = min_val >= 0 and max_val <= 1

    # Print summary
    print(f"Analyzed {len(indices)} graphs with {stats_dict['all']['count']} nodes")
    print(f"Node type distribution:")
    print(f"  - AND gates: {stats_dict['and_gates']['count']}")
    print(f"  - Input nodes: {stats_dict['inputs']['count']}")
    print(f"  - Output nodes: {stats_dict['outputs']['count']}")

    print("\nTruth table value statistics:")
    print(f"  - Range: [{stats_dict['all']['min']}, {stats_dict['all']['max']}]")
    print(f"  - Mean: {stats_dict['all']['mean']:.4f}")
    print(f"  - Median: {stats_dict['all']['median']:.4f}")
    print(f"  - Standard deviation: {stats_dict['all']['std']:.4f}")
    print(f"  - Max: {stats_dict['all']['max']}")
    print(f"  - Min: {stats_dict['all']['min']}")

    print("\nUnique truth table values:")
    if len(stats_dict['all']['unique_values']) <= 20:
        print(f"  - All values: {stats_dict['all']['unique_values']}")
    else:
        print(
            f"  - {len(stats_dict['all']['unique_values'])} unique values (showing first 10): {stats_dict['all']['unique_values'][:10]}...")

    # Generate plots if requested
    if plot and len(all_tt_values) > 0:
        plt.figure(figsize=(15, 10))

        # Histogram of all truth table values
        plt.subplot(2, 2, 1)
        plt.hist(all_tt_values, bins=30, alpha=0.7)
        plt.title('Distribution of All Truth Table Values')
        plt.xlabel('Value')
        plt.ylabel('Count')

        # Histogram of AND gate truth table values
        if len(and_gate_tt_values) > 0:
            plt.subplot(2, 2, 2)
            plt.hist(and_gate_tt_values, bins=30, alpha=0.7, color='orange')
            plt.title('Distribution of AND Gate Truth Table Values')
            plt.xlabel('Value')
            plt.ylabel('Count')

        # Box plot of truth table values by node type
        plt.subplot(2, 2, 3)
        data_to_plot = []
        labels = []

        if len(and_gate_tt_values) > 0:
            data_to_plot.append(and_gate_tt_values)
            labels.append('AND Gates')

        if len(input_tt_values) > 0:
            data_to_plot.append(input_tt_values)
            labels.append('Input Nodes')

        if len(output_tt_values) > 0:
            data_to_plot.append(output_tt_values)
            labels.append('Output Nodes')

        if data_to_plot:
            plt.boxplot(data_to_plot, labels=labels)
            plt.title('Truth Table Values by Node Type')
            plt.ylabel('Value')

        # If values appear to be binary, show pie chart of 0/1 distribution
        if is_binary and len(and_gate_tt_values) > 0:
            plt.subplot(2, 2, 4)
            zero_count = np.sum(and_gate_tt_values == 0)
            one_count = np.sum(and_gate_tt_values == 1)
            plt.pie([zero_count, one_count], labels=['0', '1'], autopct='%1.1f%%')
            plt.title('Binary Value Distribution for AND Gates')

        plt.tight_layout()
        plt.show()

    return stats_dict


def sample_truth_table_distributions(dataset, num_graphs=5, truth_table_idx=3):
    """
    Sample and display truth table distributions for a few individual graphs.

    Args:
        dataset: An AIGDataset instance
        num_graphs: Number of graphs to sample
        truth_table_idx: Index of the truth table value in node features
    """
    if num_graphs > len(dataset):
        num_graphs = len(dataset)

    indices = np.random.choice(len(dataset), num_graphs, replace=False)

    plt.figure(figsize=(15, num_graphs * 3))

    for i, idx in enumerate(indices):
        data = dataset[idx]

        # Identify node types
        is_and_gate = (data.x[:, 0] == 0) & (data.x[:, 1] == 1) & (data.x[:, 2] == 0)

        # Extract truth table values for AND gates
        tt_values = data.x[is_and_gate, truth_table_idx].cpu().numpy()

        # Count nodes by type
        and_gate_count = is_and_gate.sum().item()
        input_count = ((data.x[:, 0] == 1) & (data.x[:, 1] == 0) & (data.x[:, 2] == 0)).sum().item()
        output_count = ((data.x[:, 0] == 0) & (data.x[:, 1] == 0) & (data.x[:, 2] == 1)).sum().item()

        plt.subplot(num_graphs, 2, 2 * i + 1)
        if len(tt_values) > 0:
            plt.hist(tt_values, bins=20, alpha=0.7)
            plt.title(f'Graph {idx}: AND Gate Truth Table Values')
            plt.xlabel('Value')
            plt.ylabel('Count')
        else:
            plt.text(0.5, 0.5, 'No AND gates in this graph', horizontalalignment='center')
            plt.title(f'Graph {idx}')

        # Pie chart of node types
        plt.subplot(num_graphs, 2, 2 * i + 2)
        plt.pie([and_gate_count, input_count, output_count],
                labels=['AND Gates', 'Inputs', 'Outputs'],
                autopct='%1.1f%%')
        plt.title(f'Graph {idx}: Node Type Distribution')

    plt.tight_layout()
    plt.show()


def analyze_dataset(dataset_path, sample_size=100, truth_table_idx=3):
    """
    Load and analyze a dataset from a saved file.

    Args:
        dataset_path: Path to the processed dataset
        sample_size: Number of samples to analyze
        truth_table_idx: Index of the truth table value in node features
    """
    dataset = AIGDataset(root=os.path.dirname(dataset_path),
                         processed_file=os.path.basename(dataset_path))

    print(f"Dataset loaded: {len(dataset)} graphs")

    # Analyze overall statistics
    stats = inspect_truth_table_values(dataset, num_samples=sample_size,
                                       truth_table_idx=truth_table_idx)

    # Sample individual graphs
    sample_truth_table_distributions(dataset, num_graphs=5,
                                     truth_table_idx=truth_table_idx)

    return stats


# Example usage:
if __name__ == "__main__":
    # For using with an existing dataset
    dataset = AIGDataset(root="./", processed_file="data.pt")
    stats = inspect_truth_table_values(dataset, num_samples=100)
    sample_truth_table_distributions(dataset, num_graphs=5)

    # For analyzing a dataset from file
    # stats = analyze_dataset("./processed/data.pt")
    pass