import os
import pickle
import networkx as nx
from aigverse import read_aiger_into_aig, to_edge_list, simulate, simulate_nodes
import numpy as np
from typing import List, Tuple
import math


MAX_SIZE = 120
MAX_INPUTS = 12

# Directory where all AIG folders are stored
base_dir = './aigs/rand_aigs_20k'


# Define one-hot encodings for node types and edge labels
node_type_encoding = {
    "0": [0, 0, 0],
    "PI": [1, 0, 0],  #  [Index placeholder] + [One-hot encoding]
    "AND": [0, 1, 0],
    "PO": [0, 0, 1]
}

edge_label_encoding = {
    "INV": [1, 0],  # Inverted edge
    "REG": [0, 1]  # Regular edge
}


# Function to save all graphs to a single pickle file
def save_all_graphs(all_graphs, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(all_graphs, f)
    print(f"Saved {len(all_graphs)} graphs to {output_file}")


def generate_binary_inputs(num_inputs: int) -> List[List[int]]:
    """Generates all possible binary input combinations for a given number of inputs."""
    return [[(i >> bit) & 1 for bit in range(num_inputs - 1, -1, -1)] for i in range(2 ** num_inputs)]


def get_nodes(aig, G):
    """Add nodes to the graph with one-hot encoded node types."""
    # Get zero node info
    input_patterns = list(zip(*generate_binary_inputs(aig.num_pis())))

    G.add_node(0, type=node_type_encoding["0"], feature=0.0)
    for pi in aig.pis():  # Add input nodes
        binary_inputs = "".join(map(str, list(input_patterns[pi - 1])))
        G.add_node(pi, type=node_type_encoding["PI"], feature=normalize_log(int(binary_inputs, 2)))

    n_to_tt = simulate_nodes(aig)
    for gate in aig.gates():  # Add gate nodes
        binary_truths = n_to_tt[gate].to_binary()
        G.add_node(gate, type=node_type_encoding["AND"], feature=normalize_log(int(binary_truths,2)))
    return G


def get_edges(aig, G):
    """Add edges to the graph with one-hot encoded edge labels."""
    edges = to_edge_list(aig, inverted_weight=1, regular_weight=0)
    for e in edges:
        # Assign one-hot encoded edge labels
        onehot_label = np.array(edge_label_encoding["INV"] if e.weight == 1 else edge_label_encoding["REG"], dtype=np.float32)
        G.add_edge(e.source, e.target, type=onehot_label)
    return G


def get_out_features(ind):
    "Get node features includes the index of the output node, node type, output patterns"
    tts = simulate(aig)
    binary_truths = tts[ind].to_binary()
    return [int(bit) for bit in binary_truths]


def get_outs(aig, G, size):
    """Add output nodes and edges to the graph with one-hot encoded output nodes."""
    tts = simulate(aig)

    for ind, po in enumerate(aig.pos()):
        binary_truths = tts[ind].to_binary()
        # Get out node
        new_out_node_id = size + ind
        G.add_node(new_out_node_id, type=node_type_encoding["PO"], feature=normalize_log(int(binary_truths,2)))
        # Get out edge
        onehot_label = np.array(edge_label_encoding["INV"] if aig.is_complemented(po) else edge_label_encoding["REG"],
                                dtype=np.float32)
        pre_node = aig.get_node(po)
        G.add_edge(pre_node, new_out_node_id, type=onehot_label)

    return G

def normalize_log(value):
    return math.log2(1 + value) / math.log2(1 + 2**(2**MAX_INPUTS))

def get_condition(aig,graph_size):
    # tensor
    # iterate over nodes in graph
    condition_list = [0]

    full_condition_list = []

    input_patterns = list(zip(*generate_binary_inputs(aig.num_pis())))
    # TODO check size and orientation to make sure i get it right
    for pi in aig.pis():
        # convert binary to integer
        binary_inputs = "".join(map(str, list(input_patterns[pi - 1])))
        # append to list
        condition_list.append(normalize_log(int(binary_inputs,2)))
        full_condition_list.append(normalize_log(int(binary_inputs, 2)))

    condition_list += [0.0] * aig.num_gates() #TODO change to simulation?

    n_to_tt = simulate_nodes(aig)
    for gate in aig.gates():  # Add gate nodes
        binary_t = n_to_tt[gate].to_binary()
        full_condition_list.append(normalize_log(int(binary_t, 2)))

    tts = simulate(aig)

    for ind, po in enumerate(aig.pos()):
        binary_truths = tts[ind].to_binary()
        condition_list.append(normalize_log(int(binary_truths,2)))
        full_condition_list.append(normalize_log(int(binary_truths,2)))
        # Convert to int and append to condition_list



    assert len(condition_list) == graph_size
    return condition_list, full_condition_list


    # get truth tables # Generate all possible binary input combinations
    #
    # get input patterns
    # get intermediate and gate patterns ? might bias towards shape ? and number of intermediate nodes?
    # get output patterns
    # make into integers + list(input_patterns[pi-1])


# Function to parse .aig file and create a directed graph
def get_graph(aig, graph_size):
    """Create the graph, process nodes and edges, and apply one-hot encoding"""
    condition, full_condition = get_condition(aig, graph_size)
    # Make graph with features
    G = nx.DiGraph(inputs=aig.num_pis(),
                   outputs=aig.num_pos(),
                   tts=condition,
                   full_tts=full_condition,
                   output_tts=[[int(tt.get_bit(i)) for i in range(tt.num_bits())] for tt in simulate(aig)])
    # Add nodes with one-hot encoding
    G = get_nodes(aig, G)
    # Check if the number of nodes matches
    num_nodes_G = G.number_of_nodes()
    aig_size = aig.size()
    assert num_nodes_G == aig_size, f"Node count mismatch: G has {num_nodes_G}, AIG has {aig_size}"
    # Add edges with one-hot encoding
    G = get_edges(aig, G)
    # Add output nodes and edges
    G = get_outs(aig, G, aig_size)
    pos = aig.num_pos()
    assert G.number_of_nodes() == aig_size + pos, f"Node count mismatch: G has {G.number_of_nodes()}, Should be {aig_size + pos}"

    return G


all_graphs = []

# Iterate through all subfolders and .aig files
for filename in os.listdir(base_dir):
    if filename.endswith('.aig'):  # Process only .aig files
        file_path = os.path.join(base_dir, filename)

        # Create an aig object from the .aig file
        aig = read_aiger_into_aig(file_path)

        graph_size =(aig.num_pis() + aig.num_pos() +aig.num_gates() +1)

        # # Add size check
        if graph_size > MAX_SIZE: # size is normal nodes and inputs, the additional output nodes i add, and constant 0
            print(f"Skipping graph {filename} in {base_dir}, too large ({aig.size()} nodes)")
            continue  # Skip this graph if it's too large

        if aig.num_pis() > MAX_INPUTS: # size is normal nodes and inputs, the additional output nodes i add, and constant 0
            print(f"Skipping graph {filename} in {base_dir}, larger than {MAX_INPUTS}, ({aig.num_pis()} inputs)")
            continue  # Skip this graph if it's too large

        Graph = get_graph(aig, graph_size)

        # Add the graph to the list
        all_graphs.append(Graph)

print("Dataset Size",len(all_graphs))
# Save all the graphs into one pickle file at the end
save_all_graphs(all_graphs, "./gpt_graphs.pkl")





