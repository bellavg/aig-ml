import argparse
import os
from aig_dataset import AIGDataset  # Make sure aig_dataset.py is in the same directory

def main():
    parser = argparse.ArgumentParser(description="Preprocess AIG graphs in chunks")

    # Dataset parameters (only the necessary ones for processing)
    parser.add_argument("--data_path", type=str, default="../complete_tt_graphs.pkl",
                        help="Path to AIG dataset pickle file")
    parser.add_argument("--processed_dir", type=str, default="../data", help="Directory for processed data")
    parser.add_argument("--process_chunk_size", type=int, default=None,
                        help="Process data in chunks of this size (for large pickle files)")

    args = parser.parse_args()

    # Instantiate the AIGDataset with the chunk size
    dataset = AIGDataset(
        file_path=args.data_path,
        processed_dir=args.processed_dir,
        num_graphs=None,  # Not directly used for chunking in the process method
    )

    # Call the process method to generate the PyG .pt file
    dataset.process()

if __name__ == "__main__":
    main()