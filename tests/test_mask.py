import torch
import unittest
from ..masking import create_masked_aig_with_edges
from torch_geometric.data import Data

class TestCreateMaskedAIGWithEdges(unittest.TestCase):
    def test_masking_behavior(self):
        """
        Builds a small synthetic PyG Data object with 5 nodes:
          - Node 0 => [1,0,0] (PI)
          - Node 1 => [0,1,0] (AND)
          - Node 2 => [0,1,0] (AND)
          - Node 3 => [0,0,1] (PO)
          - Node 4 => [0,1,0] (AND)
        Edges: (0->1), (1->2), (2->3), (1->4), (4->3).
        Then forces node_mask_prob=1.0 to guarantee all ANDs are masked.
        Checks that the function zeroes out the AND node features and removes
        edges originating or ending in them.
        """
        # Make the node features (for simplicity, no extra 'feature' column)
        x = torch.tensor([
            [1,0,0,1],  # PI
            [0,1,0,1],  # AND
            [0,1,0,1],  # AND
            [0,0,1,1],  # PO
            [0,1,0,1],  # AND
        ], dtype=torch.float)

        # Build edge_index: shape = [2, E]
        #   For instance 0->1, 1->2, 2->3, 1->4, 4->3
        edge_index = torch.tensor([
            [0, 1, 2, 1, 4],
            [1, 2, 3, 4, 3]
        ], dtype=torch.long)

        # Optionally define edge attributes
        edge_attr = torch.ones(edge_index.size(1), 2)  # e.g. shape [5,2]

        # Create the PyG data object
        aig = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Force a 100% chance (1.0) to mask AND gates so test is deterministic
        masked_aig = create_masked_aig_with_edges(aig, node_mask_prob=1.0)

        # 1) Check that x_target and edge_index_target are clones of original
        self.assertTrue(torch.equal(masked_aig.x_target, aig.x))
        self.assertTrue(torch.equal(masked_aig.edge_index_target, aig.edge_index))
        self.assertTrue(torch.equal(masked_aig.edge_attr_target, aig.edge_attr))

        # 2) Check that all AND gates got masked => node_mask is True for nodes 1,2,4
        expected_mask = torch.tensor([False, True, True, False, True], dtype=torch.bool)
        self.assertTrue(torch.equal(masked_aig.node_mask, expected_mask))

        # 3) Those masked nodes should have been zeroed in masked_aig.x
        self.assertTrue(torch.allclose(masked_aig.x[1], torch.zeros(4)))
        self.assertTrue(torch.allclose(masked_aig.x[2], torch.zeros(4)))
        self.assertTrue(torch.allclose(masked_aig.x[4], torch.zeros(4)))

        # 4) The edges out of or into masked nodes (1,2,4) must be removed in new edge_index
        # Original edges: 0->1,1->2,2->3,1->4,4->3 => all involve masked nodes except 2->3
        # Actually node 2 is also masked, so 2->3 is out. So all edges removed => no edges remain
        self.assertEqual(masked_aig.edge_index.size(1), 0)  # Expect zero edges

        # 5) The edge_mask should be True for all old edges
        self.assertTrue(masked_aig.edge_mask.all().item())

        # That’s it!

if __name__ == '__main__':
    unittest.main(verbosity=2)