import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class MyStaticSTGDatasetLoader(object):
    def __init__(self, X:np.ndarray, A:np.ndarray):
        '''X: Node Feature Matrix (node_num, channel_num, sample_num)
           A: Adjacency Matrix (node_num, node_num)
        '''
        super(MyStaticSTGDatasetLoader, self).__init__()
        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, num_channels_out: int = 1):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, :num_channels_out, i + num_timesteps_in : j]).numpy().squeeze())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12, num_channels_out: int = 1
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator as an instance of the static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)*
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out, num_channels_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
