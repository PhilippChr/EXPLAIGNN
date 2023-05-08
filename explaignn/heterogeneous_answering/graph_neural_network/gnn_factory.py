from explaignn.heterogeneous_answering.graph_neural_network.gcn import GCN
from explaignn.heterogeneous_answering.graph_neural_network.gat import GAT
from explaignn.heterogeneous_answering.graph_neural_network.hgnn import HeterogeneousGNN
from explaignn.heterogeneous_answering.graph_neural_network.hgcn import HeterogeneousGCN


class GNNFactory:
    @staticmethod
    def get_gnn(config):
        """Get a GNN, based on the given config."""
        if config["gnn_model"] == "gcn":
            return GCN(config)
        elif config["gnn_model"] == "gat":
            return GAT(config)
        elif config["gnn_model"] == "heterogeneous_gnn":
            return HeterogeneousGNN(config)
        elif config["gnn_model"] == "heterogeneous_gcn":
            return HeterogeneousGCN(config)
        else:
            raise ValueError(f'Unknown GNN model: {config["gnn_model"]}')
