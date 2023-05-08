from explaignn.heterogeneous_answering.graph_neural_network.answering.bilinear_answering import (
    BilinearAnswering,
)
from explaignn.heterogeneous_answering.graph_neural_network.answering.linear_answering import (
    LinearAnswering,
)
from explaignn.heterogeneous_answering.graph_neural_network.answering.multitask_bilinear_answering import (
    MultitaskBilinearAnswering,
)
from explaignn.heterogeneous_answering.graph_neural_network.answering.multitask_linear_answering import (
    MultitaskLinearAnswering,
)


class AnsweringFactory:
    @staticmethod
    def get_answering(config):
        """Get an answering mechanism/layer, based on the given config."""
        if config["gnn_answering"] == "linear":
            return LinearAnswering(config)
        elif config["gnn_answering"] == "bilinear":
            return BilinearAnswering(config)
        elif config["gnn_answering"] == "multitask_linear":
            return MultitaskLinearAnswering(config)
        elif config["gnn_answering"] == "multitask_bilinear":
            return MultitaskBilinearAnswering(config)
        else:
            raise ValueError(f'Unknown answering layer: {config["gnn_answering"]}')
