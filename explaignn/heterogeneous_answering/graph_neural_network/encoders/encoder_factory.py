from explaignn.heterogeneous_answering.graph_neural_network.encoders.alternating_encoder import (
    AltEncoder,
)
from explaignn.heterogeneous_answering.graph_neural_network.encoders.alternating_encoder_cross_SR import (
    AltEncoderCrossSR,
)
from explaignn.heterogeneous_answering.graph_neural_network.encoders.full_encoder import FullEncoder
from explaignn.heterogeneous_answering.graph_neural_network.encoders.full_encoder_cross_SR import (
    FullEncoderCrossSR,
)


class EncoderFactory:
    @staticmethod
    def get_encoder(config):
        """Get an encoder, based on the given config."""
        if config["gnn_encoder"] == "full_encoder":
            return FullEncoder(config)
        elif config["gnn_encoder"] == "full_encoder_cross_SR":
            return FullEncoderCrossSR(config)
        elif config["gnn_encoder"] == "alternating_encoder":
            return AltEncoder(config)
        elif config["gnn_encoder"] == "alternating_encoder_cross_SR":
            return AltEncoderCrossSR(config)
        else:
            raise ValueError(f'Unknown encoder: {config["gnn_encoder"]}')
