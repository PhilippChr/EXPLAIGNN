import torch

from explaignn.library.utils import get_logger


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config
        self.logger = get_logger(__name__, config)

    def encode_srs_batch(self, srs):
        """Encode all SRs in the batch."""
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def encode_evidences_batch(self, evidences, *args):
        """Encode all evidences in the batch."""
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def encode_entities_batch(self, entities, *args):
        """Encode all entities in the batch."""
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )
