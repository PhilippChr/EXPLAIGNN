import torch
import torch.nn.functional as F

from explaignn.heterogeneous_answering.graph_neural_network.answering.answering_factory import (
    AnsweringFactory,
)
from explaignn.heterogeneous_answering.graph_neural_network.encoders.encoder_factory import (
    EncoderFactory,
)


class HeterogeneousGCN(torch.nn.Module):
    """
    Heterogeneous graph convolutional network:
    - different weight matrices per node type
    - no attention
    """

    def __init__(self, config):
        super(HeterogeneousGNN, self).__init__()
        self.config = config

        # load parameters
        self.num_layers = config["gnn_num_layers"]
        self.emb_dimension = config["gnn_emb_dimension"]
        self.dropout = config["gnn_dropout"] if "gnn_dropout" in config else 0.0

        # encoder
        self.encoder = EncoderFactory.get_encoder(config)

        # GNN layers
        for i in range(self.num_layers):
            # updating entities
            setattr(
                self,
                "w_ev_ent_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )

            # updating evidences
            setattr(
                self,
                "w_ent_ev_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )

        # answering
        self.answering = AnsweringFactory.get_answering(config)

        # move layers to cuda
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, batch, train=False):
        """Forward step of heterogeneous GCN."""
        # get data
        srs = batch["sr"]
        entities = batch["entities"]
        evidences = batch["evidences"]
        ent_to_ev = batch["ent_to_ev"]
        ev_to_ent = batch["ev_to_ent"]

        # encoding
        sr_vec = self.encoder.encode_srs_batch(srs)
        evidences_mat = self.encoder.encode_evidences_batch(
            evidences, srs
        )  # size: batch_size x num_ev x emb
        entities_mat = self.encoder.encode_entities_batch(
            entities, srs, evidences_mat, ev_to_ent, sr_vec
        )  # size: batch_size x num_ent x emb

        ## apply graph neural updates
        for i in range(self.num_layers):
            ## UPDATE ENTITIES
            w_ev_ent = getattr(self, "w_ev_ent_" + str(i))  # size: emb x emb
            ev_messages_ent = torch.bmm(ent_to_ev, evidences_mat)  # batch_size x num_ent x emb
            ev_messages_ent = w_ev_ent(ev_messages_ent)  # batch_size x num_ent x emb

            # activation function
            entities_mat = F.relu(ev_messages_ent + entities_mat)  # batch_size x num_ent x emb

            ## UPDATE EVIDENCES
            w_ent_ev = getattr(self, "w_ent_ev_" + str(i))  # size: emb x emb
            ent_messages_ev = torch.bmm(ev_to_ent, entities_mat)  # batch_size x num_ev x emb
            ent_messages_ev = w_ent_ev(ent_messages_ev)

            # activation function
            evidences_mat = F.relu(ent_messages_ev + evidences_mat)  # batch_size x num_ev x emb

            ## PREPARE FOR NEXT LAYER
            entities_mat = F.dropout(entities_mat, self.dropout, training=train)
            evidences_mat = F.dropout(evidences_mat, self.dropout, training=train)

        ## obtain answer probabilities, loss, qa-metrics
        res = self.answering(batch, train, entities_mat, sr_vec, evidences_mat)
        return res
