import torch
import torch.nn.functional as F

from explaignn.heterogeneous_answering.graph_neural_network.answering.answering_factory import (
    AnsweringFactory,
)
from explaignn.heterogeneous_answering.graph_neural_network.encoders.encoder_factory import (
    EncoderFactory,
)


class GCN(torch.nn.Module):
    """GNN that applies the same projections for all nodes; no attention mechanism in graph updates."""

    def __init__(self, config):
        super(GCN, self).__init__()

        self.config = config

        # load parameters
        self.num_layers = config["gnn_num_layers"]
        self.emb_dimension = config["gnn_emb_dimension"]
        self.dropout = config["gnn_dropout"] if "gnn_dropout" in config else 0.0

        # encoder
        self.encoder = EncoderFactory.get_encoder(config)

        # GNN layers
        for i in range(self.num_layers):
            # one weight matrix for all connections
            setattr(
                self,
                "w_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )

        # answering
        self.answering = AnsweringFactory.get_answering(config)

        # move layers to cuda
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, batch, train=False):
        """
        Forward step of GCN.
        """
        # get data
        srs = batch["sr"]
        entities = batch["entities"]
        evidences = batch["evidences"]

        ent_to_ev = batch["ent_to_ev"]  # size: batch_size x num_ent x num_ev
        ev_to_ent = batch["ev_to_ent"]  # size: batch_size x num_ev x num_ent

        # encoding
        sr_vec = self.encoder.encode_srs_batch(srs)
        evidences_mat = self.encoder.encode_evidences_batch(
            evidences, srs
        )  # size: batch_size x num_ev x emb
        entities_mat = self.encoder.encode_entities_batch(
            entities, srs, evidences_mat, ev_to_ent, sr_vec
        )  # size: batch_size x num_ent x emb

        # apply graph neural updates
        for i in range(self.num_layers):
            # linear projection function
            w = getattr(self, "w_" + str(i))  # size: emb x emb

            # UPDATE ENTITIES
            # message passing: evidences -> evidences
            ev_messages_ent = torch.bmm(ent_to_ev, evidences_mat)  # batch_size x num_ent x emb
            ev_messages_ent = w(ev_messages_ent)  # batch_size x num_ent x emb
            # updates: entities -> entities
            ent_messages_ent = w(entities_mat)  # batch_size x num_ent x emb
            # activation function
            entities_mat = F.relu(ev_messages_ent + ent_messages_ent)  # batch_size x num_ent x emb

            # UPDATE EVIDENCES
            # message passing: entities -> evidences
            ent_messages_ev = torch.bmm(ev_to_ent, entities_mat)  # batch_size x num_ev x emb
            ent_messages_ev = w(ent_messages_ev)  # batch_size x num_ev x emb
            # updates: evidences -> evidences
            ev_messages_ev = w(evidences_mat)  # batch_size x num_ev x emb
            # activation function
            evidences_mat = F.relu(ev_messages_ev + ent_messages_ev)  # batch_size x num_ev x emb

            # PREPARE FOR NEXT LAYER
            entities_mat = F.dropout(entities_mat, self.dropout, training=train)
            evidences_mat = F.dropout(evidences_mat, self.dropout, training=train)

        # obtain answer probabilities, loss, qa-metrics
        res = self.answering(batch, train, entities_mat, sr_vec, evidences_mat)
        return res
