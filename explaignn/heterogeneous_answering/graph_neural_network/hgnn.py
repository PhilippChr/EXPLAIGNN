import torch
import torch.nn.functional as F

from explaignn.heterogeneous_answering.graph_neural_network.answering.answering_factory import (
    AnsweringFactory,
)
from explaignn.heterogeneous_answering.graph_neural_network.encoders.encoder_factory import (
    EncoderFactory,
)


class HeterogeneousGNN(torch.nn.Module):
    """
    Heterogeneous graph neural network, as used in EXPLAIGNN (SIGIR 2023 full paper).
    - Node-type specific projections
    - Attention on SR
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
                "w_ev_att_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )
            setattr(
                self,
                "w_ent_ent_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )
            setattr(
                self,
                "w_ev_ent_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )

            # updating evidences
            setattr(
                self,
                "w_ent_att_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )
            setattr(
                self,
                "w_ev_ev_" + str(i),
                torch.nn.Linear(in_features=self.emb_dimension, out_features=self.emb_dimension),
            )
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
        """Forward step of HeterogeneousGNN."""
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
            w_ev_att = getattr(self, "w_ev_att_" + str(i))  # size: emb x emb
            w_ev_ent = getattr(self, "w_ev_ent_" + str(i))  # size: emb x emb

            # compute the evidence attention
            projected_evs = w_ev_att(evidences_mat)
            ev_att_scores = torch.bmm(projected_evs, sr_vec.unsqueeze(dim=2))
            ev_att_scores = F.softmax(ev_att_scores, dim=1)

            # multiply with adjacency
            evidence_weights = ev_att_scores * ev_to_ent.unsqueeze(dim=0)
            evidence_weights = evidence_weights.squeeze(dim=0).transpose(1, 2)

            # normalize
            vec = torch.sum(evidence_weights, keepdim=True, dim=2)
            vec[vec == 0] = 1
            evidence_weights = evidence_weights / vec

            # message passing: evidences -> entities
            ev_messages_ent = torch.bmm(evidence_weights, evidences_mat)
            ev_messages_ent = w_ev_ent(ev_messages_ent)

            # activation function
            entities_mat = F.relu(ev_messages_ent + entities_mat)

            ## UPDATE EVIDENCES
            w_ent_att = getattr(self, "w_ent_att_" + str(i))  # size: emb x emb
            w_ent_ev = getattr(self, "w_ent_ev_" + str(i))  # size: emb x emb

            # compute the entity attention
            projected_ents = w_ent_att(entities_mat) # size: batch_size x num_ent x emb
            ent_att_scores = torch.bmm(projected_ents, sr_vec.unsqueeze(dim=2)) # size: batch_size x num_ent x 1
            ent_att_scores = F.softmax(ent_att_scores, dim=1) # size: batch_size x num_ent x 1

            # multiply with adjacency
            entity_weights = ent_att_scores * ent_to_ev.unsqueeze(dim=0)
            entity_weights = entity_weights.squeeze(dim=0).transpose(1, 2)

            # normalize
            vec = torch.sum(entity_weights, keepdim=True, dim=2)
            vec[vec == 0] = 1
            entity_weights = entity_weights / vec

            # message passing: entities -> evidences
            ent_messages_ev = torch.bmm(entity_weights, entities_mat)
            ent_messages_ev = w_ent_ev(ent_messages_ev)

            # activation function
            evidences_mat = F.relu(ent_messages_ev + evidences_mat)

            # apply dropout
            entities_mat = F.dropout(entities_mat, self.dropout, training=train)
            evidences_mat = F.dropout(evidences_mat, self.dropout, training=train)

        ## obtain answer probabilities, loss, qa-metrics
        res = self.answering(batch, train, entities_mat, sr_vec, evidences_mat)
        return res
