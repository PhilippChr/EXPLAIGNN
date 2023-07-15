import torch
import torch.nn.functional as F
import transformers

from explaignn.heterogeneous_answering.graph_neural_network.encoders.encoder import Encoder


class AltEncoder(Encoder):
    def __init__(self, config):
        super(AltEncoder, self).__init__(config)
        self.initialize(config)

        # instantiate linear encoding layer
        if self.config["gnn_encoder_linear"]:
            self.encoder_linear = torch.nn.Linear(in_features=self.max_input_length, out_features=1)

        self.w_ev_att = torch.nn.Linear(
            in_features=self.emb_dimension, out_features=self.emb_dimension
        )

        # move to cuda
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.w_ev_att = self.w_ev_att.cuda()
            if self.config["gnn_encoder_linear"]:
                self.encoder_linear = self.encoder_linear.cuda()

    def encode_srs_batch(self, srs):
        """Encode all SRs in the batch."""
        return self._encode(srs, max_input_length=self.max_input_length_sr)

    def encode_evidences_batch(self, evidences, srs):
        """Encode all evidences (paired with the SR) in the batch."""
        batch_size = len(evidences)
        num_evidences = len(evidences[0])

        def _prepare_input(evidence):
            evidence_text = evidence["evidence_text"]
            return f"{evidence_text}"

        # flatten input
        flattened_input = [
            _prepare_input(evidence)
            for i, evidences_for_inst in enumerate(evidences)
            for evidence in evidences_for_inst
        ]  # size: (batch_size * num_ev) x emb
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ev)
        encodings = encodings.view(batch_size, num_evidences, -1)
        return encodings

    def encode_entities_batch(self, entities, srs, evidences_mat, ev_to_ent, sr_vec, *args):
        """
        Encode all entities in the batch, using the
        neighboring evidences as input.
        This is similar as in entity ranking, in which you
        use the lead text or even question-relevant passages
        to encode an entity.
        (e.g. BERT-ER: Query-specific BERT Entity Representations for Entity Ranking, Chatterjee et al., SIGIR 2022)
        """
        # compute the evidence attention
        projected_evs = self.w_ev_att(evidences_mat)
        ev_att_scores = torch.bmm(projected_evs, sr_vec.unsqueeze(dim=2))
        ev_att_scores = F.softmax(ev_att_scores, dim=1)
        ev_att_scores = ev_att_scores.clamp(min=1e-30, max=1e20) if self.config.get("gnn_clamping", False) else ev_att_scores

        # multiply with adjacency
        evidence_weights = ev_att_scores * ev_to_ent.unsqueeze(dim=0)
        evidence_weights = evidence_weights.squeeze(dim=0).transpose(1, 2)

        # normalize
        vec = torch.sum(evidence_weights, keepdim=True, dim=2)
        vec[vec == 0] = 1
        evidence_weights = evidence_weights / vec

        # initialize entity by surrounding evidences, weighted by attention
        entity_encodings = torch.bmm(evidence_weights, evidences_mat)  # batch_size x num_ent x emb
        return entity_encodings

    def _encode(self, flattened_input, max_input_length):
        """Encode the given input strings."""
        # tokenize
        tokenized_input = self.tokenizer(
            flattened_input,
            padding="max_length",
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )

        # move to cuda
        if torch.cuda.is_available():
            tokenized_input = tokenized_input.to(torch.device("cuda"))

        # LM encode
        outputs = self.model(**tokenized_input)  # size: flattened_len x max_length x emb
        lm_encodings = outputs.last_hidden_state

        # move to cuda
        if torch.cuda.is_available():
            lm_encodings = lm_encodings.cuda()

        # encoder linear
        if self.config["gnn_encoder_linear"]:
            encodings = self.encoder_linear(
                lm_encodings.transpose(1, 2)
            ).squeeze()  # size: flattened_len x max_length x emb
            encodings = F.relu(encodings)  # size: flattened_len x emb
            return encodings
        # no encoder linear (mean of embeddings)
        else:
            encodings = torch.mean(lm_encodings, dim=1)  # size: flattened_len x emb
            return encodings
