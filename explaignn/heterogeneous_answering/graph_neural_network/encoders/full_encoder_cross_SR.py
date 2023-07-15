import pickle
import torch
import torch.nn.functional as F
import transformers

from explaignn.heterogeneous_answering.graph_neural_network.encoders.encoder import Encoder
from explaignn.library.string_library import StringLibrary


class FullEncoderCrossSR(Encoder):
    def __init__(self, config):
        super(FullEncoderCrossSR, self).__init__(config)
        self.initialize(config)

        if config.get("gnn_add_entity_type"):
            self.string_lib = StringLibrary(config)
            with open(self.config["path_to_types"], "rb") as fp:
                self.type_dict = pickle.load(fp)

        # instantiate linear encoding layer
        if self.config["gnn_encoder_linear"]:
            self.encoder_linear = torch.nn.Linear(in_features=self.max_input_length, out_features=1)

        # move to cuda
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if self.config["gnn_encoder_linear"]:
                self.encoder_linear = self.encoder_linear.cuda()

    def encode_srs_batch(self, srs):
        """Encode all SRs in the batch."""
        # srs = self.normalize_srs(srs)
        return self._encode(srs, max_input_length=self.max_input_length_sr)

    def encode_evidences_batch(self, evidences, srs):
        """Encode all evidences (paired with the SR) in the batch."""
        batch_size = len(evidences)
        num_evidences = len(evidences[0])

        def _prepare_input(evidence, sr):
            evidence_text = evidence["evidence_text"]
            return f"{sr}{self.sep_token}{evidence_text}"

        # flatten input
        flattened_input = [
            _prepare_input(evidence, srs[i])
            for i, evidences_for_inst in enumerate(evidences)
            for evidence in evidences_for_inst
        ]  # size: (batch_size * num_ev) x emb
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ev)
        encodings = encodings.view(batch_size, num_evidences, -1)
        return encodings

    def encode_entities_batch(self, entities, srs, *args):
        """Encode all entities (paired with the SR) in the batch."""
        batch_size = len(entities)
        num_entities = len(entities[0])

        def _prepare_input(entity, sr):
            entity_label = entity["label"]
            if "gnn_add_entity_type" in self.config and self.config["gnn_add_entity_type"]:
                entity_type = _get_entity_type(entity)
                return f"{sr}{self.sep_token}{entity_label}{self.sep_token}{entity_type}"
            else:
                return f"{sr}{self.sep_token}{entity_label}"

        def _get_entity_type(entity):
            """Get the type for the entity."""
            if self.string_lib.is_year(entity["label"]):
                return "year"
            elif self.string_lib.is_timestamp(entity["id"]):
                return "date"
            elif self.string_lib.is_number(entity["id"]):
                return "number"
            elif self.string_lib.is_entity(entity["id"]):
                t = self.type_dict.get(entity["id"])
                if t is None:
                    return ""
                return t["label"]
            else:
                return "string"

        # flatten input
        flattened_input = [
            _prepare_input(entity, srs[i])
            for i, entities_for_inst in enumerate(entities)
            for entity in entities_for_inst
        ]  # size: (batch_size * num_ent) x emb
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ent)
        encodings = encodings.view(batch_size, num_entities, -1)
        return encodings

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
