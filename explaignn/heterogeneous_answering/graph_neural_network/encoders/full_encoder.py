import pickle
import torch
import torch.nn.functional as F
import transformers

from explaignn.heterogeneous_answering.graph_neural_network.encoders.encoder import Encoder
from explaignn.library.string_library import StringLibrary


class FullEncoder(Encoder):
    def __init__(self, config):
        super(FullEncoder, self).__init__(config)

        # load params
        self.emb_dimension = config["gnn_emb_dimension"]

        self.max_input_length_sr = config["gnn_enc_sr_max_input"]
        self.max_input_length_ev = config["gnn_enc_ev_max_input"]
        self.max_input_length_ent = config["gnn_enc_ent_max_input"]

        if config.get("gnn_add_entity_type"):
            self.string_lib = StringLibrary(config)
            with open("_data/types.pickle", "rb") as fp:
                self.type_dict = pickle.load(fp)

        # load LM
        if config["gnn_encoder_lm"] == "BERT":
            self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = transformers.BertModel.from_pretrained("bert-base-uncased")
            self.sep_token = "[SEP]"
        elif "gnn_hidden_layers" in config:
            lm_config = transformers.DistilBertConfig(n_layers=config["gnn_hidden_layers"])
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.model = transformers.DistilBertModel(lm_config)
            self.sep_token = "[SEP]"
        elif "gnn_hidden_layers_pretrained" in config:
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.model.transformer.n_layers = config["gnn_hidden_layers_pretrained"]
            self.model.transformer.layer = self.model.transformer.layer[
                : config["gnn_hidden_layers_pretrained"]
            ]
            self.sep_token = "[SEP]"
        elif config["gnn_encoder_lm"] == "DistilBERT":
            self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.sep_token = "[SEP]"
        elif config["gnn_encoder_lm"] == "DistilRoBERTa":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
            self.model = transformers.AutoModel.from_pretrained("distilroberta-base")
            self.sep_token = " </s>"
        else:
            raise Exception("Unknown architecture for Encoder module specified in config.")

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
        return self._encode(srs, max_input_length=self.max_input_length_sr)

    def encode_evidences_batch(self, evidences, *args):
        """Encode all evidences in the batch."""
        batch_size = len(evidences)
        num_evidences = len(evidences[0])

        def _prepare_input(evidence):
            evidence_text = evidence["evidence_text"]
            return evidence_text

        # flatten input
        flattened_input = [
            _prepare_input(evidence)
            for i, evidences_for_inst in enumerate(evidences)
            for evidence in evidences_for_inst
        ]  # size: (batch_size * num_ev) x emb
        encodings = self._encode(flattened_input, max_input_length=self.max_input_length_ev)
        encodings = encodings.view(batch_size, num_evidences, -1)
        return encodings

    def encode_entities_batch(self, entities, *args):
        """Encode all entities in the batch."""
        batch_size = len(entities)
        num_entities = len(entities[0])

        def _prepare_input(entity):
            entity_label = entity["label"]
            if "gnn_add_entity_type" in self.config and self.config["gnn_add_entity_type"]:
                entity_type = _get_entity_type(entity)
                return f"{entity_label}{self.sep_token}{entity_type}"
            else:
                return f"{entity_label}"

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
            _prepare_input(entity)
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
