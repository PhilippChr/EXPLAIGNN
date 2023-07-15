import torch
import transformers

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


    def initialize(self, config):
        # load params
        self.emb_dimension = config["gnn_emb_dimension"]

        self.max_input_length_sr = config["gnn_enc_sr_max_input"]
        self.max_input_length_ev = config["gnn_enc_ev_max_input"]
        self.max_input_length_ent = config["gnn_enc_ent_max_input"]

        # initialize LM
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
        elif config["gnn_encoder_lm"] == "RoBERTa":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
            self.model = transformers.AutoModel.from_pretrained("roberta-base")
            self.sep_token = " </s>"
        elif config["gnn_encoder_lm"] == "opt350":
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-350m")
            self.model = transformers.AutoModel.from_pretrained("facebook/opt-350m")
            self.sep_token = "#"
        else:
            raise Exception("Unknown architecture for Encoder module specified in config.")
            