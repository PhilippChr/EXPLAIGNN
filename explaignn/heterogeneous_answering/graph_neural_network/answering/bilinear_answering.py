import torch
import torch.nn.functional as F

from explaignn.heterogeneous_answering.graph_neural_network.answering.answering import Answering


class BilinearAnswering(Answering):
    """
    Bilinear answering layer, that gets the entity encodings, and the structured representations as input,
    and predicts the answer scores/probabilities for each entity.
    """

    def __init__(self, config):
        super(BilinearAnswering, self).__init__(config)

        self.emb_dimension = config["gnn_emb_dimension"]
        self.dropout = config["gnn_dropout"] if "gnn_dropout" in config else 0.0

        # bilinear answering
        self.answer_linear_projection = torch.nn.Linear(
            in_features=self.emb_dimension, out_features=self.emb_dimension
        )

        # loss fct
        self.loss_fct = torch.nn.BCELoss()

    def forward(self, batch, train, entity_mat, sr_vec, *args):
        """
        Arguments:
                - batch: the input batch
                - train: boolean
                - entity_mat (batch_size x num_ent x emb_dim): the entity encodings
                - sr_vec (batch_size x emb_dim): the SR vector
                - *args: other answering mechanisms require additional arguments
        """
        projected_entities = self.answer_linear_projection(entity_mat)  # batch_size x num_ent x emb
        projected_entities = F.dropout(
            projected_entities, self.dropout, training=train
        )  # batch_size x num_ent x emb
        outputs = torch.bmm(projected_entities, sr_vec.unsqueeze(dim=2))  # batch_size x num_ent x 1
        logits = F.softmax(outputs, dim=1)  # batch_size x num_ent x 1
        mask = batch["entity_mask"]
        logits = logits.squeeze() * mask
        logits = logits.clamp(min=1e-30) if self.config.get("gnn_clamping", False) else logits

        loss = None
        accuracies = list()
        qa_metrics = list()
        answer_predictions = list()
        if "entity_labels" in batch:
            # flatten logits
            loss = self.loss_fct(logits.view(-1), batch["entity_labels"].view(-1).float())

        # compute ranked answers and QA metrics (if possible)
        if not train:
            answer_predictions = self.add_ranked_answers(batch, logits.unsqueeze(dim=2))
            if "gold_answers" in batch:
                qa_metrics = self.evaluate(batch, answer_predictions)
        return {
            "loss": loss,
            "accuracies": accuracies,
            "qa_metrics": qa_metrics,
            "answer_predictions": answer_predictions,
        }
