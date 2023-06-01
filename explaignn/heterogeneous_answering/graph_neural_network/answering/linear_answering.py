import torch
import torch.nn.functional as F

from explaignn.heterogeneous_answering.graph_neural_network.answering.answering import Answering


class LinearAnswering(Answering):
    """
    Linear answering layer, that gets the entity encodings as input, and predicts their answer scores/probabilities.
    """

    def __init__(self, config):
        super(LinearAnswering, self).__init__(config)

        self.emb_dimension = config["gnn_emb_dimension"]
        self.dropout = config["gnn_dropout"] if "gnn_dropout" in config else 0.0

        # answer linear
        self.answer_linear = torch.nn.Linear(in_features=self.emb_dimension, out_features=2)

        # weigh positive labels higher than negatives (helped substantially)
        weights = torch.FloatTensor([3, 50])
        if torch.cuda.is_available():
            weights = weights.cuda()

        # loss function
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, batch, train, entity_mat, *args):
        """
        Arguments:
                - batch: the input batch
                - train: boolean
                - entity_mat (batch_size x num_ent x emb_dim): the entity encodings
                - *args: other answering mechanisms require additional arguments
        """
        outputs = self.answer_linear(entity_mat).squeeze(dim=2)  # batch_size x num_ent x 2
        outputs = F.dropout(outputs, self.dropout, training=train)  # batch_size x num_ent x 2
        logits = F.log_softmax(outputs, dim=1)  # batch_size x num_ent x 2
        mask = (
            batch["entity_mask"].unsqueeze(2).expand(-1, -1, 2)
        )  # expand to match the 2 outputs per entity
        logits = logits * mask
        logits = logits.clamp(min=1e-30) if self.config.get("gnn_clamping", False) else logits

        loss = None
        accuracies = list()
        qa_metrics = list()
        answer_predictions = list()
        if "entity_labels" in batch:
            loss = self.loss_fct(logits.view(-1, 2), batch["entity_labels"].view(-1))
            accuracies = self.compute_balanced_accuracy(logits, batch)

        # compute QA metrics
        if not train and "gold_answers" in batch:
            answer_predictions = self.add_ranked_answers(batch, logits)
            qa_metrics = self.evaluate(batch, answer_predictions)
        return {
            "loss": loss,
            "accuracies": accuracies,
            "qa_metrics": qa_metrics,
            "answer_predictions": answer_predictions,
        }
