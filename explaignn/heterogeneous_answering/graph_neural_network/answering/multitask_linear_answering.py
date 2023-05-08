import torch
import torch.nn.functional as F

from explaignn.heterogeneous_answering.graph_neural_network.answering.answering import Answering


class MultitaskLinearAnswering(Answering):
    def __init__(self, config):
        super(MultitaskLinearAnswering, self).__init__(config)

        self.emb_dimension = config["gnn_emb_dimension"]
        self.dropout = config["gnn_dropout"] if "gnn_dropout" in config else 0.0

        # bilinear answering
        self.answer_linear = torch.nn.Linear(in_features=self.emb_dimension, out_features=2)
        self.evidences_linear = torch.nn.Linear(in_features=self.emb_dimension, out_features=2)

        # weigh positive labels higher than negatives (helped substantially)
        weights = torch.FloatTensor([3, 50])
        if torch.cuda.is_available():
            weights = weights.cuda()

        self.loss_fct = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, batch, train, entity_mat, sr_vec, ev_mat, *args):
        """
        Arguments:
                - batch: the input batch
                - train: boolean
                - entity_mat (batch_size x num_ent x emb_dim): the entity encodings
                - sr_vec (batch_size x emb_dim): the SR vector
                - ev_mat (batch_size x num_ev x emb_dim): the evidence encodings
                - *args: other answering mechanisms require additional arguments
        """
        outputs_ent = self.answer_linear(entity_mat).squeeze(dim=2)  # batch_size x num_ent x 2
        outputs_ent = F.dropout(
            outputs_ent, self.dropout, training=train
        )  # batch_size x num_ent x 2
        ent_logits = F.log_softmax(outputs_ent, dim=1)  # batch_size x num_ent x 2
        entity_mask = (
            batch["entity_mask"].unsqueeze(2).expand(-1, -1, 2)
        )  # expand to match the 2 outputs per entity
        answer_logits = ent_logits.squeeze() * entity_mask

        outputs_ev = self.evidences_linear(ev_mat).squeeze(dim=2)  # batch_size x num_ent x 2
        outputs_ev = F.dropout(outputs_ev, self.dropout, training=train)  # batch_size x num_ent x 2
        ev_logits = F.log_softmax(outputs_ev, dim=1)  # batch_size x num_ent x 2
        evidence_mask = (
            batch["evidence_mask"].unsqueeze(2).expand(-1, -1, 2)
        )  # expand to match the 2 outputs per evidence
        ev_logits = ev_logits.squeeze() * evidence_mask

        loss = None
        accuracies = list()
        qa_metrics = list()
        answer_predictions = list()
        evidence_predictions = list()
        if "entity_labels" in batch and "evidence_labels" in batch:
            # loss for answer prediction
            answer_loss = self.loss_fct(ent_logits.view(-1, 2), batch["entity_labels"].view(-1))
            accuracies = self.compute_balanced_accuracy(ent_logits, batch, "entity_labels")

            # loss for answering evidence prediction
            evidence_loss = self.loss_fct(ev_logits.view(-1, 2), batch["evidence_labels"].view(-1))

            # combined loss
            if "gnn_multitask_random" in self.config:
                weights = F.softmax(
                    torch.randn(2), dim=-1
                )  # Random Loss Weighting (RLW): https://openreview.net/pdf?id=OdnNBNIdFul
                answer_weight = weights[0]
                ev_weight = weights[1]
            elif "gnn_multitask_answer_weight" in self.config:
                answer_weight = self.config["gnn_multitask_answer_weight"]
                ev_weight = self.config["gnn_multitask_ev_weight"]
            else:
                answer_weight = 0.5
                ev_weight = 0.5
            loss = answer_weight * answer_loss + ev_weight * evidence_loss

        # compute QA metrics
        if not train:
            answer_predictions = self.add_ranked_answers(batch, answer_logits)
            if "gnn_max_output_evidences" in self.config:
                evidence_predictions = self.add_top_evidences(batch, ev_logits, answer_logits)
            if "gold_answers" in batch:
                qa_metrics = self.evaluate(batch, answer_predictions, evidence_predictions)

        return {
            "loss": loss,
            "accuracies": accuracies,
            "qa_metrics": qa_metrics,
            "answer_predictions": answer_predictions,
            "ev_logits": ev_logits,
            "evidence_predictions": evidence_predictions,
        }
