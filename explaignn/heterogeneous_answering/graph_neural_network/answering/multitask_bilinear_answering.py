import torch
import torch.nn.functional as F

from explaignn.heterogeneous_answering.graph_neural_network.answering.answering import Answering


class MultitaskBilinearAnswering(Answering):
    def __init__(self, config):
        super(MultitaskBilinearAnswering, self).__init__(config)

        self.emb_dimension = config["gnn_emb_dimension"]
        self.dropout = config["gnn_dropout"] if "gnn_dropout" in config else 0.0

        # bilinear answering
        self.answer_linear_projection = torch.nn.Linear(
            in_features=self.emb_dimension, out_features=self.emb_dimension
        )
        self.evidences_linear_projection = torch.nn.Linear(
            in_features=self.emb_dimension, out_features=self.emb_dimension
        )

        # loss fct
        self.loss_fct = torch.nn.BCELoss()

    def forward(self, batch, train, entity_mat, sr_vec, ev_mat, iteration=None, *args):
        """
        Arguments:
                - batch: the input batch
                - train: boolean
                - entity_mat (batch_size x num_ent x emb_dim): the entity encodings
                - sr_vec (batch_size x emb_dim): the SR vector
                - ev_mat (batch_size x num_ev x emb_dim): the evidence encodings
                - *args: other answering mechanisms require additional arguments
        """
        projected_entities = self.answer_linear_projection(entity_mat)  # batch_size x num_ent x emb
        projected_entities = F.dropout(
            projected_entities, self.dropout, training=train
        )  # batch_size x num_ent x emb
        answer_outputs = torch.bmm(
            projected_entities, sr_vec.unsqueeze(dim=2)
        )  # batch_size x num_ent x 1
        answer_logits = F.softmax(answer_outputs, dim=1)  # batch_size x num_ent x 1
        entity_mask = batch["entity_mask"]
        answer_logits = answer_logits.squeeze() * entity_mask
        answer_logits = answer_logits.clamp(min=1e-30) if self.config.get("gnn_clamping", False) else answer_logits

        projected_evidences = self.evidences_linear_projection(ev_mat)  # batch_size x num_ev x emb
        projected_evidences = F.dropout(
            projected_evidences, self.dropout, training=train
        )  # batch_size x num_ev x emb
        ev_outputs = torch.bmm(
            projected_evidences, sr_vec.unsqueeze(dim=2)
        )  # batch_size x num_ev x 1
        ev_logits = F.softmax(ev_outputs, dim=1)  # batch_size x num_ev x 1
        evidence_mask = batch["evidence_mask"]
        ev_logits = ev_logits.squeeze() * evidence_mask
        ev_logits = ev_logits.clamp(min=1e-30) if self.config.get("gnn_clamping", False) else ev_logits

        loss = None
        accuracies = list()
        qa_metrics = list()
        answer_predictions = list()
        evidence_predictions = list()
        if "entity_labels" in batch and "evidence_labels" in batch:
            # loss for answer prediction
            answer_loss = self.loss_fct(
                answer_logits.view(-1), batch["entity_labels"].view(-1).float()
            )

            # loss for answering evidence prediction
            evidence_loss = self.loss_fct(
                ev_logits.view(-1), batch["evidence_labels"].view(-1).float()
            )

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
            # print(f"Loss: {loss}, answer_loss: {answer_loss} ({torch.sum(batch['entity_mask'])}), evidence_loss: {evidence_loss} ({torch.sum(batch['evidence_mask'])})")

        # compute ranked answers and QA metrics (if possible)
        if not train:
            answer_predictions = self.add_ranked_answers(batch, answer_logits.unsqueeze(dim=2))
            if "gnn_max_output_evidences" in self.config:
                evidence_predictions = self.add_top_evidences(
                    batch, ev_logits.unsqueeze(dim=2), answer_logits.unsqueeze(dim=2)
                )
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
