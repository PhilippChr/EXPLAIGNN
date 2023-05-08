import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

import explaignn.evaluation as evaluation


def collate_fn(batch):
    """Collate the input data for the batch."""

    def _is_vector(obj):
        return (type(obj) is torch.Tensor) or (
            type(obj).__module__ == np.__name__ and obj.dtype == np.float32
        )

    elem = batch[0]
    # collate instances and mappings (id->entity/evidence) separately
    instances = {
        key: default_collate([d[key] for d in batch]) for key in elem if _is_vector(elem[key])
    }
    mappings = {key: [d[key] for d in batch] for key in elem if not _is_vector(elem[key])}
    instances.update(mappings)
    return instances


class ContinueWithNext(Exception):
    pass


class DatasetGNN(Dataset):
    def __init__(self, config, data_path=None, train=False):
        self.config = config

        # load data
        self.instances = DatasetGNN.prepare_data(config, data_path, train)

    def __getitem__(self, idx):
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    @staticmethod
    def prepare_data(config, data_path, train=False):
        """
        Create matrix representations of each instance.
        This entails collecting the SR, evidences and entities,
        and creating the adjacency matrices.
        Embeddings are method specific, and are therefore derived in
        the inheriting GNN class(es).
        Here, we randomly sample some (value specified in config via gnn_train_max_evidences)
        negative examples from all evidences, but ensure that answer-connecting evidences are kept.
        """
        # load data
        dataset = list()
        with open(data_path, "rb") as fp:
            line = fp.readline()
            counter = 0
            while line:
                counter += 1
                conversation = json.loads(line)
                dataset.append(conversation)
                line = fp.readline()
                if config.get("dev") and counter > 10:
                    break

        # initialize
        instances = list()
        # go through conversations
        for conversation in tqdm(dataset):
            instances += DatasetGNN.prepare_conversation(config, conversation, train)
        ans_pres_list = [1 if sum(instance["evidence_labels"]) > 0 else 0 for instance in instances]
        ans_pres = sum(ans_pres_list) / len(ans_pres_list)
        ans_pres = round(ans_pres, 3)
        num_questions = len(ans_pres_list)
        print("Maximum answer presence", ans_pres)
        print("Number of questions", num_questions)
        return instances

    @staticmethod
    def prepare_conversation(config, conversation, train=False):
        """
        Create matrix representations of each instance in the conversation.
        """
        instances = list()
        for turn in conversation["questions"]:
            try:
                instance = DatasetGNN.prepare_turn(config, turn, train)
                instances.append(instance)
            except ContinueWithNext:
                continue
        return instances

    @staticmethod
    def prepare_turns(config, turns, train=False):
        """Prepare all turns."""
        return [DatasetGNN.prepare_turn(config, turn, train) for turn in turns]

    @staticmethod
    def prepare_turn(config, turn, train=False):
        # load data
        sr = turn["structured_representation"]
        question = turn["question"]
        evidences = turn["top_evidences"]
        gold_answers = turn["answers"] if "answers" in turn else list()

        # load params
        max_entities = config["gnn_max_entities"]
        max_evidences = config["gnn_max_evidences"]
        max_pos_evidences = config["gnn_train_max_pos_evidences"] if train else 0

        if len(evidences) > max_evidences:
            print("Assertion failed for ", json.dumps(turn))
        assert len(evidences) <= max_evidences

        if config["gnn_shuffle_evidences"]:
            random.shuffle(evidences)

        # skip instances for which the answer is not present, and existential questions
        if train:
            if not evaluation.answer_presence(evidences, gold_answers)[0] or gold_answers[0][
                "label"
            ].lower() in ["yes", "no"]:
                raise ContinueWithNext("No answer present in top evidences.")

        # mappings from entities and evidences to their index in the matrices
        entity_to_id = dict()
        evidence_to_id = dict()

        # mappings from local ids in matrixes to entity/evidence
        id_to_entity = np.zeros(max_entities, dtype=object)
        id_to_evidence = np.zeros(max_evidences, dtype=object)

        # initialize evidence and entity matrixes: |E| x d, |Ɛ| x d
        entities_list = list()
        evidences_list = list()

        # matrices that map from entities to evidences and vice-versa; |E| x |Ɛ|, |Ɛ| x |E|
        ent_to_ev = np.zeros((max_entities, max_evidences), dtype=np.float32)
        ev_to_ent = np.zeros((max_evidences, max_entities), dtype=np.float32)

        # gold answer distribution
        entity_labels = np.zeros(max_entities, dtype=np.float32)
        evidence_labels = np.zeros(max_evidences, dtype=np.float32)

        num_entities = 0
        num_evidences = 0

        # sampling
        if train:
            answering_evidences = [
                evidence
                for evidence in evidences
                if evaluation.evidence_has_answer(evidence, gold_answers)
            ]

            # prune instances with too many answering evidences (likely to be spurious path)
            if max_pos_evidences and len(answering_evidences) > max_pos_evidences:
                raise ContinueWithNext(
                    f"Too many answering evidences: {len(answering_evidences)} answering, with max_pos_evidences={max_pos_evidences}."
                )

        question_entity_ids = set()
        # iterate through evidences retrieved
        for i, evidence in enumerate(evidences):
            # load text
            evidence_text = evidence["evidence_text"]

            # add evidence emb
            g_ev_id = evidence_to_id.get(evidence_text)
            if g_ev_id is None:
                g_ev_id = num_evidences
                id_to_evidence[g_ev_id] = evidence
                num_evidences += 1
                evidence_to_id[evidence_text] = g_ev_id
                evidence["g_id"] = g_ev_id
                evidences_list.append(evidence)
            else:
                evidence["g_id"] = g_ev_id

            # add evidence to answering evidences distribution (for multitask learning)
            if evaluation.evidence_has_answer(evidence, gold_answers):
                evidence_labels[g_ev_id] = 1
                evidences_list[g_ev_id]["is_answering_evidence"] = True
                evidence["is_answering_evidence"] = True
            else:
                evidences_list[g_ev_id]["is_answering_evidence"] = False
                evidence["is_answering_evidence"] = False

            # get question entities (entities the evidence was retrieved for)
            question_entity_ids.update([qe["id"] for qe in evidence["retrieved_for_entity"]])

            # get evidence entities (will be connected to evidence)
            entities = evidence["wikidata_entities"]
            for entity in entities:
                entity["g_id"] = None
                entity_id = entity["id"]
                g_ent_id = entity_to_id.get(entity_id)
                # check if entity already known
                if g_ent_id is None:
                    ## add entity emb
                    # continue if max_entities reached
                    if num_entities == (max_entities - 1):
                        # print(turn["question_id"], "Dropping one entity;;;")
                        continue

                    g_ent_id = num_entities
                    id_to_entity[g_ent_id] = entity
                    num_entities += 1
                    entity_to_id[entity_id] = g_ent_id
                    entity["g_id"] = g_ent_id
                    entities_list.append(entity)
                else:
                    entity["g_id"] = g_ent_id

                # add entity to answer distribution
                if evaluation.candidate_in_answers(entity, gold_answers):
                    entity_labels[g_ent_id] = 1
                    entities_list[g_ent_id]["is_answer"] = True
                    entity["is_answer"] = True
                else:
                    entities_list[g_ent_id]["is_answer"] = False
                    entity["is_answer"] = False

                # set entries in adj. matrixes
                ent_to_ev[g_ent_id, g_ev_id] = 1
                ev_to_ent[g_ev_id, g_ent_id] = 1

        # set requires grad to False
        ent_to_ev = torch.from_numpy(ent_to_ev).to_sparse()
        ent_to_ev.requires_grad = False
        ev_to_ent = torch.from_numpy(ev_to_ent).to_sparse()
        ev_to_ent.requires_grad = False
        entity_labels = torch.from_numpy(entity_labels).type(torch.LongTensor)
        entity_labels.requires_grad = False
        evidence_labels = torch.from_numpy(evidence_labels).type(torch.LongTensor)
        evidence_labels.requires_grad = False

        # masking
        entity_mask = num_entities * [1] + (max_entities - num_entities) * [0]
        entity_mask = torch.FloatTensor(entity_mask)
        evidence_mask = num_evidences * [1] + (max_evidences - num_evidences) * [0]
        evidence_mask = torch.FloatTensor(evidence_mask)

        # padding
        entities_list = entities_list + (max_entities - num_entities) * [{"id": "", "label": ""}]
        evidences_list = evidences_list + (max_evidences - num_evidences) * [
            {"evidence_text": "", "wikidata_entities": [], "retrieved_for_entity": None}
        ]

        # add question entity flag
        for entity_id in question_entity_ids:
            if not entity_id in entity_to_id:
                continue
            g_ent_id = entity_to_id.get(entity_id)
            entities_list[g_ent_id]["is_question_entity"] = True
            # mask question entities (no answer candidates)
            if config["gnn_mask_question_entities"]:
                entity_labels[g_ent_id] = 0

        entity_labels_sum = torch.sum(entity_labels)
        if train and not entity_labels_sum:
            raise ContinueWithNext(
                f"Answer pruned via max_entities (or via masking) restriction: max_entities of {max_entities} reached."
            )

        # normalize adjacency matrix (ent_to_ev)
        vec = torch.sum(ent_to_ev.to_dense(), dim=0)
        vec[vec == 0] = 1
        ent_to_ev = ent_to_ev.to_dense() / vec

        # normalize adjacency matrix (ev_to_ent)
        vec = torch.sum(ev_to_ent.to_dense(), dim=0)
        vec[vec == 0] = 1
        ev_to_ent = ev_to_ent.to_dense() / vec

        # create final object
        instance = {
            "question_id": turn["question_id"],
            "entities": entities_list,
            "entity_mask": entity_mask,
            "evidences": evidences_list,
            "evidence_mask": evidence_mask,
            "ent_to_ev": ent_to_ev,
            "ev_to_ent": ev_to_ent,
            "entity_labels": entity_labels,
            "evidence_labels": evidence_labels,
            "id_to_entity": id_to_entity,
            "id_to_evidence": id_to_evidence,
            "sr": sr,
            "question": question,
            "gold_answers": gold_answers,
        }
        return instance
