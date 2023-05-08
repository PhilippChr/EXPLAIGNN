import os
import random
import torch

from explaignn.question_understanding.question_understanding import QuestionUnderstanding
from explaignn.question_understanding.structured_representation.structured_representation_model import (
    StructuredRepresentationModel,
)


class StructuredRepresentationModule(QuestionUnderstanding):
    def __init__(self, config, use_gold_answers):
        """Initialize SR module."""
        super(StructuredRepresentationModule, self).__init__(config, use_gold_answers)

        # create model
        self.sr_model = StructuredRepresentationModel(config)
        self.model_loaded = False

        self.history_separator = config["history_separator"]
        self.sr_delimiter = config["sr_delimiter"]

    def train(self):
        """Train the model on silver SR data."""
        # train model
        self.logger.info(f"Starting training...")
        data_dir = self.config["path_to_annotated"]
        train_path = os.path.join(data_dir, "annotated_train.json")
        dev_path = os.path.join(data_dir, "annotated_dev.json")
        self.sr_model.train(train_path, dev_path)
        self.sr_model.save()
        self.logger.info(f"Finished training.")

    def inference_on_conversation(self, conversation):
        """Run inference on a single conversation."""
        # load SR model (if required)
        self.load()

        with torch.no_grad():
            # SR model inference
            history_turn_texts = list()
            for i, turn in enumerate(conversation["questions"]):
                self.inference_on_turn(turn, history_turn_texts)

                # only append answer if there is a next question
                if i + 1 < len(conversation["questions"]):
                    if self.use_gold_answers:
                        answer_text = ", ".join([answer["label"] for answer in turn["answers"]])
                    else:
                        answer_text = (
                            turn["ranked_answers"][0]["answer"]["label"]
                            if turn["ranked_answers"]
                            else "NONE"
                        )
                    history_turn_texts.append(answer_text)
        return conversation

    def inference_on_turn(self, turn, history_turn_texts):
        """Run inference on a single turn."""
        # load SR model (if required)
        self.load()

        with torch.no_grad():
            # SR model inference
            question = turn["question"]
            history_turn_texts.append(question)

            # skip turns which have been processed already (optimization for turn-wise runs)
            if "structured_representation" in turn:
                return turn

            # prepare input (omit gold answer(s))
            rewrite_input = self.history_separator.join(history_turn_texts)

            # run inference
            sr = self._inference(rewrite_input)
            turn["structured_representation"] = sr
            return turn

    def _inference(self, input_str):
        """Apply the model on the given input."""

        def _normalize_input(_input_str):
            _input_str = _input_str.replace(",", " ").replace(
                self.config["history_separator"].strip(), " "
            )
            _input_str = (
                _input_str.replace("?", " ").replace(".", " ").replace("!", " ").replace("'", "")
            )
            return _input_str

        def _normalize_sr(_sr):
            # drop type ("hallucination" is desired there)
            _sr = _sr.rsplit(self.config["sr_delimiter"], 1)[0]
            _sr = _sr.replace(",", " ").replace(self.config["sr_delimiter"].strip(), " ")
            return _sr

        def _hallucination(_input_words, _sr_words):
            """Check if the model hallucinated (except for type)."""
            hits = [word in " ".join(_input_words) for word in _sr_words]
            if False in hits:
                return True
            return False

        # try to avoid hallucination: get top-k SRs, and take first
        # that only output words that are there in input (except for type)
        if self.config.get("sr_avoid_hallucination"):
            srs = self.sr_model.inference_top_k(input_str)
            srs = [self._format_sr(sr) for sr in srs]

            # get input words
            input_words = _normalize_input(input_str).split()
            for sr in srs:
                sr_words = _normalize_sr(sr).split()
                # return first SR without hallucination
                if not _hallucination(input_words, sr_words):
                    return sr

            # if hallucination is there in all SR candidates, return the top-ranked
            return srs[0]
        # top-k
        elif self.config.get("sr_top_k"):
            srs = self.sr_model.inference_top_k(input_str)
            # format SR properly
            srs = [self._format_sr(sr) for sr in srs]
            return srs
        # compute top-k SRs and aggregate results
        elif self.config.get("sr_top_k_aggregation"):
            srs = self.sr_model.inference_top_k(input_str)

            all_contexts = dict()
            all_entities = dict()
            all_relations = dict()
            all_types = dict()

            for sr in srs:
                sr_list = sr.split("||", 3)
                if len(sr_list) < 4:
                    continue
                sr_list = [slot.strip() if slot else "_" for slot in sr_list]
                sr_context, sr_entities, sr_relation, sr_type = sr_list

                # collect data
                if not sr_context in all_contexts:
                    all_contexts[sr_context] = 0
                all_contexts[sr_context] += 1
                if not sr_entities in all_entities:
                    all_entities[sr_entities] = 0
                all_entities[sr_entities] += 1
                if not sr_relation in all_relations:
                    all_relations[sr_relation] = 0
                all_relations[sr_relation] += 1
                if not sr_type in all_types:
                    all_types[sr_type] = 0
                all_types[sr_type] += 1
        # default CONVINSE inference
        else:
            sr = self.sr_model.inference_top_1(input_str)
            sr = self._format_sr(sr)
            return sr

    def _format_sr(self, sr):
        """Make sure the SR has 3 delimiters."""
        slots = sr.split(self.sr_delimiter.strip(), 3)
        if len(slots) < 4 and not slots[0]:
            # type missing
            slots = slots + [""]
        elif len(slots) < 4:
            # topic missing
            slots = [""] + slots
        if len(slots) < 4:
            # in case there are still less than 4 slots
            slots = slots + (4 - len(slots)) * [""]
        sr = self.sr_delimiter.join(slots)
        return sr

    def adjust_sr_for_ablation(self, sr, ablation_type):
        """
        Adjust the given SR based on the specific ablation type.
        """
        sr = self._format_sr(sr)
        slots = sr.split(self.sr_delimiter, 3)
        context, entity, pred, ans_type = slots
        if ablation_type == "no_context":
            sr = f"{entity.strip()} {self.sr_delimiter} {pred.strip()} {self.sr_delimiter} {ans_type.strip()}"
        elif ablation_type == "no_entity":
            sr = f"{context.strip()} {self.sr_delimiter} {pred.strip()} {self.sr_delimiter} {ans_type.strip()}"
        elif ablation_type == "no_pred":
            sr = f"{context.strip()} {self.sr_delimiter} {entity.strip()} {self.sr_delimiter} {ans_type.strip()}"
        elif ablation_type == "no_type":
            sr = f"{context.strip()} {self.sr_delimiter} {entity.strip()} {self.sr_delimiter} {pred.strip()}"
        elif ablation_type == "no_structure":
            slots = [context, entity, pred, ans_type]
            random.shuffle(slots)
            sr = f"{slots[0].strip()} {self.sr_delimiter} {slots[1].strip()} {self.sr_delimiter} {slots[2].strip()} {self.sr_delimiter} {slots[3].strip()}"
        elif ablation_type == "full":
            sr = f"{context.strip()} {self.sr_delimiter} {entity.strip()} {self.sr_delimiter} {pred.strip()} {self.sr_delimiter} {ans_type.strip()}"
        else:
            raise Exception(f"Unknown ablation type: {ablation_type}")
        return sr

    def load(self):
        """Load the SR model."""
        # only load if not already done so
        if not self.model_loaded:
            self.sr_model.load()
            self.sr_model.set_eval_mode()
            self.model_loaded = True
