import json
import os
from tqdm import tqdm

from explaignn.library.utils import store_json_with_mkdir, store_jsonl_with_mkdir, get_logger


class HeterogeneousAnswering:
    def __init__(self, config):
        """Initialize HA module."""
        self.config = config
        self.logger = get_logger(__name__, config)

    def train(self, sources=("kb", "text", "table", "info"), random_seed=None):
        """Method used in case no training required for HA phase."""
        self.logger.info("Module used does not require training.")

    def inference(self):
        """Run HA on data and add answers for each source combination."""
        input_dir = self.config["path_to_annotated"]

        qu = self.config["qu"]
        ers = self.config["ers"]
        ha = self.config["ha"]
        method_name = self.config["name"]

        source_combinations = self.config["source_combinations"]
        for sources in source_combinations:
            sources_string = "_".join(sources)

            input_path = os.path.join(
                input_dir, qu, ers, sources_string, f"test_ers-{method_name}.jsonl"
            )
            output_path = os.path.join(
                input_dir, qu, ers, sources_string, ha, f"test_ha-{method_name}.json"
            )
            self.inference_on_data_split(input_path, output_path, sources)

    def inference_on_data_split(self, input_path, output_path, sources, train=False, jsonl=False):
        """
        Run HA on given data split.
        When train is set to `True`, teacher forcing for th
        GNN inference on the train set is enabled.
        """
        # open data
        input_turns = list()
        data = list()
        with open(input_path, "r") as fp:
            line = fp.readline()
            counter = 0
            while line:
                conversation = json.loads(line)
                input_turns += [turn for turn in conversation["questions"]]
                data.append(conversation)
                line = fp.readline()
                counter += 1
                if self.config.get("dev") and counter > 100:
                    break

        # inference
        self.inference_on_turns(input_turns, sources, train)

        # log results
        self.log_results(input_turns)

        # plot answer presence (added for iterative GNN)
        if "top_evidences" in input_turns[0]:
            ans_pres_list = [turn["answer_presence"] for turn in input_turns]
            ans_pres = sum(ans_pres_list) / len(ans_pres_list)
            ans_pres = round(ans_pres, 3)

            num_ev_list = [len(turn["top_evidences"]) for turn in input_turns]
            num_ev = sum(num_ev_list) / len(num_ev_list)
            num_ev = round(num_ev, 3)

            res_str = f"Gold answers - {sources_str} - Ans. pres. ({num_questions}): {ans_pres}"
            self.logger.info(res_str)
            res_str = f"Gold answers - {sources_str} - Num. evidences ({num_questions}): {num_ev}"
            self.logger.info(res_str)

        # store processed data
        if jsonl:
            store_jsonl_with_mkdir(data, output_path)
        else:
            store_json_with_mkdir(data, output_path)

    def inference_on_data(self, input_data, sources=("kb", "text", "table", "info"), train=False):
        """Run HA on given data."""
        input_turns = [turn for conv in input_data for turn in conv["questions"]]
        self.inference_on_turns(input_turns, sources, train)
        return input_data

    def inference_on_turns(self, input_turns, sources=("kb", "text", "table", "info"), train=False):
        """
        Run HA on a set of turns.
        When train is set to `True`, teacher forcing for th
        GNN inference on the train set is enabled.
        """
        for turn in tqdm(input_turns):
            self.inference_on_turn(turn, sources, train)
        return input_turns

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info"), *args):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def log_results(self, turns):
        """
        Compute the averaged result and log it.
        """
        # compute results
        p_at_1_list = [turn["p_at_1"] for turn in turns]
        p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
        p_at_1 = round(p_at_1, 3)
        mrr_list = [turn["mrr"] for turn in turns]
        mrr = sum(mrr_list) / len(mrr_list)
        mrr = round(mrr, 3)
        hit_at_5_list = [turn["h_at_5"] for turn in turns]
        hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
        hit_at_5 = round(hit_at_5, 3)
        num_questions = len(p_at_1_list)

        # log results
        sources_str = "_".join(sources)
        res_str = f"Gold answers - {sources_str} - P@1 ({num_questions}): {p_at_1}"
        self.logger.info(res_str)
        res_str = f"Gold answers - {sources_str} - MRR ({num_questions}): {mrr}"
        self.logger.info(res_str)
        res_str = f"Gold answers - {sources_str} - Hit@5 ({num_questions}): {hit_at_5}"
        self.logger.info(res_str)

    def load(self):
        pass

    def get_supporting_evidences(self, turn):
        """
        Get the supporting evidences for the answer.
        Default: return evidences connected to the predicted answer,
        in a post-hoc manner. This variant is model-agnostic.
        The actual computations of the HA approach are not considered.
        """
        if not turn["ranked_answers"]:
            return []
        answer_entity = turn["ranked_answers"][0]["answer"]
        supporting_evidences = [
            ev
            for ev in turn["top_evidences"]
            if answer_entity["id"] in [item["id"] for item in ev["wikidata_entities"]]
        ]
        return supporting_evidences

    def _get_evidence_sample(self, evidences):
        """Obtain a (random) subset of the answering evidences for the context or as supporting evidences."""
        evidences = evidences[: self.config["ha_max_supporting_evidences"]]
        return evidences

    def _get_data_dirs(self, sources):
        root_data_dir = self.config["path_to_intermediate_results"]
        qu = self.config["qu"]
        ers = self.config["ers"]
        ha = self.config["ha"]
        sources_str = "_".join(sources)
        input_dir = os.path.join(root_data_dir, qu, ers, sources_str)
        output_dir = os.path.join(root_data_dir, qu, ers, sources_str, ha)
        return input_dir, output_dir
