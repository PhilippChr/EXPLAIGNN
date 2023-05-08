import copy
import json
import os
import sys
import time
import torch
import random
import numpy as np

# ers
from explaignn.evidence_retrieval_scoring.clocq_bm25 import ClocqBM25
from explaignn.evidence_retrieval_scoring.rers import RERS

# ha
from explaignn.heterogeneous_answering.fid_module.fid_module import FiDModule
from explaignn.heterogeneous_answering.graph_neural_network.graph_neural_network import GNNModule
from explaignn.heterogeneous_answering.graph_neural_network.iterative_gnns import IterativeGNNs
from explaignn.heterogeneous_answering.seq2seq_answering.seq2seq_answering_module import (
    Seq2SeqAnsweringModule,
)
from explaignn.library.utils import (
    get_config,
    get_logger,
    get_result_logger,
    store_json_with_mkdir,
)

# qu
from explaignn.question_understanding.naive_concat.naive_concat import NaiveConcat
from explaignn.question_understanding.question_resolution.question_resolution_module import (
    QuestionResolutionModule,
)
from explaignn.question_understanding.question_rewriting.question_rewriting_module import (
    QuestionRewritingModule,
)
from explaignn.question_understanding.structured_representation.structured_representation_module import (
    StructuredRepresentationModule,
)

# reproducibility
SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"


class Pipeline:
    def __init__(self, config, use_gold_answers):
        """Create the pipeline based on the config."""
        # load config
        self.config = config
        self.logger = get_logger(__name__, config)
        self.result_logger = get_result_logger(config)
        self.use_gold_answers = use_gold_answers

        # load individual modules
        self.qu = self._load_qu(use_gold_answers)
        self.ers = self._load_ers()
        self.ha = self._load_ha()

        self.name = config["name"]

    def train(self, sources_str, modules_to_train=("qu", "ers", "ha")):
        """
        Train the given pipeline in the standard manner.
        First, train the QU phase (if required), run the inference on all sets,
        and then train the ERS (if required), run inference in all sets, and
        finally train the HA model.
        """
        sources = sources_str.split("_")

        # Question Understanding (QU)
        if "qu" in modules_to_train:
            self.qu.train()
            self.qu.inference()
        self.qu = None  # free up memory

        # Evidence Retrieval and Scoring (ERS)
        if "ers" in modules_to_train:
            self.ers.train()
            self.ers.inference(sources)
        self.ers = None  # free up memory

        # Heterogeneous Answering (HA)
        if "ha" in modules_to_train:
            self.ha.train(sources, random_seed=SEED)

    def source_combinations(self):
        """
        Run the pipeline using gold answers on all source combinations in the CONVINSE paper.
        """
        source_combinations = [
            "kb_text_table_info",
            "kb",
            "text",
            "table",
            "info",
            "kb_text",
            "kb_table",
            "kb_info",
            "text_table",
            "text_info",
            "table_info",
        ]
        self.run_all(source_combinations, clean_up=False)

    def run_all(self, sources_str, clean_up=True, dev=False):
        """
        Run the pipeline using gold answers for the next turns.
        `sources_str` can either be a single string, or a list of strings (several source combinations).
        The parameter `clean_up` controls whether the QU and ERS modules would
        be unset after their inference, for freeing up some memory.
        """
        # define output path
        if not isinstance(sources_str, list):
            source_combinations = [sources_str]
        else:
            source_combinations = sources_str
        output_dir = self.set_output_dir(source_combinations[0])

        # open data
        input_dir = self.config["benchmark_path"]
        if dev:
            dev_path = self.config["dev_input_path"]
            input_path = os.path.join(input_dir, dev_path)
        else:
            test_path = self.config["test_input_path"]
            input_path = os.path.join(input_dir, test_path)
        with open(input_path, "r") as fp:
            data = json.load(fp)

        self.logger.debug(f"len(input_data) {len(data)}")

        self.qu.inference_on_data(data)
        self.logger.info("Done with QU.")
        if clean_up:
            self.qu = None  # free up memory
        output_path = f"{output_dir}/res_{self.name}_qu.json"
        store_json_with_mkdir(data, output_path)

        # run inference on data
        for sources_str in source_combinations:
            input_data = copy.deepcopy(data)

            # define output path
            output_dir = self.set_output_dir(sources_str)

            sources = sources_str.split("_")
            self.ers.inference_on_data(input_data, sources)
            self.ers.store_cache()
            self.logger.info("Done with ER.")
            if clean_up:
                self.ers.evr = None  # free up memory
                self.ers = None  # free up memory
            output_path = f"{output_dir}/res_{self.name}_ers.json"
            store_json_with_mkdir(input_data, output_path)

            self.ha.inference_on_data(input_data, sources)
            self.logger.info("Done with HA.")
            output_path = f"{output_dir}/res_{self.name}_gold_answers.json"
            store_json_with_mkdir(input_data, output_path)

            # compute results
            p_at_1_list = [turn["p_at_1"] for conv in input_data for turn in conv["questions"]]
            p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
            p_at_1 = round(p_at_1, 3)
            num_questions = len(p_at_1_list)
            # log result
            res_str = f"Gold answers - {sources_str} - P@1 ({num_questions}): {p_at_1}"
            self.logger.info(res_str)
            self.result_logger.info(res_str)

            # compute results
            mrr_list = [turn["mrr"] for conv in input_data for turn in conv["questions"]]
            mrr = sum(mrr_list) / len(mrr_list)
            mrr = round(mrr, 3)
            # log result
            res_str = f"Gold answers - {sources_str} - MRR ({num_questions}): {mrr}"
            self.logger.info(res_str)
            self.result_logger.info(res_str)

            # compute results
            hit_at_5_list = [turn["h_at_5"] for conv in input_data for turn in conv["questions"]]
            hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
            hit_at_5 = round(hit_at_5, 3)
            # log result
            res_str = f"Gold answers - {sources_str} - H@5 ({num_questions}): {hit_at_5}"
            self.logger.info(res_str)
            self.result_logger.info(res_str)

    def run_turn_wise(self, sources_str, dev=False):
        """
        Run the instantiated pipeline, using the predicted answers of previous turns
        for generating the output of the QU phase.
        """

        def _next_turn_exists(_current_turn_id, _benchmark):
            """Check if there is a next turn. Assumes that all conversations have same length."""
            return any([(_current_turn_id < len(_conv["questions"])) for _conv in _benchmark])

        def _first_k_turns(_conversation, _k):
            """Get first k turns of conversation."""
            return _conversation["questions"][:_k]

        # define output path
        output_dir = self.set_output_dir(sources_str)

        # open data
        input_dir = self.config["benchmark_path"]
        if dev:
            dev_path = self.config["dev_input_path"]
            input_path = os.path.join(input_dir, dev_path)
        else:
            test_path = self.config["test_input_path"]
            input_path = os.path.join(input_dir, test_path)
        with open(input_path, "r") as fp:
            benchmark = json.load(fp)
            if self.config["dev"]:
                benchmark = benchmark[:5]

        # prepare input for first turn
        current_turn_id = 0
        output_prev_turn = benchmark.copy()

        # iterate through turns
        while _next_turn_exists(current_turn_id, benchmark):
            self.logger.info(f"Starting with turn {current_turn_id}")
            # prepare for next turn
            input_data = copy.deepcopy(benchmark)
            for i, conv in enumerate(input_data):
                if current_turn_id >= len(conv["questions"]):
                    input_data[i] = output_prev_turn[i]
                    continue
                conv_output_prev_turn = output_prev_turn[i].copy()
                conv["questions"] = _first_k_turns(conv_output_prev_turn, current_turn_id) + [
                    conv["questions"][current_turn_id]
                ]
                conv["questions"][-1]["context"] = (
                    conv["questions"][-2]["context"].copy() if current_turn_id else list()
                )

            ### QU
            self.logger.info(f"QU for turn {current_turn_id}")
            output_path = f"{output_dir}/qu_input_turn_{current_turn_id}.json"
            store_json_with_mkdir(input_data, output_path)
            self.qu.inference_on_data(input_data)

            ### ER
            self.logger.info(f"ERS for turn {current_turn_id}")
            output_path = f"{output_dir}/ers_input_turn_{current_turn_id}.json"
            input_turns = [
                conv["questions"][current_turn_id]
                for conv in input_data
                if current_turn_id < len(conv["questions"])
            ]
            sources = sources_str.split("_")
            store_json_with_mkdir(input_turns, output_path)
            self.ers.inference_on_turns(input_turns, sources)

            ### HA
            self.logger.info(f"HA for turn {current_turn_id}")
            self.ha.inference_on_turns(input_turns, sources)

            # prepare for next turn
            output_prev_turn = input_data
            current_turn_id += 1

            # store intermediate res
            output_path = f"{output_dir}/res_turn_{current_turn_id}.json"
            store_json_with_mkdir(output_prev_turn, output_path)

        # compute results
        p_at_1_list = [turn["p_at_1"] for conv in output_prev_turn for turn in conv["questions"]]
        p_at_1 = sum(p_at_1_list) / len(p_at_1_list)
        p_at_1 = round(p_at_1, 3)
        num_questions = len(p_at_1_list)
        # log result
        res_str = f"Iterative pipeline - {sources_str}, use_gold_answers={self.use_gold_answers} - P@1 ({num_questions}): {p_at_1}"
        self.logger.info(res_str)
        self.result_logger.info(res_str)

        # compute results
        mrr_list = [turn["mrr"] for conv in output_prev_turn for turn in conv["questions"]]
        mrr = sum(mrr_list) / len(mrr_list)
        mrr = round(mrr, 3)
        # log result
        res_str = f"Iterative pipeline - {sources_str}, use_gold_answers={self.use_gold_answers} - MRR ({num_questions}): {mrr}"
        self.logger.info(res_str)
        self.result_logger.info(res_str)

        # compute results
        hit_at_5_list = [turn["h_at_5"] for conv in output_prev_turn for turn in conv["questions"]]
        hit_at_5 = sum(hit_at_5_list) / len(hit_at_5_list)
        hit_at_5 = round(hit_at_5, 3)
        # log result
        res_str = f"Iterative pipeline - {sources_str}, use_gold_answers={self.use_gold_answers} - H@5 ({num_questions}): {hit_at_5}"
        self.logger.info(res_str)
        self.result_logger.info(res_str)

        # store cache
        self.qu = None
        self.ha = None
        self.ers.store_cache()

    def example(self):
        """Run pipeline on a single input turn."""
        turn = {
            "question_id": "0",
            "turn": 0,
            "answers": [{"id": "Q445772", "label": "Nikolaj Coster-Waldau"}],
            "question": "Who played Jaime Lannister in Game of Thrones?",
        }
        # pre-load models
        self.qu.load()
        self.ha.load()

        start = time.time()
        self.logger.info(f"Running QU")
        self.qu.inference_on_turn(turn, [])
        self.logger.debug(turn)
        self.logger.info(f"Time taken (QU): {time.time()-start} seconds")
        self.logger.info(f"Running ERS")
        self.ers.inference_on_turn(turn)
        self.logger.debug(turn)
        self.logger.info(f"Time taken (QU, ERS): {time.time()-start} seconds")
        self.logger.info(f"Running HA")

        self.ha.inference_on_turn(turn, "kb_text_table_info")
        self.logger.info(turn)
        self.logger.info(f"Time taken (ALL): {time.time()-start} seconds")

    def inference(self, turn, history_turns):
        """Run pipeline on single instance (e.g. for demo)."""
        self.qu.inference_on_turn(turn, history_turns)
        self.ers.inference_on_turn(turn)
        self.ha.inference_on_turn(turn)
        return turn

    def set_output_dir(self, sources_str):
        """Define path for outputs."""
        qu = self.config["qu"]
        ers = self.config["ers"]
        ha = self.config["ha"]
        path_to_intermediate_results = self.config["path_to_intermediate_results"]

        output_dir = os.path.join(path_to_intermediate_results, qu, ers, sources_str, ha)
        return output_dir

    def _load_qu(self, use_gold_answers):
        """Instantiate QU stage of CONVINSE pipeline."""
        qu = self.config["qu"]
        self.logger.info("Loading QU module")
        if qu.startswith("nc_"):
            return NaiveConcat(self.config, use_gold_answers)
        elif qu == "sr":
            return StructuredRepresentationModule(self.config, use_gold_answers)
        elif qu == "qrew":
            return QuestionRewritingModule(self.config, use_gold_answers)
        elif qu == "qres":
            return QuestionResolutionModule(self.config, use_gold_answers)
        else:
            raise ValueError(
                f"There is no available module for instantiating the QU phase called {qu}."
            )

    def _load_ers(self):
        """Instantiate ERS stage of CONVINSE pipeline."""
        ers = self.config["ers"]
        self.logger.info("Loading ERS module")
        if ers == "clocq_bm25":
            return ClocqBM25(self.config)
        elif ers == "rers":
            return RERS(self.config)
        else:
            raise ValueError(
                f"There is no available module for instantiating the ERS phase called {ers}."
            )

    def _load_ha(self):
        """Instantiate HA stage of CONVINSE pipeline."""
        ha = self.config["ha"]
        self.logger.info("Loading HA module")
        if ha == "fid":
            return FiDModule(self.config)
        elif ha == "seq2seq":
            return Seq2SeqAnsweringModule(self.config)
        elif ha == "gnn":
            return GNNModule(self.config)
        elif ha == "explaignn":
            return IterativeGNNs(self.config)
        else:
            raise ValueError(
                f"There is no available module for instantiating the HA phase called {ha}."
            )

    def load(self):
        """Load the individual modules."""
        self.qu.load()
        self.ers.load()
        self.ha.load()


def main():
    # check if provided options are valid
    if len(sys.argv) < 3:
        raise Exception(
            "Usage: python explaignn/pipeline.py <FUNCTION> <PATH_TO_CONFIG> [<SOURCES_STRING>]"
        )

    # load config
    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    # inference using predicted answers
    if function.startswith("--train"):
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config, use_gold_answers=True)
        modules_to_train_str = function.replace("--train", "")
        modules_to_train = ("qu", "ers", "ha") if not modules_to_train_str else [modules_to_train_str]
        pipeline.train(sources_str, modules_to_train)

    elif function == "--source-combinations":
        pipeline = Pipeline(config, use_gold_answers=True)
        pipeline.source_combinations()

    elif function == "--gold-answers":
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config, use_gold_answers=True)
        pipeline.run_all(sources_str)

    elif function == "--gold-answers-dev":
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config, use_gold_answers=True)
        pipeline.run_all(sources_str, dev=True)

    elif function == "--pred-answers":
        sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
        pipeline = Pipeline(config, use_gold_answers=False)
        pipeline.run_turn_wise(sources_str)

    elif function == "--example":
        pipeline = Pipeline(config, use_gold_answers=False)
        pipeline.example()

    else:
        raise Exception(f"Unknown function {function}!")


if __name__ == "__main__":
    main()
