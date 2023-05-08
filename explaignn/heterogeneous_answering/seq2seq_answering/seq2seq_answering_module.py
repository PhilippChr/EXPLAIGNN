import os
import os
import sys
import torch

import explaignn.evaluation as evaluation
import explaignn.heterogeneous_answering.seq2seq_answering.dataset_seq2seq_answering as dataset
from explaignn.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from explaignn.heterogeneous_answering.seq2seq_answering.seq2seq_answering_model import (
    Seq2SeqAnsweringModel,
)
from explaignn.library.utils import get_config, get_result_logger


class Seq2SeqAnsweringModule(HeterogeneousAnswering):
    def __init__(self, config):
        """Initialize seq2seq answering module."""
        super(Seq2SeqAnsweringModule, self).__init__(config)
        self.result_logger = get_result_logger(config)

        # create model
        self.ha_model = Seq2SeqAnsweringModel(config)
        self.model_loaded = False

    def train(self, sources=("kb", "text", "table", "info"), **kwargs):
        """Train the model on generated HA data (skip instances for which answer is not there)."""
        # train model
        self.logger.info(f"Starting training...")
        data_dir = self.config["path_to_intermediate_results"]
        # data_dir = "/home/pchristm/explaignn/CONVINSE/_intermediate_representations/convmix"

        qu = self.config["qu"]
        ers = self.config["ers"]
        sources_str = "_".join(sources)

        # train_path = os.path.join(data_dir, qu, ers, sources_str, "train_ers-convinse_e1k.jsonl")
        train_path = os.path.join(data_dir, qu, ers, sources_str, "demo_ers.jsonl")
        # train_path = os.path.join(data_dir, qu, ers, sources_str, "train_ers.jsonl")
        # dev_path = os.path.join(data_dir, qu, ers, sources_str, "dev_ers-convinse_e1k.jsonl")
        # dev_path = os.path.join(data_dir, qu, ers, sources_str, "dev_ers.jsonl")
        dev_path = os.path.join(data_dir, qu, ers, sources_str, "test_ers.jsonl")
        self.ha_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info")):
        """Run inference on a single turn."""
        with torch.no_grad():
            # load SR model (if required)
            self._load()

            # prepare input
            input_text = dataset.input_to_text(turn)

            # run inference
            generated_answer = self.ha_model.inference(input_text)
            turn["generated_answer"] = generated_answer
            ranked_answers = evaluation.get_ranked_answers(self.config, generated_answer, turn)
            turn["pred_answers"] = [
                {"id": ans["answer"]["id"], "label": ans["answer"]["label"], "rank": ans["rank"]}
                for ans in ranked_answers
            ]

            # eval
            if "answers" in turn:
                p_at_1 = evaluation.precision_at_1(ranked_answers, turn["answers"])
                turn["p_at_1"] = p_at_1
                mrr = evaluation.mrr_score(ranked_answers, turn["answers"])
                turn["mrr"] = mrr
                h_at_5 = evaluation.hit_at_5(ranked_answers, turn["answers"])
                turn["h_at_5"] = h_at_5

            # delete noise
            if turn.get("top_evidences"):
                del turn["top_evidences"]
            if turn.get("question_entities"):
                del turn["question_entities"]
            if turn.get("silver_SR"):
                del turn["silver_SR"]
            if turn.get("silver_relevant_turns"):
                del turn["silver_relevant_turns"]
            if turn.get("silver_answering_evidences"):
                del turn["silver_answering_evidences"]
            return turn

    def _load(self):
        """Load the SR model."""
        # only load if not already done so
        if not self.model_loaded:
            self.ha_model.load()
            self.ha_model.set_eval_mode()
            self.model_loaded = True


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python explaignn/heterogeneous_answering/seq2seq_answering/seq2seq_answering_module.py --<FUNCTION> <PATH_TO_CONFIG> [<SOURCES_STR>]"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        sam = Seq2SeqAnsweringModule(config)
        sources = sources_str.split("_")
        sam.train(sources=sources)

    # test: add predictions to data
    elif function == "--test":
        sam = Seq2SeqAnsweringModule(config)
        input_dir = config["path_to_annotated"]
        output_dir = config["path_to_intermediate_results"]

        qu = config["qu"]
        ers = config["ers"]
        ha = config["ha"]
        method_name = config["name"]
        architecture = config["ha_architecture"]
        sources = sources_str.split("_")
        input_path = os.path.join(input_dir, qu, ers, sources_str, f"test_ers.jsonl")
        # input_path = os.path.join(input_dir, qu, ers, sources_str, f"test_ers-convinse_e1k.jsonl")
        output_path = os.path.join(
            input_dir, qu, ers, sources_str, ha, f"test_ha_{architecture}.json"
        )
        sam.inference_on_data_split(input_path, output_path, sources)
        print("Done!")
