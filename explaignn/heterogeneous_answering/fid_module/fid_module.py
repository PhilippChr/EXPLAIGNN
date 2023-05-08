import json
import os
import sys
import time
from subprocess import Popen

import explaignn.evaluation as evaluation
import explaignn.heterogeneous_answering.fid_module.fid_utils as fid_utils
from explaignn.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from explaignn.library.utils import get_config


class FiDModule(HeterogeneousAnswering):
    def __init__(self, config):
        """Initialize the FiD module."""
        super().__init__(config)
        self.config = config
        self.path_to_fid = "explaignn/heterogeneous_answering/fid_module/FiD"
        self._initialize_conda_dir()

    def train(self, sources=("kb", "text", "table", "info"), **kwargs):
        """Train the FiD model on the dataset."""
        # set paths
        sources_string = "_".join(sources)
        input_dir = self.config["path_to_intermediate_results"]
        qu = self.config["qu"]
        ers = self.config["ers"]
        method_name = self.config["name"]
        train_path = os.path.join(
            input_dir, qu, ers, sources_string, f"train_ers-{method_name}.jsonl"
        )
        dev_path = os.path.join(input_dir, qu, ers, sources_string, f"dev_ers-{method_name}.jsonl")

        # load train data
        with open(train_path, "r") as fp:
            train_data = list()
            line = fp.readline()
            while line:
                train_data.append(json.loads(line))
                line = fp.readline()
            train_input_turns = [turn for conv in train_data for turn in conv["questions"]]
        # load dev data
        with open(dev_path, "r") as fp:
            dev_data = list()
            line = fp.readline()
            while line:
                dev_data.append(json.loads(line))
                line = fp.readline()
            dev_input_turns = [turn for conv in dev_data for turn in conv["questions"]]

        # prepare paths
        prepared_train_path, _ = self._prepare_paths()
        prepared_dev_path, _ = self._prepare_paths()

        # prepare data
        fid_utils.prepare_data(self.config, train_input_turns, prepared_train_path, train=True)
        fid_utils.prepare_data(self.config, dev_input_turns, prepared_dev_path, train=False)

        # free up memory
        del dev_input_turns
        del train_input_turns
        del dev_data
        del train_data

        # train
        self._train(prepared_train_path, prepared_dev_path, sources_string)

    def inference_on_turns(self, input_turns, sources, train=False):
        """Run HA on given turns."""
        # paths
        prepared_input_path, res_name = self._prepare_paths()

        # prepare data
        fid_utils.prepare_data(self.config, input_turns, prepared_input_path)

        # inference
        self._inference(res_name, prepared_input_path)

        # parse result
        path_to_result = f"{self.path_to_fid}/tmp_output_data/{res_name}/final_output.txt"
        generated_answers = self._parse_result(path_to_result)

        # add predicted answers to turns
        for turn in input_turns:
            self._postprocess_turn(turn, generated_answers)
        return input_turns

    def inference_on_turn(self, turn):
        """Run HA on a single turn."""
        # paths
        prepared_input_path, res_name = self._prepare_paths()

        # prepare data
        fid_utils.prepare_turn(self.config, turn, prepared_input_path, train=False)

        # inference
        self._inference(res_name, prepared_input_path)

        # parse result
        path_to_result = f"{self.path_to_fid}/tmp_output_data/{res_name}/final_output.txt"
        generated_answers = self._parse_result(path_to_result)

        # add predicted answers to turns
        self._postprocess_turn(turn, generated_answers)
        return turn

    def _train(self, prepared_train_path, prepared_dev_path, sources_string):
        benchmark = self.config["benchmark"]
        method_name = self.config["name"]
        name = f"{method_name}_{sources_string}"
        cmd = [self.path_to_fid_python_env, f"{self.path_to_fid}/train_reader.py"]
        cmd += ["--name", name]
        cmd += ["--checkpoint_dir", f"_data/{benchmark}/{method_name}/fid"]
        cmd += ["--train_data", prepared_train_path]
        cmd += ["--eval_data", prepared_dev_path]
        cmd += ["--model_size", "base"]
        cmd += ["--lr", str(self.config["fid_lr"])]
        cmd += ["--optim", str(self.config["fid_optim"])]
        cmd += ["--scheduler", str(self.config["fid_scheduler"])]
        cmd += ["--weight_decay", str(self.config["fid_weight_decay"])]
        cmd += ["--text_maxlength", str(self.config["fid_text_maxlength"])]
        cmd += ["--answer_maxlength", str(self.config["fid_answer_maxlength"])]
        cmd += ["--per_gpu_batch_size", str(self.config["fid_per_gpu_batch_size"])]
        cmd += ["--n_context", str(self.config["fid_max_evidences"])]
        cmd += ["--total_step", str(self.config["fid_total_step"])]
        cmd += ["--warmup_step", str(self.config["fid_warmup_step"])]
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()

    def _inference(self, res_name, prepared_input_path):
        """Run inference on a given question (or SR), and a set of evidences."""
        cmd = [self.path_to_fid_python_env, f"{self.path_to_fid}/test_reader.py"]
        cmd += ["--name", res_name]
        cmd += ["--model_path", self.config["fid_model_path"]]
        cmd += ["--checkpoint_dir", f"{self.path_to_fid}/tmp_output_data"]
        cmd += ["--eval_data", prepared_input_path]
        cmd += ["--n_context", str(self.config["fid_max_evidences"])]
        cmd += ["--per_gpu_batch_size", str(self.config["fid_per_gpu_batch_size"])]
        cmd += ["--write_results"]
        if self.config.get("fid_top_k"):
            cmd += ["--top_k", str(self.config["fid_top_k"])]
            cmd += ["--num_beams", str(self.config["fid_num_beams"])]
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.communicate()

    def _postprocess_turn(self, turn, generated_answers):
        ques_id = turn["question_id"]
        generated_answer = generated_answers.get(ques_id)
        turn["generated_answer"] = generated_answer

        # get ranked answers
        # top-k generated answers
        if self.config.get("fid_top_k") and self.config["fid_top_k"] > 1:
            ranked_answers = evaluation.get_ranked_answers_for_top_k_strings(
                self.config, generated_answer, turn
            )
        # best generated answer
        else:
            ranked_answers = evaluation.get_ranked_answers(self.config, generated_answer, turn)
        turn["ranked_answers"] = [
            {
                "answer": {
                    "id": ans["answer"]["id"],
                    "label": ans["answer"]["label"]
                },
                "rank": ans["rank"],
                "score": ans["score"] if "score" in ans else 0,
            }
            for ans in ranked_answers
        ]
        # eval
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

    def _parse_result(self, path_to_result):
        """
        Parse the output generated by FiD, and add predicted
        (and generated) answers to the data.
        """
        # get answers from output file
        generated_answers = dict()
        with open(path_to_result, "r") as fp:
            line = fp.readline()
            while line:
                try:
                    ques_id, answer_str = line.split(None, 1)
                except:
                    ques_id = line.strip()
                    answer_str = ""
                ques_id = ques_id.strip()
                answer_str = answer_str.strip()
                if self.config.get("fid_top_k") and int(self.config["fid_top_k"]) > 1:
                    generated_answers[ques_id] = answer_str.split("||")
                else:
                    generated_answers[ques_id] = answer_str
                line = fp.readline()
        return generated_answers

    def _prepare_paths(self):
        """Prepare random path for handling input/output with piped FiD process."""
        random_str = str(time.strftime("%y-%m-%d_%H-%M", time.localtime()))
        prepared_input_path = f"{self.path_to_fid}/tmp_input_data/data_{random_str}.jsonl"
        res_name = f"output_{random_str}"
        return prepared_input_path, res_name

    def _initialize_conda_dir(self):
        """Code to automatically detect and set the path to the FiD environment."""
        conda_dir = os.environ.get("CONDA_PREFIX", None)
        if not conda_dir:
            raise Exception(
                "Something went wrong! Tried accessing the value of the CONDA_PREFIX variable, but failed. Please make sure that you have a valid conda installation."
            )

        # in case some environment is activated (which should be `explaignn`), move one dir upwards
        if "envs" in conda_dir:
            conda_dir = os.path.dirname(conda_dir)
        else:
            conda_dir = os.path.join(conda_dir, "envs")
        self.path_to_fid_python_env = os.path.join(conda_dir, "fid", "bin", "python")

    def _load(self):
        pass


def main():
    if len(sys.argv) < 2:
        raise Exception(
            "python explaignn/heterogeneous_answering/fid_module/fid_module.py --<FUNCTION> <PATH_TO_CONFIG> [<SOURCES_STR>]"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
    sources = sources_str.split("_")
    config = get_config(config_path)

    if function == "--train":
        # train
        fid = FiDModule(config)
        sources = sources_str.split("_")
        fid.train(sources=sources)

    elif function == "--example":
        # set paths
        qu = config["qu"]
        ers = config["ers"]
        input_dir = config["path_to_intermediate_results"]
        path = os.path.join(input_dir, qu, ers, sources_str)
        input_path = os.path.join(path, "dev_ers.jsonl")

        with open(input_path, "r") as fp:
            line = fp.readline()
            conv = json.loads(line)
        turn = conv["questions"][0]

        # run inference on example
        fid = FiDModule(config)

        start = time.time()
        res = fid.inference_on_turn(turn)
        print(res)
        print(f"Spent {time.time()-start} seconds!")

    elif function == "--test":
        fid = FiDModule(config)
        input_dir = config["path_to_annotated"]
        output_dir = config["path_to_intermediate_results"]

        qu = config["qu"]
        ers = config["ers"]
        ha = config["ha"]
        method_name = config["name"]
        input_path = os.path.join(input_dir, qu, ers, sources_str, f"test_ers-{method_name}.jsonl")
        output_path = os.path.join(
            input_dir, qu, ers, sources_str, ha, f"test_ha-{method_name}.json"
        )
        fid.inference_on_data_split(input_path, output_path, sources)
        print("Done!")


if __name__ == "__main__":
    main()
