import json
import os
from tqdm import tqdm

from explaignn.library.utils import store_json_with_mkdir, get_logger


class QuestionUnderstanding:
    """Abstract class for QU phase."""

    def __init__(self, config, use_gold_answers):
        """Initialize QU module."""
        self.config = config
        self.logger = get_logger(__name__, config)
        self.use_gold_answers = use_gold_answers

    def train(self):
        """Method used in case no training required for QU phase."""
        self.logger.info("QU - Module used does not require training.")

    def inference(self):
        """Run model on data and add predictions."""
        # inference: add predictions to data
        qu = self.config["qu"]
        input_dir = self.config["path_to_annotated"]
        output_dir = self.config["path_to_intermediate_results"]
        method_name = self.config["name"]

        input_path = os.path.join(input_dir, f"annotated_train.json")
        output_path = os.path.join(output_dir, qu, f"train_qu-{method_name}.json")
        self.inference_on_data_split(input_path, output_path)

        input_path = os.path.join(input_dir, f"annotated_dev.json")
        output_path = os.path.join(output_dir, qu, f"dev_qu-{method_name}.json")
        self.inference_on_data_split(input_path, output_path)

        input_path = os.path.join(input_dir, f"annotated_test.json")
        output_path = os.path.join(output_dir, qu, f"test_qu-{method_name}.json")
        self.inference_on_data_split(input_path, output_path)

    def inference_on_data_split(self, input_path, output_path):
        """Run model on data and add predictions."""
        self.logger.info(f"QU - Starting inference on {input_path}.")

        # open data
        with open(input_path, "r") as fp:
            data = json.load(fp)

        # model inference on given data
        self.inference_on_data(data)

        # store data
        store_json_with_mkdir(data, output_path)

        # log
        self.logger.info(f"QU - Inference done on {input_path}.")

    def inference_on_data(self, input_data):
        """Run model on data and add predictions."""
        # model inference on given data
        for conversation in tqdm(input_data):
            self.inference_on_conversation(conversation)
        return input_data

    def inference_on_conversation(self, conversation):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def inference_on_turn(self, turn, history_turn_texts):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def load(self):
        pass
