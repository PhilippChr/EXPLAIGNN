import json
import os
import torch

from explaignn.library.utils import extract_mapping_incomplete_complete
from explaignn.question_understanding.structured_representation.dataset_structured_representation import input_to_text


class DatasetQuestionRewriting(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path):
        self.config = config
        self.tokenizer = tokenizer
        self.history_separator = config["history_separator"]

        benchmark_path = config["benchmark_path"]
        train_path = os.path.join(benchmark_path, config["train_input_path"])
        dev_path = os.path.join(benchmark_path, config["dev_input_path"])
        data_paths = [train_path, dev_path]
        self.mapping_incomplete_to_complete = extract_mapping_incomplete_complete(data_paths)

        input_encodings, output_encodings, dataset_length = self._load_data(path)
        self.input_encodings = input_encodings
        self.output_encodings = output_encodings
        self.dataset_length = dataset_length

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        labels = self.output_encodings["input_ids"][idx]
        item = {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": labels,
        }
        return item

    def __len__(self):
        return self.dataset_length

    def _load_data(self, path):
        """
        Opens the file, and loads the data into
        a format that can be put into the model.

        The whole history is given as input.
        The complete question, as annotated in the dataset,
        is the gold output.
        """
        # open data
        with open(path, "r") as fp:
            dataset = json.load(fp)

        inputs = list()
        outputs = list()

        for conversation in dataset:
            history = list()
            for turn in conversation["questions"]:
                # skip initial turn: no rewrite required!
                if turn["turn"] == 0:
                    continue

                # create input
                inputs.append(input_to_text(history, turn, self.history_separator))

                # create output
                question = turn["question"]
                complete = self.mapping_incomplete_to_complete.get(question)
                outputs.append(complete)

                # append to history
                history.append(turn)

        input_encodings = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=self.config["qrew_max_input_length"]
        )
        output_encodings = self.tokenizer(
            outputs, padding=True, truncation=True, max_length=self.config["qrew_max_input_length"]
        )
        dataset_length = len(inputs)

        return input_encodings, output_encodings, dataset_length
