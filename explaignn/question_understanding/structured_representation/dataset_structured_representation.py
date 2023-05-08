import json

import torch


def input_to_text(history_turns, current_turn, history_separator):
    """
    Transform the history turns and current turn into the input text.
    """
    # create history text
    history_text = history_separator.join(
        [_history_turn_to_text(history_turn, history_separator) for history_turn in history_turns]
    )

    # create input
    current_question = current_turn["question"]
    input_text = f"{history_text}{history_separator}{current_question}"
    return input_text


def _history_turn_to_text(history_turn, history_separator):
    """
    Transform the given history turn to text.
    """
    question = history_turn["question"]
    answers = history_turn["answers"]
    answers_text = " ".join([answer["label"] for answer in answers])
    history_turn_text = f"{question}{history_separator}{answers_text}"
    return history_turn_text


def output_to_text(silver_sr, sr_delimiter):
    """
    Transform the given silver abstract representation to text.
    The (recursive) list data structure is resolved and flattened.
    """
    topic, entities, relation, ans_type = silver_sr[0]

    # create individual components
    topic = " ".join(topic).strip()
    entities = " ".join(entities).strip()
    relation = " ".join(relation).strip()
    ans_type = ans_type.strip() if ans_type else ""

    # create ar text
    sr_text = (
        f"{topic} {sr_delimiter} {entities} {sr_delimiter} {relation} {sr_delimiter} {ans_type}"
    )

    # remove whitespaces in AR
    while "  " in sr_text:
        sr_text = sr_text.replace("  ", " ")
    sr_text.replace(" , ", ", ")
    sr_text = sr_text.strip()
    return sr_text


class DatasetStructuredRepresentation(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path):
        self.config = config
        self.tokenizer = tokenizer
        self.history_separator = config["history_separator"]
        self.sr_delimiter = config["sr_delimiter"]

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

        The input dataset should be annotated using
        the silver_annotation.py class.

        The whole history is given as input.
        """
        # open data
        with open(path, "r") as fp:
            dataset = json.load(fp)

        inputs = list()
        outputs = list()

        for conversation in dataset:
            history = list()
            for turn in conversation["questions"]:
                # skip examples for which no gold SR was found, or for first turn
                if not turn["silver_SR"]:
                    continue

                inputs.append(input_to_text(history, turn, self.history_separator))
                outputs.append(output_to_text(turn["silver_SR"], self.sr_delimiter))

                # append to history
                history.append(turn)

        # encode
        input_encodings = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=self.config["sr_max_input_length"]
        )
        output_encodings = self.tokenizer(
            outputs, padding=True, truncation=True, max_length=self.config["sr_max_input_length"]
        )
        dataset_length = len(inputs)

        return input_encodings, output_encodings, dataset_length
