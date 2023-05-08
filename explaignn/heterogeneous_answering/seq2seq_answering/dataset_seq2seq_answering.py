import json
import torch

import explaignn.evaluation as evaluation


def input_to_text(turn):
    sr = turn["structured_representation"]
    evidences_text = "</e>".join(evidence["evidence_text"] for evidence in turn["top_evidences"])
    input_text = f"{sr}</sr>{evidences_text}"
    return input_text


class DatasetSeq2SeqAnswering(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path):
        self.config = config
        self.tokenizer = tokenizer

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
        inputs = list()
        outputs = list()

        # open data
        with open(path, "r") as fp:
            line = fp.readline()
            while line:
                conv = json.loads(line)
                for turn in conv["questions"]:
                    for answer in turn["answers"]:
                        # get evidences that will be kept in truncated input
                        # this could actually promote hallucination
                        max_input_length = self.config["ha_max_input_length"]
                        current_length = len(self.tokenizer(turn["structured_representation"]))
                        current_length += 1  # include </sr> token

                        evidences_in_input = list()
                        for evidence in turn["top_evidences"]:
                            evidence_tokens = len(self.tokenizer(evidence["evidence_text"]))
                            evidence_tokens += 1  # include </e> token
                            current_length += evidence_tokens
                            if current_length > max_input_length:
                                break
                            evidences_in_input.append(evidence)

                        # skip examples for which the specific answer is not in the evidences
                        if not evaluation.answer_presence(evidences_in_input, [answer]):
                            continue

                        # prepare input
                        input_text = input_to_text(turn)
                        inputs.append(input_text)

                        # prepare output
                        answer_label = answer["label"]
                        outputs.append(answer_label)
                        break
                line = fp.readline()

        # encode
        input_encodings = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config["ha_max_input_length"],
            return_tensors="pt",
        )
        output_encodings = self.tokenizer(
            outputs,
            padding=True,
            truncation=True,
            max_length=self.config["ha_max_output_length"],
            return_tensors="pt",
        )
        dataset_length = len(inputs)
        return input_encodings, output_encodings, dataset_length
