import json
import os
from nltk.tokenize import word_tokenize
from pathlib import Path

from explaignn.question_understanding.structured_representation.dataset_structured_representation import (
    output_to_text,
)


def postprocess_data(input_data, quretec_output_path):
    """Postprocess predictions by Question resolution model."""
    # open outputs
    with open(quretec_output_path, "r") as fp:
        quretec_pred = json.load(fp)

    # process predictions
    counter = 0
    for conversation in input_data:
        for turn in conversation["questions"]:
            # load input
            question = turn["question"]
            if turn["turn"] == 0:
                turn["structured_representation"] = question
                continue
            history_words = quretec_pred["x_input"][counter]
            predictions = quretec_pred["y_pred"][counter]
            # create completed
            qres = _process_prediction(question, history_words, predictions)
            # create instance
            turn["structured_representation"] = qres
            # next instance
            counter += 1
    return input_data


def postprocess_turn(turn, quretec_output_path):
    """Postprocess predictions by Question resolution model."""
    # open outputs
    with open(quretec_output_path, "r") as fp:
        quretec_pred = json.load(fp)

    # process predictions
    history_words = quretec_pred["x_input"][0]
    predictions = quretec_pred["y_pred"][0]

    # create completed
    question = turn["question"]
    qres = _process_prediction(question, history_words, predictions)
    # create instance
    turn["structured_representation"] = qres
    return turn


def _process_prediction(question, history_words, predictions):
    """Construct completed question from QuReTeC predictions."""
    # set of question words
    question_words_set = set(word_tokenize(question))

    # add words that were found relevant
    for i, pred in enumerate(predictions):
        if pred == "REL":  # word found relevant
            word = history_words[i]
            # skip if word already in question
            if word in question_words_set:
                continue

            # add word to question
            question += " " + word
            question_words_set.add(word)
    # return completed question
    return question


def prepare_data_for_inference(config, data, output_path, use_gold_answers=False):
    """Prepare data in the given split."""
    dataset_for_inference = list()

    question_id = 0

    for conversation in data:
        history = list()
        for i, turn in enumerate(conversation["questions"]):
            # append question to history
            question = turn["question"]
            history.append(question)

            # first turn does not have a rewrite
            if turn["turn"] == 0:
                continue

            turn_instance = _prepare_turn_for_inference(
                config, turn, history, question_id, use_gold_answers
            )
            dataset_for_inference.append(turn_instance)
            question_id += 1

            # append answer to history (if there is a next turn)
            if i + 1 < len(conversation["questions"]):
                if use_gold_answers:
                    answer_text = ", ".join([answer["label"] for answer in turn["answers"]])
                else:
                    answer_text = turn["pred_answers"][0]["label"]
                # append to history
                history.append(answer_text)

    # store in file
    with open(output_path, "w") as fp:
        json.dump(dataset_for_inference, fp)


def _prepare_turn_for_inference(config, turn, history, question_id, use_gold_answers=False):
    """Prepare a turn for inference and return result."""
    # append question to history
    question = turn["question"]
    qres_input_separator = config["qres_input_separator"]

    # prepare input
    history_text = " ".join(history)
    tokenized_history = word_tokenize(history_text)
    tokenized_question = word_tokenize(question)

    # create instance for distant supervision
    sr = turn.get("silver_SR")
    if sr:
        sr_text = output_to_text(sr, " ")
        tokenized_sr = word_tokenize(sr_text)
        bert_ner_overlap = _create_bert_ner_overlap(
            tokenized_history, tokenized_question, tokenized_sr, qres_input_separator
        )

        instance_distant_supervision = {
            "prev_questions": history_text,
            "cur_question": question,
            "answer_text": sr_text,
            "bert_ner_overlap": bert_ner_overlap,
            "id": str(question_id),
        }
        return instance_distant_supervision
    else:
        tokenized_input = tokenized_history + [qres_input_separator] + tokenized_question
        ner_info = (
            len(tokenized_history) * ["O"]
            + [qres_input_separator]
            + (len(tokenized_question) * ["O"])
        )
        bert_ner_overlap = [tokenized_input, ner_info]

        instance_distant_supervision = {
            "prev_questions": history_text,
            "cur_question": question,
            "answer_text": question,
            "bert_ner_overlap": bert_ner_overlap,
            "id": str(question_id),
        }
        return instance_distant_supervision


def prepare_turn_for_inference(config, turn, history_turns, output_path, use_gold_answers=False):
    """Prepare a turn for inference and store in file."""
    # process history
    history = list()
    for turn in history_turns:
        # append question
        question = turn["question"]
        history.append(question)

        # append answer
        if use_gold_answers:
            answer_text = ", ".join([answer["label"] for answer in turn["answers"]])
        else:
            answer_text = turn["pred_answers"][0]["label"]
        history.append(answer_text)

    # prepare turn
    turn_instance = _prepare_turn_for_inference(config, turn, history, 0, use_gold_answers)

    # store in file
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump([turn_instance], fp)


def prepare_data_for_training(config, train_path, dev_path):
    """Prepare data in the given split."""
    input_dir = config["path_to_annotated"]
    benchmark = config["benchmark"]
    intermediate_res = config["path_to_intermediate_results"]
    output_dir = os.path.join(intermediate_res, "qres", "data")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_for_train = os.path.join(output_dir, "train.json")
    _prepare_data_split_for_training(config, train_path, output_for_train)

    output_for_train = os.path.join(output_dir, "dev.json")
    _prepare_data_split_for_training(config, dev_path, output_for_train)


def _prepare_data_split_for_training(config, input_path, output_path):
    """Prepare data in the given split."""
    with open(input_path, "r") as fp:
        data = json.load(fp)

    dataset_for_train = list()
    qres_input_separator = config["qres_input_separator"]

    question_id = 0

    for conversation in data:
        history = list()
        for turn in conversation["questions"]:
            question = turn["question"]
            answer_text = " ".join([answer["label"] for answer in turn["answers"]])

            # append to history
            history.append(f"{question} {qres_input_separator} {answer_text}")
            # first turn does not have a rewrite
            if turn["turn"] == 0:
                continue

            # prepare input
            history_text = " ".join(history[:-1])
            tokenized_history = word_tokenize(history_text)
            tokenized_question = word_tokenize(question)

            # create instance for distant supervision
            sr = turn.get("silver_SR")
            if sr:
                sr_text = output_to_text(sr, " ")
                tokenized_sr = word_tokenize(sr_text)
                bert_ner_overlap = _create_bert_ner_overlap(
                    tokenized_history, tokenized_question, tokenized_sr, qres_input_separator
                )

                instance_distant_supervision = {
                    "prev_questions": history_text,
                    "cur_question": question,
                    "answer_text": sr_text,
                    "bert_ner_overlap": bert_ner_overlap,
                    "id": str(question_id),
                }
                dataset_for_train.append(instance_distant_supervision)

    # store in files
    with open(output_path, "w") as fp:
        json.dump(dataset_for_train, fp)


def _create_bert_ner_overlap(
    tokenized_history, tokenized_question, tokenized_gold, qres_input_separator
):
    """Create the required 'bert_ner_overlap' for the QuReTeC method."""
    gold_tokens = set(tokenized_gold)
    tokenized_input = tokenized_history + [qres_input_separator] + tokenized_question
    ner_info = list()
    for token in tokenized_history:
        if token in gold_tokens:
            ner_info.append("REL")
        else:
            ner_info.append("O")
    # can not provide any ner_info here
    ner_info = ner_info + [qres_input_separator] + (len(tokenized_question) * ["O"])
    return [tokenized_input, ner_info]
