import json
import os
from pathlib import Path

from explaignn.evaluation import evidence_has_answer


def prepare_turn(config, input_turn, output_path, train=False):
    """
    Prepare the given turn for input into FiD.
    Input will be top-100 evidences per question
    as predicted by ERS stage.
    Writes the result in the given output path.
    """
    # create output dir
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # prepare
    res = _prepare_turn(config, input_turn, train)
    if res is None:
        sr = input_turn["structured_representation"]
        raise Exception(f"No evidences found for this turn! SR: {sr}.")

    # store
    with open(output_path, "w") as fp:
        fp.write(json.dumps(res))
        fp.write("\n")


def prepare_data(config, input_turns, output_path, train=False):
    """
    Prepare the given data for input into FiD.
    Input will be top-100 evidences per question
    as predicted by ERS stage.
    """
    # create output dir
    output_dir = os.path.dirname(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # process data
    with open(output_path, "w") as fp_out:
        # transform
        for turn in input_turns:
            # skip instances that are already processed
            if "pred_answers" in turn:
                continue

            res = _prepare_turn(config, turn, train)
            # skip turns for which no evidences were found
            if res is None:
                continue

            # write new
            fp_out.write(json.dumps(res))
            fp_out.write("\n")


def _prepare_turn(config, input_turn, train):
    """
    Prepare the given turn for input into FiD.
    Input will be top-100 evidences per question
    as predicted by ERS stage.
    Returns the object. For internal usage!
    """
    # construct set of answers that are present (from silver evidences)
    answer_ids = [answer["id"] for answer in input_turn["answers"]]

    # prepare target answers
    target_answers = set()
    # retrieve target answers from answering evidences -> preserve order!
    evidences = input_turn["top_evidences"]
    for evidence in evidences:
        if evidence_has_answer(evidence, input_turn["answers"]):
            for disambiguation in evidence["disambiguations"]:
                if disambiguation[1] in answer_ids:
                    target_answers.add(disambiguation[0])

    # if no answer can be found, skip instance during train/dev!
    if train and not target_answers:
        return None

    # if no answer in dataset, skip (fix for TimeQuestions dataset)
    if not input_turn["answers"]:
        return None

    evidences = input_turn["top_evidences"]
    evidences = evidences[: config["fid_max_evidences"]]

    # create data
    answers = list(target_answers) + [answer["label"] for answer in input_turn["answers"]]
    target_answer = answers[0]  # always first element of target_answers

    # error handling
    for evidence in evidences:
        if isinstance(evidence["retrieved_for_entity"], list):
            if not evidence["retrieved_for_entity"][0].get("label"):
                evidence["retrieved_for_entity"][0]["label"] = evidence["retrieved_for_entity"][0][
                    "id"
                ]

    # format evidences into FiD input format
    evidences = [
        {
            "title": evidence["retrieved_for_entity"][0]["label"]
            if isinstance(evidence["retrieved_for_entity"], list)
            else evidence["retrieved_for_entity"]["label"],
            "text": evidence["evidence_text"],
        }
        for evidence in input_turn["top_evidences"]
    ]

    # "pad" evidences
    if len(evidences) < 100:
        evidences = evidences + (100 - len(evidences)) * [{"title": "None", "text": "None"}]

    # if there are no evidences, return None (=skip instance)
    if not evidences:
        return None

    # return transformed instance
    return {
        "id": input_turn["question_id"],
        "question": input_turn["structured_representation"],
        "target": target_answer,
        "answers": answers,
        "ctxs": evidences,
    }
