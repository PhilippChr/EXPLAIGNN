from Levenshtein import distance as levenshtein_distance

from explaignn.library.string_library import StringLibrary


def answer_presence(evidences, answers, relaxed=False):
    """
    Compute the answer presence for a set of evidences
    and a parsed answer dict, and return a list of
    answering evidences.
    Return format: (boolean, [evidence-dict, ...])
    """
    # initialize
    answer_present = False
    answering_evidences = list()

    # go through evidences
    for evidence in evidences:
        if evidence_has_answer(evidence, answers, relaxed):
            # remember evidence
            answer_present = True
            answering_evidences.append(evidence)
    # return results
    return answer_present, answering_evidences


def evidence_has_answer(evidence, gold_answers, relaxed=False):
    """Check whether the given evidence has any of the answers."""
    if gold_answers and gold_answers[0]["id"].lower() in ["yes", "no"]:
        return True
    for answer_candidate in evidence["wikidata_entities"]:
        # check for year in case the item is a timestamp
        answer_candidate_id = answer_candidate["id"]
        if relaxed and StringLibrary.is_timestamp(answer_candidate_id):
            year = StringLibrary.get_year(answer_candidate_id)
            if candidate_in_answers({"id": year, "label": year}, gold_answers):
                return True

        # check if answering candidate
        if candidate_in_answers(answer_candidate, gold_answers):
            return True
    return False


def candidate_in_answers(answer_candidate, gold_answers):
    """Check if candidate is answer."""
    # get ids
    answer_candidate_id = answer_candidate["id"]
    if gold_answers and "id" in gold_answers[0]:
        gold_answer_ids = [answer["id"] for answer in gold_answers]
    else:
        gold_answer_ids = []

    # normalize
    answer_candidate_id = answer_candidate_id.lower().strip().replace('"', "").replace("+", "")
    gold_answer_ids = [
        answer.lower().strip().replace('"', "").replace("+", "") for answer in gold_answer_ids
    ]

    # perform check
    if answer_candidate_id in gold_answer_ids:
        return True

    # no match found
    return False


def mrr_score(answers, gold_answers):
    """Compute MRR score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if candidate_in_answers(answer["answer"], gold_answers):
            return 1.0 / float(answer["rank"])
    return 0.0


def precision_at_1(answers, gold_answers):
    """Compute P@1 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(1.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


def hit_at_5(answers, gold_answers):
    """Compute Hit@5 score for given answers and gold answers."""
    # check if any answer was given
    if not answers:
        return 0.0
    # go through answer candidates
    for answer in answers:
        if float(answer["rank"]) > float(5.0):
            break
        elif candidate_in_answers(answer["answer"], gold_answers):
            return 1.0
    return 0.0


def get_ranked_answers(config, generated_answer, turn):
    """
    Convert the generated answer text to a Wikidata ID (or Yes/No),
    and return the ranked answers.
    Can be used for any method that predicts an answer string (instead of a KB item).
    """
    # check if existential (special treatment)
    question = turn["question"]
    if question_is_existential(question):
        ranked_answers = [
            {"answer": {"id": "yes", "label": "yes"}, "score": 1.0, "rank": 1},
            {"answer": {"id": "no", "label": "no"}, "score": 0.5, "rank": 2},
        ]
    # no existential
    else:
        # return dummy answer in case None was found (if no evidences found)
        if generated_answer is None:
            return [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
        all_answers = list()
        mentions = set()
        for evidence in turn["top_evidences"]:
            for disambiguation in evidence["disambiguations"]:
                mention = disambiguation[0]
                id = disambiguation[1]
                if id is None or id == False:
                    continue

                # skip duplicates
                ans = str(mention) + str(id)
                if ans in mentions:
                    continue
                mentions.add(ans)

                # exact match
                if generated_answer == mention:
                    diff = 0

                # otherwise compute Levenshtein distance
                else:
                    diff = levenshtein_distance(generated_answer, mention)

                all_answers.append({"answer": {"id": id, "label": mention}, "score": diff})

        sorted_answers = sorted(all_answers, key=lambda j: j["score"])
        ranked_answers = [
            {"answer": answer["answer"], "score": answer["score"], "rank": i + 1}
            for i, answer in enumerate(sorted_answers)
        ]

    # don't return all answers
    max_answers = config["ha_max_answers"]
    ranked_answers = ranked_answers[:max_answers]
    if not ranked_answers:
        ranked_answers = [{"answer": {"id": "None", "label": "None"}, "rank": 1, "score": 0.0}]
    return ranked_answers


def get_ranked_answers_for_top_k_strings(config, generated_answers, turn):
    """
    Convert the generated top-k answer texts to Wikidata IDs (or Yes/No),
    and return the ranked answers.
    Can be used for any method that predicts top-k answer strings.
    """
    question = turn["question"]
    if question_is_existential(question):
        ranked_answers = [
            {"answer": {"id": "yes", "label": "yes"}, "score": 1.0, "rank": 1},
            {"answer": {"id": "no", "label": "no"}, "score": 0.5, "rank": 2},
        ]
    # no existential
    else:
        ranked_answers = list()
        ranked_answers_set = set()  # store all answers that have already been predicted
        for i, generated_answer in enumerate(generated_answers):
            ranked_answers_for_generation = get_ranked_answers(config, generated_answer, turn)
            # skip answers that are already there in higher ranks
            ranked_answers_for_generation = [
                ans
                for ans in ranked_answers_for_generation
                if not ans["answer"]["id"] in ranked_answers_set
            ]
            if ranked_answers_for_generation:
                ranked_answer_for_generation = ranked_answers_for_generation[0]
                ranked_answer_for_generation["rank"] = i + 1
                ranked_answers.append(ranked_answer_for_generation)
                ranked_answers_set.add(ranked_answer_for_generation["answer"]["id"])
    return ranked_answers


def question_is_existential(question):
    existential_keywords = [
        "is",
        "are",
        "was",
        "were",
        "am",
        "be",
        "being",
        "been",
        "did",
        "do",
        "does",
        "done",
        "doing",
        "has",
        "have",
        "had",
        "having",
    ]
    lowercase_question = question.lower()
    lowercase_question = lowercase_question.strip()
    for keyword in existential_keywords:
        if lowercase_question.split()[0] == keyword:
            return True
    return False
