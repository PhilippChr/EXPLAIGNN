import json
from Levenshtein import distance as levenshtein_distance

from explaignn.evaluation import answer_presence
from explaignn.evidence_retrieval_scoring.clocq_er import ClocqRetriever
from explaignn.library.string_library import StringLibrary
from explaignn.library.utils import get_logger


class ConvFlowAnnotator:
    def __init__(self, clocq, config):
        self.config = config
        self.logger = get_logger(__name__, config)

        self.string_lib = StringLibrary(config)
        self.clocq = clocq
        self.retriever = ClocqRetriever(config)

    def get_conv_flow_graph(self, conversation):
        """
        Extract a noisy history graph for the given conversation.
        Questions asking for years, countries, and existential questions
        are dropped since the answering paths are usually noisy (spurious paths).
        """
        # load data
        turns = conversation["questions"]

        # parsing answers
        question_answer_pairs = [
            (turn["question"], [answer["id"] for answer in turn["answers"]])
            for turn in turns[1:]
        ]
        first_question = turns[0]["question"]
        first_answers = [answer["id"] for answer in turns[0]["answers"]]

        # prune QA pairs in case answer is date or "yes"/"no"
        if self.config["ds_prune_noisy_qa_pairs"] and self._prune_question_answer_pair(first_answers):
            return None

        # log loaded data
        self.logger.debug(f"First question: {first_question}")
        self.logger.debug(f"First answers: {first_answers}")

        # process initial question to initialize flow graph
        (
            first_relevant_disambiguations,
            answering_evidences,
        ) = self._get_relevant_entities_initial_turn(first_question, first_answers)
        if not first_relevant_disambiguations:
            return None

        # create node for question in flow graph
        question_node = {
            "question": first_question,
            "turn": 0,
            "relevant_disambiguations": first_relevant_disambiguations,
            "relevant_context": [],
            "answering_evidences": answering_evidences,
            "type": "question",
            "parents": [],
        }

        # create node for answer in flow graph
        answer_label, answer_disambiguations = self.transform_answers(first_answers, 0)
        answer_node = {
            "question": answer_label,
            "turn": 0,
            "relevant_disambiguations": answer_disambiguations,
            "relevant_context": [],
            "type": "answer",
            "parents": [question_node],
        }

        # create initial flow graph
        conv_flow_graph = {"leafs": [answer_node], "not_answered": []}

        # process follow-up questions to populate flow-graph
        for i, (question, answers) in enumerate(question_answer_pairs):
            turn = i + 1
            # prune QA pairs with "Yes"/"No"/years or countries as answer (too noisy matches)
            if self._prune_question_answer_pair(answers):
                conv_flow_graph["not_answered"].append([turn, question, answers])
                continue
            self.logger.debug(f"Question: {question}")
            self.logger.debug(f"Answers: {answers}")
            conv_flow_graph = self._get_relevant_entities_followup(
                question, answers, turn, conv_flow_graph
            )

        # return populated flow graph
        return conv_flow_graph

    def _get_relevant_entities_initial_turn(self, question, answers):
        """
        Get relevant entities for the first turn.
        """
        evidences, disambiguations = self.retriever.retrieve_evidences(question, self.config["ds_sources"])
        answering_evidences = self._get_answering_evidences(evidences, answers)
        disambiguation_tuples = self._get_answer_connecting_disambiguations(
            disambiguations, answering_evidences, answers, question, 0
        )
        return disambiguation_tuples, answering_evidences

    def _get_relevant_entities_followup(self, question, answers, turn, conv_flow_graph):
        """
        Get relevant entities for a follow-up question. Begin your graph traversal in the leaf nodes,
        and traverse up to the root, until the answer is found.
        """
        # remember whether answer was found
        answer_found = False

        # check whether question can be answered without context
        evidences, disambiguations = self.retriever.retrieve_evidences(question, self.config["ds_sources"])
        disambiguated_items = [item["item"]["id"] for item in disambiguations]
        answering_evidences = self._get_answering_evidences(evidences, answers)
        all_entities = set(disambiguated_items)
        if answering_evidences:
            answer_found = True
            # add disambiguation tuples (kb_item_id, surface_form, label) for answering evidences
            disambiguation_tuples = self._get_answer_connecting_disambiguations(
                disambiguations, answering_evidences, answers, question, turn
            )
        else:
            disambiguation_tuples = []

        # create a new node for the follow-up question
        new_node = {
            "question": question,
            "turn": turn,
            "relevant_disambiguations": disambiguation_tuples,
            "relevant_context": [],
            "answering_evidences": answering_evidences,
            "type": "question",
            "parents": [],
            "relation_shared_with": None,
        }

        # initialize traversal of context
        explored_turns = set()
        leafs_to_remove = list()  # remember leafs that need to be removed
        prev_question_node = None

        # explore leafs first, then go one layer higher (to leaf's parents)
        leafs = conv_flow_graph["leafs"]
        while leafs:
            for node in leafs:
                turn_id = node["turn"]
                node_str = str(turn_id) + node["type"]
                self.logger.debug(f"Consider node: {node}")

                if node_str in explored_turns:  # context question was already processed
                    continue

                # remember the context question
                if node["turn"] == (turn - 1) and node["type"] == "question":
                    prev_question_node = node
                explored_turns.add(node_str)

                # load disambiguations for the context question
                context_dis_tuples = node["relevant_disambiguations"]

                # bring disambiguations into CLOCQ format
                kb_item_tuple = [
                    {"item": {"id": item, "label": label}, "question_word": surface_form}
                    for item, surface_form, label, _ in context_dis_tuples
                ]

                # retrieve evidences for the context entities
                evidences = list()
                for item, surface_form, label, _ in context_dis_tuples:
                    # check whether item already processed
                    if item in all_entities:
                        continue
                    else:
                        all_entities.add(item)
                    self.logger.debug(f"Retrieve evidences for: {item}")

                    # retrieve evidences for context entity
                    retrieved_evidences = self.retriever.retrieve_evidences_for_entity({"id": item}, self.config["ds_sources"])
                    evidences += retrieved_evidences
                    self.logger.debug(f"Retrieved {len(retrieved_evidences)} evidences for {[item, surface_form, label]}")

                # get answer connecting evidences
                answering_evidences = self._get_answering_evidences(evidences, answers)

                # bring disambiguations into CLOCQ output format
                if answering_evidences:
                    answer_found = True
                    relevant_context_dis_tuples = self._get_answer_connecting_disambiguations(
                        kb_item_tuple, answering_evidences, answers, question, turn_id
                    )
                    new_node["relevant_context"] += relevant_context_dis_tuples
                    new_node["answering_evidences"] += answering_evidences
                    new_node["parents"].append(node)
                    # if this context node was a leaf, it won't be a leaf anymore (will be parent node of current question)
                    if node in conv_flow_graph["leafs"]:
                        leafs_to_remove.append(node)

            # go through next layer
            leafs = [node for leaf in leafs for node in leaf["parents"]]

            # log next layer
            self.logger.debug(f"Leafs: {[leaf['question'] for leaf in leafs]}")

        # check if the previous question node had the same predicate
        if prev_question_node and self._check_if_answering_paths_shared(
                new_node, prev_question_node
        ):
            new_node["relation_shared_with"] = prev_question_node["turn"]
            if not prev_question_node in new_node["parents"]:
                new_node["parents"].append(prev_question_node)

        # remove leaf nodes; if this is done on the fly, nodes are potentially skipped
        for node in leafs_to_remove:
            if node in conv_flow_graph["leafs"]:
                conv_flow_graph["leafs"].remove(node)

        # only add turn to flow graph if answer was found
        if answer_found:
            answer_label, answer_disambiguations = self.transform_answers(answers, turn)
            new_answer_node = {
                "question": answer_label,
                "turn": turn,
                "relevant_disambiguations": answer_disambiguations,
                "type": "answer",
                "relevant_context": [],
                "parents": [new_node],
            }
            conv_flow_graph["leafs"].append(new_answer_node)
        else:  # otherwise, append to list of unanswered questions
            conv_flow_graph["not_answered"].append([turn, question, answers])

        self.logger.debug(f"conv_flow_graph after turn {turn}: {conv_flow_graph}")

        return conv_flow_graph

    def _get_answering_evidences(self, evidences, answers):
        """
        Among the given evidences, extract the subset that has the answers. If the
        disambiguated item is already an answer, no need to return full 1-hop of
        such an answer (full 1-hop has answer => is answering).
        """
        # check whether answer is in evidence
        gold_answers = [{"id": answer, "label": answer} for answer in answers]
        _, answering_evidences = answer_presence(evidences, gold_answers, relaxed=True)

        # filter out evidences retrieved for the answer (this would lead to lots of spurious paths)
        answering_evidences = [
            evidence
            for evidence in answering_evidences
            if not set(ent["id"] for ent in evidence["retrieved_for_entity"]) & set(answers)
        ]

        self.logger.debug(f"gold_answers: {gold_answers}")
        self.logger.debug(f"answering_evidences: {answering_evidences}")
        return answering_evidences

    def _get_answer_connecting_disambiguations(
        self, disambiguations, answering_evidences, answers, question, turn_id
    ):
        """
        Extract the relevant disambiguations using the answering evidences.
        Returns disambiguation tuples, which have the following form:
        (kb_item_id, surface_forms, label, turn).
        There are multiple surface forms, since the same KB item can potentially
        be disambiguated for several different question words.
        """
        # create dict from item_id to surface forms
        inverse_disambiguations = dict() # dict from item_id to surface forms
        local_labels = dict() # store labels for each item id (for the disambiguations to map back)
        for disambiguation in disambiguations:
            surface_form = disambiguation["question_word"]
            item_id = disambiguation["item"]["id"]
            label = disambiguation["item"]["label"]

            # skip disambiguations that are an answer
            if item_id in answers:
                continue

            # skip disambiguations for stopwords (noise!)
            surface_form_words = self.string_lib.get_question_words(surface_form, ner=None)
            if len(surface_form_words) == 0:
                continue

            # store surface form
            if item_id in inverse_disambiguations:
                # check if surface form closer to label than existing
                # -> for cases in which e.g. "rapper" and "Eminem" are both disambiguated to "Eminem".
                old_surface_form = inverse_disambiguations[item_id]
                edit_distance1 = levenshtein_distance(old_surface_form, label)
                edit_distance2 = levenshtein_distance(surface_form, label)
                inverse_disambiguations[item_id] = old_surface_form if edit_distance1 < edit_distance2 else surface_form
            elif label in question:
                # if full label in question, use this instead (only if exact match present)
                inverse_disambiguations[item_id] = label
            else:
                inverse_disambiguations[item_id] = surface_form
            local_labels[item_id] = label

        # create disambiguation tuples
        disambiguation_tuples = list()
        for evidence in answering_evidences:
            # get items that led to the answering evidences coming into the context
            for item in evidence["wikidata_entities"]:
                item_id = item["id"]
                if item_id in inverse_disambiguations and self._valid_item(item):
                    surface_form = inverse_disambiguations[item_id]
                    label = local_labels[item_id]
                    if not (item_id, surface_form, label, turn_id) in disambiguation_tuples:
                        disambiguation_tuples.append((item_id, surface_form, label, turn_id))
        return disambiguation_tuples

    def _valid_item(self, item):
        """
        Verify that the item is valid.
        1.  Checks whether the item is very frequent. For frequent items,
            the occurence in an extracted evidence could be misleading.
        2.  Checks whether item is a predicate (predicates go into relation slot, not entity slot)
        """
        item_id = item["id"]
        if item_id[0] == "P":  # predicates are dropped
            return False
        if self._item_is_country(item_id):  # countries are always frequent
            return False
        freq1, freq2 = self.clocq.get_frequency(item_id)
        freq = freq1 + freq2
        return freq < 100000

    def _check_if_answering_paths_shared(self, question_node, previous_question_node):
        """
        Check if the two question nodes share an answering path
        if yes, they most likely share the same relation.

        Used for answering "what about x?"-like questions,
        but can also introduce noise.

        Only works with predicates within KB-facts.
        """
        answering_facts = [evidence["kb_fact"] for evidence in question_node["answering_evidences"] if evidence["source"] == "kb"]
        prev_answering_facts = [evidence["kb_fact"] for evidence in previous_question_node["answering_evidences"] if evidence["source"] == "kb"]

        answering_paths = [
            answering_fact[1]["id"] for answering_fact in answering_facts if len(answering_fact) > 1
        ]
        prev_answering_paths = [
            answering_fact[1]["id"]
            for answering_fact in prev_answering_facts
            if len(answering_fact) > 1
        ]
        intersection = set(answering_paths) & set(prev_answering_paths)
        return len(intersection) > 0

    def transform_answers(self, answers, turn):
        """
        Transform the answers into a string using the labels and the corresponding disambiguation dict.
        """
        answer_labels = []
        relevant_disambiguations = []
        for item in answers:
            if item[0] == "Q":
                label = self.clocq.get_label(item)
            else:
                label = item
            relevant_disambiguations.append((item, label, label, turn))
            answer_labels.append(label)
        return str(answer_labels), relevant_disambiguations

    def _item_is_country(self, item_id):
        """
        Check if the item is of type country.
        """
        if item_id[0] != "Q":
            return False
        types = self.clocq.get_types(item_id)

        if not types or types == ["None"]:
            return False
        type_ids = [type_["id"] for type_ in types]
        if "Q6256" in type_ids:  # country type
            return True
        return False

    def _prune_question_answer_pair(self, answers):
        """
        Some answers can trigger too many incorrect reasoning paths
        -> question-answer pairs are dropped.
        """
        if answers[0] in ["Yes", "No"]:
            return True
        elif any(
            (self.string_lib.is_year(answer) or self._item_is_country(answer)) for answer in answers
        ):
            return True
        return False

    def get_question_count(self, conv_flow_graph):
        """
        Count the number of questions in the history graph.
        """
        if not conv_flow_graph:
            return 0
        explored_turns = set()
        question_count = 0
        leafs = conv_flow_graph["leafs"]
        while leafs:
            for node in leafs:
                if node["type"] == "answer":
                    continue
                elif (str(node["turn"]) + node["type"]) in explored_turns:
                    continue
                explored_turns.add(str(node["turn"]) + node["type"])
                question_count += 1
            leafs = [node for leaf in leafs for node in leaf["parents"]]
        return question_count

    def print_dict(self, python_dict):
        """
        Print python dict as json.
        """
        # if self.verbose:
        jsonobject = json.dumps(python_dict)
        self.logger.info(jsonobject)