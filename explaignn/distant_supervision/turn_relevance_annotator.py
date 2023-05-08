from explaignn.library.utils import get_logger


class TurnRelevanceAnnotator:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

    def annotate_turn_relevances(self, flow_graph, conversation):
        """
        Annotate turn relevances for the conversation from the graph.
        This method also extracts a dataset to train the turn relevance
        module in the following form:
        [relevance, (turn1, question1, answer_str1), (turn2, question2, answer_str2)]
        """
        if not flow_graph:
            return []
        questions = dict()
        answers = dict()
        explored_turns = set()
        # extract positive examples
        positive_examples = list()
        leafs = flow_graph["leafs"]
        while leafs:
            for node in leafs:
                turn_id = node["turn"]
                node_type = node["type"]
                node_str = str(turn_id) + node_type
                # skip if node already seen
                if node_str in explored_turns:
                    continue
                # remember explored turn
                explored_turns.add(node_str)
                if node["type"] == "answer":
                    answer = [
                        {"id": id_, "label": labels[0]}
                        for (id_, labels, surface_form, turn) in node["relevant_disambiguations"]
                    ]
                    answer_turn = node["turn"]
                    answers[answer_turn] = answer
                elif node["type"] == "question":
                    question = node["question"]
                    question_turn = node["turn"]
                    questions[question_turn] = question
                    self._initialize_turn_relevance(
                        question_turn, conversation
                    )  # remember that node was seen
                    parent_questions = self._get_parent_nodes(node)
                    for (parent_question_turn, parent_question) in parent_questions:
                        positive_examples.append(
                            [
                                1,
                                (parent_question_turn, parent_question),
                                (question_turn, question),
                            ]
                        )
                        self._add_relevant_turn(question_turn, parent_question_turn, conversation)
            leafs = [node for leaf in leafs for node in leaf["parents"]]

        # extract transitive turn_relevances from the positive examples
        turn_relevances = {turn: list() for turn in range(10)}
        if self.config["tr_transitive_relevances"]:
            for example in positive_examples:
                _, question1, question2 = example
                turn1, _ = question1
                turn2, _ = question2
                turn_relevances[turn2].append(turn1)
            turn_relevances = self._add_transitive_turn_relevances(turn_relevances, conversation)

        # extract negative examples
        negative_examples = list()
        for turn1 in questions:
            for turn2 in questions:
                if turn1 == turn2:
                    continue
                if not turn2 in turn_relevances[turn1]:
                    turn1_question = questions[turn1]
                    turn2_question = questions[turn2]
                    negative_examples.append([0, (turn2, turn2_question), (turn1, turn1_question)])

        # augment data with answers
        instances = positive_examples + negative_examples
        tr_data = list()
        for instance in instances:
            label, (turn1, question1), (turn2, question2) = instance
            tr_data.append(
                {
                    "relevance": label,
                    "history_turn": {
                        "turn": turn1,
                        "question": question1,
                        "answers": answers[turn1],
                    },
                    "current_turn": {
                        "turn": turn2,
                        "question": question2,
                        "answers": answers[turn2],
                    },
                }
            )

        # return data for turn relevance (required for training turn r module)
        return tr_data

    def _add_relevant_turn(self, current_turn, relevant_turn, conversation):
        """
        Add relevant_turn as relevant for the current turn.
        """
        if not relevant_turn in conversation["questions"][current_turn]["silver_relevant_turns"]:
            conversation["questions"][current_turn]["silver_relevant_turns"].append(relevant_turn)

    def _initialize_turn_relevance(self, current_turn, conversation):
        """
        Remember that node was found in graph.
        Aims to distinguish between turns for which no information was found
        due to their answer type (e.g. existentials; relevant turns: None),
        and turns which are found to be self-sufficient (relevant turns: empty list).
        """
        if conversation["questions"][current_turn]["silver_relevant_turns"] is None:
            conversation["questions"][current_turn]["silver_relevant_turns"] = list()

    def _get_parent_questions(self, node):
        """
        NOT IN USE: Extract parent question of the given node.
        Was used in earlier version!
        """
        if not node["parents"]:
            return list()
        parent_nodes = node["parents"]
        parent_nodes_copy = list()
        for parent_node in parent_nodes:
            if not parent_node["type"] == "question":
                new_parent_nodes = [
                    node for node in parent_node["parents"] if parent_node["type"] == "question"
                ]
                parent_nodes_copy += new_parent_nodes
            else:
                parent_nodes_copy.append(parent_node)
        parent_nodes = parent_nodes_copy
        parent_questions = [
            (parent_node["turn"], parent_node["question"]) for parent_node in parent_nodes
        ]
        # return question and turn
        return parent_questions

    def _get_parent_nodes(self, node):
        """
        Extract parent nodes of the given node.
        """
        if not node["parents"]:
            return list()
        parent_nodes = node["parents"]
        parent_nodes = [
            (parent_node["turn"], parent_node["question"]) for parent_node in parent_nodes
        ]
        return parent_nodes

    def _add_transitive_turn_relevances(self, turn_relevances, conversation):
        """
        Add the transitive turn_relevances from the single hop turn_relevances.
        """
        has_changed = True
        # iterate until nothing has changed in a loop
        while has_changed:
            has_changed = False
            new_turn_relevances = turn_relevances.copy()
            for child in turn_relevances:
                for parent in turn_relevances[child]:
                    for grandparent in turn_relevances[parent]:
                        if not grandparent in turn_relevances[child]:
                            new_turn_relevances[child].append(grandparent)
                            self._add_relevant_turn(child, grandparent, conversation)
                            has_changed = True
            turn_relevances = new_turn_relevances
        return turn_relevances

    def _answers_to_string(self, answers):
        """
        Transform the answer list string into text.
        """
        answers = answers.replace("[", "").replace("]", "").replace("'", "")
        return answers
