from explaignn.library.string_library import StringLibrary


class StructuredRepresentationAnnotator:
    def __init__(self, clocq, config):
        self.clocq = clocq
        self.config = config

        self.string_lib = StringLibrary(config)

        self.type_relevance_cache = dict()

    def annotate_structured_representations(self, flow_graph, conversation):
        """
        Get the abstract representations for the questions in the flow_graph.
        """
        # initialize
        structured_representations = dict()
        relation_shared_dict = list()
        explored_turns = set()
        leafs = flow_graph["leafs"]

        # search tree bottom-up
        while leafs:
            for node in leafs:
                turn_id = node["turn"]
                node_type = node["type"]
                node_str = str(turn_id) + node_type
                if node_str in explored_turns:
                    continue
                elif node["type"] == "question":
                    # extract SR
                    structured_representation = self._extract_structured_representation(
                        node, turn_id, conversation, relation_shared_dict
                    )
                    structured_representations[turn_id] = structured_representation

                    # add SR to data
                    conversation["questions"][turn_id]["silver_SR"].append(
                        structured_representation
                    )

                explored_turns.add(node_str)
            leafs = [node for leaf in leafs for node in leaf["parents"]]

        # add shared relations (if applicable)
        for turn_id, prev_turn_id in relation_shared_dict:
            prev_sr = structured_representations[prev_turn_id]
            prev_relation = prev_sr[2][-1]
            structured_representations[turn_id][2].append(prev_relation)

    def _extract_structured_representation(
        self, node, turn_id, conversation, relation_shared_dict
    ):
        """
        Extract a structured representation for the question.
        """
        question = node["question"]
        answers = conversation["questions"][turn_id]["answers"]

        disambiguations_context = node["relevant_context"]
        disambiguations_current = node["relevant_disambiguations"]

        # extract common disambiguations
        # -> avoid that same disambiguation is in context and entity slot!
        sr_context, sr_entities = self._extract_entities_and_context(
            disambiguations_context, disambiguations_current
        )

        # remove entity surface forms from question
        question = self._remove_surface_forms(question, disambiguations_current)

        # derive SR relation
        if self.config["sr_remove_stopwords"]:
            words = self.string_lib.get_question_words(question, ner=None)
            sr_relation = " ".join(words)
            sr_relation = self._normalize_relation_str(sr_relation)
            sr_relation = [sr_relation]
        else:
            # remove symbols
            sr_relation = [self._normalize_relation_str(question)]

        # remember to add shared relation (if applicable)
        if self.config["sr_relation_shared_active"]:
            # if the previous turn had the same predicate, append the previous turns relation here
            if node.get("relation_shared_with"):
                current_turn_id = node["turn"]
                prev_turn_id = node["relation_shared_with"]
                relation_shared_dict.append([current_turn_id, prev_turn_id])

        # derive SR answer type
        sr_answer_type = self._get_answer_type(question, answers)

        # create SR
        structured_representation = (
            sr_context,
            sr_entities,
            sr_relation,
            sr_answer_type,
        )
        return structured_representation

    def _remove_surface_forms(self, question, relevant_disambiguations):
        """
        Remove disambiguated surface forms from question. Sort surface forms by
        length to avoid problems: e.g. removing 'unicorn' before removing 'last unicorn'
        leads to a problem.
        """
        # derive set of surface forms
        distinct_surface_forms = set()
        for (item_id, surface_form, label, turn) in relevant_disambiguations:
            distinct_surface_forms.add(surface_form)
        # sort surface forms by string length
        distinct_surface_forms = sorted(distinct_surface_forms, key=lambda j: len(j), reverse=True)
        for surface_form in distinct_surface_forms:
            # mechanism to avoid lowering full question at this point
            start_index = question.lower().find(surface_form.lower())
            if not start_index == -1:
                end_index = start_index + len(surface_form)
                question = question[:start_index] + question[end_index:]
        return question

    def _extract_entities_and_context(self, disambiguations_context, disambiguations_current):
        """
        Returns the common disambiguations. We care only about surface forms here,
        but compare common items.
        Parameters:
        - disambiguations_context: disambiguations in previous turns
        - disambiguations_current: disambiguations in current turn
        """
        entities = set()
        context = set()
        common_disambiguations = set()
        common_disambiguations_items = set()

        # disambiguated item-ids from current question
        context_item_ids = [item for item, surface_form, label, turn in disambiguations_context]

        # other disambiguations, that are not in common, are entities
        for item, surface_form, label, turn in disambiguations_current:
            # check if entity was already in context (common disambiguation)
            if item in context_item_ids:
                common_disambiguations.add(surface_form)
                common_disambiguations_items.add(item)
            else:
                entities.add(surface_form)

        # common disambiguations are entities if no other existent
        if not entities:
            entities = common_disambiguations
            common_disambiguations = set()

        # go through disambiguations in context and check if any common disambiguations exist
        # sort in descending order by turn (most recent turn: entity, others: context)
        disambiguations_context = sorted(disambiguations_context, key=lambda j: j[3], reverse=True)
        for item, surface_form, label, turn in disambiguations_context:
            if item in common_disambiguations_items:
                continue
            elif not entities:
                # use most recent disambiguation as entity, remaining as context
                entities.add(surface_form)
            else:
                context.add(surface_form)

        # common disambiguations go into context (might be empty if already used)
        context.update(common_disambiguations)
        return list(context), list(entities)

    def _get_answer_type(self, question, answers):
        """
        Get the answer_type from the answer.
        In case the answer has multiple types, compute the most relevant type to the question.
        """
        if self.string_lib.is_year(answers[0]["label"]):
            return "year"
        elif self.string_lib.is_timestamp(answers[0]["id"]):
            return "date"
        elif self.string_lib.is_number(answers[0]["id"]):
            return "number"
        elif self.string_lib.is_entity(answers[0]["id"]):
            type_ = self._get_most_relevant_type(answers)
            if type_ is None:
                return ""
            return type_["label"]
        else:
            return "string"

    def _type_relevance(self, type_id):
        """
        Score the relevance of the type.
        """
        if self.type_relevance_cache.get(type_id):
            return self.type_relevance_cache.get(type_id)
        freq1, freq2 = self.clocq.get_frequency(type_id)
        type_relevance = freq1 + freq2
        self.type_relevance_cache[type_id] = type_relevance
        return type_relevance

    def _get_most_relevant_type(self, answers):
        """
        Get the most relevant type for the item, as given by the type_relevance funtion.
        """
        # fetch types
        all_types = list()
        for item in answers:
            item_id = item["id"]
            types = self.clocq.get_types(item_id)
            if not types:
                continue
            for type_ in types:
                if type_ != "None":
                    all_types.append(type_)
        if not all_types:
            return None
        # sort types by relevance, and take top one
        most_relevant_type = sorted(
            all_types, key=lambda j: self._type_relevance(j["id"]), reverse=True
        )[0]
        return most_relevant_type

    def _get_relevant_types(self, item):
        """
        NOT IN USE: Get only the relevant types for the item.
        E.g. Christopher Nolan has 11 different occupations, but only 3-4 are important.
        Implemented by matching with description, if no exact match found return all types.
        """
        all_types = self.clocq.get_types(item)
        description = descriptions.get(item)
        if not all_types:
            return "unknown"
        elif not description:
            return all_types
        # extract types that have an exact match in the description
        relevant_types = list()
        for candidate in all_types:
            candidate_label = candidate["label"]
            if candidate_label in description:
                relevant_types.append(candidate)
        # if no such exact match found, return all
        if not relevant_types:
            return all_types
        return relevant_types

    def _normalize_relation_str(self, relation_str):
        """Remove punctuation, whitespaces and lower the string."""
        relation_str = (
            relation_str.replace(",", "")
            .replace("!", "")
            .replace("?", "")
            .replace(".", "")
            .replace("'", "")
            .replace('"', "")
            .replace(":", "")
            .replace("’", "")
            .replace("{", "")
            .replace("}", "")
            .replace(" s ", " ")
        )
        while "  " in relation_str:
            relation_str = relation_str.replace("  ", " ")
        # relation_str = relation_str.lower()
        relation_str = relation_str.strip()
        return relation_str
