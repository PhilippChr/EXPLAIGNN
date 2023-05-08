import networkx as nx

WIKIDATA_ENTITIES_SEP = "<BR>" + 5 * "&nbsp;"


class Graph:
    def __init__(self):
        """Create a new empty graph."""
        self.nx_graph = nx.Graph()
        self.nodes_dict = dict()
        self.ev_to_ent_dict = dict()
        self.ent_to_score = dict()

    def _add_entity(self, entity):
        g_ent_id = f'ent{entity["g_id"]}'
        self.nx_graph.add_node(
            g_ent_id,
            type="entity",
            entity_type=entity["type"] if "type" in entity and entity["type"] else "None",
            label=entity["label"],
            wikidata_id=entity["id"],
            is_question_entity="is_question_entity" in entity,
            is_answer="is_answer" in entity and entity["is_answer"],
            is_predicted_answer=False,
        )

    def _add_evidence(self, evidence):
        g_ev_id = f'ev{evidence["g_id"]}'
        self.nx_graph.add_node(
            g_ev_id,
            type="evidence",
            label=evidence["evidence_text"],
            source=evidence["source"],
            wikidata_entities=WIKIDATA_ENTITIES_SEP.join(
                [f'"{e["label"]}" => {e["id"]}' for e in evidence["wikidata_entities"]]
            ),
            retrieved_for_entity=str(evidence["retrieved_for_entity"]),
            is_answering_evidence="is_answering_evidence" in evidence
            and evidence["is_answering_evidence"],
        )

    def from_instance(self, instance):
        """Create a new graph from the given dataset instance."""
        # add entity nodes
        entities = instance["entities"]
        for entity in entities:
            if not "g_id" in entity:
                continue
            self._add_entity(entity)

        # add evidence nodes
        evidences = instance["evidences"]
        for evidence in evidences:
            if not "g_id" in evidence:
                continue
            self._add_evidence(evidence)

        # add edges
        ent_to_ev = instance["ent_to_ev"]
        for i, entity in enumerate(entities):
            if not "g_id" in entity:
                continue
            g_ent_id = f'ent{entity["g_id"]}'
            connected_ev_ids = ent_to_ev[i, :]
            for j, val in enumerate(connected_ev_ids):
                if val > 0:
                    g_ev_id = f"ev{j}"
                    self.nx_graph.add_edge(g_ent_id, g_ev_id)
        return self

    def from_scoring_output(self, scored_evidences, scored_entities):
        """Create an evidence-only graph from the outputs of the scoring phase."""

        for entity in scored_entities:
            self.ent_to_score[entity["id"]] = entity["score"]

        # add evidence nodes
        for evidence in scored_evidences:
            if not "g_id" in evidence:  # padded evidence
                continue
            self._add_evidence(evidence)
            node_id = f'ev{evidence["g_id"]}'
            self.nodes_dict[node_id] = evidence
            self.ev_to_ent_dict[node_id] = [
                entity["id"]
                for entity in evidence["wikidata_entities"]
                if entity["id"] in self.ent_to_score
            ]

        for i, evidence1 in enumerate(scored_evidences):
            if not "g_id" in evidence1:  # padded evidence
                continue
            for j, evidence2 in enumerate(scored_evidences):
                # avoid duplicate checks or checks with same item
                if i >= j or not "g_id" in evidence2:
                    continue
                # derive set of entities
                entities1 = set([entity["id"] for entity in evidence1["wikidata_entities"]])
                entities2 = set([entity["id"] for entity in evidence2["wikidata_entities"]])

                # if shared entity, there is a connection
                if entities1 & entities2:
                    g_ev_id1 = f'ev{evidence1["g_id"]}'
                    g_ev_id2 = f'ev{evidence2["g_id"]}'

                    # add edge
                    self.nx_graph.add_edge(g_ev_id1, g_ev_id2)
        return self

    def from_nx_graph(self, nx_graph):
        """Create an instance of "Graph" from a given nx graph."""
        self.nx_graph = nx_graph

    def write_to_file(self, file_path="/home/pchristm/public_html/graph.gexf"):
        """Write the graph to file."""
        xml = self.to_string()
        with open(file_path, "w") as fp:
            fp.write(xml)

    def to_string(self):
        """Write the graph to String."""
        xml_lines = nx.generate_gexf(self.nx_graph)
        xml = "\n".join(xml_lines)
        for i in range(20):
            # fix attributes in xml string
            i_str = str(i)
            if f'id="{i_str}" title="' in xml:
                title = xml.split(f'id="{i_str}" title="', 1)[1].split('"', 1)[0]
                xml = xml.replace(f'id="{i_str}" title="', f'id="{title}" title="')
                xml = xml.replace(f'for="{i_str}"', f'for="{title}"')
        xml = "<?xml version='1.0' encoding='utf-8'?>\n" + xml
        return xml

    def get_answer_neighborhood(self, answer_entity):
        """Get the 2-hop neighborhood of the answer in a graph (surrounding evidences->entities)."""
        graph = Graph()
        if not "g_id" in answer_entity:
            return self
        g_ent_id = f'ent{answer_entity["g_id"]}'
        self.nx_graph.nodes[g_ent_id]["is_predicted_answer"] = True
        graph.from_nx_graph(nx.ego_graph(self.nx_graph, g_ent_id, radius=2))
        return graph
