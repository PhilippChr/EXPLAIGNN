import copy
import sys

from explaignn.evaluation import answer_presence
from explaignn.evidence_retrieval_scoring.bm25_es import BM25Scoring
from explaignn.evidence_retrieval_scoring.clocq_er import ClocqRetriever
from explaignn.evidence_retrieval_scoring.evidence_retrieval_scoring import EvidenceRetrievalScoring
from explaignn.library.utils import get_config


class RERS(EvidenceRetrievalScoring):
    # (R)estricted (E)vidence (R)etrieval (S)coring
    # -> Prune evidences based on SR slots
    def __init__(self, config):
        super(RERS, self).__init__(config)
        self.evr = ClocqRetriever(config)
        self.evs = BM25Scoring(config)
        self.sr_delimiter = config["sr_delimiter"].strip()

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info"), train=False):
        """Retrieve best evidences for SR."""
        structured_representation = turn["structured_representation"]
        evidences, _ = self.retrieve_evidences(structured_representation, sources)

        top_evidences = self.evs.get_top_evidences(structured_representation, evidences)

        # store result in instance
        turn["top_evidences"] = top_evidences
        turn["num_evidences"] = [len(top_evidences)]

        # answer presence
        if "answers" in turn:
            hit, answering_evidences = answer_presence(top_evidences, turn["answers"])
            turn["answer_presence"] = hit
            turn["answer_presence_per_src"] = {
                evidence["source"]: 1 for evidence in answering_evidences
            }
        return top_evidences

    def retrieve_evidences(
        self, structured_representation, sources=("kb", "text", "table", "info")
    ):
        """
        Retrieve evidences for the given SR.
        Skip evidences for disambiguations in the relation and type fields.
        Reuses code from retrieve_evidences and retrieve_KB functions in the ClocqRetriever class.
        """
        evidences, disambiguations = self._retrieve_kb_facts(structured_representation)

        # wikipedia evidences (only if required)
        if any(src in sources for src in ["text", "table", "info"]):
            for disambiguation in disambiguations:
                evidences += self.evr.retrieve_wikipedia_evidences(disambiguation["item"])

        # config-based filtering
        evidences = self.evr.filter_evidences(evidences, sources)
        return evidences, disambiguations

    def _retrieve_kb_facts(self, structured_representation):
        """Retrieve evidences from KB."""
        # look-up cache
        if self.evr.use_cache and structured_representation in self.evr.cache:
            return copy.deepcopy(self.evr.cache[structured_representation])

        # normalize SR
        sr_text = structured_representation.replace(self.sr_delimiter, " ")
        while "  " in sr_text:
            sr_text = sr_text.replace("  ", " ")

        # apply CLOCQ
        try:
            clocq_result = self.evr.clocq.get_search_space(
                sr_text, parameters=self.config["clocq_params"], include_labels=True, include_type=True
            )
        except:
            self.logger.info(f"Problem when retrieving search space for SR: {sr_text}")
            return [], []

        # get mentions in context and entity slot
        sr_context = structured_representation.split(self.sr_delimiter)[0].strip()
        sr_entity = structured_representation.split(self.sr_delimiter)[1].strip()
        sr_mentions_text = sr_context + " " + sr_entity

        # get entities disambiguated for context or entity slot
        disambiguations = [
            item
            for item in clocq_result["kb_item_tuple"]
            if item["question_word"] in sr_mentions_text
        ]

        sr_entities_set = set([item["item"]["id"] for item in disambiguations])

        # remember potential duplicate facts
        evidence_texts = set()

        # transform facts to evidences
        evidences = list()
        for fact in clocq_result["search_space"]:
            # identify entities the fact was retrieved for from clocq
            # -> only consider the ones of interest (i.e. retrieved for mentions in the context/entity slot)
            retrieved_for = [item for item in fact if item["id"] in sr_entities_set]

            # skip facts that are not retrieved for a mention in context or entity slot
            if not retrieved_for:
                continue

            # skip duplicates
            evidence_text = self.evr.kb_fact_to_text(fact)
            if evidence_text in evidence_texts:
                continue
            evidence_texts.add(evidence_text)

            # add evidence
            evidence = self.evr.kb_fact_to_evidence(fact, retrieved_for)
            evidences.append(evidence)

        # store result in cache
        if self.evr.use_cache:
            self.evr.cache_changed = True
            self.evr.cache[structured_representation] = (copy.deepcopy(evidences), disambiguations)

        return evidences, disambiguations

    def store_cache(self):
        """Store cache of evidence retriever."""
        self.evr.store_cache()


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception(
            "Usage: python explaignn/evidence_retrieval_scoring/rers.py --<FUNCTION> <PATH_TO_CONFIG>"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    config = get_config(config_path)

    ers = RERS(config)

    if function == "--example":
        sr = "|| many || How Pulitzer Prizes has John Updike won || number"
        evs, entities = ers.retrieve_evidences(sr)
        print(len(evs), entities)
