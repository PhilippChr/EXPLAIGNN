import os
import sys

from explaignn.evaluation import answer_presence
from explaignn.evidence_retrieval_scoring.bm25_es import BM25Scoring
from explaignn.evidence_retrieval_scoring.clocq_er import ClocqRetriever
from explaignn.evidence_retrieval_scoring.evidence_retrieval_scoring import EvidenceRetrievalScoring
from explaignn.library.utils import get_config


class ClocqBM25(EvidenceRetrievalScoring):
    def __init__(self, config):
        super(ClocqBM25, self).__init__(config)
        self.evr = ClocqRetriever(config)
        self.evs = BM25Scoring(config)

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info"), train=False):
        """Retrieve best evidences for SR."""
        structured_representation = turn["structured_representation"]
        evidences, _ = self.evr.retrieve_evidences(structured_representation, sources)

        top_evidences = self.evs.get_top_evidences(structured_representation, evidences)
        turn["top_evidences"] = top_evidences
        # answer presence
        hit, answering_evidences = answer_presence(top_evidences, turn["answers"])
        turn["answer_presence"] = hit
        return top_evidences

    def store_cache(self):
        """Store cache of evidence retriever."""
        self.evr.store_cache()