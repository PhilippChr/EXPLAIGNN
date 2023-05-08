import copy
import os
import os
import pickle
import re
import time
from clocq.CLOCQ import CLOCQ
from clocq.interface.CLOCQInterfaceClient import CLOCQInterfaceClient
from filelock import FileLock
from pathlib import Path

from explaignn.evidence_retrieval_scoring.wikipedia_retriever.wikipedia_retriever import (
    WikipediaRetriever,
)
from explaignn.library.string_library import StringLibrary
from explaignn.library.utils import get_logger

ENT_PATTERN = re.compile("^Q[0-9]+$")
PRE_PATTERN = re.compile("^P[0-9]+$")

KB_ITEM_SEPARATOR = ", "


class ClocqRetriever:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

        # load cache
        self.use_cache = config["ers_use_cache"]
        if self.use_cache:
            self.cache_path = config["ers_cache"]
            self._init_cache()
            self.cache_changed = False

        # initialize clocq for KB-facts and disambiguations
        if config["clocq_use_api"]:
            self.clocq = CLOCQInterfaceClient(host=config["clocq_host"], port=config["clocq_port"])
        else:
            self.clocq = CLOCQ()

        # initialize wikipedia-retriever
        self.wiki_retriever = WikipediaRetriever(config)
        if config["qu"] == "sr":
            self.sr_delimiter = config["sr_delimiter"].strip()
        else:
            self.sr_delimiter = " "

    def retrieve_evidences(self, structured_representation, sources):
        """
        Retrieve evidences and question entities
        for the given SR (or other question/text).

        This function is used for initial evidence
        retrieval. These evidences are filtered in the
        next step.

        Can also be used from external modules to access
        all evidences for the given SR (if possible from cache).
        """
        # remove delimiter from SR
        structured_representation = structured_representation.replace(self.sr_delimiter, " ")
        while "  " in structured_representation:
            structured_representation = structured_representation.replace("  ", " ")

        # first get question entities (and KB-facts if required)
        evidences, disambiguations = self.retrieve_kb_facts(structured_representation)

        # wikipedia evidences (only if required)
        if any(src in sources for src in ["text", "table", "info"]):
            for disambiguation in disambiguations:
                evidences += self.retrieve_wikipedia_evidences(disambiguation["item"])

        # config-based filtering
        evidences = self.filter_evidences(evidences, sources)
        return evidences, disambiguations

    def retrieve_evidences_for_entity(self, item, sources=("kb", "text", "table", "info")):
        """Retrieve evidences for the given entity (used for silver annotation)."""
        evidences = list()
        # kb
        if "kb" in sources:
            evidences += self.retrieve_evidences_from_kb(item)
        # wikipedia
        if any(src in sources for src in ["text", "table", "info"]):
            evidences += self.retrieve_wikipedia_evidences(item)

        # config-based filtering
        evidences = self.filter_evidences(evidences, sources)
        return evidences

    def retrieve_wikipedia_evidences(self, question_entity):
        """
        Retrieve evidences from Wikipedia for the given question entity.
        """
        # retrieve result
        evidences = self.wiki_retriever.retrieve_wp_evidences(question_entity)

        assert not evidences is None  # evidences should never be None
        return evidences

    def retrieve_kb_facts(self, structured_representation):
        """
        Retrieve KB facts for the given SR (or other question/text).
        Also returns the question entities, for usage in Wikipedia retriever.
        """
        # look-up cache
        if self.use_cache and structured_representation in self.cache:
            return copy.deepcopy(self.cache[structured_representation])

        self.logger.debug(
            f"No cache hit: Retrieving search space for: {structured_representation}."
        )

        # apply CLOCQ
        clocq_result = self.clocq.get_search_space(
            structured_representation,
            parameters=self.config["clocq_params"],
            include_labels=True,
            include_type=True,
        )

        # get question entities (predicates dropped)
        disambiguations = [
            item
            for item in clocq_result["kb_item_tuple"]
            if not item["item"]["id"] is None and ENT_PATTERN.match(item["item"]["id"])
        ]

        question_items_set = set([item["item"]["id"] for item in clocq_result["kb_item_tuple"]])

        # remember potential duplicate facts
        evidence_texts = set()

        # transform facts to evidences
        evidences = list()
        for fact in clocq_result["search_space"]:
            # entities the fact was retrieved for from clocq
            retrieved_for = [item for item in fact if item["id"] in question_items_set]

            # skip duplicates
            evidence_text = self.kb_fact_to_text(fact)
            if evidence_text in evidence_texts:
                continue
            evidence_texts.add(evidence_text)

            # add evidence
            evidence = self.kb_fact_to_evidence(fact, retrieved_for)
            evidences.append(evidence)

        # store result in cache
        if self.use_cache:
            self.cache_changed = True
            self.cache[structured_representation] = (copy.deepcopy(evidences), disambiguations)
        return evidences, disambiguations

    def kb_fact_to_evidence(self, kb_fact, retrieved_for):
        """Transform the given KB-fact to an evidence."""

        def _format_fact(kb_fact):
            """Correct format of fact (if necessary)."""
            for item in kb_fact:
                if StringLibrary.is_timestamp(item["label"]):
                    item["label"] = StringLibrary.convert_timestamp_to_date(item["id"])
                item["label"] = item["label"].replace('"', "")
            return kb_fact

        def _get_wikidata_entities(kb_fact):
            """Return wikidata_entities for fact."""
            items = list()
            for item in kb_fact:
                # skip undesired answers
                if not _is_potential_answer(item["id"]):
                    continue
                # append to set
                item["label"] = item["label"].replace('"', "")
                items.append(item)
                # augment candidates with years (for differen granularity of answer)
                if StringLibrary.is_timestamp(item["id"]):
                    year = StringLibrary.get_year(item["id"])
                    new_item = {"id": StringLibrary.convert_year_to_timestamp(year), "label": year}
                    items.append(new_item)
            return items

        def _is_potential_answer(item_id):
            """Return if item_id could be the answer."""
            # keep all KB-items except for predicates
            if PRE_PATTERN.match(item_id):
                return False
            return True

        # evidence text
        evidence_text = self.kb_fact_to_text(_format_fact(kb_fact))
        evidence = {
            "evidence_text": evidence_text,
            "wikidata_entities": _get_wikidata_entities(kb_fact),
            "disambiguations": [
                (item["label"], item["id"])
                for item in kb_fact
                # if ENT_PATTERN.match(item["id"])
                # -> changed on 2023-01-11: not clear why only entities should be added
            ],
            "retrieved_for_entity": retrieved_for,
            "source": "kb",
            "kb_fact": kb_fact
        }
        return evidence

    def retrieve_evidences_from_kb(self, item):
        """Retrieve evidences from KB for the given item (used in DS)."""
        facts = self.clocq.get_neighborhood(
            item["id"], p=self.config["clocq_p"], include_labels=True
        )
        return [self.kb_fact_to_evidence(kb_fact, [item]) for kb_fact in facts]

    def filter_evidences(self, evidences, sources):
        """
        Filter the set of evidences according to their source.
        """
        filtered_evidences = list()
        for evidence in evidences:
            if len(evidence["wikidata_entities"]) == 1:
                continue
            if len(evidence["wikidata_entities"]) > self.config["evr_max_entities"]:
                continue
            if evidence["source"] in sources:
                filtered_evidences.append(evidence)
        return filtered_evidences

    def kb_fact_to_text(self, kb_fact):
        """Verbalize the KB-fact."""
        return KB_ITEM_SEPARATOR.join([item["label"] for item in kb_fact])

    def store_cache(self):
        """Store the cache to disk."""
        if not self.use_cache:  # store only if cache in use
            return
        if not self.cache_changed:  # store only if cache changed
            return
        # check if the cache was updated by other processes
        if self._read_cache_version() == self.cache_version:
            # no updates: store and update version
            self.logger.info(f"Writing ER cache at path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                self._write_cache(self.cache)
                self._write_cache_version()
        else:
            # update! read updated version and merge the caches
            self.logger.info(f"Merging ER cache at path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                # read updated version
                updated_cache = self._read_cache()
                # overwrite with changes in current process (most recent)
                updated_cache.update(self.cache)
                # store
                self._write_cache(updated_cache)
                self._write_cache_version()
        # store extended wikipedia dump (if any changes occured)
        self.wiki_retriever.store_dump()

    def reset_cache(self):
        """Reset the cache for new population."""
        self.logger.warn(f"Resetting ER cache at path {self.cache_path}.")
        with FileLock(f"{self.cache_path}.lock"):
            self.cache = {}
            self._write_cache(self.cache)
            self._write_cache_version()

    def _init_cache(self):
        """Initialize the cache."""
        if os.path.isfile(self.cache_path):
            # remember version read initially
            self.logger.info(f"Loading ER cache from path {self.cache_path}.")
            with FileLock(f"{self.cache_path}.lock"):
                self.cache_version = self._read_cache_version()
                self.logger.debug(f"Cache version: {self.cache_version}")
                self.cache = self._read_cache()
            self.logger.info(f"ER cache successfully loaded.")
        else:
            self.logger.info(f"Could not find an existing ER cache at path {self.cache_path}.")
            self.logger.info("Populating ER cache from scratch!")
            self.cache = {}
            self._write_cache(self.cache)
            self._write_cache_version()

    def _read_cache(self):
        """
        Read the current version of the cache.
        This can be different from the version used in this file,
        given that multiple processes may access it simultaneously.
        """
        # read file content from cache shared across QU methods
        with open(self.cache_path, "rb") as fp:
            cache = pickle.load(fp)
        return cache

    def _write_cache(self, cache):
        """Write to the cache."""
        cache_dir = os.path.dirname(self.cache_path)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as fp:
            pickle.dump(cache, fp)
        return cache

    def _read_cache_version(self):
        """Read the cache version (hashed timestamp of last update) from a dedicated file."""
        if not os.path.isfile(f"{self.cache_path}.version"):
            self._write_cache_version()
        with open(f"{self.cache_path}.version", "r") as fp:
            cache_version = fp.readline().strip()
        return cache_version

    def _write_cache_version(self):
        """Write the current cache version (hashed timestamp of current update)."""
        with open(f"{self.cache_path}.version", "w") as fp:
            version = str(time.time())
            fp.write(version)
        self.cache_version = version
