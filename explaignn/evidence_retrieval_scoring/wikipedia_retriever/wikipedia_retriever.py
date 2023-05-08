import os
import pickle
import re
import requests
import spacy
import sys
import time
from bs4 import BeautifulSoup
from filelock import FileLock
from pathlib import Path

import explaignn.library.wikipedia_library as wiki
from explaignn.evidence_retrieval_scoring.wikipedia_retriever.evidence_annotator import (
    EvidenceAnnotator,
)
from explaignn.evidence_retrieval_scoring.wikipedia_retriever.infobox_parser import (
    InfoboxParser,
    infobox_to_evidences,
)
from explaignn.evidence_retrieval_scoring.wikipedia_retriever.table_parser import (
    extract_wikipedia_tables,
    json_tables_to_evidences,
)
from explaignn.evidence_retrieval_scoring.wikipedia_retriever.text_parser import (
    extract_text_snippets,
)
from explaignn.library.utils import get_config, get_logger

API_URL = "http://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "CONVINSE-experiments/1.0 (https://convinse.mpi-inf.mpg.de; pchristm@mpi-inf.mpg.de)"
}
PARAMS = {
    "prop": "extracts|revisions",
    "format": "json",
    "action": "query",
    "explaintext": "",
    "rvprop": "content",
}

YEAR_PATTERN = re.compile("^[0-9][0-9][0-9][0-9]$")
WIKI_DATE_PATTERN = re.compile("[0-9]+ [A-Z][a-z]* [0-9][0-9][0-9][0-9]")


class WikipediaRetriever:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(__name__, config)

        # whether Wikipedia evidences are retrieved on the fly (i.e. from the Wikipedia API)
        self.use_cache = config["ers_use_cache"]
        self.on_the_fly = config["ers_on_the_fly"]

        # initialize dump
        if self.use_cache:
            self.path_to_dump = self.config["ers_wikipedia_dump"]
            self._init_wikipedia_dump()
            self.dump_changed = False

        if self.on_the_fly:
            # open dicts
            with open(config["path_to_wikidata_mappings"], "rb") as fp:
                self.wikidata_mappings = pickle.load(fp)
            with open(config["path_to_wikipedia_mappings"], "rb") as fp:
                self.wikipedia_mappings = pickle.load(fp)

            # initialize evidence annotator (used for (text)->Wikipedia->Wikidata)
            self.annotator = EvidenceAnnotator(config, self.wikidata_mappings)

            # load nlp pipeline
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("sentencizer")
        self.logger.debug("WikipediaRetriever successfully initialized!")

    def retrieve_wp_evidences(self, question_entity):
        """
        Retrieve evidences from Wikipedia for the given Wikipedia title.
        Always returns the full set of evidences (text, table, infobox).
        Filtering is done via filter_evidences function.
        """
        question_entity_id = question_entity["id"]
        if self.use_cache and question_entity_id in self.wikipedia_dump:
            self.logger.debug(f"Found Wikipedia evidences in dump!")
            return self.wikipedia_dump.get(question_entity_id)

        if not self.on_the_fly:
            self.logger.debug(
                f"No Wikipedia evidences in dump, but on-the-fly retrieval not active!"
            )
            return []

        # get Wikipedia title
        wiki_path = self.wikipedia_mappings.get(question_entity_id)
        if not wiki_path:
            self.logger.debug(
                f"No Wikipedia link found for this Wikidata ID: {question_entity_id}."
            )
            if self.use_cache:
                self.wikipedia_dump[question_entity_id] = []  # remember
            return []
        self.logger.debug(f"Retrieving Wikipedia evidences for: {wiki_path}.")
        self.dump_changed = True

        # retrieve Wikipedia soup
        wiki_title = wiki.wiki_path_to_title(wiki_path)
        soup = self._retrieve_soup(wiki_title)
        if soup is None:
            if self.use_cache:
                self.wikipedia_dump[question_entity_id] = []  # remember
            return []

        # retrieve Wikipedia markdown
        wiki_md = self._retrieve_markdown(wiki_title)

        # extract anchors
        doc_anchor_dict = self._build_document_anchor_dict(soup)

        # retrieve evidences
        infobox_evidences = self._retrieve_infobox_entries(wiki_title, soup, doc_anchor_dict)
        table_records = self._retrieve_table_records(wiki_title, wiki_md)
        text_snippets = self._retrieve_text_snippets(wiki_title, wiki_md)

        # prune e.g. too long evidences
        evidences = infobox_evidences + table_records + text_snippets
        evidences = self.filter_and_clean_evidences(evidences)

        # add `retrieved_for_entity` information
        for evidence in evidences:
            evidence["retrieved_for_entity"] = [question_entity]

        ## add wikidata entities (for table and text)
        # evidences with no wikidata entities (except for the wiki_path) are dropped
        self.annotator.annotate_wikidata_entities(wiki_path, evidences, doc_anchor_dict)

        # store result in dump
        if self.use_cache:
            self.wikipedia_dump[question_entity_id] = evidences

        self.logger.debug(f"Evidences successfully retrieved for {question_entity_id}.")
        return evidences

    def filter_and_clean_evidences(self, evidences):
        """
        Drop evidences which do not suffice specific
        criteria. E.g. such evidences could be too
        short, long, or contain too many symbols.
        """
        filtered_evidences = list()
        for evidence in evidences:
            evidence_text = evidence["evidence_text"]
            ## filter evidences
            # too short
            if len(evidence_text) < self.config["evr_min_evidence_length"]:
                continue
            # too long
            if len(evidence_text) > self.config["evr_max_evidence_length"]:
                continue
            # ratio of letters very low
            letters = sum(c.isalpha() for c in evidence_text)
            if letters < len(evidence_text) / 2:
                continue

            ## clean evidence
            evidence_text = self.clean_evidence(evidence_text)
            evidence["evidence_text"] = evidence_text
            filtered_evidences.append(evidence)
        return filtered_evidences

    def clean_evidence(self, evidence_text):
        """Clean the given evidence text."""
        evidence_text = re.sub(r"\[[0-9]*\]", "", evidence_text)
        return evidence_text

    def _retrieve_infobox_entries(self, wiki_title, soup, doc_anchor_dict):
        """
        Retrieve infobox entries for the given Wikipedia entity.
        """
        # get infobox (only one infobox possible)
        infoboxes = soup.find_all("table", {"class": "infobox"})
        if not infoboxes:
            return []
        infobox = infoboxes[0]

        # parse infobox content
        p = InfoboxParser(doc_anchor_dict)
        infobox_html = str(infobox)
        p.feed(infobox_html)

        # transform parsed infobox to evidences
        infobox_parsed = p.tables[0]
        evidences = infobox_to_evidences(infobox_parsed, wiki_title)
        return evidences

    def _retrieve_table_records(self, wiki_title, wiki_md):
        """
        Retrieve table records for the given Wikipedia entity.
        """
        # extract wikipedia tables
        tables = extract_wikipedia_tables(wiki_md)

        # extract evidences from tables
        evidences = json_tables_to_evidences(tables, wiki_title)
        return evidences

    def _retrieve_text_snippets(self, wiki_title, wiki_md):
        """
        Retrieve text snippets for the given Wikidata entity.
        """
        evidences = extract_text_snippets(wiki_md, wiki_title, self.nlp)
        return evidences

    def _build_document_anchor_dict(self, soup):
        """
        Establishes a dictionary that maps from Wikipedia text
        to the Wikipedia entity (=link). Is used to map to
        Wikidata entities (via Wikipedia) later.
        Format: text -> Wikidata entity.
        """
        # prune navigation bar
        for div in soup.find_all("div", {"class": "navbox"}):
            div.decompose()

        # go through links
        anchor_dict = dict()
        for tag in soup.find_all("a"):
            # anchor text
            text = tag.text.strip()
            if len(text) < 3:
                continue
            # duplicate anchor text (keep first)
            # -> later ones can be more specific/incorrect
            if anchor_dict.get(text):
                continue

            # wiki title (=entity)
            href = tag.attrs.get("href")
            if not wiki.is_wikipedia_path(href):
                continue
            wiki_path = wiki.format_wiki_path(href)

            anchor_dict[text] = wiki_path
        return anchor_dict

    def _retrieve_soup(self, wiki_title):
        """
        Retrieve Wikipedia html for the given Wikipedia Title.
        """
        wiki_path = wiki.wiki_title_to_path(wiki_title)
        link = f"https://en.wikipedia.org/wiki/{wiki_path}"
        try:
            html = requests.get(link, headers=HEADERS).text
            soup = BeautifulSoup(html, features="html.parser")
        except:
            self.logger.info(f"Exception when retrieving Wikipedia page soup for {wiki_title}.")
            return None
        return soup

    def _retrieve_markdown(self, wiki_title):
        """
        Retrieve the content of the given wikipedia title.
        """
        params = PARAMS.copy()
        params["titles"] = wiki_title
        try:
            # make request
            r = requests.get(API_URL, params=params)
            res = r.json()
            pages = res["query"]["pages"]
            page = list(pages.values())[0]
        except:
            return None
        return page

    def _init_wikipedia_dump(self):
        """
        Initialize the Wikipedia dump. The consists of a mapping
        from Wikidata IDs to Wikipedia evidences in the expected format.
        """
        if os.path.isfile(self.path_to_dump):
            # remember version read initially
            self.logger.info(f"Loading Wikipedia dump from path {self.path_to_dump}.")
            with FileLock(f"{self.path_to_dump}.lock"):
                self.dump_version = self._read_dump_version()
                self.logger.debug(f"Cache version: {self.dump_version}")
                self.wikipedia_dump = self._read_dump()
            self.logger.info(f"Wikipedia dump successfully loaded.")
        else:
            self.logger.info(
                f"Could not find an existing Wikipedia dump at path {self.path_to_dump}."
            )
            self.logger.info("Populating Wikipedia dump from scratch!")
            self.wikipedia_dump = {}
            self._write_dump(self.wikipedia_dump)
            self._write_dump_version()

    def store_dump(self):
        """Store the Wikipedia dump to disk."""
        # store mappings from wikipedia paths to wikidata links
        self.annotator.store_cache()

        # store dump
        if not self.use_cache:  # store only if Wikipedia dump in use
            return
        if not self.dump_changed:  # store only if Wikipedia dump  changed
            return
        # check if the Wikipedia dump  was updated by other processes
        if self._read_dump_version() == self.dump_version:
            # no updates: store and update version
            self.logger.info(f"Writing Wikipedia dump at path {self.path_to_dump}.")
            with FileLock(f"{self.path_to_dump}.lock"):
                self._write_dump(self.wikipedia_dump)
                self._write_dump_version()
        else:
            # update! read updated version and merge the dumps
            self.logger.info(f"Merging Wikipedia dump at path {self.path_to_dump}.")
            with FileLock(f"{self.path_to_dump}.lock"):
                # read updated version
                updated_dump = self._read_dump()
                # overwrite with changes in current process (most recent)
                updated_dump.update(self.wikipedia_dump)
                # store
                self._write_dump(updated_dump)
                self._write_dump_version()


    def _read_dump(self):
        """
        Read the current version of the dump.
        This can be different from the version used in this file,
        given that multiple processes may access it simultaneously.
        """
        # read file content from wikipedia dump shared across QU methods
        with open(self.path_to_dump, "rb") as fp:
            wikipedia_dump = pickle.load(fp)
        return wikipedia_dump

    def _write_dump(self, dump):
        """Store the dump."""
        dump_dir = os.path.dirname(self.path_to_dump)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        with open(self.path_to_dump, "wb") as fp:
            pickle.dump(dump, fp)
        return dump

    def _read_dump_version(self):
        """Read the dump version (hashed timestamp of last update) from a dedicated file."""
        if not os.path.isfile(f"{self.path_to_dump}.version"):
            self._write_dump_version()
        with open(f"{self.path_to_dump}.version", "r") as fp:
            dump_version = fp.readline().strip()
        return dump_version

    def _write_dump_version(self):
        """Write the current dump version (hashed timestamp of current update)."""
        with open(f"{self.path_to_dump}.version", "w") as fp:
            version = str(time.time())
            fp.write(version)
        self.dump_version = version


#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    # RUN: python explaignn/evidence_retrieval_scoring/wikipedia_retriever/wikipedia_retriever.py config/convmix/explaignn.yml

    # load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/convmix/explaignn_demo.yml"
    config = get_config(config_path)

    # create retriever
    retriever = WikipediaRetriever(config)

    # retrieve evidences
    start = time.time()
    question_entity = {"id": "Q5", "label": "Human"}
    evidences = retriever.retrieve_wp_evidences(question_entity)
    print("Time consumed", time.time() - start)

    # show evidences
    for evidence in evidences:
        print(evidence)
