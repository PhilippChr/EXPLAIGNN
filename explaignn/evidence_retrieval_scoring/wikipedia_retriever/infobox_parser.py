"""
Infobox parser for Wikipedia tables.
Inspired by https://github.com/roskoff/HTMLTableParser/blob/master/HTMLTableParser.py.
"""
from html.parser import HTMLParser

import explaignn.library.wikipedia_library as wiki

CELL_SEPARATOR = ", "
COMPONENT_SEPARATOR = ", "


def infobox_to_evidences(parsed_infobox, wiki_title):
    """
    Extracts the evidences from the given parsed infobox.
    The parsed infobox is output of the InfoboxParser class
    below.
    """
    # initialize
    evidences = list()
    last_header_row = ""

    # iterate through rows
    for row in parsed_infobox:
        if len(row) == 1 and row[0]["cell_type"] == "header":
            # remember header
            last_header_row = row[0]["text"]
        else:
            # create evidence text
            row_text = CELL_SEPARATOR.join([cell["text"] for cell in row if cell["text"]])
            if last_header_row:  # non-empty header row will be prepended
                evidence_components = [last_header_row, row_text]
                evidence_text = COMPONENT_SEPARATOR.join(evidence_components)
            else:
                evidence_text = row_text

            # wikipedia entities (=paths)
            wiki_paths = [entity for cell in row for entity in cell["entities"]]

            # dict from string to wiki_path
            disambiguations = [
                (string, wiki_path)
                for cell in row
                for string, wiki_path in (cell["anchor_dict"]).items()
            ]

            ## do not consider wiki_paths with hashtags
            # wiki_title will be prepended later to avoid noisy matches
            # hashtag indicates a paragraph on entity, rather than entity
            wiki_paths = [wiki_path for wiki_path in wiki_paths if not "#" in wiki_path]

            # create evidence
            evidence = {
                "evidence_text": evidence_text,
                "wikipedia_paths": wiki_paths,
                "wp_disambiguations": disambiguations,
                "source": "info",
            }
            evidences.append(evidence)
    return evidences


class InfoboxParser(HTMLParser):
    """
    This class serves as a html infobox parser.
    Assumes the following table format:
            header_row1
            attribute_name | attribute_value
            ...
            header_row2
            attribute_name | attribute_value
    """

    def __init__(
        self,
        anchor_dict,
        decode_html_entities=False,
        data_separator=" ",
    ):
        HTMLParser.__init__(self, convert_charrefs=decode_html_entities)
        self._data_separator = data_separator
        self._in_td = False
        self._in_th = False
        self._current_table = []
        self._current_row = []
        self._current_cell = {"entities": [], "text": [], "anchor_dict": dict()}

        self._after_href = False
        self._last_href = None

        # initialize tables object
        self.tables = []

        # multiple mentions of same entity are not all tagged with URL
        # -> remember such entities (phrase->entity entries)
        self.anchor_dict = anchor_dict

    def get_anchor_dict(self):
        return self.anchor_dict

    def handle_starttag(self, tag, attrs):
        """
        We need to remember the opening point for the content of interest.
        The other tags (<table>, <tr>) are only handled at the closing point.
        """
        if tag == "td":
            self._in_td = True
        if tag == "th":
            self._in_th = True

        for name, value in attrs:
            if name == "href" and wiki.is_wikipedia_path(value):
                wiki_path = wiki.format_wiki_path(value)
                if wiki.is_wikipedia_path(wiki_path):
                    self._current_cell["entities"].append(wiki_path)
                self._after_href = True
                self._last_href = wiki_path

    def handle_data(self, data):
        """This is where we save content to a cell."""
        if self._in_td or self._in_th:
            self._current_cell["text"].append(data)
        if self._after_href:
            # store data->entity in dicts
            self.anchor_dict[data] = self._last_href
            if wiki.is_wikipedia_path(self._last_href):
                self._current_cell["anchor_dict"][data] = self._last_href
            # delete flag
            self._after_href = False

    def handle_endtag(self, tag):
        """
        Here we exit the tags. If the closing tag is </tr>, we know that we
        can save our currently parsed cells to the current table as a row and
        prepare for a new row. If the closing tag is </table>, we save the
        current table and prepare for a new one.
        """
        if tag == "td":
            self._in_td = False
        elif tag == "th":
            self._in_th = False

        if tag in ["td", "th"]:
            cell_text = self._data_separator.join(self._current_cell["text"]).strip()
            cell_text = cell_text.replace("\n", ", ")
            cell_text = cell_text.replace("\xa0", " ")
            cell_entities = self._current_cell["entities"]
            cell_anchor_dict = self._current_cell["anchor_dict"]
            if not cell_entities:
                for anchor in self.anchor_dict:
                    if anchor in cell_text:
                        anchor_entity = self.anchor_dict[anchor]
                        cell_entities.append(anchor_entity)
                        cell_anchor_dict[anchor] = anchor_entity
            cell_type = "data" if tag == "td" else "header"
            final_cell = {
                "entities": cell_entities,
                "text": cell_text,
                "anchor_dict": cell_anchor_dict,
                "cell_type": cell_type,
            }
            self._current_row.append(final_cell)
            self._current_cell = {"entities": [], "text": [], "anchor_dict": dict()}
        elif tag == "tr":
            self._current_table.append(self._current_row)
            self._current_row = []
        elif tag == "table":
            self.tables.append(self._current_table)
            self._current_table = []
