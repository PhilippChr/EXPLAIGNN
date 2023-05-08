import mwparserfromhell as mwp
from wikitables import WikiTable
from wikitables.util import ftag


def json_tables_to_evidences(tables, wiki_title):
    """
    Convert the table parsed by wikitables-module to evidences.
    """
    evidences = list()
    # for each table in document
    for table in tables:
        # row-wise table processing
        for row in table.rows:
            # wiki_title will be prepended later to avoid noisy matches
            evidence_text = ", ".join([f"{key} is {value}" for key, value in row.items() if value])

            # create evidence
            evidence = {"evidence_text": evidence_text, "source": "table"}
            evidences.append(evidence)
    return evidences


def extract_wikipedia_tables(wiki_md):
    """
    Retrieve json-tables from the wikipedia page.
    """
    if not wiki_md or not wiki_md.get("revisions"):
        return []

    # load content
    content = wiki_md["revisions"][0]["*"]
    title = wiki_md["title"]

    # extract tables using wikitables-module
    try:
        tables = _import_tables(content, title)
    except:
        tables = []
    return tables


def _import_tables(content, title, lang="en"):
    """
    Extract tables from the given markdown content
    using the wikitables module and mwparser.
    """
    raw_tables = mwp.parse(content).filter_tags(matches=ftag("table"))

    def _table_gen():
        for idx, table in enumerate(raw_tables):
            name = "%s[%s]" % (title, idx)
            yield WikiTable(name, table)

    return list(_table_gen())
