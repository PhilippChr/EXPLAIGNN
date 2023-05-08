"""
Library for different string and path functions
for the Wikipedia retriever.
"""


def format_wiki_path(value):
    """Reformat Wikipedia entity link."""
    return value.replace("/wiki/", "")


def is_wikipedia_path(value):
    """Check if the value is a Wikipedia entity."""
    if not value:
        return False
    elif not value.startswith("/wiki"):
        return False
    elif value.startswith("/wiki/File:"):
        return False
    elif "Category:" in value:
        return False
    elif "Special:" in value:
        return False
    return True


def wiki_title_to_path(wiki_title):
    wiki_path = wiki_title.replace(" ", "_")
    wiki_path = wiki_path.replace("'", "%27")
    wiki_path = wiki_path.replace("-", "_")
    return wiki_path


def wiki_path_to_title(wiki_path):
    wiki_title = wiki_path.replace("_", " ")
    wiki_title = wiki_title.replace("%27", "'")
    wiki_title = wiki_title.replace("%20", " ")
    wiki_title = wiki_title.replace("%28", "(")
    wiki_title = wiki_title.replace("%29", ")")
    wiki_title = wiki_title.replace("%2C", ",")
    wiki_title = wiki_title.replace("%21", "!")
    wiki_title = wiki_title.replace("%24", "$")
    wiki_title = wiki_title.replace("%25", "%")
    return wiki_title
