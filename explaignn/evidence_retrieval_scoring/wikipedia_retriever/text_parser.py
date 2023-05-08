import re


def extract_text_snippets(wiki_md, wiki_title, nlp):
    """
    Extract text snippets from the given
    Markdown text.
    """
    if not wiki_md or not wiki_md.get("extract"):
        return []

    # load content
    content = wiki_md["extract"]

    # remove noise and load doc
    clean_content = _filter_noise(content)
    doc = nlp(clean_content)

    # split the given document into sentences
    evidences = list()
    for sent in doc.sents:
        # drop empty sentences
        if not sent.text.strip():
            continue

        # wiki_title will be prepended later to avoid noisy matches
        evidence_text = sent.text.strip()

        # create evidence object
        evidence = {
            # entities are added later by EvidenceAnnotator
            "evidence_text": evidence_text,
            "source": "text",
        }
        evidences.append(evidence)
    return evidences


def _filter_noise(wiki_content):
    """
    Filter headings and whitespaces from the document.
    """
    # remove sections
    content = wiki_content.split("== Citations ==")[0]
    content = wiki_content.split("== Footnotes ==")[0]
    content = wiki_content.split("== References ==")[0]
    content = wiki_content.split("== Further reading ==")[0]
    # clean text
    content = re.sub(r"==.*?==+", "", content)
    content = content.replace("\n", " ")
    while "  " in content:
        content = content.replace("  ", " ")
    return content
