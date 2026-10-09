import math
import re
from collections import Counter

from policy_loader import load_policy_documents


TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(\+?\d[\d\s().-]{6,}\d)"
)


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def load_documents():
    return load_policy_documents()


def has_phone_number(text):
    return PHONE_PATTERN.search(text) is not None


def retrieve_documents(query, documents=None, k=5):
    if documents is None:
        documents = load_documents()

    query_terms = tokenize(query)
    if not query_terms:
        return []

    document_terms = [
        tokenize(
            document.page_content
            + " "
            + document.metadata.get("topic", "")
        )
        for document in documents
    ]

    document_count = len(documents)
    document_frequency = Counter(
        term
        for terms in document_terms
        for term in set(terms)
    )

    query_counts = Counter(query_terms)
    scored = []

    for document, terms in zip(documents, document_terms):
        term_counts = Counter(terms)
        score = 0.0

        for term, query_count in query_counts.items():
            if term not in term_counts:
                continue

            inverse_document_frequency = math.log(
                (document_count + 1)
                / (document_frequency[term] + 1)
            ) + 1.0

            score += (
                query_count
                * term_counts[term]
                * inverse_document_frequency
            )

        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:k]]


def format_context(documents):
    blocks = []

    for document in documents:
        metadata = document.metadata
        blocks.append(
            "[{policy_id} topic={topic}]\n{content}".format(
                policy_id=metadata.get("policy_id", "unknown"),
                topic=metadata.get("topic", "unknown"),
                content=document.page_content,
            )
        )

    return "\n\n".join(blocks)
