"""
A RAG system should not pass weak or mismatched evidence into an answer just because it appeared at the top of a
search result.
In this lab, you will build the retrieval-control layer for a policy support system. Your code will decide which policy
records are strong enough to use as evidence later.

The scoring helper is already imported from hidden support code.
This is not document ingestion and it is not answer generation.

Your program will:
    embed the user’s query;
    apply metadata filters such as region;
    score eligible policy records;
    apply threshold and top_k;
    return readable evidence without internal vectors;
    print an evidence decision.
    The output should feel like a compact evidence report:
    STATUS: STRONG_EVIDENCE
    QUERY: holiday allowance
    EVIDENCE:
    0.991 | uk-leave | leave.md | UK employees receive 25 days annual leave.
"""


import json
from pathlib import Path
from openai import OpenAI

from retrieval_helpers import similarity_score
from retrieval_runner import build_search_records, print_decision


import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
# Complete: embed_text, matches_filters, search, evidence_status.

def embed_text(text):
    """Return an embedding for one piece of text."""

    response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
)
    return response.data[0].embedding


def matches_filters(record, filters):
    """Return True when a record is eligible for the requested filters."""
    if not filters:
        return True
    for key, excepted_value in filters.items():
        if key in record.keys():
            print(record.get(key))
            if record.get(key) != excepted_value and filters.get(key) != "ALL":
                return False
        else:
            return False
    return True


def search(query, records, top_k=3, threshold=0.5, filters=None):
    """Return ranked evidence records that pass the retrieval controls."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_vector = embed_text(query)
    useful_records = []

    for record in records:
        score = similarity_score(query_vector, record["vector"])
        if score >= threshold and matches_filters(record, filters):
        # keep this result
            record['score'] = score
            useful_records.append(record)
    if not useful_records:
        return []
    useful_records = sorted(useful_records, key=lambda x: x["score"], reverse=True)

    return useful_records[:top_k]


def evidence_status(results):
    """Return STRONG_EVIDENCE, WEAK_EVIDENCE, or NO_EVIDENCE.
        NO_EVIDENCE     no results passed the threshold
        STRONG_EVIDENCE best score is at least 0.75
        WEAK_EVIDENCE   results exist, but the best score is below 0.75
    """
    if not results:
        return "NO_EVIDENCE"
    for result in results:
        if result["score"] >= 0.75:
            return "STRONG_EVIDENCE"
        return "WEAK_EVIDENCE"


def main():
    query = "company car parking rules"
    threshold = 0.5
    filters = {"region": "ALL"}

    records = json.loads(Path("policies.json").read_text(encoding="utf-8"))
    records = build_search_records(records, embed_text)

    results = search(
        query,
        records,
        top_k=2,
        threshold=threshold,
        filters=filters,
    )
    print_decision(query, results, evidence_status)


if __name__ == "__main__":
    main()
