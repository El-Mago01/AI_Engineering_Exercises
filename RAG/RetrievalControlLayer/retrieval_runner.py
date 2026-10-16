def build_search_records(records, embed_text):
    """Attach an embedding to each supplied policy record."""
    search_records = []
    for record in records:
        search_record = dict(record)
        search_record["vector"] = embed_text(record["text"])
        search_records.append(search_record)
    return search_records


def print_decision(query, results, evidence_status):
    """Print a compact evidence decision for the terminal."""
    status = evidence_status(results)
    print(f"STATUS: {status}")
    print(f"QUERY: {query}")

    if not results:
        print("REASON: No policy record passed the minimum evidence threshold.")
        return

    print("EVIDENCE:")
    for item in results:
        print(f"{item['score']:.3f} | {item['id']} | {item['source']} | {item['text']}")
