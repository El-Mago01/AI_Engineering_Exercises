"""
For this PDF:
loading extracts the text from sample.pdf;
chunking keeps nearby words together;
embeddings turn each chunk into a vector;
search compares the query vector with each chunk vector;
the result should preserve the source and chunk text.
"""


from pathlib import Path

from pdf_search_helpers import embed_text, load_pdf_text, similarity_score



def chunk_text(text, chunk_size=400, overlap=80):
    """
    Return a list of overlapping text chunks.
    """
    words = text.strip().split(" ")
    if overlap >= chunk_size:
        raise ValueError("overlap size is too large. Must be smaller than chunk_size")
    chunks = []
    index = 0
    while index < len(words)-1:
        if index - overlap < 0: # the first slide of the window
            this_chunk = words[index:index + chunk_size]
            index = chunk_size
        else:
            if (len(words) - index) >= chunk_size: # The normal slide of the window, not at the end, not at the beginning
                this_chunk = words[index - overlap:index + chunk_size]
                index += chunk_size

            else: #This is the case when the sliding window is at the end of the text. The last round for the left-overs
                this_chunk = words[index-overlap:]
                index += chunk_size
        this_chunk = " ".join(this_chunk)
        chunks.append(this_chunk)
    return chunks


def build_index(pdf_path):
    """
    Build searchable records from a PDF.
    Each record should keep readable text and its embedding.
    """
    records = []
    total_text = load_pdf_text(pdf_path)
    chunks = chunk_text(total_text, 40, 5)
    item = 1
    for text in chunks:
        chunk = {
            "source": pdf_path,
            "id": item,
            "text": text,
            "embedding": embed_text(text),
        }
        item += 1
        records.append(chunk)
    return records



def search_index(chunks, query, top_k=3):
    """
    Return the strongest matching chunks for a query.

    The returned results should not include raw embeddings.
    """
    results = []
    query_vector = embed_text(query)
    for chunk in chunks:
        score = similarity_score(query_vector, chunk["embedding"])
        analysis = {
            "text": chunk["text"],
            "score": score,
            "id": "sample-" + str(chunk["id"]),
            "source": "sample.pdf",
        }
        results.append(analysis)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def main():
    print(chunk_text("Reset VPN access from the portal before escalation.", chunk_size=5, overlap=2))
    print(chunk_text('     ', chunk_size=5, overlap=2))

    pdf_path = "sample.pdf"
    query = "How do I fix VPN access?"
    chunks = build_index(pdf_path)
    results = search_index(chunks, query)

    print("PDF Search Tool")
    print(f"PDF: {pdf_path}")
    print(f"Query: {query}")
    if not results:
        print("No results yet. Complete the three TODO functions, then run again.")
        return

    for result in results:
        print(f"{result['score']:.3f} | {result['source']} | {result['text'][:30]}")


if __name__ == "__main__":
    main()

