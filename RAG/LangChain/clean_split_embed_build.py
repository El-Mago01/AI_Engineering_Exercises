"""
This exercise comes from codio 'Document ingestion pipeline"
https://codio.com/mgorren/document-ingestion-pipeline/guides
Supplied materials
docs/leave.md contains excessive blank lines.
docs/security.txt contains repeated spaces.
docs/empty.txt checks that empty files are skipped safely.

This is the document preparation step before semantic retrieval.
You are not building a chatbot. You are not generating answers. You are preparing clean source records that a
later retriever could search.

messy docs -> clean text -> chunks -> OpenAI embeddings -> records -> chunks.jsonl
What you are building
You will complete four parts in ingestion.py:
Function	    Job
clean_text()	Remove repeated spaces while keeping paragraph breaks.
split_text()	Turn cleaned text into smaller word chunks.
embed_text()	Create an OpenAI embedding for each chunk.
build_records()	Create records with id, source, category, chunk_index, text, and embedding.
The starter code handles writing chunks.jsonl.
OpenAI is set up at the top of ingestion.py. Your job is to use the supplied client
inside embed_text() and then use embed_text() when building records.

Definition of done
Your program should:
read .md and .txt files from docs;
clean messy spacing;
keep useful paragraph boundaries;
split cleaned text into readable chunks;
create simple records with id, source, category, chunk_index, text, and embedding;
skip empty documents;
write valid JSONL to chunks.jsonl.
JSONL stands for JSON Lines. It is not one large JSON list.


"""

import json
import re
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def discover_files(folder):
    """Return supported source files in a predictable order."""
    supported_files = []

    for path in folder.iterdir():
        if path.is_file() and path.suffix in [".md", ".txt"]:
            supported_files.append(path)

    return sorted(supported_files)


def clean_text(text):
    """Remove messy spacing while keeping paragraph breaks.
       It does the following:
            remove obvious extraction noise such as repeated slashes like ///;
            collapse repeated spaces and tabs inside each paragraph;
            trim unnecessary space around paragraphs;
            preserve one blank line between meaningful paragraphs;
            return '' when no meaningful content remains.

    """

    # TODO: split the text into paragraphs.
    text = text.replace("///", "").strip()
    text = text.replace("--", "").strip()

    # TODO: remove repeated spaces and tabs inside each paragraph.
    if chr(9) in text:
        text = text.replace('\t', ' ')
    while "  " in text:
        text = text.replace("  ", " ")
    # TODO: join non-empty paragraphs with one blank line.
    text_lines = text.splitlines()
    useful_lines = []
    paragraph_marker = False
    for line in text_lines:
        if len(line) == 0 and paragraph_marker == False:
            paragraph_marker = True
            useful_lines.append("\n")
        elif len(line) != 0:
            paragraph_marker = False
            line = line.strip() + "\n"
            useful_lines.append(line)
    text= "".join(useful_lines).strip()
    print(text)
    return text

def split_text(text, chunk_size=4):
    """Split cleaned text into word chunks.
    keep chunking simple:
        split the cleaned text into words;
        create chunks with at most chunk_size words;
        raise ValueError if chunk_size is less than 1;
        return a list of chunk strings.
    """
    if chunk_size < 1:
        raise ValueError("chunk size must be greater than 0"
                         )
    # TODO: split text into words.
    words = text.strip().split()
    splitted_text = []
    index = 0
    while index < len(words):
        if len(words) - index < chunk_size:
            word_row = words[index:len(words)]
            splitted_text.append(" ".join(word_row))
            break
        splitted_text.append(" ".join(words[index:index+chunk_size]))
        index += chunk_size
        # if len(words) - (index) < chunk_size:
        #     splitted_text.append(" ".join(words[index:index + chunk_size]))
        #     break
        # word_row.append(words[index])
        # if len(word_row) % chunk_size == 0:
        #     splitted_text.append(" ".join(word_row))
        #     word_row = []



    # TODO: return chunks with at most chunk_size words each.
    print(splitted_text)
    return splitted_text

def embed_text(text):
    """Create an OpenAI embedding for one chunk of text.
    Each chunk needs an embedding before it can be searched by meaning later.
    The result is a list of numbers. Store that list later in the record’s embedding field.
    You will run the full automated checks on the Unit Tests page.

    """
    # TODO: Call client.embeddings.create() with:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    print(response.data[0].embedding)
    return response.data[0].embedding

    # TODO: Return response.data[0].embedding.



def build_records(folder):
    """Create searchable vector records from every supported file."""
    records = []

    for path in discover_files(folder):
        cleaned = clean_text(path.read_text(encoding="utf-8"))

        if cleaned == "":
            continue

        chunks = split_text(cleaned)

        # for index, chunk in enumerate(chunks, start=1):
        #     # chunk_index, text, and embedding.
        #
        #     records.append({
        #         "id": path.stem + "-" + str(index),
        #         "source": path.name,
        #         "category": path.stem,
        #         "chunk_index": index,
        #         "text": chunk,
        #         "embedding": embed_text(chunk),
        #     })
    for record in records:
        print(record["id"], record["source"], record["category"], record["chunk_index"])
    return records


def write_jsonl(output_path, records):
    """Write one JSON record per line."""
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def main():
    """Build records from docs and write chunks.jsonl."""
    records = build_records(Path("docs"))
    write_jsonl(Path("chunks.jsonl"), records)
    print(f"Wrote {len(records)} records to chunks.jsonl")


if __name__ == "__main__":
    main()

