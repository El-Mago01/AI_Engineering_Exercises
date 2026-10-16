"""
Evidence Hunt Brief
A support team has three short help-centre notes, but the useful answer is not always in the file a user would guess first.

Your job is to build a compact LangChain evidence finder.
It should search the notes by meaning, then print the strongest readable evidence for a question.

The workflow is:
load notes -> split documents -> embed chunks -> search for evidence
You will complete one file: evidence_hunt.py


The finished program prints a compact evidence report:
LANGCHAIN EVIDENCE HUNT
Query: how long do refunds take?

Best evidence:
0.82 | returns.md | Approved card refunds normally appear within five to seven working days.

"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#To log the progress of DirectoryLoader you need to install tqdm, `pip install tqdm`
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

api_key = os.environ.get("OPENAI_API_KEY")
DOCS_PATH = Path("docs")
EMBEDDING_MODEL = "text-embedding-3-small"


def load_notes(folder):
    """Load the Markdown support notes as LangChain Document objects.
    Use LangChain’s DirectoryLoader with TextLoader to load the Markdown notes from docs/.
    Each returned item is a LangChain Document.
    Each document has:
        page_content -> the readable note text
        metadata     -> facts about where it came from
    """
    loader = DirectoryLoader(
        str(folder),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    return documents


def split_notes(documents):
    """Split loaded notes while preserving document metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=220,
        chunk_overlap=40,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_store(chunks):
    """Build an in-memory LangChain vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
    store = InMemoryVectorStore.from_documents(chunks, embeddings)
    return store


def hunt(query, store, top_k=3):
    """Return the strongest evidence chunks for a query."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    matches = store.similarity_search_with_score(query, k=top_k)
    results = []
    for item in matches:
        result = {
        "score" : item[1],
        "source" : item[0].metadata['source'],
        "text" : item[0].page_content
        }
        results.append(result)
    return results


def print_evidence(query, results):
    """Print a compact evidence report."""
    print("LANGCHAIN EVIDENCE HUNT")
    print(f"Query: {query}")
    print()

    if not results:
        print("Best evidence:")
        print("No evidence found.")
        return

    print("Best evidence:")
    for item in results:
        print(f"{item["score"]:.2f} | {item["source"]} | {item["text"]}")


def main():
    query = "What do I need to provide in order to get a refund?"
    documents = load_notes(DOCS_PATH)
    chunks = split_notes(documents)
    store = build_store(chunks)
    results = hunt(query, store, top_k=3)
    print_evidence(query, results)


if __name__ == "__main__":
    main()
