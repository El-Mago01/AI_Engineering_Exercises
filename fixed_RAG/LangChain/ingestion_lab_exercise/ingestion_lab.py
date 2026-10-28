import re
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


SOURCE = Path("docs/returns_policy.md")

def clean_text(text: str) -> str:
    useful_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("NAV:", "COOKIE:", "UPDATED:")):
            continue
        useful_lines.append(stripped)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(useful_lines)).strip()


def load_documents():
    documents = TextLoader(str(SOURCE), encoding="utf-8").load()

    documents = TextLoader(str(SOURCE), encoding="utf-8").load()
    if len(documents) == 0:
        raise "No documents loaded"
    for document in documents:
        document.page_content = clean_text(document.page_content)
        document.metadata.update(
            {
                "source": SOURCE.name,
                "department": "customer-care",
                "document_type": "policy",
            }
        )
    return documents


def split_documents(documents, chunk_size: int = 260, chunk_overlap: int = 40):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents(
        [document.page_content for document in documents]
    )
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = f"returns-{index:02d}"
        chunk.metadata["source"] = SOURCE.name
    return chunks


def print_report(documents, chunks) -> None:
    print("LANGCHAIN INGESTION REPORT")
    print(f"source documents: {len(documents)}")
    print(f"chunks created:   {len(chunks)}\n")
    for chunk in chunks:
        preview = chunk.page_content.replace("\n", " ")[:90]
        print(
            f"{chunk.metadata['chunk_id']} | "
            f"{len(chunk.page_content):>3} chars | "
            f"{chunk.metadata['source']} | {preview}"
        )


def main() -> None:
    documents = load_documents()
    chunks = split_documents(documents)
    chunks = split_documents(documents, chunk_size=140, chunk_overlap=25)

    print_report(documents, chunks)


if __name__ == "__main__":
    main()
