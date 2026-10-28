def load_pdf_text(pdf_path):
    """Extract readable text from a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return load_pdf_text_with_fitz(pdf_path)

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_pdf_text_with_fitz(pdf_path):
    """Fallback PDF extractor for environments that provide PyMuPDF."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("A PDF reader is required to extract sample.pdf text.") from exc

    pages = []
    with fitz.open(pdf_path) as document:
        for page in document:
            pages.append(page.get_text())
    return "\n".join(pages)


def embed_text(text):
    """Create one embedding for a piece of text."""
    from openai import OpenAI
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


def similarity_score(left, right):
    """Return a similarity score for two embedding vectors."""
    left_values = list(left)
    right_values = list(right)
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_size = sum(value * value for value in left_values) ** 0.5
    right_size = sum(value * value for value in right_values) ** 0.5
    if left_size == 0 or right_size == 0:
        return 0.0
    return dot / (left_size * right_size)
