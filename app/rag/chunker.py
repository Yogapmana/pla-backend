from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_markdown(content: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Split markdown content into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(content)
    # Filter out very short chunks that carry no information
    return [c.strip() for c in chunks if len(c.strip()) > 50]
