from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

def chunk_markdown(content: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Split markdown content into semantic chunks based on headers, then by size if needed."""
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    
    # 1. First, split by Markdown headers to keep semantic sections together
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(content)
    
    # 2. Then, use Recursive splitter to break down sections that are still too large
    # but since it's already split by headers, we focus on paragraphs/sentences
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    
    final_splits = char_splitter.split_documents(md_header_splits)
    
    # Return as strings and filter out very short chunks
    return [doc.page_content.strip() for doc in final_splits if len(doc.page_content.strip()) > 50]
