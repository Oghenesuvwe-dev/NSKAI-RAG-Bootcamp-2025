from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    """Split documents into smaller chunks for better retrieval"""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_index'] = i
    
    return chunks