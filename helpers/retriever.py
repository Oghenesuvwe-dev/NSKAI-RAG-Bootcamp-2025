def create_retriever(vectorstore):
    """Create simple retriever"""
    
    # Simple retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    return retriever