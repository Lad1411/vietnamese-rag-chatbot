from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking_func(docs):
    """
        Divide documents to smaller chunks

        Args:
            docs: list of documents
    """

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size= 1000,
        chunk_overlap=200,
        length_function=len
    )

    return splitter.split_documents(docs)
