from .data_loader import document_loader
from .chunking import chunking_func
from .embedding import create_vector_db, create_keyword_retriever


class IngestionPipeline:
    def __init__(self,tokenizer, doc_dir = '/home/lad/AI/vietnamese-rag-chatbot/docs'):
        self.doc_dir = doc_dir
        self.tokenizer = tokenizer


    def ingest(self):
        """Runs the full ingestion process."""
        # Load documents from a directory
        documents = document_loader(self.doc_dir)

        if not documents:
            print("No documents found!")
            return None
        # Chunking documents that already loaded
        chunks = chunking_func(documents)
        # Store to vector db
        vector_db = create_vector_db(chunks=chunks)
        # Key word retriever
        keyword_retriever = create_keyword_retriever(chunks=chunks, vietnamesetokenizer=self.tokenizer)
        # print(vector_db)
        return vector_db, keyword_retriever
