from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


def create_keyword_retriever(chunks, vietnamesetokenizer):
    """
        Create Vietnamese keyword retriever
        Args:
            chunks: list of langchain Documents
            vietnamesetokenizer: tokenizer function
    """

    retriever = BM25Retriever.from_documents(
        documents=chunks,
        preprocess_func=vietnamesetokenizer
    )
    return retriever

def create_vector_db(chunks, db_dir='../vector_db', embed_model = '/home/lad/AI/my_local_models/vietnamese-bi-encoder'):
    """
        Create Chroma vector DB
        Args:
            chunks: list of langchain Documents
            db_dir: directory of vector db
            embed_model: path of embedding model
    """
    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model
    )

    vector_store = Chroma(
        collection_name='langchain_store',
        embedding_function=embed_model,
        persist_directory=db_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )

    batch_size = 5000
    total_chunks = len(chunks)
    progressbar = tqdm(range(0, total_chunks, batch_size), desc="Ingesting to ChromaDB")

    for i in progressbar:
        batch_chunks = chunks[i: (i+batch_size)]
        vector_store.add_documents(batch_chunks)
    return vector_store


