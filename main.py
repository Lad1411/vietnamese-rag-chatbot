from RAG_pipeline import RAG_pipeline
from utils import vietnamese_tokenizer
from Ingestion import ingestion_pipeline, chunking
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# 1. Run Ingestion (Load DB and Retriever)
print("=========================== Ingestion ===========================")
ingest_pip = ingestion_pipeline.IngestionPipeline(tokenizer=vietnamese_tokenizer)
vectordb, keyword_retriever = ingest_pip.ingest()
print('Initialize ingestion successfully')

# 2. Initialize RAG Pipeline
print("=========================== Retriever ===========================")
rag_pip = RAG_pipeline(
    keyword_retriever=keyword_retriever,
    vector_db=vectordb,
    chunking_func= chunking.chunking_func
)
print('Initialize retriever successfully')

if __name__ == '__main__':
    while True:
        query = input('Nhập câu hỏi: ')
        if query == "/bye":
            break
        response = rag_pip.generate(query=query)

        print(f"Bot: {response}")
        print("\nNhập /bye để thoát.")






