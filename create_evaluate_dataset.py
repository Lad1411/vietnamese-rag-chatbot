from RAG_pipeline import RAG_pipeline
from utils import vietnamese_tokenizer
from Ingestion import ingestion_pipeline, chunking
import os
from datasets import load_dataset, Dataset
import pandas as pd

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("Load the evaluate dataset")
eval_dataset = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG")["train"]
sub_eval = eval_dataset.select(range(200))


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

# Create generation evaluation dataset
# answers = []
#
# for idx, row_dict in enumerate(sub_eval):
#     answer = rag_pip.generate(query=row_dict["question"], context_text= row_dict["context"])
#     answers.append(answer)
#     print("Processed: {}/200".format(idx+1))
#
# gen_eval_df = sub_eval.to_pandas()
# gen_eval_df.rename(columns={"answer": "ground_truth"}, inplace=True)
# gen_eval_df.drop(labels="context", axis=1, inplace=True)
# gen_eval_df["answer"] = answers
#
# gen_eval_df.to_csv("Generation_eval_dataset.csv")
# print("Save to Generation_eval_dataset.csv")

# Create retrieval evaluation and end-to-end dataset
retrieved_contexts = []
responses = []
for idx, row_dict in enumerate(sub_eval):
    answer, context = rag_pip.generate(query=row_dict["question"], evaluation=True)
    retrieved_contexts.append(context)
    responses.append(answer)
    print("Processed: {}/200".format(idx+1))

eval_df = sub_eval.to_pandas()
eval_df.rename(columns={"answer": "ground_truth"}, inplace=True)
eval_df.drop(labels="context", axis=1, inplace=True)
eval_df["contexts"] = retrieved_contexts
eval_df["answer"] = responses

eval_df.to_csv("Evaluate_dataset.csv")
print("Save to Evaluate_dataset.csv")







