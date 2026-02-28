from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_community import GoogleSearchAPIWrapper
from collections import defaultdict
from itertools import chain
from dotenv import load_dotenv
import os
from langchain_core.documents import Document

load_dotenv()

def unique_by_key(iterable, key):
    """Yield unique elements of an iterable based on a key function.

    Args:
        iterable: The iterable to filter.
        key: A function that returns a hashable key for each element.

    Yields:
        Unique elements of the iterable based on the key function.
    """
    seen = set()
    for e in iterable:
        if (k := key(e)) not in seen:
            seen.add(k)
            yield e

class CustomEnsembleRetriever(EnsembleRetriever):
    def weighted_reciprocal_rank(self, doc_lists):
        """
        Return: list[list[doc, relevant_score]]
        """
        if len(doc_lists) != len(self.weights):
            msg = "Number of rank lists must be equal to the number of weights."
            raise ValueError(msg)

        # Associate each doc's content with its RRF score for later sorting by it
        # Duplicated contents across retrievers are collapsed & scored cumulatively
        rrf_score: dict[str, float] = defaultdict(float)
        for doc_list, weight in zip(doc_lists, self.weights, strict=False):
            for rank, doc in enumerate(doc_list, start=1):
                rrf_score[
                    (
                        doc.page_content
                        if self.id_key is None
                        else doc.metadata[self.id_key]
                    )
                ] += weight / (rank + self.c)
        # Docs are deduplicated by their contents then sorted by their scores
        all_docs = chain.from_iterable(doc_lists)
        sorted_docs =  sorted(
            unique_by_key(
                all_docs,
                lambda doc: (
                    doc.page_content
                    if self.id_key is None
                    else doc.metadata[self.id_key]
                ),
            ),
            reverse=True,
            key=lambda doc: rrf_score[
                doc.page_content if self.id_key is None else doc.metadata[self.id_key]
            ],
        )

        relevant_docs = []
        for doc in sorted_docs:
            relevant_docs.append([doc, rrf_score[doc.page_content if self.id_key is None else doc.metadata[self.id_key]]])
        return relevant_docs

class Fallback:
    def __init__(self, chunking_func, vectordb):
        self.chunking_func = chunking_func
        self.vectordb = vectordb

    def process_web_results(self, documents):
        docs = []
        for res in documents:
            # Create a document for each search result
            content = res.get("snippet", "")
            metadata = {"source": res.get("link"), "title": res.get("title")}
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def fallback(self, new_docs):
        pro_new_docs = self.process_web_results(new_docs)
        chunks = self.chunking_func(pro_new_docs)
        # Add to vector_db
        self.vectordb.add_documents(chunks)

        return chunks


class RAGRetriever:
    def __init__(self, keyword_retriever, chunking_func, vectordb):
        self.chunking_func = chunking_func

        semantic_retriever = vectordb.as_retriever()
        self.ensemble_retriever = CustomEnsembleRetriever(
            retrievers=[semantic_retriever, keyword_retriever],
            weights=[0.5, 0.5]
        )

        self.fall_back = Fallback(chunking_func=self.chunking_func, vectordb=vectordb)

        GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        self.search = GoogleSearchAPIWrapper()

    def retrieve(self, query, top_k=5, threshold=0.016):
        """
            Retrieve relevant chunks for a query
            Args:
                query: The string of search query
                top_k: Number of relevant documents returned
                threshold: Threshold score for chunks
        """
        docs = self.ensemble_retriever.invoke(input=query)
        # print(docs)
        relevant_docs = [doc[0] for doc in docs if doc[1]>=threshold][:top_k]

        # No relevant docs -> use Google search
        if len(relevant_docs) == 0:
            print("Không tìm thấy thông tin hữu ich -> Tra Google")
            raw_relevant_docs = self.search.results(query, top_k)

            relevant_docs = self.fall_back.fallback(new_docs=raw_relevant_docs)

        return relevant_docs
