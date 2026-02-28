# Vietnamese RAG Chatbot

An advanced, locally-hosted Retrieval-Augmented Generation (RAG) chatbot designed specifically for the Vietnamese language. This pipeline ensures high-quality, context-aware responses by combining local document retrieval with intelligent web search fallbacks.

## 🚀 Features

- **LLM Engine:** Powered by the `AITeamVN/Vi-Qwen2-7B-RAG` model for native, high-fidelity Vietnamese text generation.
- **Vector Database:** Utilizes **ChromaDB** for fast, local vector storage and retrieval.
- **Hybrid Search:** Implements a robust retrieval pipeline combining **Semantic Search** (dense vector embeddings) and **BM25** (sparse keyword matching) to maximize document relevance.
- **Smart Fallback Mechanism:** If the hybrid search fails to find a high-confidence match in the local vector database, the system automatically falls back to **Google Search** to retrieve real-time, external information.
- **Customizable Prompting:** Easy-to-edit system prompts to adjust the assistant's behavior and response boundaries.


## 🛠️ How to Use the Repo

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system. 

### 2. Installation
Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/vietnamese-rag-chatbot.git](https://github.com/your-username/vietnamese-rag-chatbot.git)
cd vietnamese-rag-chatbot
pip install -r requirements.txt
```

### 3. Environment Setup
Generate API key at https://developers.google.com/custom-search/docs/paid_element#api_key


Create a .env file in the `Retrieval_pipeline` directory of the project to configure your API keys (required for the Google Search fallback):
```bash
GOOGLE_CSE_ID = ""
GOOGLE_API_KEY = ""
```

### 4. Running the chatbot
```bash
python main.py
```


## 📊 Evaluation
This project is evaluated using **RAGAS** (Retrieval-Augmented Generation Assessment) to measure retrieval quality, generation correctness, and overall end-to-end RAG performance. 

The evaluation framework ensures the Vietnamese RAG chatbot produces accurate, grounded, and context-aware responses.

---

### 1. Generation Evaluation

We evaluate the generation module independently to verify that the language model produces correct answers based on the provided context.

#### Correctness Metric (RAGAS)

```python
correctness_metric = DiscreteMetric(
    name="correctness",
    prompt="""Compare the model response to the expected answer and determine if it's correct.
Consider the response correct if it:
1. Contains the key information from the expected answer
2. Is factually accurate based on the provided context
3. Adequately addresses the question asked

Return 'pass' if the response is correct, 'fail' if it's incorrect.

Question: {question}
Expected Answer: {expected_answer}
Model Response: {response}

Evaluation:""",
    allowed_values=["pass", "fail"],
)
```

### 2. Retrieval evaluation
```python
context_metric = DiscreteMetric(
    name="context",
    prompt="""Compare the model retrieved context to the expected answer and determine if it is adequate.
Consider the context adequate if it:
1. Contains the key facts necessary to answer the question correctly.
2. The information in the expected answer can be fully supported using only the retrieved context.
3. Minor wording differences or paraphrasing are acceptable.

Return 'pass' if the response is adequate, 'fail' if important information is missing

Question: {question}
Expected Answer: {expected_answer}
Retrieved context = {contexts}
Evaluation:""",
    allowed_values=["pass", "fail"]
)
```

Reported Metric: > End-to-End Pass Rate (%)

### 4. Result

| Evaluation Type | Metric                  | Score |
| :--- |:------------------------|:------|
| Retrieval | Context metric          | 83.5% |
| Generation | Correctness (Pass Rate) | 100%  |
| End-to-End | Correctness (Pass Rate) | 97.5% |


## 🤖 Chatbot Demo
![img.png](img.png)![img_1.png](img_1.png)