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
