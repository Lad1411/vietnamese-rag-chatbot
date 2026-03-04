FROM ubuntu

RUN apt-get update
RUN apt-get -y install python3 python3-pip
RUN apt install python3.12-venv -y

WORKDIR /chatbot

COPY requirements.txt .
RUN python3 -m venv venv
ENV PATH="/chatbot/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY docs/ docs/
COPY Ingestion/ Ingestion/
COPY Retrieval_pipeline/ Retrieval_pipeline/
COPY my_local_models/ my_local_models/
COPY RAG_pipeline.py .
COPY main.py .
COPY utils.py .
COPY vietnamese-stopwords-dash.txt .

VOLUME /home/lad/AI/vietnamese-rag-chatbot/Vi-Qwen2-7B-RAG:/Vi-Qwen2-7B-RAG

CMD ["python3", "main.py"]