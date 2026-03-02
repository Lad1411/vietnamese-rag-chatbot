FROM ubuntu

RUN apt-get update
RUN apt-get -y install python3-pip

WORKDIR ./vietnamse_chatbot

COPY . .

RUN pip3 install -r requirements.txt