FROM python:3.7

RUN mkdir /opt/ml
RUN mkdir /opt/ml/input
RUN mkdir /opt/ml/output
RUN mkdir /opt/ml/model
RUN mkdir /opt/ml/code
RUN mkdir /opt/ml/user
WORKDIR /opt/ml/code

COPY requirements.txt /opt/ml/code
RUN pip3 install -r requirements.txt

COPY ./src /opt/ml/code
COPY ./auth/ai-vqc-fb43ef70f5ec.json /opt/ml/user

ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/ml/user/ai-vqc-fb43ef70f5ec.json

