FROM anibali/pytorch:1.13.0-cuda11.8-ubuntu22.04

USER root

#RUN chmod 777 /predictions/

WORKDIR /

COPY . .

RUN pip install -r requirements.txt

#RUN python inference.py
