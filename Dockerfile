FROM python:3.6.3-jessie

ADD riot_neuralnet riot_neuralnet
ADD images images
ADD requirements.txt .

RUN apt-get update && apt-get install -y gfortran libopenblas-dev liblapack-dev

RUN pip install -r requirements.txt
CMD python riot_neuralnet/main.py
