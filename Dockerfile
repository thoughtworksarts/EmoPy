FROM python:2.7.14-jessie

ADD singleImageTDNN.py .
ADD setup.py .

RUN pip install numpy

CMD python setup.py install
#CMD pip install -U setuptools && pip list
