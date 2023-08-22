FROM nvcr.io/nvidia/pytorch:19.11-py3

USER root
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y zip htop screen libgl1-mesa-glx net-tools lrzsz

RUN pip uninstall torch torchtext torchvision --y
RUN pip install seaborn thop torch torchvision torchaudio

RUN git clone https://github.com/JunnYu/mish-cuda
RUN cd mish-cuda && python setup.py build install --record mish-install.log

RUN git clone https://github.com/fbcotter/pytorch_wavelets
RUN cd pytorch_wavelets && pip install .

WORKDIR /home

ENV PATH="/home/.local/bin:${PATH}"

RUN mkdir /logs && chmod -R 777 /logs

ENV FVCORE_CACHE="/tmp"
