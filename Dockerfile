FROM nvidia/cuda:10.2-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install python3 python3-pip libpcre3 libpcre3-dev git libjpeg8-dev zlib1g-dev

WORKDIR /opt

RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN git clone https://github.com/movienet/movienet-tools.git

WORKDIR /opt/movienet-tools

RUN pip3 install -r requirements.txt

RUN pip3 install -U setuptools

# This number needs to be adjusted according to your GPU
# https://developer.nvidia.com/cuda-gpus
# For example, if you are using "Geforce RTX GeForce GTX 1080", you should set it as "6.1+PTX"
ARG TORCH_CUDA_ARCH_LIST="5.2+PTX"

RUN python3 setup.py develop
RUN python3 scripts/download_models.py
RUN apt-get -y install ffmpeg libsm6 libxext6

CMD /bin/bash
