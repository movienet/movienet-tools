FROM nvidia/cuda:10.2-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"

# Run the following if you are in mainland China
RUN rm /etc/apt/sources.list
COPY ./docker/tsinghua.u18.04.sources.list /etc/apt/sources.list

RUN apt-get -y update && apt-get -y install --no-install-recommends build-essential python3 python3-dev python3-pip libpcre3 libpcre3-dev git libjpeg8-dev zlib1g-dev ffmpeg libsm6 libxext6

WORKDIR /opt
COPY ./ /opt/movienet-tools

# Add "-i https://pypi.tuna.tsinghua.edu.cn/simple some-package" if you are in mainland China
RUN pip3 install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && pip3 install --no-cache-dir -U setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /opt/movienet-tools

# Add "-i https://pypi.tuna.tsinghua.edu.cn/simple some-package" if you are in mainland China
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# This number needs to be adjusted according to your GPU
# https://developer.nvidia.com/cuda-gpus
# For example, if you are using "Geforce RTX GeForce GTX 1080", you should set it as "6.1+PTX"
# Note that cuda 10 does not support ARCH version higher than 7.5
ARG TORCH_CUDA_ARCH_LIST="7.5+PTX"

RUN python3 setup.py develop
#RUN python3 scripts/download_models.py

CMD /bin/bash
