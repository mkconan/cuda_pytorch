FROM nvidia/cuda:11.0-devel-ubuntu20.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch Pillow numpy
WORKDIR /LayerCNN
 
COPY train_time_ver27.py /LayerCNN/
COPY model_skip50.py /LayerCNN/

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs