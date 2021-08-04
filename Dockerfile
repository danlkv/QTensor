FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN    yes | apt update
RUN    yes | apt install python3 python3-pip git htop vim
RUN    yes | apt install mpich

WORKDIR /app
RUN    git clone --recursive -b dev https://github.com/danlkv/QTensor.git
RUN    cd QTensor/qtree && pip install .
RUN    cd QTensor && pip install .

RUN    pip install quimb pyrofiler cartesian-explorer pynauty opt_einsum
RUN    pip install mpi4py
RUN    pip install torch
RUN    pip install cupy-cuda110

ENTRYPOINT ["python3"]
