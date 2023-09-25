FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

RUN    yes | apt update
RUN    yes | apt install python3 python3-pip git htop vim

WORKDIR /app
RUN    git clone --recursive -b qtree-sample-api https://github.com/danlkv/QTensor.git
COPY . . 
RUN    cd QTensor/qtree && pip install .
RUN    cd QTensor && pip install .
RUN    pip install quimb pyrofiler cartesian-explorer opt_einsum
RUN    pip install --no-binary pynauty pynauty
# RUN    pip install torch
# RUN    pip install cupy-cuda110

ENTRYPOINT ["bash"]
