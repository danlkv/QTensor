FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04

RUN    yes | apt update
RUN    yes | apt install python3 python3-pip

WORKDIR /root
RUN    git clone --recursive -b dev https://github.com/danlkv/QTensor.git
RUN    cd QTensor/qtree && pip install .
RUN    cd QTensor && pip install .

RUN    pip install quimb cotengra pyrofiler cartesian-explorer pynauty opt_einsum
RUN    pip install pytorch
RUN    pip install cupy-112

ENTRYPOINT ["python3"]
