FROM tiagopeixoto/graph-tool:latest

RUN    yes | pacman -Syu
RUN    yes | pacman -S git openssh gcc make cmake neovim python-pip 
RUN    # A better shell, optional
RUN    yes | pacman -S fish

WORKDIR /root
RUN    git clone --recursive -b dev https://github.com/danlkv/QTensor.git
RUN    cd QTensor/qtree && pip install .
RUN    cd QTensor && pip install .

EXPOSE 8888

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
