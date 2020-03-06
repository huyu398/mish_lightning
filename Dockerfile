FROM huyu398/deep_learning_env:cuda10_python3.7

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update \
 && apt -y install git \
 && apt -y clean \
 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/NVIDIA/apex /root/apex
RUN pip install https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp37-cp37m-linux_x86_64.whl numpy==1.18.1
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /root/apex

WORKDIR /workspace

# RUN apt update \
#  && apt -y install gfortran liblapack-dev libopencv-dev \
#  && apt -y clean \
#  && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt   ./requirements.txt
RUN pip install -r requirements.txt
# RUN python -c "from torchvision import datasets; datasets.MNIST(root='./data', train=True, download=True)"
# RUN python -c "from torchvision import models; models.vgg19(pretrained=True)"

RUN pip install ipdb

RUN mkdir -p ./data/MNIST/raw
RUN wget -P ./data/MNIST/raw \
    http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz \
    http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz \
    http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz  \
    http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
RUN python -c "from torchvision import datasets; datasets.MNIST(root='./data', train=True, download=True)"
RUN python -c "from torchvision import datasets; datasets.CIFAR10(root='./data', train=True, download=True)"
