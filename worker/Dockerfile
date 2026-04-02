FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace

COPY files /workspace/files

RUN pip install --no-cache-dir protobuf==6.31.0 grpcio==1.73.0 grpcio-tools==1.73.0 torch torchvision scipy tqdm tensorflow
RUN pip3 install --upgrade "torch==2.7.0+cu126" --index-url https://download.pytorch.org/whl/cu126

CMD ["tail", "-f", "/dev/null"]