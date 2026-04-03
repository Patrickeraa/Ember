FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace

COPY files /workspace/files

RUN pip install --no-cache-dir protobuf==6.31.0 grpcio==1.73.0 grpcio-tools==1.73.0 torch torchvision scipy tqdm tensorflow


CMD ["tail", "-f", "/dev/null"]