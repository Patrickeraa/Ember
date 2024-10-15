FROM nvcr.io/nvidia/pytorch:23.05-py3

WORKDIR /workspace

COPY files /workspace/files

RUN pip install --no-cache-dir grpcio grpcio-tools protobuf==5.27.2 torch torchvision scipy


CMD ["tail", "-f", "/dev/null"]