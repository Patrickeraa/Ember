FROM python:3.9-slim
WORKDIR /workspace
COPY . .
RUN pip install --no-cache-dir protobuf==6.31.0 grpcio==1.73.0 grpcio-tools==1.73.0 torch torchvision scipy tqdm

CMD ["tail", "-f", "/dev/null"]