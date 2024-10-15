FROM python:3.9-slim
WORKDIR /workspace
COPY . .
RUN pip install --no-cache-dir grpcio grpcio-tools protobuf==5.27.2 torch torchvision scipy

CMD ["tail", "-f", "/dev/null"]