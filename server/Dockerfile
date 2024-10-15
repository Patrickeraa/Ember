FROM python:3.9-slim
WORKDIR /workspace
COPY . .
RUN pip install --no-cache-dir grpcio grpcio-tools torch torchvision scipy

CMD ["tail", "-f", "/dev/null"]