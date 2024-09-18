FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir grpcio grpcio-tools torch torchvision

CMD ["tail", "-f", "/dev/null"]