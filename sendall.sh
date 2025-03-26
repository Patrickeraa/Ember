docker exec grworker1 rm -rf /workspace/files
docker exec grworker2 rm -rf /workspace/files
docker exec grserver rm -rf /workspace/files
docker cp /home/patrick.araujo/GRPC/worker/files grworker1:/workspace
docker cp /home/patrick.araujo/GRPC/worker/files grworker2:/workspace
docker cp /home/patrick.araujo/GRPC/server/files grserver:/workspace