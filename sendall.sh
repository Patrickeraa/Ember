docker exec grworker1 rm -rf workspace/files
docker exec grworker2 rm -rf workspace/files
docker exec grserver rm -rf workspace/files
docker cp /home/paraujo/GRPC/worker/files grworker1:/workspace
docker cp /home/paraujo/GRPC/worker/files grworker2:/workspace
docker cp /home/paraujo/GRPC/server/files grserver:/workspace