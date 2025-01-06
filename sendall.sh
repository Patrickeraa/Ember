docker exec grworker1 rm -rf /workspace/files
docker exec grworker2 rm -rf /workspace/files
docker exec grserver rm -rf /workspace/files
docker cp /home/paraujo/POC/worker/files grworker1:/workspace
docker cp /home/paraujo/POC/worker/files grworker2:/workspace
docker cp /home/paraujo/POC/server/files grserver:/workspace