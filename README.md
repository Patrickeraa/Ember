# Ember Distributed Framework
Ember is a open source machine learning framework based on PyTorch created to facilitate the process of distributed training. This is a beta version which includes all the necessary to simulate the training environment on a single machine using Docker. 

## Utilization 
Navigate to the project folder and create the containers utilizing ``` docker compose up -d ```. Note that the -d flag is important to keep the containers running in detached mode, to access the containers you can use:

 ``` docker exec -it {container_name} /bin/bash```

With the containers created, open a terminal for each worker node and one for the server.

## Dataloading on Server
Ember utilizes 
To setup a dataset sender reach up to the node where you data is located, then run:

``python server.py --config ./config/{your json name}``


