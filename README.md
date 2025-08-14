# Ember Distributed Framework
Ember is a open source machine learning framework based on PyTorch created to facilitate the process of distributed training. This is a beta version which includes all the necessary to simulate the training environment on a single machine using Docker. 

## Utilization 
Navigate to the project folder and create the containers utilizing ``` docker compose up -d ```. Note that the -d flag is important to keep the containers running in detached mode, to access the containers you can use:

 ``` docker exec -it {container_name} /bin/bash```

With the containers created, open a terminal for each worker node and one for the server.

## Dataloading on Server
Ember utilizes a JSON config file system on client and server to streamline the process of configuring the dataloader, the file consists of:

```
{
    "transforms": [
        {
            "name": "ToTensor",
            "args": []
        },
        {
            "name": "Normalize",
            "args": [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
        },
        { Other arguments that can be used to transform the data}
    ],
    "dataset_path": {
        "train": "{your datapath to train folder}",
        "test": "{your datapath to test folder}"
    }
  }
```
The list of transformations avaiable by the JSON config file are in line with the usual transformations avaiable by the PyTorch dataloader functionalities as follows:

* Grayscale
* ToTensor
* Normalize
* Resize
* RandomCrop
* RandomHorizontalFlip

You can pass the related parameters of the above transformations via the "args" line, which passes the values to the dataloader.

To run the server as a dataset sender reach up to the node where you data is located, then run:

``python server.py --config ./config/{your json config name}``

The terminal will indicate informations about your transforms and the classes present on the dataloader. 

## Worker Setup

Each worker runs with a similar config file, that passes useful information about the training process, such as:

    * nodes
    * gpus
    * nr
    * epochs
    * request_size
    * batch_size

Each separate worker needs a different file with a unique incremental value for the `nr` clause that represents its rank on the system. The first node is the rank 0. 

