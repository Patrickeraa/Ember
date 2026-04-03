# Ember: Asynchronous Dynamic Data Serving for PyTorch Distributed Training

Ember is a open source machine learning framework based on PyTorch created to facilitate the process of distributed training. This branch contains a summarized flow to run a training loop using ember simulating a distributed setting using docker compose.

Ember can be customizable, simple, and a efficient distributed
training system. First, Ember streams data asynchronously to different nodes using
Google Remote Procedure Calls (gRPC) instead of distributing entire datasets before
training, as in state-of-the-art approaches. Second, Ember nodes keep only the shards
of the dataset that are currently being used by training devices and that will be used soon,
significantly reducing memory usage. Third, Ember starts training as soon as the first
batch of training data arrives on all GPUs, thus accelerating training by minimizing GPU
idling time. Finally, Ember exposes a single-node-like API, hiding all the complexity of
dataset distribution.

## README Structure

The repository is organized in two main folders, server and worker. The distinction separates the files present in the main server loading functions and the codes loaded to each worker node. The docker-compose in the root of the project orchestrates the launch of the entire example in one go as explained below.

**note:** the containers used in this simulation can weight up to 30gb due to Nvidia's native packaging and additional libraries used to run the distributed pipeline. 

## Badges considered (Selos Considerados)
The authors of this work consider applying to the following badges: "Artefatos Disponíveis (SeloD)", "Artefatos Funcionais (SeloF)", "Artefatos Sustentáveis (SeloS)", and "Experimentos Reprodutíveis (SeloR)".

## Basic Information

The execution ambient will consist of three main containers, one for the server and two for each one of the two nodes. The objective is to simulate a distributed environment in a single machine to facilitate reproductability. 

The container images already contains every library needed for the testing and reproduction of the smaller model testing using MNIST. Since the project goal is to be used with different datasets and models, the example utilizes a smaller model and dataset to make this training loop faster. 

**Note:** A GPU is strongly recommended to make the training step faster.

## Dependencies

The containers already contains all the necessary libraries.

## Security

The execution of the local cluster doesn't have any network issue or other dangers.

## Instalation
Make sure you have Docker and Docker-Compose installed, which can be obtained for Linux and Windows on: https://docs.docker.com/engine/install/

All additional dependencies will be installed on the steps bellow.

## Utilization and minimun test
Navigate to root project folder and create the containers utilizing ``` docker compose up -d ```. Note that the -d flag is important to keep the containers running in detached mode, to be able to open different terminals to each worker and run separate commands in each one.

With the containers created

 ``` docker exec -it {container_name} /bin/bash```

With the containers created, open a terminal for each worker node and one for the server. Example:

 ``` 
 docker exec -it grworker1 /bin/bash
 docker exec -it grworker2 /bin/bash
 docker exec -it grserver /bin/bash
 ```

## Launching the Server proccess
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

### Running the server
To run the example pipeline you can download the dataset present on https://github.com/teavanist/MNIST-JPG and load the zipped folder inside the server container, adding the correct folder location to the ```config.json ``` file in the server files.

**example:**
```
{
  "transforms": [
      {
          "name": "Grayscale",
          "args": [1]
      },
      {
          "name": "ToTensor",
          "args": []
      },
      {
          "name": "Normalize",
          "args": [[0.5], [0.5]]
      },
      {
            "name": "Resize",
            "args": [28]
      }
  ],
  "dataset_path": {
    # SPECIFY HERE
      "train": "/workspace/datasets/MNIST/train",
      "test": "/workspace/datasets/MNIST/test",
      "val": "/workspace/datasets/MNIST/val"
  },
  "validation": false,
  "data_type": "image"
}
```

**Important:** the ````/workspace/datasets/``` path is a default configuration setted up in the ```docker-compose``` configuration file and can be changed.

### Worker Setup

Each worker runs with a similar config file, that passes useful information about the training process, such as:

    * nodes
    * gpus
    * nr
    * epochs
    * request_size
    * batch_size

Each separate worker needs a different file with a unique incremental value for the `nr` clause that represents its rank on the system. The first node is the rank 0. 

the ```config``` folder on the worker folders already have the necessary parameters to run a 2 worker example. To start the training run the following command on worker 0:

```
python ember_iterable.py --config ./config/config1.json
```

this will queue the worker process and wait for the total number of workers specified on the config to start the training. Then run on the second worker terminal:

```
python ember_iterable.py --config ./config/config2.json
```
The training will start, the server will divide the dataset and start the streaming process to send chunks to be processed by each node. This is the overall workflow with ```ember_iterable.py``` which is the default training process. 


### Experiments and Claims

- A more complex test can be executed using ```instrumented_test.py``` to visualize the fluctuations in data efficiency vs. epoch time while generating a confusion matrix of the results. This training loop takes longer due to the amount of loops and configurations, but can be used to replicate the image present in the paper.

- A comparison to analyze memory comsumption between streamed dataset and default distributed workflow can be obtained by running ```ember_mem.py``` and ```ember_train_main.py```.

- A simpler training loop without data streaming can be used to run a baseline default pytorch distributed training with ```ember_control.py``` to compare the amount of memory used during training on the workers when the entire dataset in loaded in memory. 

**Note:** Since Ray framework, one baseline used to comparison in the article requires its own setup, files and container configuration, a simpler baseline was presented in the repository to ease the reproduction of some results.

## LICENSE

This project is licensed under MIT license. See LICENSE file for more details.