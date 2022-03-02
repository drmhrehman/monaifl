from pathlib import Path
cwd = str(Path.cwd())

import sys
sys.path.append('.')
import os
from hub import Client
import time
import logging
import concurrent.futures
import copy
import time
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)
import torch as t
from coordinator import FedAvg

modelpath = os.path.join(cwd, "save","models","hub")
modelName = "monai-test.pth.tar"

modelFile = os.path.join(modelpath, modelName)


w_loc = []

nodes ={  1: {'address': 'localhost:50051'},
          2: {'address': 'localhost:50052'},
       }

clients = (Client(nodes[1]['address']), Client(nodes[2]['address']))

def model_spread_plan(client):
    try:
        if(client.status() == "alive"):
            # spreading model to nodes
            client.bootstrap()
    except:
        logger.info(f"client {client.address} is dead...")

def train_plan(client):
    try:
        if(client.status() == "alive"):
            # initializing training on nodes
            client.train()
    except:
        logger.info(f"client {client.address} is dead...")


def collect_models(client):
    checkpoint = None
    try:
        if(client.status() == "alive"):
            logger.info(f"Collecting with Node: {client.address}...")
            checkpoint = client.gather()
    except:
        logger.info(f"client {client.address} is dead...")
    return checkpoint['weights']


def aggregate_weights(weights):
    logger.info(f"Aggregating data from {len(weights)} nodes...")
    global_weights = FedAvg(weights)

    hub_checkpoint = {
        'weights': global_weights
    }
    t.save(hub_checkpoint, modelFile)
    logger.info(f"Aggregation completed")


def test_plan(client):
    try:
        if(client.status() == "alive"):
            # testing models on nodes
            client.test()
    except:
        logger.info(f"client {client.address} is dead...")
    
def stop_now(client):
    try:
        if(client.status() == "alive"):
            # asking nodes to stop
            client.stop()
    except:
        logger.info(f"client {client.address} is dead...")
    

if __name__ == '__main__':
    logger.info("Central Hub initialized")

    global_round = 2

    for round in range(global_round):
        if (round==0):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                result = executor.map(model_spread_plan, clients)    

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(train_plan, clients)    

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(collect_models, clients)

        weights = list(result)
        aggregate_weights(weights)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = executor.map(test_plan, clients)    
        
        logger.info(f"Global round {round+1}/{global_round} completed")
        print("-------------------------------------------------")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(stop_now, clients)  
    
    # all processes are excuted 
    logger.info(f"Done! Model Training is completed across all sites and current global model is available at following location...{modelFile}")