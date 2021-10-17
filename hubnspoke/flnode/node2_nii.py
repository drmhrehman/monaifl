from pathlib import Path
import threading
cwd = str(Path.cwd())
print(cwd)
import os
import sys
sys.path.append('.')

from concurrent import futures
from io import BytesIO
import grpc
import json
from common import monaifl_pb2_grpc as monaifl_pb2_grpc
from common.monaifl_pb2 import ParamsResponse
from common.utils import Mapping
import torch as t
from flnode.start_pipeline_nii import instantiateMonaiAlgo
import logging
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)


modelName = "segment-test.pth.tar"

headmodelpath = os.path.join(cwd, "save","models","node2","head")
headModelFile = os.path.join(headmodelpath, modelName)

trunkmodelpath = os.path.join(cwd, "save","models","node2","trunk")
trunkModelFile = os.path.join(trunkmodelpath, modelName)

configpath = os.path.join(cwd, "save","configs","node2")
configName = 'config_nii.json'
configFile = os.path.join(configpath, configName)

w_loc = []
request_data = Mapping()
ma = instantiateMonaiAlgo(0.2, 0.2, 'MedicalDecathlon2')

class MonaiFLService(monaifl_pb2_grpc.MonaiFLServiceServicer):
    def __init__(self, stop_event):
        self._stop_event = stop_event
    
    def ModelTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        t.save(request_data, headModelFile)
        if os.path.isfile(headModelFile):
            request_data.update(reply="model received")
            logger.info(f"global model saved at: {headModelFile}")
            logger.info("FL node is ready for training and waiting for training configurations")
        else:
            request_data.update(reply="error while receiving the model")
            logger.error("FL node is not ready for training")
        
        logger.info("returning answer to the central Hub...")
        buffer = BytesIO()
        t.save(request_data['reply'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def MessageTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        print(request_data['trainer']['epochs'])
        logger.info('received training configurations')
        #logger.info(f"local epochs to run: {request_data['epochs']}")
        # training and checkpoints
        with open(configFile, 'w') as json_file:
            json.dump(request_data, json_file, indent=4)
        logger.info("starting training...")
        
        ma.epochs = int(request_data['trainer']['epochs'])
        checkpoint = Mapping()
        checkpoint = ma.train()
        logger.info("saving trained local model...")
        t.save(checkpoint, trunkModelFile)
        logger.info(f"local model saved at: {trunkModelFile}")
        logger.info("sending training completed message to the the Central Hub...")
        buffer = BytesIO()
        request_data.update(reply="training started")
        t.save(request_data['reply'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def NodeStatus(self, request, context):
        logger.info("received status request")
                
        request_data.update(reply="alive")
        logger.info("node status: alive")
        
        logger.info("sending node status to the central hub...")
        buffer = BytesIO()
        t.save(request_data['reply'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def TrainedModel(self, request, context):
        buffer = BytesIO()
        if os.path.isfile(trunkModelFile):
                logger.info(f"sending trained model {trunkModelFile} to the central Hub...") 
                checkpoint = t.load(trunkModelFile, map_location='cpu')
                t.save(checkpoint, buffer)

        return ParamsResponse(para_response=buffer.getvalue())
    
    def ReportTransfer(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        t.save(request_data, headModelFile)
        if os.path.isfile(headModelFile):
            request_data.update(reply="model received for testing")
            logger.info(f"global model saved at: {headModelFile}")
        else:
            request_data.update(reply="error while receiving the model")

        logger.info('received test request')

        response_data = Mapping()
        response_data = ma.predict(headModelFile)

        logger.info("sending test report to the Central Hub...")       
        buffer = BytesIO()
        t.save(response_data['report'], buffer)
        return ParamsResponse(para_response=buffer.getvalue())
    
    def StopMessage(self, request, context):
        request_bytes = BytesIO(request.para_request)
        request_data = t.load(request_bytes, map_location='cpu')
        logger.info('received stop request')   
        logger.info("sending stopping status to the Central Hub...")
        buffer = BytesIO()
        response_data = Mapping()
        response_data.update(reply="stopping")
        t.save(response_data, buffer)
        logger.info('node stopping...thanks for using MONAI-FL...see you soon.')  
        self._stop_event.set() 
        return ParamsResponse(para_response=buffer.getvalue())

def serve():
    stop_event = threading.Event()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),options=[
               ('grpc.max_send_message_length', 1000*1024*1024),
               ('grpc.max_receive_message_length', 1000*1024*1024)])
    monaifl_pb2_grpc.add_MonaiFLServiceServicer_to_server(
        MonaiFLService(stop_event), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    logger.info("FL node is up and waiting for training configurations...")
    stop_event.wait()
    server.stop(5)

if __name__ == "__main__":
    serve()