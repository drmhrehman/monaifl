from client import Client
from common.utils import Mapping
from startpipeline import instantiateMonaiAlgo
if __name__ == '__main__':
    ma, class_names = instantiateMonaiAlgo(0.2, 0.5, 'MedNIST1')
    client = Client("client1", "localhost:50051")

    client.bootstrap(ma.model, ma.optimizer)

    # training and checkpoints
    checkpoint = Mapping()
    checkpoint = ma.train()
    # print(checkpoint)

    #aggregation request
    client.aggregate(ma.model, ma.optimizer, checkpoint)
    report = Mapping()
    report = ma.predict(client, class_names)

    #performs testing on the test dataset and then reports back the summary of training
    client.report(report)