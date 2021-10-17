import sys
sys.path.append('.')

import os
import torch
from sklearn.metrics import classification_report
from flnode.pipeline2.algo import Algo
from common.utils import Mapping
from monai.metrics import compute_roc_auc
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
import numpy as np
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"



class MonaiAlgo(Algo):
    def __init__(self):
        self.model = None
        self.loss = None
        self.optimizer = None
        self.epochs = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.model_dir = None
        self.metric = None


    def train(self):
        print('In train loop')
        # Set deterministic training for reproducibility
        #set_determinism(seed=0)
        device = torch.device(DEVICE)
        val_interval = 1
        best_metric = -1
        best_metric_epoch = -1
        epoch_loss_values = list()
        metric_values = list()

        self.model.to(device)
        for epoch in range(self.epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.epochs}")
            self.model.train()
            for batch_idx, (data_batch) in enumerate(self.train_loader):
                data, target = data_batch['img'].to(DEVICE), data_batch['seg'].to(DEVICE)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()


            if (epoch + 1) % val_interval == 0:
                print('Validating')
                self.model.eval()
                with torch.no_grad():
                    val_metrics = []
                    for batch_idx, (data_batch) in enumerate(self.val_loader):
                        data, target = data_batch['img'].to(DEVICE), data_batch['seg'].to(DEVICE)

                        output_logits = sliding_window_inference(data,
                                                                 sw_batch_size=2,
                                                                 roi_size=(128, 128, 128),
                                                                 predictor=self.model,
                                                                 overlap=0.25,
                                                                 do_sigmoid=False)
                        output = torch.sigmoid(output_logits)

                        #loss = self.criterion(output, target)
                        val_metrics.append(self.metric(output, target, include_background=False).cpu().numpy())
                mean_val_metric = np.mean(val_metrics)
                print(mean_val_metric)

        checkpoint = Mapping()
        checkpoint.update(epoch=epoch, weights=self.model.state_dict(), metric=mean_val_metric)
        #print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        return checkpoint



    def load_model(self, modelFile):
        path = modelFile
        self.model.load_state_dict(torch.load(path))

    def save_model(self, model, path):
        pass
        # json.dump(model, path)

    def predict(self, headModelFile):
        set_determinism(seed=0)
        device = torch.device(DEVICE)
        self.load_model(headModelFile)
        self.model.to(device)
        self.model.eval()

        dice_scores = []
        with torch.no_grad():
            for data_batch in self.test_loader:
                data, target = data_batch['img'].to(DEVICE), data_batch['seg'].to(DEVICE)

                output_logits = sliding_window_inference(data,
                                                         sw_batch_size=2,
                                                         roi_size=(128, 128, 128),
                                                         predictor=self.model,
                                                         overlap=0.25,
                                                         do_sigmoid=False)
                output = torch.sigmoid(output_logits)

                # loss = self.criterion(output, target)
                dice_scores.append(self.metric(output, target, include_background=False).cpu().numpy())
        test_report = Mapping()
        test_report.update(report=dice_scores, target_names='dice', digits=4)
        
        return test_report
