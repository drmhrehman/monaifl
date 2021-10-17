import sys
sys.path.append('.')

import os
import torch
from sklearn.metrics import classification_report
from trainer.pipeline.algo import Algo
from common.utils import Mapping
from monai.metrics import compute_roc_auc
from monai.utils import set_determinism

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
        self.act = None
        self.to_onehot = None
        self.model_dir = None

    def train(self, X, y, models, rank):
        epoch_num = 4

    def train(self):
        # """## Set deterministic training for reproducibility"""
        set_determinism(seed=0)
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
            epoch_loss = 0
            step = 0
            for batch_data in self.train_loader:
                step += 1
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}/{len(self.train_ds) // self.train_loader.batch_size}, train_loss: {loss.item():.4f}")
                epoch_len = len(self.train_ds) // self.train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            self.model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in self.val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, self.model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)

                y_onehot = self.to_onehot(y[None, ...]).T
                y_pred_act = self.act(y_pred)
                auc_metric = compute_roc_auc(y_pred_act, y_onehot)
                del y_pred_act, y_onehot
                metric_values.append(auc_metric)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if auc_metric > best_metric:
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1
                    best_model = self.model.state_dict()
                #    torch.save(self.model.state_dict(), os.path.join(self.model_dir, "best_metric_model.pth"))
                #    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current AUC: {auc_metric:.4f}"
                    f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        checkpoint = Mapping()
        #np_arr = torch_tensor
        checkpoint.update(epoch=best_metric_epoch, weights=best_model, metric=best_metric)
        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        return checkpoint


    def load_model(self, client):
        path = client.modelFile
        self.model.load_state_dict(torch.load(path))
        print("model loaded and creating report...")

    def save_model(self, checkpoint, trunkModelFile):
        torch.save(checkpoint, trunkModelFile)
        print("model saved at: ", trunkModelFile)
        # json.dump(model, path)

    def predict(self, client, class_names):
        set_determinism(seed=0)
        device = torch.device(DEVICE) 
        self.load_model(client)
        self.model.to(device)
        self.model.eval()

        y_true = []
        y_pred = []
        with torch.no_grad():
            for test_data in self.test_loader:
                test_images, test_labels = (
                    test_data[0].to(device),
                    test_data[1].to(device),
                )
                pred = self.model(test_images).argmax(dim=1)
                for i in range(len(pred)):
                    y_true.append(test_labels[i].item())
                    y_pred.append(pred[i].item())
            
        test_report = Mapping()
        test_report.update(report=classification_report(
        y_true, y_pred, target_names=class_names, digits=4))
        
        return test_report
