import os
import sys
sys.path.append('.')
import torch as t
from monai.networks.nets import densenet121
from monai.transforms import (Activations, AddChannel, AsDiscrete, Compose, LoadImage, RandFlip, RandRotate, RandZoom,
    ScaleIntensity, ToTensor,)
from pipeline.monaiopener import MonaiOpener, MedNISTDataset
from pipeline.monaialgo import MonaiAlgo
from common.utils import Mapping

from pathlib import Path
import logging
logging.basicConfig(format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.NOTSET)

def instantiateMonaiAlgo(frac_val = 0.1, frac_test = 0.1, dataset_name='MedNIST1'):
    cwd = Path.cwd()
    print(cwd)
    datasetName = dataset_name
    data_path = '../data_provider/synthetic_dataset/src/'
    #data_path = os.path.join(cwd, "flnode")
    data_dir = os.path.join(data_path, datasetName)
    folders = os.listdir(data_dir)

    mo = MonaiOpener(data_dir)
    logger.info("----------------------------")
    logger.info("Dataset Summary")
    print("----------------------------")
    mo.data_summary(folders)
    train_x, train_y, val_x, val_y, test_x, test_y = mo.get_x_y(folders, frac_val, frac_test)
    logger.info(f"Training count: {len(train_x)}, Validation count: {len(val_x)}, Test count: {len(test_x)}")

    # getting class names
    class_names = mo.class_names
    ##transforms
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            ToTensor(),
        ]
    )

    val_transforms = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])

    # monai algorithm object
    ma = MonaiAlgo()

    ma.act = Activations(softmax=True)
    ma.to_onehot = AsDiscrete(to_onehot=True, n_classes=mo.num_class)

    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = t.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)

    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = t.utils.data.DataLoader(val_ds, batch_size=128, num_workers=2)

    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = t.utils.data.DataLoader(test_ds, batch_size=128, num_workers=2)

    # model initiliatization
    ma.model = densenet121(spatial_dims=2, in_channels=1, out_channels=mo.num_class)#.to(device)

    # model loss function
    ma.loss_function = t.nn.CrossEntropyLoss()

    # model optimizer
    ma.optimizer = t.optim.Adam(ma.model.parameters(), 1e-5)

    # training/validation/testing datasets
    ma.train_ds = train_ds
    ma.val_ds = val_ds
    ma.test_ds = test_ds

    # training/validation/testing data loaders
    ma.train_loader = train_loader
    ma.val_loader = val_loader
    ma.test_loader = test_loader
    
    return ma, class_names

if __name__ == '__main__':
    ma, class_names = instantiateMonaiAlgo()
    

