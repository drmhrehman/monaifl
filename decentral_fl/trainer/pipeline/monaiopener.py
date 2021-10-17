import os

import sys
sys.path.append('.')

import os
import pandas as pd
from trainer.pipeline.opener import Opener
import PIL
import numpy as np
import matplotlib.pyplot as plt
import torch


class MonaiOpener(Opener):
    def __init__(self, data_dir, num_class=0):
        self.data_dir = data_dir
        self.num_class = num_class
        self.class_names = None

    def data_summary(self, folders):
        self.class_names = folders
        self.num_class = len(self.class_names)
        # print("Root Directory (dataset): " + self.data_dir)

        image_files = [
            [
                os.path.join(self.data_dir, self.class_names[i], x)
                for x in os.listdir(os.path.join(self.data_dir, self.class_names[i]))
            ]
            for i in range(self.num_class)
        ]
        num_each = [len(image_files[i]) for i in range(self.num_class)]
        image_files_list = []
        image_class = []
        for i in range(self.num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        num_total = len(image_class)
        image_width, image_height = PIL.Image.open(image_files_list[0]).size

        print(f"Total image count: {num_total}")
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Label names: {self.class_names}")
        print(f"Label counts: {num_each}")

        plt.subplots(3, 3, figsize=(8, 8))
        for i, k in enumerate(np.random.randint(num_total, size=9)):
            im = PIL.Image.open(image_files_list[k])
            arr = np.array(im)
            plt.subplot(3, 3, i + 1)
            plt.xlabel(self.class_names[image_class[k]])
            plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
        plt.tight_layout()
        # plt.show()

    def get_x_y(self, folders, frac_val, frac_test):
        train_x = list()
        train_y = list()
        val_x = list()
        val_y = list()
        test_x = list()
        test_y = list()
        
        self.class_names = folders
        self.num_class = len(self.class_names)
        #print("Root Directory (dataset): " + self.data_dir)
        
        image_files = [
            [
                os.path.join(self.data_dir, self.class_names[i], x)
                for x in os.listdir(os.path.join(self.data_dir, self.class_names[i]))
            ]
            for i in range(self.num_class)
        ]
        num_each = [len(image_files[i]) for i in range(self.num_class)]
        image_files_list = []
        image_class = []
        for i in range(self.num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        num_total = len(image_class)

        for i in range(num_total):
            rann = np.random.random()
            if rann < frac_val:
                val_x.append(image_files_list[i])
                val_y.append(image_class[i])
            elif rann < (frac_val+frac_test):
                test_x.append(image_files_list[i])
                test_y.append(image_class[i])
            else:
                train_x.append(image_files_list[i])
                train_y.append(image_class[i])
        return (train_x, train_y, val_x, val_y, test_x, test_y)

    def save_predictions(self, y_pred, path):
        with open(path, 'w') as fp:
            y_pred.to_csv(fp, index=False)

    def get_predictions(self, path):
        return pd.read_csv(path)

    def fake_X(self, n_samples):
        return []  # compute random fake data

    def fake_y(self, n_samples):
        return []  # compute random fake data

    def get_X(self, folders):
        return [
            # print(folders)
            folders
        ]

    def get_y(self, folders):
        return [
            folders
            # print(folders)
            # pd.read_csv(folders)#os.path.join(folders, 'y.csv'))
            # for folder in folders
        ]

class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]
