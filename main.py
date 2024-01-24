import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import gc
import cv2
import albumentations as A
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from typing import *
from src import *
from model import UNet

KIDNEY_1_DENSE = '/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/images'
KIDNEY_1_LABELS = '/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/labels'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == "__main__":
    
    model = UNet().to(DEVICE)
    unet = Trainer(model,
                   X_path='/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/images',
                   y_path='/kaggle/input/blood-vessel-segmentation/train/kidney_1_dense/labels',
                   device=DEVICE,
                   batch_size=4
                   )
    unet.train()
    test_dataset = TestDataset() # Создаем тестовый датасет для загрузки решения
    unet.submit(test_dataset) # Делаем предсказание на тестовой выборкеS
