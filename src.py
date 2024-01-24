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

def rle_encode(mask):
    """Представляет решение в формате соревнования. Принимает на вход маску
       из 0 и 1"""
    pixel = mask.flatten()
    pixel = np.concatenate([[0], pixel, [0]])
    run = np.where(pixel[1:] != pixel[:-1])[0] + 1
    run[1::2] -= run[::2]
    rle = ' '.join(str(r) for r in run)
    if rle == '':
        rle = '1 0'
    return rle

class DiceScore(nn.Module):
    """Метрика Surface Dice"""
    def __init__(self, smooth: float = 0.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, y_true, y_pred):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        intersection = (y_pred @ y_true)
        sum_ = (y_pred.sum() + y_true.sum())
        return (2 * intersection + self.smooth) / (sum_ + self.smooth + 1e-6)
    
class DiceLoss(nn.Module):
    """Лосс на основе Dice"""
    def __init__(self, smooth: float = 0.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, y_true, y_pred):
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        intersection = (y_pred @ y_true)
        sum_ = (y_pred.sum() + y_true.sum())
        return 1 - (2 * intersection + self.smooth) / (sum_ + self.smooth + 1e-4)

class KidneyDataset(Dataset):
    """Датасет с почками"""
    def __init__(self, 
                 features_path, 
                 features,
                 target_path=None, 
                 target=None,
                 augment=True,
                 test=False,
                 image_size=(1024, 768)
                ):
        super().__init__()
        self.features_path = features_path
        self.features = features
        self.test = test
        self.target_path = target_path
        self.target = target
        self.augment = augment
        self.height , self.width = image_size[::-1]
        self.transform = None
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3)
            ])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.test:
            X_path = os.path.join(self.features_path, self.features[idx])
            X = cv2.imread(X_path, cv2.IMREAD_GRAYSCALE) / 255.0
            X = cv2.resize(X, (self.height, self.width))
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            return X
        X_path = os.path.join(self.features_path, self.features[idx])
        X = cv2.imread(X_path, cv2.IMREAD_GRAYSCALE) / 255.0
        X = cv2.resize(X, (self.height, self.width))
        y_path = os.path.join(self.target_path, self.target[idx])
        y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        y = cv2.resize(y, (self.height, self.width))
        y = y / 255.0
        transformed = self.transform(image=X, mask=y)
        X = torch.tensor(transformed['image'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        y = torch.tensor(transformed['mask'], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return X, y

class Trainer:
    """
    API for model training and validation
    
    Parameters:
    
    model: torch.nn.Module - your model for training and validation
    X_path: str - path to folder with images for semantic segmentation (features)
    y_path: str - path to folder with segmented images (target)
    optimizer: Any - optimizer instance (like torch.optim.Adam, torch.optim.rmsprop etc)
    epochs: int - number of epochs for train
    batch_size: int - batch size
    device: Any - device for train and validation (CPU/GPU)
    val_size: float - fraction of labeled data for validation. Must be in interval (0, 1)
    seed: int - seed for pseudorandom generator
    
    """
    def __init__(self,
                 model: nn.Module, 
                 X_path: str,
                 y_path: str,
                 lr: int = 0.0001,
                 epochs: int = 5,
                 batch_size: int = 16,
                 device: Any = None,
                 val_size: float = 0.25,
                 seed: int = 0
                ):
        self.device = device if device else torch.device('cpu')
        torch.random.manual_seed(seed)
        self.model = model.to(self.device)
        self.X_path = X_path
        self.y_path = y_path
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.metric = DiceScore().to(self.device)
        self.loss = DiceLoss(smooth=0.05).to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.best_score = 0.0
        X_train, X_val, y_train, y_val = train_test_split(os.listdir(self.X_path),
                                                          os.listdir(self.y_path),
                                                          test_size=val_size,
                                                          random_state=seed
                                                         )
        
        self.train_data = KidneyDataset(X_path, X_train, y_path, y_train)
        self.val_data = KidneyDataset(X_path, X_val, y_path, y_val)
        self.train_dataloader = DataLoader(self.train_data, 
                                           batch_size=self.batch_size,
                                           shuffle=True
                                          )
        self.val_dataloader = DataLoader(self.val_data, 
                                         batch_size=self.batch_size,
                                         shuffle=False
                                        )
        
    def train_epoch_(self):
        #===================train_part=============================
        self.model.train()
        for iteration, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            X, y = batch
            preds = self.model(X)
            X = X.to(self.device)
            y = y.to(self.device)
            loss = self.loss(y, preds)
            if (iteration + 1) % 10 == 0:
                print('    {}    Batch {}   Loss = {:1.4f}'.format(datetime.now(), 
                                                                   iteration + 1,
                                                                   loss.item()
                                                                  )
                     )
            loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
        #===========================================================
        
        #====================Validation part========================
        self.model.eval()
        val_metric = 0.0
        with torch.no_grad():
            for iteration, batch in enumerate(self.val_dataloader):
                X, y = batch
                preds = self.model(X)
                val_metric += self.metric(y, preds).item()
            val_metric /= iteration + 1
            print('\n=========================================\n')
            print('    Validation Score = {:1.4f}'.format(val_metric))
            if val_metric > self.best_score:
                self.best_score = val_metric
                torch.save(self.model.state_dict(), '/kaggle/working/model.pt')
                print('Model improved. Saving...')
        #============================================================
    def train(self, epochs=5):
        for epoch in range(1, epochs + 1):
            print('   Epoch {}'.format(epoch))
            print('=========================================\n')
            self.train_epoch_()
    def submit(self, test_dataset):
        self.model.load_state_dict(torch.load('/kaggle/working/model.pt'))
        self.model.eval()
        submit_dict = {'id': [], 
                       'rle': []}
        for img in test_dataset:
            idx, X = img
            pred = self.model(X)
            pred = cv2.resize(pred
                              .squeeze(dim=(0, 1))
                              .cpu()
                              .detach()
                              .numpy(),
                              (912, 1303)
                             )
            pred = np.where(pred > 0.6, 1, 0)
            submit_dict['id'].append(idx)
            submit_dict['rle'].append(rle_encode(pred))
        submission = pd.DataFrame(data=submit_dict)
        display(submission)
        submission.to_csv('submission.csv', index=None)

class TestDataset(Dataset):
    """Тестовый датасет"""
    def __init__(self):
        super().__init__()
        self.idx = ['kidney_5_0000', 
                    'kidney_5_0001', 
                    'kidney_5_0002', 
                    'kidney_6_0000', 
                    'kidney_6_0001', 
                    'kidney_6_0002']
        self.paths1 = '/kaggle/input/blood-vessel-segmentation/test/kidney_5/images'
        self.paths2 = '/kaggle/input/blood-vessel-segmentation/test/kidney_6/images'
    def __getitem__(self, idx):
        if idx < 3:
            X_path = os.path.join(self.paths1, sorted(os.listdir(self.paths1))[idx])
        else:
            X_path = os.path.join(self.paths2, sorted(os.listdir(self.paths2))[idx % 3])
        X = cv2.imread(X_path, cv2.IMREAD_GRAYSCALE) / 255.0
        X = cv2.resize(X, (768, 1024))
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        return self.idx[idx], X
    def __len__(self):
        return 6
