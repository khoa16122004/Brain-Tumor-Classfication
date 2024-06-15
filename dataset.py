# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
from torchvision.io import read_image
from torchvision import transforms

from torchvision.transforms import ToTensor





class BrainTumorDataset(Dataset):    

    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir

        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
        transforms.Resize((224, 224)), # tuy shape input
        transforms.ToTensor()])

        
        image = transform(image)
        label = self.img_labels.iloc[idx, 1]

        return image, label
    

