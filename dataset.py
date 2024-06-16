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
from config import *




class BrainTumorDataset(Dataset):    

    def __init__(self, annotations_file, img_dir):
        with open(annotations_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        
        self.lines = lines
        self.img_dir = img_dir

        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        file_name, label = self.lines[idx].split(",")
        label = int(label)
        
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert('RGB')

        transform = transforms.Compose([
        transforms.Resize((224, 224)), # tuy shape input
        transforms.ToTensor()])

        
        image = transform(image)

        return image, label
    


