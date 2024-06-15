# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim import SGD, Optimizer, Adam
from utils import *
from tqdm import tqdm
from config import *
from dataset import BrainTumorDataset
from architech import Classification

# parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
#                     help='initial learning rate', dest='lr')
# parser.add_argument('--outdir', type=str, help='folder to save denoiser and training log)')
# parser.add_argument('--epochs', default=10, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--arch', type=str, choices=CLASSIFIERS_ARCHITECTURES)
# parser.add_argument('--dataset', type=str, choices=DATASETS)
# parser.add_argument('--optimizer', default='Adam', type=str,
#                     help='SGD, Adam, or Adam then SGD', choices=['SGD', 'Adam', 'AdamThenSGD'])
# parser.add_argument('--gpu', default=None, type=str,
#                     help='id(s) for CUDA_VISIBLE_DEVICES')




os.environ['CUDA_VISIBLE_DEVICES'] = device

if not os.path.exists(OUTDIR_TRAIN):
    os.mkdir(OUTDIR_TRAIN)


train_dataset = BrainTumorDataset(ARGUMENT_PATH, ARGUMENT_DIR)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Classification().train().cuda()
criterion = nn.CrossEntropyLoss(size_average=None, reduce=None, reduction='mean').cuda()
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

acc_meter = AverageMeter()
losses_meter = AverageMeter()

best_acc = None

for epoch in tqdm(range(epochs)):
    for data in train_loader:
        imgs, labels = data
        imgs = imgs.cuda()
        labels = labels.cuda()
        output = model(imgs).cuda()
        loss = criterion(output, labels)
        acc = accuracy(output, labels)
        

        losses_meter.update(loss, imgs.shape[0])
        acc_meter.update(acc[0], imgs.shape[0])

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Loss: ", losses_meter.avg)
    print("acc: ", acc_meter.avg)

    if epoch % 40:
        torch.save(model.state_dict(), f"trained/ep{epoch}.pth")

        
    if not best_acc or acc.avg > best_acc:
        best_acc = acc.avg
        torch.save(model.state_dict(), r"trained/best.pth")

